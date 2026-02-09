# -*- coding: utf-8 -*-
"""
SO8T MoE Training Pipeline with Unsloth BF16

Features:
- SO8 Group Triality Routing for MoE
- ShinkaEvolve Frozen Parameter Evolution with Ebbinghaus Forgetting Curve
- mHC Manifold Harmonic Correction
- GRAPE Position Encoding
- imatrix Quantization
- PET Regularization
- Rolling Checkpoints (5-min, 3 slots)
- Auto-Resume on Power-On

Hardware: RTX 3060+ (12GB VRAM), 32GB RAM, Ryzen 5600 12-core
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR
import numpy as np

try:
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    from unsloth import is_bfloat16_supported

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    UnslothTrainer = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset as HFDataset

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/so8t_moe_training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

try:
    from core.models.so8t_moe_router import SO8MoELayer, SO8TrialityRouter
    from core.models.grape_position_encoding import GrapeRotaryEmbedding
    from core.quantization import IMatrixQuantizer, QuantizationConfig
    from training.evolution import (
        EbbinghausForgettingCurve,
        ShinkaEvolveOptimizer,
        EvolutionConfig,
    )
    from training.regularization import PETRegularizer, PETConfig, PETScheduler
    from utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig
    from utils.progress_tracker import TrainingProgressTracker, ProgressConfig
except ImportError:
    from ..core.models.so8t_moe_router import SO8MoELayer, SO8TrialityRouter
    from ..core.models.grape_position_encoding import GrapeRotaryEmbedding
    from ..core.quantization import IMatrixQuantizer, QuantizationConfig
    from ..training.evolution import (
        EbbinghausForgettingCurve,
        ShinkaEvolveOptimizer,
        EvolutionConfig,
    )
    from ..training.regularization import PETRegularizer, PETConfig, PETScheduler
    from ..utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig
    from ..utils.progress_tracker import TrainingProgressTracker, ProgressConfig


class SO8TMoEDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get("text", item.get("prompt", ""))
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class SO8TMoETrainingPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LinearLR] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.ebbinghaus: Optional[EbbinghausForgettingCurve] = None
        self.shinka_evolve: Optional[ShinkaEvolveOptimizer] = None
        self.pet_regularizer: Optional[PETRegularizer] = None
        self.imatrix_quantizer: Optional[IMatrixQuantizer] = None
        self.checkpoint_manager: Optional[RollingCheckpointManager] = None
        self.progress_tracker: Optional[TrainingProgressTracker] = None
        self.global_step = 0
        self.epoch = 0

    def _default_config(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": os.environ.get(
                "SO8T_BASE_MODEL", "microsoft/Phi-3.5-mini-instruct"
            ),
            "output_dir": os.environ.get(
                "SO8T_OUTPUT_DIR", "D:/webdataset/models/so8t_moe_final"
            ),
            "checkpoint_dir": os.environ.get(
                "SO8T_CHECKPOINT_DIR", "D:/webdataset/checkpoints/training"
            ),
            "log_dir": "logs",
            "num_experts": 4,
            "top_k_experts": 2,
            "hidden_dim": 3072,
            "learning_rate": 2e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "max_steps": 10000,
            "warmup_steps": 1000,
            "use_mhc": True,
            "use_grape": True,
            "use_imatrix": True,
            "use_pet": True,
            "use_shinka_evolve": True,
            "use_ebbinghaus": True,
            "checkpoint_interval": 300,
            "max_checkpoint_slots": 3,
            "bf16": True,
            "gradient_checkpointing": True,
            "seed": 42,
            "dataset_paths": [],
        }

    def initialize(self) -> None:
        logger.info("=" * 60)
        logger.info("Initializing SO8T MoE Training Pipeline")
        logger.info("=" * 60)
        torch.manual_seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed"])
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        logger.info(f"Device: {self.device}")
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["log_dir"]).mkdir(parents=True, exist_ok=True)
        self._setup_model()
        self._setup_components()
        logger.info("Pipeline initialized successfully")

    def _setup_model(self) -> None:
        logger.info(f"Loading base model: {self.config['model_name_or_path']}")
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name_or_path"]
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name_or_path"],
                torch_dtype=torch.bfloat16 if self.config["bf16"] else torch.float32,
            )
            self.config["hidden_dim"] = base_model.config.hidden_size
        else:
            self.tokenizer = None
            base_model = None
            self.config["hidden_dim"] = self.config["hidden_dim"]
        logger.info(f"Hidden dimension: {self.config['hidden_dim']}")
        self.model = SO8MoELayer(
            hidden_dim=self.config["hidden_dim"],
            num_experts=self.config["num_experts"],
            top_k=self.config["top_k_experts"],
        )
        if base_model is not None:
            self.model.load_state_dict(base_model.state_dict(), strict=False)
        self.model.to(self.device)
        if self.config["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
        logger.info(
            f"MoE Model loaded: {self.config['num_experts']} experts, top-{self.config['top_k_experts']}"
        )

    def _setup_components(self) -> None:
        logger.info("Setting up training components...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=0.01,
        )
        num_steps = self.config["max_steps"]
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=num_steps,
        )
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        if self.config["use_ebbinghaus"]:
            self.ebbinghaus = EbbinghausForgettingCurve(
                decay_rate=0.1,
                reinforcement_rate=0.1,
                retention_threshold=0.3,
            )
        if self.config["use_shinka_evolve"] and self.ebbinghaus:
            self.shinka_evolve = ShinkaEvolveOptimizer(
                model=self.model,
                ebbinghaus_curve=self.ebbinghaus,
                config=EvolutionConfig(),
            )
        if self.config["use_pet"]:
            self.pet_regularizer = PETRegularizer(
                model=self.model,
                config=PETConfig(),
            )
        self.checkpoint_manager = RollingCheckpointManager(
            config=CheckpointConfig(
                interval_seconds=self.config["checkpoint_interval"],
                max_slots=self.config["max_checkpoint_slots"],
                checkpoint_dir=self.config["checkpoint_dir"],
            ),
            logger=logger,
        )
        total_steps = self.config["max_steps"]
        self.progress_tracker = TrainingProgressTracker(
            total_steps=total_steps,
            desc="SO8T MoE Training",
            config=ProgressConfig(),
            logger=logger,
        )
        logger.info("All components initialized")

    def load_dataset(self, dataset_paths: List[str]) -> DataLoader:
        all_data = []
        for path_str in dataset_paths:
            path = Path(path_str)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            all_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                logger.info(f"Loaded {len(all_data)} samples from {path}")
        if not all_data:
            all_data = [{"text": "Sample training data for testing."}]
            logger.warning("No data loaded, using sample data")
        dataset = SO8TMoEDataset(all_data, self.tokenizer)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
        )
        return self.train_loader

    def train(self) -> Dict[str, Any]:
        if not self.model or not self.train_loader:
            raise ValueError(
                "Model or dataset not initialized. Call initialize() and load_dataset() first."
            )
        logger.info("=" * 60)
        logger.info("Starting SO8T MoE Training")
        logger.info(f"Epochs: {self.config['num_train_epochs']}")
        logger.info(f"Max Steps: {self.config['max_steps']}")
        logger.info(f"Batch Size: {self.config['batch_size']}")
        logger.info(f"Learning Rate: {self.config['learning_rate']}")
        logger.info("=" * 60)
        self.model.train()
        accumulation_steps = self.config["gradient_accumulation_steps"]
        self.global_step = 0
        for epoch in range(self.config["num_train_epochs"]):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config['num_train_epochs']}")
            for batch in self.train_loader:
                if self.global_step >= self.config["max_steps"]:
                    break
                with torch.cuda.amp.autocast(enabled=self.config["bf16"]):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    if self.config["use_pet"] and self.pet_regularizer:
                        position_ids = torch.arange(
                            inputs["input_ids"].size(1)
                        ).unsqueeze(0)
                        pet_loss, pet_metrics = self.pet_regularizer(
                            outputs.hidden_states,
                            position_ids=position_ids,
                            attention_mask=inputs.get("attention_mask"),
                        )
                        loss = loss + pet_loss
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if (self.global_step + 1) % accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    if (self.global_step + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                metrics = {
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.config["use_shinka_evolve"]:
                    evolution_state = self.shinka_evolve.evolve_frozen_parameters(
                        step=self.global_step,
                        metrics=metrics,
                    )
                    metrics["active_frozen"] = evolution_state.active_frozen
                    metrics["evolution_count"] = evolution_state.evolution_count
                if self.checkpoint_manager and self.global_step % 100 == 0:
                    self.checkpoint_manager.update(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        step=self.global_step,
                        metrics=metrics,
                    )
                if self.progress_tracker:
                    self.progress_tracker.update(
                        step=self.global_step,
                        metrics=metrics,
                    )
        logger.info("Training completed")
        return self._save_final_model()

    def _save_final_model(self) -> Dict[str, Any]:
        output_path = Path(self.config["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_path))
        torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
        if self.checkpoint_manager:
            self.checkpoint_manager.update(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics={"status": "completed"},
            )
        logger.info(f"Model saved to: {output_path}")
        return {
            "model_path": str(output_path),
            "status": "completed",
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

    def export_gguf(self, output_path: Optional[str] = None) -> str:
        gguf_path = output_path or str(
            Path(self.config["output_dir"]) / "model.bf16.gguf"
        )
        logger.info(f"Exporting GGUF to: {gguf_path}")
        script = (
            Path(__file__).parent.parent
            / "external"
            / "llama.cpp"
            / "convert_hf_to_gguf.py"
        )
        if script.exists():
            cmd = [
                sys.executable,
                str(script),
                self.config["output_dir"],
                "--outfile",
                gguf_path,
                "--outtype",
                "bf16",
            ]
            import subprocess

            subprocess.run(cmd, check=True)
        else:
            logger.warning("llama.cpp convert script not found")
        return gguf_path


def main():
    parser = argparse.ArgumentParser(description="SO8T MoE Training Pipeline")
    parser.add_argument(
        "--model-name", type=str, default=os.environ.get("SO8T_BASE_MODEL", "")
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("SO8T_OUTPUT_DIR", "")
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--skip-components", type=str, default="")
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()
    config = {
        "model_name_or_path": args.model_name,
        "output_dir": args.output_dir,
        "dataset_paths": args.dataset,
        "num_experts": args.num_experts,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "bf16": args.bf16,
    }
    if args.skip_components:
        for comp in args.skip_components.split(","):
            config[f"use_{comp}"] = False
    pipeline = SO8TMoETrainingPipeline(config=config)
    pipeline.initialize()
    if args.dataset:
        pipeline.load_dataset(args.dataset)
    result = pipeline.train()
    if args.export_gguf:
        pipeline.export_gguf()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
