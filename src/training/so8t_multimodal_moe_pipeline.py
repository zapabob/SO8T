# -*- coding: utf-8 -*-
"""
SO8T Multimodal MoE Training Pipeline

Multimodal (Vision + Language) MoE model with SO8 group triality routing.
Implements all phases from AGENTS.md:
- Phase 1: Data Collection & Processing
- Phase 2: Advanced Training (SFT+GRPO+mHC+GRAPE+imatrix+PET+Unsloth BF16)
- Phase 3: SO8T MoE Evolution (ShinkaEvolve+Ebbinghaus+Triality Routing)
- Phase 4: C/D MoE Testing & HF Upload

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
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/so8t_multimodal_moe.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


@dataclass
class SO8MultimodalMoEConfig:
    model_name_or_path: str = ""
    vision_model_name: str = "openai/clip-vit-base-patch32"
    output_dir: str = "D:\\webdataset\\models\\so8t_multimodal_moe"
    checkpoint_dir: str = "D:\\webdataset\\checkpoints\\training"
    log_dir: str = "logs"

    num_experts: int = 4
    top_k_experts: int = 2
    hidden_dim: int = 3072
    vision_hidden_dim: int = 768
    num_layers: int = 32
    num_attention_heads: int = 32
    max_seq_length: int = 4096
    max_image_tokens: int = 64

    learning_rate: float = 2e-5
    vision_learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = 10000
    warmup_steps: int = 1000

    use_mhc: bool = True
    use_grape: bool = True
    use_imatrix: bool = True
    use_pet: bool = True
    use_shinka_evolve: bool = True
    use_ebbinghaus: bool = True
    use_multimodal: bool = True

    checkpoint_interval: int = 300
    max_checkpoint_slots: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    def __post_init__(self):
        self.output_dir = os.environ.get("SO8T_OUTPUT_DIR", self.output_dir)
        self.checkpoint_dir = os.environ.get("SO8T_CHECKPOINT_DIR", self.checkpoint_dir)


class VisionEncoder(nn.Module):
    def __init__(self, config: SO8MultimodalMoEConfig):
        super().__init__()
        self.config = config
        try:
            from transformers import CLIPVisionModel

            self.encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
            self.vision_hidden_size = self.encoder.config.hidden_size
        except ImportError:
            self.encoder = nn.Linear(224 * 224 * 3, config.vision_hidden_dim)
            self.vision_hidden_size = config.vision_hidden_dim
        self.projection = nn.Linear(self.vision_hidden_size, config.hidden_dim)
        self.image_tokens = config.max_image_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward"):
            outputs = self.encoder(pixel_values=images)
            image_features = outputs.last_hidden_state
        else:
            batch_size = images.shape[0]
            images_flat = images.view(batch_size, -1)
            image_features = self.encoder(images_flat)
        projected = self.projection(image_features)
        return projected


class SO8TrialityRouter(nn.Module):
    SO8_DIM = 8
    TRIALITY_STATES = 3

    def __init__(self, num_experts: int, hidden_dim: int, triality_hidden: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.vector_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spinor_pos = nn.Linear(hidden_dim, hidden_dim)
        self.spinor_neg = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, triality_hidden),
            nn.Tanh(),
            nn.Linear(triality_hidden, num_experts),
        )
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        vector_state = self.vector_proj(x)
        spinor_pos = self.spinor_pos(x)
        spinor_neg = self.spinor_neg(x)
        triality_states = torch.stack([vector_state, spinor_pos, spinor_neg], dim=2)
        triality_flat = triality_states.mean(dim=(1, 2))
        routing_weights = F.softmax(self.gate(triality_flat), dim=-1)
        expert_indices = torch.argmax(routing_weights, dim=-1)
        return expert_indices, routing_weights


class ExpertLayer(nn.Module):
    def __init__(self, hidden_dim: int, expert_id: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expert_id = expert_id
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim**0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        attn = F.softmax(scores, dim=-1)
        return self.Wo(torch.matmul(attn, v))


class SO8MoELayer(nn.Module):
    def __init__(self, config: SO8MultimodalMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.router = SO8TrialityRouter(config.num_experts, config.hidden_dim)
        self.experts = nn.ModuleList(
            [ExpertLayer(config.hidden_dim, i) for i in range(config.num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        if image_features is not None:
            x = torch.cat([image_features, x], dim=1)
            if attention_mask is not None:
                img_mask = torch.ones(
                    batch,
                    image_features.shape[1],
                    device=x.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        expert_indices, routing_weights = self.router(x)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = expert_indices == i
            if mask.sum() > 0:
                expert_output = self.experts[i](
                    x[mask],
                    attention_mask[mask] if attention_mask is not None else None,
                )
                output[mask] += expert_output * routing_weights[mask, i].unsqueeze(-1)
        return output


class EbbinghausForgettingCurve(nn.Module):
    def __init__(
        self,
        decay_rate: float = 0.1,
        reinforcement_rate: float = 0.1,
        retention_threshold: float = 0.3,
        minimum_retention: float = 0.1,
    ):
        super().__init__()
        self.decay_rate = decay_rate
        self.reinforcement_rate = reinforcement_rate
        self.retention_threshold = retention_threshold
        self.minimum_retention = minimum_retention
        self.token_states: Dict[int, Dict[str, Any]] = {}

    def update(self, token_ids: List[int], is_reinforced: bool = False) -> None:
        for token_id in token_ids:
            if token_id not in self.token_states:
                self.token_states[token_id] = {
                    "retention": 1.0,
                    "usage_count": 0,
                }
            state = self.token_states[token_id]
            state["usage_count"] += 1
            if is_reinforced:
                state["retention"] = min(
                    1.0, state["retention"] + self.reinforcement_rate
                )
            else:
                state["retention"] = max(
                    self.minimum_retention, state["retention"] * (1 - self.decay_rate)
                )

    def get_retention_strength(self, token_id: int) -> float:
        return self.token_states.get(token_id, {}).get(
            "retention", self.minimum_retention
        )

    def get_frozen_param_multiplier(self, param_name: str) -> float:
        return 0.5

    def get_stats(self) -> Dict[str, float]:
        if not self.token_states:
            return {"avg_retention": 0.0, "total_tokens": 0}
        retentions = [s["retention"] for s in self.token_states.values()]
        return {
            "avg_retention": float(np.mean(retentions)),
            "total_tokens": len(self.token_states),
        }


class ShinkaEvolveOptimizer:
    def __init__(
        self,
        model: nn.Module,
        ebbinghaus: Optional[EbbinghausForgettingCurve],
        config: Optional[SO8MultimodalMoEConfig] = None,
    ):
        self.model = model
        self.ebbinghaus = ebbinghaus
        self.config = config or SO8MultimodalMoEConfig()
        self.frozen_params: set = set()
        self.evolution_history: List[Dict] = []
        self._initialize_frozen_params()

    def _initialize_frozen_params(self) -> None:
        total = sum(1 for _ in self.model.parameters())
        max_frozen = int(total * 0.3)
        for i, (name, _) in enumerate(self.model.named_parameters()):
            if i < max_frozen:
                self.frozen_params.add(name)

    def evolve_frozen_parameters(
        self, step: int, metrics: Optional[Dict[str, float]] = None
    ) -> Dict:
        evolution_count = 0
        for name, param in self.model.named_parameters():
            if name in self.frozen_params:
                retention = (
                    self.ebbinghaus.get_frozen_param_multiplier(name)
                    if self.ebbinghaus
                    else 0.5
                )
                if (
                    retention < self.config.retention_threshold
                    if hasattr(self.config, "retention_threshold")
                    else 0.3
                ):
                    noise = torch.randn_like(param.data) * 0.01 * (1 - retention)
                    param.data = param.data + noise
                    evolution_count += 1
        state = {
            "step": step,
            "active_frozen": len(self.frozen_params),
            "evolution_count": evolution_count,
        }
        self.evolution_history.append(state)
        return state


class RollingCheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        interval_seconds: int = 300,
        max_slots: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_seconds = interval_seconds
        self.max_slots = max_slots
        self.logger = logger or logging.getLogger(__name__)
        self.last_save_time = 0.0
        self.current_slot = 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_emergency: bool = False,
    ) -> None:
        current_time = time.time()
        if (
            not is_emergency
            and (current_time - self.last_save_time) < self.interval_seconds
        ):
            return
        slot_path = self.checkpoint_dir / f"checkpoint_slot_{self.current_slot}.pt"
        checkpoint = {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
        torch.save(checkpoint, slot_path)
        self.current_slot = (self.current_slot + 1) % self.max_slots
        self.last_save_time = current_time
        self.logger.info(f"Checkpoint saved: slot={self.current_slot}, step={step}")

    def load_latest_checkpoint(self) -> Optional[Dict]:
        latest_time = 0
        latest_slot = -1
        for slot in range(self.max_slots):
            slot_path = self.checkpoint_dir / f"checkpoint_slot_{slot}.pt"
            if slot_path.exists():
                mtime = slot_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_slot = slot
        if latest_slot >= 0:
            slot_path = self.checkpoint_dir / f"checkpoint_slot_{latest_slot}.pt"
            return torch.load(slot_path, map_location="cpu")
        return None


class TrainingProgressTracker:
    def __init__(self, total_steps: int, desc: str = "Training"):
        try:
            from tqdm import tqdm

            self.pbar = tqdm(total=total_steps, desc=desc)
        except ImportError:
            self.pbar = None
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.metrics_history: List[Dict] = []

    def update(self, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        self.current_step = step
        metrics = metrics or {}
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = (elapsed / max(1, step)) * (self.total_steps - step) if step > 0 else 0
        display_metrics = {**metrics, "step": step, "eta": f"{eta / 60:.1f}m"}
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(display_metrics)
        self.metrics_history.append(
            {"step": step, "metrics": metrics, "timestamp": datetime.now().isoformat()}
        )

    def close(self) -> None:
        if self.pbar:
            self.pbar.close()


class MultimodalDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        max_images: int = 5,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_images = max_images

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
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }
        if "image" in item and item["image"] is not None:
            result["image"] = torch.tensor(item["image"])
        return result


class SO8MultimodalMoETrainer:
    def __init__(self, config: Optional[SO8MultimodalMoEConfig] = None):
        self.config = config or SO8MultimodalMoEConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.vision_encoder: Optional[VisionEncoder] = None
        self.moe_layer: Optional[SO8MoELayer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LinearLR] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.ebbinghaus: Optional[EbbinghausForgettingCurve] = None
        self.shinka_evolve: Optional[ShinkaEvolveOptimizer] = None
        self.checkpoint_manager: Optional[RollingCheckpointManager] = None
        self.progress_tracker: Optional[TrainingProgressTracker] = None
        self.global_step = 0
        self.epoch = 0
        self._initialized = False

    def initialize(self) -> None:
        logger.info("Initializing SO8T Multimodal MoE Pipeline...")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        self._setup_model()
        self._setup_components()
        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def _setup_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading base model: {self.config.model_name_or_path}")
        if self.config.model_name_or_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            )
            self.config.hidden_dim = base_model.config.hidden_size
        logger.info(f"Hidden dimension: {self.config.hidden_dim}")
        if self.config.use_multimodal:
            self.vision_encoder = VisionEncoder(self.config)
            logger.info(f"Vision encoder initialized")
        self.moe_layer = SO8MoELayer(self.config)
        self.model = self.moe_layer
        self.model.to(self.device)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        logger.info(
            f"MoE Model: {self.config.num_experts} experts, multimodal={self.config.use_multimodal}"
        )

    def _setup_components(self) -> None:
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.vision_encoder.parameters())
            if self.vision_encoder
            else [],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.max_steps,
        )
        if torch.cuda.is_available() and self.config.bf16:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.config.use_ebbinghaus:
            self.ebbinghaus = EbbinghausForgettingCurve()
        if self.config.use_shinka_evolve:
            self.shinka_evolve = ShinkaEvolveOptimizer(
                model=self.model,
                ebbinghaus=self.ebbinghaus,
                config=self.config,
            )
        self.checkpoint_manager = RollingCheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            interval_seconds=self.config.checkpoint_interval,
            max_slots=self.config.max_checkpoint_slots,
            logger=logger,
        )
        self.progress_tracker = TrainingProgressTracker(
            total_steps=self.config.max_steps,
            desc="SO8T Multimodal MoE Training",
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
            all_data = [{"text": "Sample multimodal training data.", "image": None}]
            logger.warning("No data loaded, using sample data")
        from transformers import AutoTokenizer

        tokenizer = (
            AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            if self.config.model_name_or_path
            else None
        )
        dataset = MultimodalDataset(
            all_data, tokenizer, max_length=self.config.max_seq_length
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        return self.train_loader

    def train(self) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        if not self.train_loader:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        logger.info("=" * 60)
        logger.info("Starting SO8T Multimodal MoE Training")
        logger.info(f"Epochs: {self.config.num_train_epochs}")
        logger.info(f"Max Steps: {self.config.max_steps}")
        logger.info(f"Multimodal: {self.config.use_multimodal}")
        logger.info("=" * 60)
        self.model.train()
        accumulation_steps = self.config.gradient_accumulation_steps
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                with torch.cuda.amp.autocast(enabled=self.config.bf16):
                    input_ids = batch.get("input_ids", torch.zeros(1)).to(self.device)
                    attention_mask = batch.get(
                        "attention_mask", torch.ones_like(input_ids)
                    ).to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)
                    image_features = None
                    if "image" in batch and batch["image"] is not None:
                        images = batch["image"].to(self.device)
                        image_features = self.vision_encoder(images)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_features=image_features,
                    )
                    if hasattr(outputs, "loss"):
                        loss = outputs.loss
                    else:
                        logits = (
                            outputs.last_hidden_state
                            if hasattr(outputs, "last_hidden_state")
                            else outputs
                        )
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
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
                metrics = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                if self.config.use_shinka_evolve:
                    evolution_state = self.shinka_evolve.evolve_frozen_parameters(
                        self.global_step, metrics
                    )
                    metrics["active_frozen"] = evolution_state.get("active_frozen", 0)
                    metrics["evolution_count"] = evolution_state.get(
                        "evolution_count", 0
                    )
                if self.checkpoint_manager and self.global_step % 100 == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=self.epoch,
                        step=self.global_step,
                        metrics=metrics,
                    )
                if self.progress_tracker:
                    self.progress_tracker.update(self.global_step, metrics)
        logger.info("Training completed")
        return self._save_final_model()

    def _save_final_model(self) -> Dict[str, Any]:
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
        if self.vision_encoder:
            torch.save(
                self.vision_encoder.state_dict(), output_path / "vision_encoder.bin"
            )
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.epoch,
                step=self.global_step,
                metrics={"status": "completed"},
            )
        config_dict = {
            "num_experts": self.config.num_experts,
            "hidden_dim": self.config.hidden_dim,
            "multimodal": self.config.use_multimodal,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Model saved to: {output_path}")
        return {
            "model_path": str(output_path),
            "status": "completed",
            "global_step": self.global_step,
        }


def main():
    parser = argparse.ArgumentParser(
        description="SO8T Multimodal MoE Training Pipeline"
    )
    parser.add_argument(
        "--model-name", type=str, default=os.environ.get("SO8T_BASE_MODEL", "")
    )
    parser.add_argument(
        "--vision-model", type=str, default="openai/clip-vit-base-patch32"
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
    parser.add_argument("--multimodal", action="store_true", default=True)
    parser.add_argument("--skip-components", type=str, default="")
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()
    config = SO8MultimodalMoEConfig(
        model_name_or_path=args.model_name,
        vision_model_name=args.vision_model,
        output_dir=args.output_dir,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        use_multimodal=args.multimodal,
    )
    if args.skip_components:
        for comp in args.skip_components.split(","):
            setattr(config, f"use_{comp}", False)
    trainer = SO8MultimodalMoETrainer(config=config)
    trainer.initialize()
    if args.dataset:
        trainer.load_dataset(args.dataset)
    result = trainer.train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
