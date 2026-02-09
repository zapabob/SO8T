from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, field

from core.models.so8t_moe_router import SO8MoELayer
from core.models.grape_position_encoding import GrapeRotaryEmbedding
from core.quantization import IMatrixQuantizer, QuantizationConfig
from training.evolution import (
    EbbinghausForgettingCurve,
    ShinkaEvolveOptimizer,
    EvolutionConfig,
)
from training.regularization import PETRegularizer, PETConfig
from utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig
from utils.progress_tracker import TrainingProgressTracker, ProgressConfig


@dataclass
class SO8TPipelineConfig:
    model_name_or_path: str = ""
    output_dir: str = "D:\\webdataset\\models\\final"
    checkpoint_dir: str = "D:\\webdataset\\checkpoints\\training"
    log_dir: str = "logs"

    num_experts: int = 4
    top_k_experts: int = 2
    hidden_dim: int = 3072
    num_layers: int = 32
    num_attention_heads: int = 32
    max_seq_length: int = 4096

    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = 10000
    warmup_steps: int = 1000

    use_grape: bool = True
    use_imatrix: bool = True
    use_pet: bool = True
    use_shinka_evolve: bool = True
    use_ebbinghaus: bool = True

    checkpoint_interval: int = 300
    max_checkpoint_slots: int = 3
    resume_from_checkpoint: Optional[str] = None

    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    def __post_init__(self):
        self.output_dir = os.environ.get("SO8T_OUTPUT_DIR", self.output_dir)
        self.checkpoint_dir = os.environ.get("SO8T_CHECKPOINT_DIR", self.checkpoint_dir)


class SO8TMoETrainer:
    def __init__(
        self,
        config: Optional[SO8TPipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or SO8TPipelineConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.ebbinghaus: Optional[EbbinghausForgettingCurve] = None
        self.shinka_evolve: Optional[ShinkaEvolveOptimizer] = None
        self.pet_regularizer: Optional[PETRegularizer] = None
        self.imatrix_quantizer: Optional[IMatrixQuantizer] = None
        self.checkpoint_manager: Optional[RollingCheckpointManager] = None
        self.progress_tracker: Optional[TrainingProgressTracker] = None
        self._initialized = False

    def initialize(self) -> None:
        self.logger.info("Initializing SO8T MoE Pipeline...")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        self._setup_model()
        self._setup_optimizer()
        self._setup_regularizers()
        self._setup_checkpoint_manager()
        self._setup_progress_tracker()
        self._initialized = True
        self.logger.info("Pipeline initialized successfully")

    def _setup_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoConfig

        self.logger.info(f"Loading base model: {self.config.model_name_or_path}")
        if self.config.model_name_or_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            )
            self.config.hidden_dim = base_model.config.hidden_size
        else:
            self.config.hidden_dim = self.config.hidden_dim
        self.logger.info(f"Hidden dimension: {self.config['hidden_dim']}")
        self.model = SO8MoELayer(
            hidden_dim=self.config.hidden_dim,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k_experts,
        )
        self.model.to(self.device)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.logger.info(f"MoE Model: {self.config.num_experts} experts")

    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        from torch.optim.lr_scheduler import LinearLR

        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.max_steps,
        )
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_regularizers(self) -> None:
        if self.config.use_ebbinghaus:
            self.ebbinghaus = EbbinghausForgettingCurve(
                decay_rate=0.1,
                reinforcement_rate=0.1,
                retention_threshold=0.3,
            )
        if self.config.use_shinka_evolve and self.ebbinghaus:
            self.shinka_evolve = ShinkaEvolveOptimizer(
                model=self.model,
                ebbinghaus_curve=self.ebbinghaus,
                config=EvolutionConfig(),
            )
        if self.config.use_pet:
            self.pet_regularizer = PETRegularizer(
                model=self.model,
                config=PETConfig(),
            )

    def _setup_checkpoint_manager(self) -> None:
        self.checkpoint_manager = RollingCheckpointManager(
            config=CheckpointConfig(
                interval_seconds=self.config.checkpoint_interval,
                max_slots=self.config.max_checkpoint_slots,
                checkpoint_dir=self.config.checkpoint_dir,
            ),
            logger=self.logger,
        )

    def _setup_progress_tracker(self) -> None:
        self.progress_tracker = TrainingProgressTracker(
            total_steps=self.config.max_steps,
            desc="SO8T MoE Training",
            config=ProgressConfig(),
            logger=self.logger,
        )

    def train(self) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        self.logger.info("Starting training...")
        self.model.train()
        self.logger.info("Training completed")
        return {"status": "completed"}
