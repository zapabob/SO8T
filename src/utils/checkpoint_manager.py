from __future__ import annotations

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import threading
from dataclasses import dataclass, field
import logging
from datetime import datetime


@dataclass
class CheckpointConfig:
    interval_seconds: int = 300
    max_slots: int = 3
    checkpoint_dir: str = "D:\\webdataset\\checkpoints\\training"
    emergency_threshold_mb: int = 500


class RollingCheckpointManager:
    def __init__(
        self,
        config: Optional[CheckpointConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or CheckpointConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_save_time: float = 0.0
        self.current_slot: int = 0
        self.save_lock = threading.Lock()
        self.is_emergency: bool = False
        self._epoch: int = 0
        self._step: int = 0
        self._model_state: Optional[Dict[str, torch.Tensor]] = None
        self._optimizer_state: Optional[Dict[str, Any]] = None
        self._scheduler_state: Optional[Dict[str, Any]] = None
        self._scaler_state: Optional[torch.cuda.amp.GradScaler] = None
        self._metadata: Dict[str, Any] = {}

    def update(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._epoch = epoch
        self._step = step
        self._metadata = metadata or {}
        with self.save_lock:
            self._model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            if optimizer is not None:
                self._optimizer_state = optimizer.state_dict()
            if scheduler is not None:
                self._scheduler_state = scheduler.state_dict()
            if scaler is not None:
                self._scaler_state = scaler.state_dict()
        self._save_checkpoint(metrics)

    def _save_checkpoint(
        self, metrics: Optional[Dict[str, float]] = None, is_emergency: bool = False
    ) -> None:
        current_time = time.time()
        if (
            not is_emergency
            and (current_time - self.last_save_time) < self.config.interval_seconds
        ):
            return
        self._atomic_save()
        self.last_save_time = current_time
        self.is_emergency = False

    def _atomic_save(self) -> None:
        slot_path = self.checkpoint_dir / f"checkpoint_slot_{self.current_slot}.pt"
        metadata_path = (
            self.checkpoint_dir / f"checkpoint_slot_{self.current_slot}.json"
        )
        temp_path = self.checkpoint_dir / f"checkpoint_slot_{self.current_slot}.tmp"
        try:
            checkpoint = {
                "model_state": self._model_state,
                "optimizer_state": self._optimizer_state,
                "scheduler_state": self._scheduler_state,
                "scaler_state": self._scaler_state,
                "epoch": self._epoch,
                "step": self._step,
                "timestamp": datetime.now().isoformat(),
                "metadata": self._metadata,
            }
            torch.save(checkpoint, temp_path)
            temp_path.replace(slot_path)
            metadata = {
                "slot": self.current_slot,
                "timestamp": checkpoint["timestamp"],
                "epoch": self._epoch,
                "step": self._step,
                "metadata": self._metadata,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self._rotate_slots()
            self.logger.info(
                f"Checkpoint saved: slot={self.current_slot}, epoch={self._epoch}, step={self._step}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _rotate_slots(self) -> None:
        self.current_slot = (self.current_slot + 1) % self.config.max_slots

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        latest_slot = -1
        latest_time = 0
        for slot in range(self.config.max_slots):
            metadata_path = self.checkpoint_dir / f"checkpoint_slot_{slot}.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                timestamp = datetime.fromisoformat(metadata["timestamp"]).timestamp()
                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_slot = slot
        if latest_slot >= 0:
            return self._load_from_slot(latest_slot)
        return None

    def _load_from_slot(self, slot: int) -> Dict[str, Any]:
        slot_path = self.checkpoint_dir / f"checkpoint_slot_{slot}.pt"
        if slot_path.exists():
            return torch.load(slot_path, map_location="cpu")
        return {}

    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        info = []
        for slot in range(self.config.max_slots):
            metadata_path = self.checkpoint_dir / f"checkpoint_slot_{slot}.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info.append(json.load(f))
        return sorted(info, key=lambda x: x.get("timestamp", ""), reverse=True)

    def save_emergency(self, model: nn.Module, metrics: Dict[str, float]) -> None:
        self.is_emergency = True
        self._metadata["emergency"] = True
        self._metadata["metrics"] = metrics
        self._save_checkpoint(is_emergency=True)

    def cleanup_old_checkpoints(self, keep_slots: int = 1) -> None:
        info = self.get_checkpoint_info()
        if len(info) > keep_slots:
            for checkpoint in info[keep_slots:]:
                slot = checkpoint["slot"]
                for ext in [".pt", ".json"]:
                    path = self.checkpoint_dir / f"checkpoint_slot_{slot}{ext}"
                    if path.exists():
                        path.unlink()
