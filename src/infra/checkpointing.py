"""
Infrastructure helpers: rolling checkpoints and emergency saves.
"""
from __future__ import annotations

from typing import Optional

try:
    from utils.checkpoint_manager import RollingCheckpointManager, EmergencyCheckpointManager
except Exception:  # pragma: no cover
    RollingCheckpointManager = None
    EmergencyCheckpointManager = None


def build_checkpoint_manager(base_dir, max_keep: int = 5, save_interval_sec: int = 300):
    if RollingCheckpointManager is None:
        return None
    manager = RollingCheckpointManager(
        base_dir=base_dir,
        max_keep=max_keep,
        save_interval_sec=save_interval_sec,
    )
    return manager


def register_emergency_checkpoint(manager, model, tokenizer) -> Optional[object]:
    if manager is None or EmergencyCheckpointManager is None:
        return None
    emergency = EmergencyCheckpointManager(manager)
    emergency.register_model(model, tokenizer)
    return emergency
