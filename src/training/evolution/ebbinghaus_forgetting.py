from __future__ import annotations

from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class TokenRetentionState:
    token_id: int
    retention_strength: float
    usage_count: int
    last_accessed: float
    reinforced_count: int = 0
    decayed_count: int = 0


class EbbinghausForgettingCurve(nn.Module):
    def __init__(
        self,
        decay_rate: float = 0.1,
        reinforcement_rate: float = 0.1,
        retention_threshold: float = 0.3,
        minimum_retention: float = 0.1,
        forgetting_half_life: float = 1000.0,
    ):
        super().__init__()
        self.decay_rate = decay_rate
        self.reinforcement_rate = reinforcement_rate
        self.retention_threshold = retention_threshold
        self.minimum_retention = minimum_retention
        self.forgetting_half_life = forgetting_half_life
        self.register_buffer("_global_time", torch.tensor(0.0))
        self._token_states: Dict[int, TokenRetentionState] = {}
        self._param_retention: Dict[str, float] = {}

    def update(
        self,
        token_ids: List[int],
        timestamps: Optional[List[float]] = None,
        is_reinforced: Optional[List[bool]] = None,
    ) -> None:
        if timestamps is None:
            timestamps = [float(self._global_time)] * len(token_ids)
        if is_reinforced is None:
            is_reinforced = [False] * len(token_ids)
        self._global_time += 1.0
        for token_id, ts, reinforced in zip(token_ids, timestamps, is_reinforced):
            if token_id not in self._token_states:
                self._token_states[token_id] = TokenRetentionState(
                    token_id=token_id,
                    retention_strength=1.0,
                    usage_count=0,
                    last_accessed=ts,
                )
            state = self._token_states[token_id]
            state.usage_count += 1
            state.last_accessed = ts
            if reinforced:
                state.retention_strength = min(
                    1.0,
                    state.retention_strength + self.reinforcement_rate,
                )
                state.reinforced_count += 1
            else:
                time_decay = math.exp(
                    -self.decay_rate
                    * (ts - state.last_accessed)
                    / self.forgetting_half_life
                )
                state.retention_strength = max(
                    self.minimum_retention,
                    state.retention_strength * (1 - self.decay_rate) * time_decay,
                )
                state.decayed_count += 1

    def get_retention_strength(self, token_id: int) -> float:
        if token_id in self._token_states:
            return self._token_states[token_id].retention_strength
        return self.minimum_retention

    def get_frozen_param_multiplier(
        self, param_name: str, base_strength: float = 0.5
    ) -> float:
        if param_name in self._param_retention:
            return self._param_retention[param_name] * base_strength
        return base_strength

    def set_param_retention(self, param_name: str, retention: float) -> None:
        self._param_retention[param_name] = retention

    def get_stats(self) -> Dict[str, float]:
        if not self._token_states:
            return {
                "avg_retention": 0.0,
                "total_tokens": 0,
                "high_retention_count": 0,
                "low_retention_count": 0,
            }
        retentions = [s.retention_strength for s in self._token_states.values()]
        return {
            "avg_retention": float(np.mean(retentions)),
            "total_tokens": len(self._token_states),
            "high_retention_count": sum(
                1 for r in retentions if r > self.retention_threshold
            ),
            "low_retention_count": sum(
                1 for r in retentions if r <= self.retention_threshold
            ),
            "reinforcement_rate": self.reinforcement_rate,
            "decay_rate": self.decay_rate,
        }

    def get_eviction_candidates(self, top_k: int = 100) -> List[int]:
        candidates = [
            (tid, s.retention_strength)
            for tid, s in self._token_states.items()
            if s.retention_strength < self.retention_threshold
        ]
        candidates.sort(key=lambda x: x[1])
        return [tid for tid, _ in candidates[:top_k]]

    def get_retention_vector(
        self, token_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        if token_ids is None:
            token_ids = list(self._token_states.keys())
        return torch.tensor(
            [self.get_retention_strength(tid) for tid in token_ids],
            dtype=torch.float32,
        )


import math
