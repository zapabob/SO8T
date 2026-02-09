from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
from pathlib import Path


@dataclass
class EvolutionConfig:
    evolution_interval: int = 100
    mutation_scale: float = 0.01
    retention_threshold: float = 0.3
    max_frozen_ratio: float = 0.3
    manifold_scaling_factor: float = 1.0
    crossover_rate: float = 0.2
    elitism_count: int = 2


@dataclass
class EvolutionState:
    step: int
    active_frozen: int
    total_params: int
    evolution_count: int
    avg_retention: float
    manifold_scaling: float
    timestamp: float = field(default_factory=time.time)


class ShinkaEvolveOptimizer:
    def __init__(
        self,
        model: nn.Module,
        ebbinghaus_curve: nn.Module,
        config: Optional[EvolutionConfig] = None,
    ):
        self.model = model
        self.ebbinghaus = ebbinghaus_curve
        self.config = config or EvolutionConfig()
        self.frozen_params: Set[str] = set()
        self.evolution_history: List[EvolutionState] = []
        self._param_groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "retention": 0.5,
                "last_evolution": 0,
                "fitness": 0.0,
                "age": 0,
            }
        )
        self._initialize_frozen_params()

    def _initialize_frozen_params(self) -> None:
        total = sum(1 for _ in self.model.parameters())
        max_frozen = int(total * self.config.max_frozen_ratio)
        for i, (name, _) in enumerate(self.model.named_parameters()):
            if i < max_frozen:
                self.frozen_params.add(name)
                self._param_groups[name]["retention"] = 1.0

    def evolve_frozen_parameters(
        self, step: int, metrics: Optional[Dict[str, float]] = None
    ) -> EvolutionState:
        evolution_count = 0
        active_frozen = 0
        retentions = []
        for name, param in self.model.named_parameters():
            if name in self.frozen_params:
                active_frozen += 1
                retention = self.ebbinghaus.get_frozen_param_multiplier(
                    name,
                    self._param_groups[name]["retention"]
                    * self.config.manifold_scaling_factor,
                )
                retentions.append(retention)
                self._param_groups[name]["retention"] = retention
                if retention < self.config.retention_threshold:
                    if self._should_evolve(step, name):
                        self._mutate_parameter(param, retention, step)
                        evolution_count += 1
                        self._param_groups[name]["last_evolution"] = step
                        self._param_groups[name]["age"] = 0
                self._param_groups[name]["age"] += 1
        state = EvolutionState(
            step=step,
            active_frozen=active_frozen,
            total_params=sum(1 for _ in self.model.parameters()),
            evolution_count=evolution_count,
            avg_retention=float(np.mean(retentions)) if retentions else 0.0,
            manifold_scaling=self.config.manifold_scaling_factor,
        )
        self.evolution_history.append(state)
        self._adapt_manifold_scaling(metrics)
        return state

    def _should_evolve(self, step: int, param_name: str) -> bool:
        last_evo = self._param_groups[param_name]["last_evolution"]
        return (step - last_evo) >= self.config.evolution_interval

    def _mutate_parameter(
        self, param: nn.Parameter, retention: float, step: int
    ) -> None:
        mutation_scale = self.config.mutation_scale * (1 - retention)
        noise = torch.randn_like(param.data) * mutation_scale
        param.data = param.data + noise
        if param.grad is not None:
            param.grad = None

    def _adapt_manifold_scaling(
        self, metrics: Optional[Dict[str, float]] = None
    ) -> None:
        if metrics is None:
            return
        if "loss" in metrics and metrics["loss"] > 0.5:
            self.config.manifold_scaling_factor *= 1.05
        elif "loss" in metrics and metrics["loss"] < 0.1:
            self.config.manifold_scaling_factor *= 0.95
        self.config.manifold_scaling_factor = np.clip(
            self.config.manifold_scaling_factor, 0.5, 2.0
        )

    def get_frozen_param_names(self) -> List[str]:
        return list(self.frozen_params)

    def set_frozen_params(self, names: Set[str]) -> None:
        self.frozen_params = names

    def get_evolution_history(self) -> List[Dict]:
        return [
            {
                "step": s.step,
                "active_frozen": s.active_frozen,
                "evolution_count": s.evolution_count,
                "avg_retention": s.avg_retention,
                "manifold_scaling": s.manifold_scaling,
            }
            for s in self.evolution_history
        ]

    def save_state(self, path: Path) -> None:
        state = {
            "frozen_params": list(self.frozen_params),
            "evolution_history": self.get_evolution_history(),
            "config": {
                "evolution_interval": self.config.evolution_interval,
                "mutation_scale": self.config.mutation_scale,
                "retention_threshold": self.config.retention_threshold,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "manifold_scaling_factor": self.config.manifold_scaling_factor,
            },
            "param_groups": dict(self._param_groups),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        with open(path, "r") as f:
            state = json.load(f)
        self.frozen_params = set(state["frozen_params"])
        self._param_groups = defaultdict(
            lambda: {
                "retention": 0.5,
                "last_evolution": 0,
                "fitness": 0.0,
                "age": 0,
            }
        )
        for name, data in state.get("param_groups", {}).items():
            self._param_groups[name] = data
