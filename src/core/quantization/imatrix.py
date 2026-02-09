from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
import time
from collections import defaultdict


@dataclass
class QuantizationConfig:
    num_bins: int = 256
    imatrix_smooth: float = 1e-6
    clip_range: Tuple[float, float] = (-3.0, 3.0)
    quantize_embeddings: bool = True
    quantize_linear: bool = True


class IMatrixQuantizer:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None,
        calibration_batches: int = 100,
        batch_size: int = 8,
    ):
        self.model = model
        self.config = config or QuantizationConfig()
        self.calibration_batches = calibration_batches
        self.batch_size = batch_size
        self.importance_matrix: Optional[torch.Tensor] = None
        self.activation_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "mean": 0.0,
                "std": 1.0,
                "max": 0.0,
                "min": 0.0,
                "percentile_99": 1.0,
            }
        )
        self._calibration_data: List[torch.Tensor] = []

    def calibrate(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model.eval()
        self.model.to(device)
        all_activations: List[torch.Tensor] = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.calibration_batches:
                    break
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                activations = self._extract_activations(outputs)
                all_activations.append(activations)
        stacked = torch.cat(all_activations, dim=0)
        self._compute_importance_matrix(stacked)
        self._compute_activation_stats(stacked)

    def _extract_activations(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output.last_hidden_state
        return hidden.view(-1, hidden.size(-1))

    def _compute_importance_matrix(self, activations: torch.Tensor) -> None:
        abs_activations = activations.abs()
        per_token_importance = abs_activations.mean(dim=0)
        self.importance_matrix = per_token_importance / (
            per_token_importance.mean() + 1e-8
        )
        self.importance_matrix = torch.clamp(
            self.importance_matrix,
            self.config.clip_range[0],
            self.config.clip_range[1],
        )

    def _compute_activation_stats(self, activations: torch.Tensor) -> None:
        mean = activations.mean(dim=0)
        std = activations.std(dim=0) + self.config.imatrix_smooth
        max_val = activations.max(dim=0)[0]
        min_val = activations.min(dim=0)[0]
        percent_99 = np.percentile(activations.cpu().numpy(), 99, axis=0)
        self.activation_stats = {
            "mean": mean.cpu().numpy(),
            "std": std.cpu().numpy(),
            "max": max_val.cpu().numpy(),
            "min": min_val.cpu().numpy(),
            "percentile_99": percent_99,
        }

    def quantize_model(
        self,
        dtype: torch.dtype = torch.qint8,
        device: torch.device = torch.device("cuda"),
    ) -> nn.Module:
        quantized_model = self.model
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                importance = self.importance_matrix
                if module.weight.shape[1] != importance.shape[0]:
                    continue
                scale = module.weight.abs().max(dim=1)[0] / (importance + 1e-8)
                if dtype == torch.qint8:
                    qweight = torch.quantize_per_channel(
                        module.weight.data,
                        scales=scale,
                        zero_points=torch.zeros_like(scale),
                        axis=1,
                        dtype=dtype,
                    )
                    module.weight = nn.Parameter(qweight)
                else:
                    module.weight = nn.Parameter(
                        module.weight.data * scale.unsqueeze(0)
                    )
        return quantized_model

    def get_quantization_stats(self) -> Dict[str, Any]:
        return {
            "importance_matrix_shape": list(self.importance_matrix.shape)
            if self.importance_matrix is not None
            else None,
            "activation_stats": dict(self.activation_stats),
            "num_calibration_batches": self.calibration_batches,
            "batch_size": self.batch_size,
            "config": {
                "num_bins": self.config.num_bins,
                "clip_range": self.config.clip_range,
                "quantize_embeddings": self.config.quantize_embeddings,
                "quantize_linear": self.config.quantize_linear,
            },
        }

    def save_calibration(self, path: Path) -> None:
        state = {
            "importance_matrix": self.importance_matrix.cpu().numpy().tolist()
            if self.importance_matrix is not None
            else None,
            "activation_stats": self.activation_stats,
            "calibration_batches": self.calibration_batches,
            "batch_size": self.batch_size,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_calibration(self, path: Path) -> None:
        with open(path, "r") as f:
            state = json.load(f)
        if state["importance_matrix"] is not None:
            self.importance_matrix = torch.tensor(state["importance_matrix"])
        self.activation_stats = state["activation_stats"]
