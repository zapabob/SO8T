from __future__ import annotations

import torch

from agents.so8t.model import SO8Gate


def test_so8_gate_preserves_shape_and_orthogonality() -> None:
    gate = SO8Gate(d_model=32, order=["R_safe", "R_cmd", "R_env"])
    inputs = torch.randn(2, 5, 32)
    outputs = gate(inputs)
    assert outputs.shape == inputs.shape

    for label in ["R_env", "R_safe", "R_cmd"]:
        matrix = gate._matrix(label)
        identity = torch.matmul(matrix, matrix.transpose(-1, -2))
        diff = torch.max(torch.abs(identity - torch.eye(8).to(identity)))
        assert diff < 1e-5
