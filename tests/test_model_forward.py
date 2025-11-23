from __future__ import annotations

import torch

from agents.so8t.model import ModelConfig, build_model


def test_model_forward_pass() -> None:
    config = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.1,
        num_labels=3,
        max_seq_len=64,
        gate_order=["R_env", "R_safe", "R_cmd"],
    )
    model = build_model(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]
    pet_loss = outputs["pet_loss"]

    assert logits.shape == (4, config.num_labels)
    assert pet_loss.ndim == 0
    assert pet_loss.item() >= 0.0
