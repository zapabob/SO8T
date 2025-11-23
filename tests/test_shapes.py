import torch

from so8t_core.attention_so8 import SO8SelfAttention
from so8t_core.transformer import SO8TModel, SO8TModelConfig


def test_attention_shapes():
    attn = SO8SelfAttention(hidden_size=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.context.shape == (2, 10, 64)


def test_model_output_shape():
    model = SO8TModel(SO8TModelConfig(hidden_size=64, num_attention_heads=8, num_hidden_layers=2, intermediate_size=128))
    tokens = torch.ones(2, 16, dtype=torch.long)
    hidden, pet = model(tokens)
    assert hidden.shape == (2, 16, 64)
    assert pet.shape == ()
