import torch
import torch.nn as nn
import math
from ..layers.nkat_thinking import NKAT_ThinkingBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        return x + self.pe[:, :x.size(1), :]

class StandardTransformer(nn.Module):
    """
    Baseline: Vanilla Transformer Encoder
    Uses Standard Attention + Standard FFN (Linear-ReLU-Linear)
    """
    def __init__(self, in_dim=40, d_model=64, nhead=4, num_layers=2, out_dim=4):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (Batch, Seq=1, Dim=40) -> We treat input history as features here for simplicity
        # Or if input is (Batch, Seq=10, Dim=4), we embed properly.
        # To keep consistent with previous MLP setup: Input is flattened history (B, 40)
        # We project it to (B, 1, d_model) to act as a sequence.

        x = self.embedding(x).unsqueeze(1) # (B, 1, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Pooling
        return self.decoder(x)

class NKATTransformer(nn.Module):
    """
    NKAT: Physics-Augmented Transformer
    Uses Standard Attention + NKAT_ThinkingBlock (instead of Standard FFN)
    """
    def __init__(self, in_dim=40, d_model=64, nhead=4, num_layers=2, out_dim=4):
        super().__init__()
        assert d_model % 8 == 0
        self.embedding = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Custom Encoder Layers
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'nkat': NKAT_ThinkingBlock(d_model), # Replaces FFN
                'norm2': nn.LayerNorm(d_model)
            }))

        self.decoder = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # Embed: (B, 40) -> (B, 1, d_model)
        x = self.embedding(x).unsqueeze(1)
        x = self.pos_encoder(x)

        for layer in self.layers:
            # 1. Self-Attention
            residual = x
            x2, _ = layer['attn'](x, x, x)
            x = layer['norm1'](residual + x2)

            # 2. NKAT Block (Geometry + Physics)
            # NKAT block handles its own residual/norm inside, but let's be explicit
            x = layer['nkat'](x)

        x = x.mean(dim=1)
        return self.decoder(x)
