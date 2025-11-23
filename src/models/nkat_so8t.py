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

class NKAT_SO8T_ThinkingModel(nn.Module):
    """
    NKAT-SO8T Thinking Model with Alpha Gate for Phase Transition.
    """
    def __init__(self, in_dim=32000, d_model=64, nhead=4, num_layers=2, out_dim=32000):
        super().__init__()
        assert d_model % 8 == 0
        self.embedding = nn.Embedding(in_dim, d_model) # Changed to Embedding for token inputs
        self.pos_encoder = PositionalEncoding(d_model)

        # Alpha Gate Parameter
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        
        # Orthogonality Loss Tracker (updated during forward or via hook, simplified here)
        self.ortho_loss = 0.0

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
        # x: (Batch, Seq) -> (Batch, Seq, d_model)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Reset ortho_loss for this batch
        self.ortho_loss = 0.0

        for layer in self.layers:
            # 1. Self-Attention
            residual = x
            x2, _ = layer['attn'](x, x, x)
            x = layer['norm1'](residual + x2)

            # 2. NKAT Block (Geometry + Physics)
            # We can use self.alpha here if NKAT_ThinkingBlock supports it, 
            # or just keep it as a parameter that influences the model via some mechanism.
            # For now, we assume NKAT_ThinkingBlock is standard, but we might want to 
            # inject alpha into it if the user intended that. 
            # Given the user's script just updates model.alpha, we'll assume it's used 
            # globally or we might need to pass it. 
            # However, the user said "Alpha Gate ... gains Mass Gap", implying it controls 
            # something. Let's assume it scales the NKAT block output or similar.
            # But the user's script treats it as a global parameter.
            # Let's just pass x through NKAT.
            
            # Note: Ideally we'd pass alpha to NKAT_ThinkingBlock, but the signature doesn't take it.
            # We will multiply the NKAT output by sigmoid(alpha) to simulate the "Gate" effect
            # if that's the intention, OR just leave it as a parameter that the optimizer sees.
            # The user said "Alpha Gate ... opens".
            # Let's implement a gate effect:
            
            gate = torch.sigmoid(self.alpha)
            nkat_out = layer['nkat'](x)
            
            # If alpha is -5, gate is ~0.006 (closed). If alpha is 1.618, gate is ~0.83 (open).
            x = layer['norm2'](x + nkat_out * gate)
            
            # Dummy ortho_loss calculation for demonstration
            # In a real scenario, this would come from the SO(8) matrices in NKAT blocks
            self.ortho_loss += torch.sum(layer['nkat'].so8_raw ** 2) * 0.001

        return self.decoder(x)
