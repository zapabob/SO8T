"""
ALLOW / ESCALATE / DENY classifier head using triality outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LABELS = ("ALLOW", "ESCALATE", "DENY")


@dataclass
class TrialityOutput:
    logits: Tensor
    probabilities: Tensor
    predicted: Tensor

    def top_label(self) -> str:
        return LABELS[self.predicted.item()]


class TrialityHead(nn.Module):
    """
    Lightweight MLP head projecting hidden states to triality logits.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, len(LABELS))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden_state: Tensor, mask: Optional[Tensor] = None) -> TrialityOutput:
        if mask is not None:
            hidden_state = hidden_state * mask.unsqueeze(-1)
        pooled = hidden_state[:, -1]
        hidden = torch.tanh(self.linear(self.dropout(pooled)))
        logits = self.out(self.dropout(hidden))
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        return TrialityOutput(logits=logits, probabilities=probs, predicted=pred)
