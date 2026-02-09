from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SO8GroupTransform(nn.Module):
    SO8_DIM = 8
    TRIALITY_STATES = 3

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vector_projection = nn.Linear(hidden_dim, hidden_dim)
        self.positive_spinor = nn.Linear(hidden_dim, hidden_dim)
        self.negative_spinor = nn.Linear(hidden_dim, hidden_dim)
        self.triality_matrix = nn.Parameter(torch.randn(self.SO8_DIM, self.SO8_DIM))
        nn.init.orthogonal_(self.triality_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        vector_state = self.vector_projection(x)
        spinor_pos = self.positive_spinor(x)
        spinor_neg = self.negative_spinor(x)
        stacked = torch.stack([vector_state, spinor_pos, spinor_neg], dim=2)
        return stacked


class SO8TrialityRouter(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, triality_hidden: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.triality_transform = SO8GroupTransform(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, triality_hidden),
            nn.Tanh(),
            nn.Linear(triality_hidden, num_experts),
        )
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        triality_states = self.triality_transform(x)
        triality_flat = triality_states.mean(dim=(1, 2))
        routing_weights = F.softmax(self.gate(triality_flat), dim=-1)
        expert_indices = torch.argmax(routing_weights, dim=-1)
        return expert_indices, routing_weights


class ExpertLayer(nn.Module):
    def __init__(self, hidden_dim: int, expert_id: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expert_id = expert_id
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        attn = F.softmax(scores, dim=-1)
        return self.Wo(torch.matmul(attn, v))


class SO8MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = SO8TrialityRouter(num_experts, hidden_dim)
        self.experts = nn.ModuleList(
            [ExpertLayer(hidden_dim, i) for i in range(num_experts)]
        )
        self.noisy_std = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq, hidden = x.shape
        expert_indices, routing_weights = self.router(x)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (expert_indices == i).any(dim=-1)
            if mask.sum() > 0:
                expert_output = self.experts[i](x[mask], attention_mask)
                output[mask] += expert_output * routing_weights[mask, i].unsqueeze(-1)
        return output
