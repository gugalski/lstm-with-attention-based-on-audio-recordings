from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .utils import ModelConfig


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # hidden: (batch, seq, hidden_dim)
        scores = self.v(torch.tanh(self.W(hidden))).squeeze(-1)  # (batch, seq)
        weights = torch.softmax(scores, dim=-1)                   # (batch, seq)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)     # (batch, hidden_dim)
        return context, weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        lstm_out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.attention = BahdanauAttention(lstm_out_dim)
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (batch, seq, input_size)
        out, _ = self.lstm(x)                      # (batch, seq, hidden_dim)
        context, weights = self.attention(out)
        context = self.dropout(self.norm(context))
        logits = self.classifier(context)           # (batch, num_classes)
        return logits, weights
