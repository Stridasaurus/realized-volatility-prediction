"""The frozen small-LSTM family. Spec: specs/model-harness/SPEC.md.

Family is frozen (manifesto s4/s8): Optuna tunes sizes within it; GRU/TCN are noted
as defensible equivalents and never swept.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallLSTM(nn.Module):
    """LSTM over a lookback window of feature vectors -> scalar log-target prediction."""

    def __init__(
        self, n_features: int, hidden: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, F) -> (B,)
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)
