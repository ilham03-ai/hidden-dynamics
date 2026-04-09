from __future__ import annotations

import torch
from torch import nn


class UncertaintyHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.network(latent))
