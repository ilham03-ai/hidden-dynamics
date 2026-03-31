from __future__ import annotations

import torch
from torch import nn


class LatentTransition(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        delta = self.network(torch.cat([latent, action_one_hot], dim=-1))
        return latent + delta
