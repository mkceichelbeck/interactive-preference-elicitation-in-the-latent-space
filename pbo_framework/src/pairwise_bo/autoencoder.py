"""Vanilla autoencoder components used by latent BO mode."""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    """Autoencoder for compressing and reconstructing listing feature vectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)
