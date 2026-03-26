from __future__ import annotations

import torch
import torch.nn as nn


class BetaEstimator(nn.Module):
    def __init__(self, beta_min: float = 0.3, beta_max: float = 2.0) -> None:
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        dark = torch.min(rgb, dim=1, keepdim=True).values
        bright = torch.max(rgb, dim=1, keepdim=True).values

        mean_luma = luma.flatten(1).mean(dim=1, keepdim=True)
        std_luma = luma.flatten(1).std(dim=1, keepdim=True)
        mean_dark = dark.flatten(1).mean(dim=1, keepdim=True)
        mean_bright = bright.flatten(1).mean(dim=1, keepdim=True)

        stats = torch.cat([mean_luma, std_luma, mean_dark, mean_bright], dim=1)
        beta_norm = torch.sigmoid(self.mlp(stats))
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta_norm
        return beta.view(-1, 1, 1, 1)
