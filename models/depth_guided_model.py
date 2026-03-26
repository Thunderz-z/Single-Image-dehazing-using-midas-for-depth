from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.aod_net import AODNet
from models.beta_estimator import BetaEstimator


class DepthGuidedAODNet(nn.Module):
    def __init__(self, beta_min: float = 0.3, beta_max: float = 2.0) -> None:
        super().__init__()
        self.backbone = AODNet(in_channels=4)
        self.beta_estimator = BetaEstimator(beta_min=beta_min, beta_max=beta_max)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        depth = depth.clamp(0.0, 1.0)
        fused = torch.cat([rgb, depth], dim=1)

        dehazed = self.backbone(fused_input=fused, rgb=rgb)
        beta = self.beta_estimator(rgb)
        transmission = torch.exp(-beta * depth).clamp(0.0, 1.0)

        return {
            "dehazed": dehazed,
            "beta": beta,
            "transmission": transmission,
            "depth": depth,
        }
