from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from losses.depth_regularization_loss import DepthRegularizationLoss
from losses.edge_loss import EdgeLoss
from losses.physics_loss import AtmosphericConsistencyLoss
from losses.ssim_loss import SSIMLoss


class CompositeDehazeLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.3,
        edge_weight: float = 0.2,
        depth_reg_weight: float = 0.1,
        physics_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.depth_reg_weight = depth_reg_weight
        self.physics_weight = physics_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.depth_reg = DepthRegularizationLoss()
        self.physics = AtmosphericConsistencyLoss()

    def forward(
        self,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        dehazed: torch.Tensor,
        transmission: torch.Tensor,
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l1_loss = self.l1(dehazed, clean)
        ssim_loss = self.ssim(dehazed, clean)
        edge_loss = self.edge(dehazed, clean)
        depth_reg_loss = self.depth_reg(dehazed, depth)
        physics_loss = self.physics(hazy, dehazed, transmission)

        total = (
            self.l1_weight * l1_loss
            + self.ssim_weight * ssim_loss
            + self.edge_weight * edge_loss
            + self.depth_reg_weight * depth_reg_loss
            + self.physics_weight * physics_loss
        )

        return {
            "total": total,
            "l1": l1_loss,
            "ssim": ssim_loss,
            "edge": edge_loss,
            "depth_reg": depth_reg_loss,
            "physics": physics_loss,
        }
