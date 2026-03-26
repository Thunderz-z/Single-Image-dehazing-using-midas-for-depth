from __future__ import annotations

import torch
import torch.nn as nn


class DepthRegularizationLoss(nn.Module):
    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def _grad_x(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, 1:] - x[:, :, :, :-1]

    @staticmethod
    def _grad_y(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def forward(self, dehazed: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        img_gray = dehazed.mean(dim=1, keepdim=True)
        dx_img = torch.abs(self._grad_x(img_gray))
        dy_img = torch.abs(self._grad_y(img_gray))

        dx_depth = torch.abs(self._grad_x(depth))
        dy_depth = torch.abs(self._grad_y(depth))

        weight_x = torch.exp(-self.alpha * dx_depth)
        weight_y = torch.exp(-self.alpha * dy_depth)

        loss = (dx_img * weight_x).mean() + (dy_img * weight_y).mean()
        return loss
