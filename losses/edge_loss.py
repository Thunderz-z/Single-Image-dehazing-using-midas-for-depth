from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        grads = []
        for c in range(x.shape[1]):
            xc = x[:, c : c + 1]
            sobel_x = self.sobel_x.to(device=xc.device, dtype=xc.dtype)
            sobel_y = self.sobel_y.to(device=xc.device, dtype=xc.dtype)
            gx = F.conv2d(xc, sobel_x, padding=1)
            gy = F.conv2d(xc, sobel_y, padding=1)
            grads.append(torch.sqrt(gx * gx + gy * gy + 1e-8))
        return torch.cat(grads, dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = self._gradient(pred)
        target_g = self._gradient(target)
        return F.l1_loss(pred_g, target_g)
