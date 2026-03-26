from __future__ import annotations

import torch
import torch.nn as nn

from utils.metrics import ssim


class SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_val = ssim(pred, target)
        return (1.0 - ssim_val).mean()
