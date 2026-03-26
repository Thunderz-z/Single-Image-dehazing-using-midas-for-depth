from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtmosphericConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def estimate_atmospheric_light(hazy: torch.Tensor) -> torch.Tensor:
        # Simple bright-pixel atmospheric light estimate per image and channel.
        return hazy.amax(dim=(2, 3), keepdim=True)

    def forward(self, hazy: torch.Tensor, dehazed: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
        transmission = transmission.clamp(0.05, 1.0)
        atmosphere = self.estimate_atmospheric_light(hazy)
        reconstructed_hazy = dehazed * transmission + atmosphere * (1.0 - transmission)
        return F.l1_loss(reconstructed_hazy, hazy)
