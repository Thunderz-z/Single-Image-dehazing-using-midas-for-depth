from __future__ import annotations

import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fused_input: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(fused_input))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat([x2, x3], dim=1)
        x4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = self.relu(self.conv5(cat3))

        # Original AOD-Net reconstruction formula.
        dehazed = k * rgb - k + 1.0
        return dehazed
