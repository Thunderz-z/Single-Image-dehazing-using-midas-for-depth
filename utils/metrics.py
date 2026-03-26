from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.repeat(channels, 1, 1, 1)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.flatten(1).mean(dim=1)


def try_no_reference_metrics(image: torch.Tensor) -> Optional[Dict[str, float]]:
    try:
        import piq
    except Exception:
        return None

    image = image.clamp(0.0, 1.0)
    out: Dict[str, float] = {}

    # NIQE availability differs across PIQ versions.
    if hasattr(piq, "niqe"):
        try:
            out["niqe"] = float(piq.niqe(image).mean().item())
        except Exception:
            pass
    elif hasattr(piq, "NIQE"):
        try:
            niqe_metric = piq.NIQE()
            out["niqe"] = float(niqe_metric(image).mean().item())
        except Exception:
            pass

    if hasattr(piq, "brisque"):
        try:
            out["brisque"] = float(piq.brisque(image).mean().item())
        except Exception:
            pass
    elif hasattr(piq, "BRISQUELoss"):
        try:
            brisque_metric = piq.BRISQUELoss(data_range=1.0)
            out["brisque"] = float(brisque_metric(image).mean().item())
        except Exception:
            pass

    if not out:
        return None
    return out
