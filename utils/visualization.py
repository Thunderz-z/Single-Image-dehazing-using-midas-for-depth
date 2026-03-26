from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def _to_numpy_gray(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image = image.detach().cpu().numpy()
    return np.clip(image, 0.0, 1.0)


def save_depth_map(depth: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy_gray(depth)
    plt.imsave(output_path.as_posix(), arr, cmap="magma")


def save_transmission_map(transmission: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy_gray(transmission)
    plt.imsave(output_path.as_posix(), arr, cmap="viridis")


def save_comparison_panel(
    hazy: torch.Tensor,
    pred: torch.Tensor,
    output_path: Path,
    gt: Optional[torch.Tensor] = None,
    title: str = "Dehazing Comparison",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hazy_np = _to_numpy_rgb(hazy)
    pred_np = _to_numpy_rgb(pred)
    gt_np = _to_numpy_rgb(gt) if gt is not None else None

    cols = 3 if gt_np is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(hazy_np)
    axes[0].set_title("Hazy Input")
    axes[0].axis("off")

    axes[1].imshow(pred_np)
    axes[1].set_title("Dehazed Output")
    axes[1].axis("off")

    if gt_np is not None:
        axes[2].imshow(gt_np)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)
