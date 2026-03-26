from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


MidasTransform = Callable[[np.ndarray], torch.Tensor]


def load_midas(model_type: str = "MiDaS_small", device: str = "cuda") -> Tuple[torch.nn.Module, MidasTransform]:
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return model, transform


def infer_depth_map(
    pil_image: Image.Image,
    midas_model: torch.nn.Module,
    midas_transform: MidasTransform,
    device: torch.device,
) -> torch.Tensor:
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    input_batch = midas_transform(rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(pil_image.height, pil_image.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    d_min = prediction.amin(dim=(1, 2), keepdim=True)
    d_max = prediction.amax(dim=(1, 2), keepdim=True)
    normalized = (prediction - d_min) / (d_max - d_min + 1e-8)
    return normalized.unsqueeze(1)


def save_depth_png(depth_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = depth_tensor.squeeze().detach().cpu().numpy()
    depth = np.clip(depth, 0.0, 1.0)
    depth_u16 = (depth * 65535.0).astype(np.uint16)
    Image.fromarray(depth_u16, mode="I;16").save(output_path)


def load_depth_png(depth_path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(depth_path), dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 65535.0
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0)
