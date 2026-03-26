from __future__ import annotations

import random
from typing import Dict

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _resize_if_needed(image: Image.Image, size: int) -> Image.Image:
    if min(image.size) < size:
        return TF.resize(image, size=size)
    return image


def _resize_depth_if_needed(depth: torch.Tensor, size: int) -> torch.Tensor:
    h, w = depth.shape[-2:]
    if min(h, w) < size:
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return depth


def train_transform(sample: Dict[str, object], image_size: int) -> Dict[str, object]:
    hazy = sample["hazy"]
    clean = sample["clean"]
    depth = sample["depth"]

    hazy = _resize_if_needed(hazy, image_size)
    clean = _resize_if_needed(clean, image_size)
    depth = _resize_depth_if_needed(depth, image_size)

    i, j, h, w = T.RandomCrop.get_params(hazy, output_size=(image_size, image_size))
    hazy = TF.crop(hazy, i, j, h, w)
    clean = TF.crop(clean, i, j, h, w)
    depth = depth[:, i : i + h, j : j + w]

    if random.random() < 0.5:
        hazy = TF.hflip(hazy)
        clean = TF.hflip(clean)
        depth = torch.flip(depth, dims=[2])

    hazy_t = TF.to_tensor(hazy)
    clean_t = TF.to_tensor(clean)
    depth = depth.clamp(0.0, 1.0)

    return {"hazy": hazy_t, "clean": clean_t, "depth": depth}


def val_transform(sample: Dict[str, object], image_size: int) -> Dict[str, object]:
    hazy = TF.resize(sample["hazy"], size=[image_size, image_size])
    clean = TF.resize(sample["clean"], size=[image_size, image_size])

    depth = sample["depth"]
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return {
        "hazy": TF.to_tensor(hazy),
        "clean": TF.to_tensor(clean),
        "depth": depth.clamp(0.0, 1.0),
    }


def pil_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def collate_to_device(batch):
    keys = batch[0].keys()
    output = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            output[key] = [item[key] for item in batch]
    return output
