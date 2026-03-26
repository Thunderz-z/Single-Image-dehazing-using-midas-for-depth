from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models import DepthGuidedAODNet
from utils.checkpoint import load_checkpoint
from utils.depth import infer_depth_map, load_depth_png, load_midas
from utils.visualization import save_comparison_panel, save_depth_map, save_transmission_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single image inference for depth-guided dehazing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/infer")
    parser.add_argument("--depth", type=str, default="")
    parser.add_argument("--midas-model", type=str, default="MiDaS_small")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _save_rgb_tensor(img: torch.Tensor, path: Path) -> None:
    arr = img.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), map_location=device.type)

    cfg = ckpt.get("config", {})
    beta_min = float(cfg.get("model", {}).get("beta_min", 0.3))
    beta_max = float(cfg.get("model", {}).get("beta_max", 2.0))

    model = DepthGuidedAODNet(beta_min=beta_min, beta_max=beta_max).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image = Image.open(args.input).convert("RGB")
    hazy_t = TF.to_tensor(image).unsqueeze(0).to(device)

    if args.depth:
        depth_path = Path(args.depth)
        if depth_path.suffix.lower() == ".npy":
            depth_np = np.load(depth_path)
            depth = torch.from_numpy(depth_np).float()
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
        else:
            depth = load_depth_png(depth_path)
        depth = depth.unsqueeze(0).to(device)
        depth = F.interpolate(depth, size=hazy_t.shape[-2:], mode="bilinear", align_corners=False)
    else:
        midas_model, midas_transform = load_midas(model_type=args.midas_model, device=device.type)
        depth = infer_depth_map(image, midas_model, midas_transform, device)

    with torch.no_grad():
        outputs = model(hazy_t, depth)

    dehazed = outputs["dehazed"][0].clamp(0.0, 1.0)
    _save_rgb_tensor(dehazed, output_dir / "dehazed.png")

    save_depth_map(outputs["depth"][0], output_dir / "depth_map.png")
    save_transmission_map(outputs["transmission"][0], output_dir / "transmission_map.png")
    save_comparison_panel(
        hazy=hazy_t[0],
        pred=dehazed,
        gt=None,
        output_path=output_dir / "comparison.png",
        title="Inference Result",
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
