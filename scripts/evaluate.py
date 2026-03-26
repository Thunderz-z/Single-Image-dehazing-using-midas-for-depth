from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.paired_dehaze_dataset import PairedDehazeDataset
from models import DepthGuidedAODNet
from utils.checkpoint import load_checkpoint
from utils.metrics import psnr, ssim, try_no_reference_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth-guided dehazing model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--split", type=str, choices=["reside", "ohaze", "both"], default="both")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--with-no-ref", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device, with_no_ref: bool = False) -> Dict[str, float]:
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_n = 0

    niqe_sum = 0.0
    brisque_sum = 0.0
    niqe_count = 0
    brisque_count = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)

        outputs = model(hazy, depth)
        pred = outputs["dehazed"]

        bsz = hazy.size(0)
        total_psnr += float(psnr(pred, clean).sum().item())
        total_ssim += float(ssim(pred, clean).sum().item())
        total_n += bsz

        if with_no_ref:
            for i in range(bsz):
                nr = try_no_reference_metrics(pred[i : i + 1])
                if nr is not None:
                    if "niqe" in nr:
                        niqe_sum += nr["niqe"]
                        niqe_count += 1
                    if "brisque" in nr:
                        brisque_sum += nr["brisque"]
                        brisque_count += 1

    result = {
        "psnr": total_psnr / max(total_n, 1),
        "ssim": total_ssim / max(total_n, 1),
    }

    if with_no_ref and niqe_count > 0:
        result["niqe"] = niqe_sum / niqe_count
    if with_no_ref and brisque_count > 0:
        result["brisque"] = brisque_sum / brisque_count

    return result


def build_dataset(cfg: Dict, split_name: str, image_size: int) -> PairedDehazeDataset:
    if split_name == "reside":
        return PairedDehazeDataset(
            hazy_dir=str(Path(cfg["data"]["reside_root"]) / "test" / "hazy"),
            clean_dir=str(Path(cfg["data"]["reside_root"]) / "test" / "GT"),
            depth_dir=str(Path(cfg["data"]["depth_cache_root"]) / "reside_test_hazy"),
            image_size=image_size,
            train=False,
        )

    return PairedDehazeDataset(
        hazy_dir=str(Path(cfg["data"]["ohaze_root"]) / "hazy"),
        clean_dir=str(Path(cfg["data"]["ohaze_root"]) / "GT"),
        depth_dir=str(Path(cfg["data"]["depth_cache_root"]) / "ohaze_hazy"),
        image_size=image_size,
        train=False,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location=device.type)

    beta_min = float(checkpoint.get("config", {}).get("model", {}).get("beta_min", 0.3))
    beta_max = float(checkpoint.get("config", {}).get("model", {}).get("beta_max", 2.0))

    model = DepthGuidedAODNet(beta_min=beta_min, beta_max=beta_max).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    splits = [args.split] if args.split != "both" else ["reside", "ohaze"]

    out = {}
    for split_name in splits:
        dataset = build_dataset(cfg, split_name, args.image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        out[split_name] = evaluate_loader(model, loader, device, with_no_ref=args.with_no_ref)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
