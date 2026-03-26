from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.paired_dehaze_dataset import PairedDehazeDataset
from losses import CompositeDehazeLoss
from models import DepthGuidedAODNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import psnr, ssim
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune model on O-HAZE")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--output", type=str, default="checkpoints/best_ohaze_finetuned.pth")
    parser.add_argument("--log", type=str, default="checkpoints/ohaze_finetune_log.csv")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="OHaze-Train", leave=False):
        batch = to_device(batch, device)
        hazy, clean, depth = batch["hazy"], batch["clean"], batch["depth"]

        optimizer.zero_grad(set_to_none=True)
        out = model(hazy, depth)
        losses = criterion(
            hazy=hazy.float(),
            clean=clean.float(),
            dehazed=out["dehazed"].float(),
            transmission=out["transmission"].float(),
            depth=out["depth"].float(),
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(losses["total"].item()) * hazy.size(0)
        total_n += hazy.size(0)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_on_loader(model, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="OHaze-Eval", leave=False):
        batch = to_device(batch, device)
        hazy, clean, depth = batch["hazy"], batch["clean"], batch["depth"]

        out = model(hazy, depth)
        pred = out["dehazed"].clamp(0.0, 1.0)

        losses = criterion(
            hazy=hazy.float(),
            clean=clean.float(),
            dehazed=out["dehazed"].float(),
            transmission=out["transmission"].float(),
            depth=out["depth"].float(),
        )

        total_loss += float(losses["total"].item()) * hazy.size(0)
        total_psnr += float(psnr(pred, clean).sum().item())
        total_ssim += float(ssim(pred, clean).sum().item())
        total_n += hazy.size(0)

    return {
        "val_loss": total_loss / max(total_n, 1),
        "val_psnr": total_psnr / max(total_n, 1),
        "val_ssim": total_ssim / max(total_n, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(Path(args.checkpoint), map_location=device.type)
    beta_min = float(ckpt.get("config", {}).get("model", {}).get("beta_min", 0.3))
    beta_max = float(ckpt.get("config", {}).get("model", {}).get("beta_max", 2.0))

    model = DepthGuidedAODNet(beta_min=beta_min, beta_max=beta_max).to(device)
    model.load_state_dict(ckpt["model"])

    criterion = CompositeDehazeLoss(
        l1_weight=float(cfg["loss"]["l1_weight"]),
        ssim_weight=float(cfg["loss"]["ssim_weight"]),
        edge_weight=float(cfg["loss"]["edge_weight"]),
        depth_reg_weight=float(cfg["loss"]["depth_reg_weight"]),
        physics_weight=float(cfg["loss"]["physics_weight"]),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=float(cfg["train"]["weight_decay"]))

    dataset = PairedDehazeDataset(
        hazy_dir=str(Path(cfg["data"]["ohaze_root"]) / "hazy"),
        clean_dir=str(Path(cfg["data"]["ohaze_root"]) / "GT"),
        depth_dir=str(Path(cfg["data"]["depth_cache_root"]) / "ohaze_hazy"),
        image_size=args.image_size,
        train=True,
    )
    eval_dataset = PairedDehazeDataset(
        hazy_dir=str(Path(cfg["data"]["ohaze_root"]) / "hazy"),
        clean_dir=str(Path(cfg["data"]["ohaze_root"]) / "GT"),
        depth_dir=str(Path(cfg["data"]["depth_cache_root"]) / "ohaze_hazy"),
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    out_ckpt = Path(args.output)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_ssim = -1.0
    write_header = not log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"])
        if write_header:
            writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            stats = eval_on_loader(model, eval_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": stats["val_loss"],
                "val_psnr": stats["val_psnr"],
                "val_ssim": stats["val_ssim"],
            }
            writer.writerow(row)
            log_file.flush()

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_ssim": best_ssim,
                "config": cfg,
            }
            save_checkpoint(state, out_ckpt.with_name("last_ohaze_finetuned.pth"))

            if stats["val_ssim"] > best_ssim:
                best_ssim = stats["val_ssim"]
                state["best_ssim"] = best_ssim
                save_checkpoint(state, out_ckpt)

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"val_loss={stats['val_loss']:.4f} val_psnr={stats['val_psnr']:.3f} "
                f"val_ssim={stats['val_ssim']:.4f}"
            )


if __name__ == "__main__":
    main()
