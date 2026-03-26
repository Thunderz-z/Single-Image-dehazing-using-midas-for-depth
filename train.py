from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW

from datasets import create_dataloaders
from losses import CompositeDehazeLoss
from models import DepthGuidedAODNet
from utils import psnr, ssim, seed_everything
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import save_comparison_panel, save_depth_map, save_transmission_map


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train depth-guided AOD-Net dehazing model")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs from config when > 0")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Limit train steps per epoch for debugging")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Limit val steps per epoch for debugging")
    return parser.parse_args()


def load_config(path: str) -> Dict:
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


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    criterion,
    device: torch.device,
    use_amp: bool,
    max_steps: int = 0,
) -> Dict[str, float]:
    model.train()
    loss_meter = RunningAverage()

    for step_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        batch = to_device(batch, device)
        hazy, clean, depth = batch["hazy"], batch["clean"], batch["depth"]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(hazy, depth)

        # Keep losses in float32 to improve AMP stability.
        losses = criterion(
            hazy=hazy.float(),
            clean=clean.float(),
            dehazed=outputs["dehazed"].float(),
            transmission=outputs["transmission"].float(),
            depth=outputs["depth"].float(),
        )

        if not torch.isfinite(losses["total"]):
            continue

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(float(losses["total"].item()), n=hazy.size(0))

        if max_steps > 0 and step_idx >= max_steps:
            break

    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool,
    save_visuals: bool,
    visual_dir: Path,
    epoch: int,
    max_steps: int = 0,
) -> Dict[str, float]:
    model.eval()

    loss_meter = RunningAverage()
    psnr_meter = RunningAverage()
    ssim_meter = RunningAverage()

    first_batch_saved = False

    for step_idx, batch in enumerate(tqdm(loader, desc="Val", leave=False), start=1):
        batch = to_device(batch, device)
        hazy, clean, depth = batch["hazy"], batch["clean"], batch["depth"]

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(hazy, depth)
        losses = criterion(
            hazy=hazy.float(),
            clean=clean.float(),
            dehazed=outputs["dehazed"].float(),
            transmission=outputs["transmission"].float(),
            depth=outputs["depth"].float(),
        )

        pred_clamped = outputs["dehazed"].clamp(0.0, 1.0)

        batch_psnr = psnr(pred_clamped, clean).mean().item()
        batch_ssim = ssim(pred_clamped, clean).mean().item()

        loss_meter.update(float(losses["total"].item()), n=hazy.size(0))
        psnr_meter.update(float(batch_psnr), n=hazy.size(0))
        ssim_meter.update(float(batch_ssim), n=hazy.size(0))

        if save_visuals and not first_batch_saved:
            visual_dir.mkdir(parents=True, exist_ok=True)
            sample_id = Path(batch["hazy_path"][0]).stem
            save_depth_map(outputs["depth"][0], visual_dir / f"epoch{epoch:03d}_{sample_id}_depth.png")
            save_transmission_map(
                outputs["transmission"][0], visual_dir / f"epoch{epoch:03d}_{sample_id}_transmission.png"
            )
            save_comparison_panel(
                hazy=hazy[0],
                pred=pred_clamped[0],
                gt=clean[0],
                output_path=visual_dir / f"epoch{epoch:03d}_{sample_id}_panel.png",
                title=f"Validation Epoch {epoch}",
            )
            first_batch_saved = True

        if max_steps > 0 and step_idx >= max_steps:
            break

    return {
        "val_loss": loss_meter.avg,
        "val_psnr": psnr_meter.avg,
        "val_ssim": ssim_meter.avg,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.get("seed", 42))

    if args.epochs > 0:
        cfg["train"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")

    train_loader, val_loader = create_dataloaders(
        reside_root=cfg["data"]["reside_root"],
        depth_cache_root=cfg["data"]["depth_cache_root"],
        image_size=int(cfg["data"]["image_size"]),
        train_batch_size=int(cfg["data"]["train_batch_size"]),
        val_batch_size=int(cfg["data"]["val_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )

    model = DepthGuidedAODNet(
        beta_min=float(cfg["model"]["beta_min"]),
        beta_max=float(cfg["model"]["beta_max"]),
    ).to(device)

    criterion = CompositeDehazeLoss(
        l1_weight=float(cfg["loss"]["l1_weight"]),
        ssim_weight=float(cfg["loss"]["ssim_weight"]),
        edge_weight=float(cfg["loss"]["edge_weight"]),
        depth_reg_weight=float(cfg["loss"]["depth_reg_weight"]),
        physics_weight=float(cfg["loss"]["physics_weight"]),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_path = save_dir / "train_log.csv"

    start_epoch = 1
    best_psnr = -1.0

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), map_location=device.type)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_psnr = float(ckpt.get("best_psnr", -1.0))

    write_header = not logs_path.exists()
    with open(logs_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"])
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                use_amp,
                max_steps=args.max_train_steps,
            )
            val_stats = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                save_visuals=bool(cfg["eval"]["save_visuals"]),
                visual_dir=Path(cfg["eval"]["visual_dir"]),
                epoch=epoch,
                max_steps=args.max_val_steps,
            )

            row = {"epoch": epoch, **train_stats, **val_stats}
            writer.writerow(row)
            log_file.flush()

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_psnr": best_psnr,
                "config": cfg,
            }
            save_checkpoint(state, save_dir / "last.pth")

            if val_stats["val_psnr"] > best_psnr:
                best_psnr = val_stats["val_psnr"]
                state["best_psnr"] = best_psnr
                save_checkpoint(state, save_dir / "best.pth")

            print(
                f"Epoch {epoch:03d} | train_loss={train_stats['train_loss']:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} val_psnr={val_stats['val_psnr']:.3f} "
                f"val_ssim={val_stats['val_ssim']:.4f}"
            )


if __name__ == "__main__":
    main()
