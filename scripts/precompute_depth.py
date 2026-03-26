from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys

from PIL import Image
from tqdm import tqdm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.depth import infer_depth_map, load_midas, save_depth_png


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTS:
            yield p


def process_folder(
    input_dir: Path,
    output_dir: Path,
    midas_model,
    midas_transform,
    device: torch.device,
    overwrite: bool,
    max_images: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(list_images(input_dir))
    if max_images > 0:
        images = images[:max_images]
    for image_path in tqdm(images, desc=f"Depth {input_dir.name}"):
        out_path = output_dir / f"{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        depth = infer_depth_map(image, midas_model, midas_transform, device)
        save_depth_png(depth[0], out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute MiDaS depth maps for dehazing datasets")
    parser.add_argument("--reside-root", type=str, default="archive/RESIDE-6K")
    parser.add_argument("--ohaze-root", type=str, default="o-haze/O-HAZY")
    parser.add_argument("--depth-cache-root", type=str, default="artifacts/depth_cache")
    parser.add_argument("--midas-model", type=str, default="MiDaS_small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, transform = load_midas(model_type=args.midas_model, device=device.type)

    depth_root = Path(args.depth_cache_root)

    process_folder(
        input_dir=Path(args.reside_root) / "train" / "hazy",
        output_dir=depth_root / "reside_train_hazy",
        midas_model=model,
        midas_transform=transform,
        device=device,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )

    process_folder(
        input_dir=Path(args.reside_root) / "test" / "hazy",
        output_dir=depth_root / "reside_test_hazy",
        midas_model=model,
        midas_transform=transform,
        device=device,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )

    if (Path(args.ohaze_root) / "hazy").exists():
        process_folder(
            input_dir=Path(args.ohaze_root) / "hazy",
            output_dir=depth_root / "ohaze_hazy",
            midas_model=model,
            midas_transform=transform,
            device=device,
            overwrite=args.overwrite,
            max_images=args.max_images,
        )

    print(f"Depth cache written to: {depth_root}")


if __name__ == "__main__":
    main()
