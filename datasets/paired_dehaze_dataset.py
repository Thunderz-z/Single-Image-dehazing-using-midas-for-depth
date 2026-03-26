from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import train_transform, val_transform
from utils.depth import load_depth_png


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


@dataclass
class PairRecord:
    hazy_path: Path
    clean_path: Path
    depth_path: Path


def _list_images(folder: Path) -> List[Path]:
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTS:
            files.append(p)
    return files


def _canonical_id(stem: str) -> str:
    return stem.split("_")[0].lstrip("0") or "0"


def _match_clean(hazy_path: Path, clean_index: Dict[str, Path]) -> Optional[Path]:
    candidates = [
        hazy_path.stem.lower(),
        hazy_path.stem,
        _canonical_id(hazy_path.stem),
        _canonical_id(hazy_path.stem).zfill(4),
    ]
    for key in candidates:
        if key in clean_index:
            return clean_index[key]
    return None


def _build_clean_index(clean_files: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for f in clean_files:
        keys = [f.stem, f.stem.lower(), _canonical_id(f.stem), _canonical_id(f.stem).zfill(4)]
        for key in keys:
            if key not in index:
                index[key] = f
    return index


class PairedDehazeDataset(Dataset):
    def __init__(
        self,
        hazy_dir: str,
        clean_dir: str,
        depth_dir: str,
        image_size: int = 384,
        train: bool = True,
    ) -> None:
        self.hazy_dir = Path(hazy_dir)
        self.clean_dir = Path(clean_dir)
        self.depth_dir = Path(depth_dir)
        self.image_size = image_size
        self.train = train

        hazy_files = _list_images(self.hazy_dir)
        clean_files = _list_images(self.clean_dir)
        clean_index = _build_clean_index(clean_files)

        records: List[PairRecord] = []
        missing_pairs = 0
        missing_depth = 0

        for hazy in hazy_files:
            clean = _match_clean(hazy, clean_index)
            if clean is None:
                missing_pairs += 1
                continue

            depth_path = self.depth_dir / f"{hazy.stem}.png"
            if not depth_path.exists():
                missing_depth += 1
                continue

            records.append(PairRecord(hazy_path=hazy, clean_path=clean, depth_path=depth_path))

        if not records:
            raise RuntimeError(
                f"No valid pairs found. Missing pairs: {missing_pairs}, missing depth: {missing_depth}. "
                f"Hazy dir={self.hazy_dir}, clean dir={self.clean_dir}, depth dir={self.depth_dir}"
            )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        hazy = Image.open(record.hazy_path).convert("RGB")
        clean = Image.open(record.clean_path).convert("RGB")
        depth = load_depth_png(record.depth_path)

        sample = {
            "hazy": hazy,
            "clean": clean,
            "depth": depth,
        }

        if self.train:
            sample = train_transform(sample, self.image_size)
        else:
            sample = val_transform(sample, self.image_size)

        sample["hazy_path"] = str(record.hazy_path)
        sample["clean_path"] = str(record.clean_path)
        return sample


def create_dataloaders(
    reside_root: str,
    depth_cache_root: str,
    image_size: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    reside_root = Path(reside_root)
    depth_cache_root = Path(depth_cache_root)

    train_dataset = PairedDehazeDataset(
        hazy_dir=(reside_root / "train" / "hazy").as_posix(),
        clean_dir=(reside_root / "train" / "GT").as_posix(),
        depth_dir=(depth_cache_root / "reside_train_hazy").as_posix(),
        image_size=image_size,
        train=True,
    )

    val_dataset = PairedDehazeDataset(
        hazy_dir=(reside_root / "test" / "hazy").as_posix(),
        clean_dir=(reside_root / "test" / "GT").as_posix(),
        depth_dir=(depth_cache_root / "reside_test_hazy").as_posix(),
        image_size=image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
