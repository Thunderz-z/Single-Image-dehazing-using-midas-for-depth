# Depth-Guided Single Image Dehazing (PyTorch)

This project implements a modular depth-guided dehazing system with:
- RESIDE-6K paired training/testing (ITS/SOTS proxy)
- MiDaS monocular depth precomputation
- 4-channel AOD-Net backbone (RGB + depth)
- Adaptive beta estimation from image brightness statistics
- Physics-inspired transmission prior: t(x) = exp(-beta * depth)
- Composite loss: L1 + SSIM + edge + depth regularization + atmospheric consistency
- Training, evaluation, inference, and visualization utilities

## Project Structure

- models/
- losses/
- datasets/
- utils/
- scripts/
- train.py
- infer.py
- config/default.yaml

## 1) Environment Setup

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Depth Precomputation (MiDaS)

This will download MiDaS weights on first run and write compressed 16-bit PNG depth maps.

```powershell
.\.venv\Scripts\python.exe scripts\precompute_depth.py --reside-root archive/RESIDE-6K --ohaze-root o-haze/O-HAZY --depth-cache-root artifacts/depth_cache --midas-model MiDaS_small
```

Depth cache folders created:
- artifacts/depth_cache/reside_train_hazy
- artifacts/depth_cache/reside_test_hazy
- artifacts/depth_cache/ohaze_hazy

## 3) Train

```powershell
.\.venv\Scripts\python.exe train.py --config config/default.yaml
```

Checkpoints and logs:
- checkpoints/last.pth
- checkpoints/best.pth
- checkpoints/train_log.csv

## 4) Evaluate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint checkpoints/best.pth --config config/default.yaml --split both --with-no-ref
```

## 5) Single Image Inference

```powershell
.\.venv\Scripts\python.exe infer.py --checkpoint checkpoints/best.pth --input path/to/hazy.jpg --output-dir outputs/infer
```

Outputs include:
- dehazed image
- depth map visualization
- transmission map visualization
- side-by-side comparison panel

## Disk Usage Note

Depth maps are stored as 16-bit PNG to control storage growth. This is much smaller than float arrays and helps keep total project space below constrained limits.
