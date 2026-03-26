# Methodology

## 1. Introduction

This project implements a depth-guided single-image dehazing system in PyTorch. The implemented system combines RGB hazy input with monocular depth, predicts a dehazed image using a modified AOD-Net backbone, and constrains learning using a composite objective that includes reconstruction, structural, edge, depth-aware smoothness, and atmospheric consistency terms. The project includes depth precomputation, supervised training on paired hazy/clear data, checkpointed inference, quantitative evaluation, and final result packaging with visual artifacts and reports.

The implemented data sources in configuration and scripts are:
- RESIDE-6K for training and test evaluation.
- O-HAZY for additional evaluation and fine-tuning.

## 2. Methodology

### 2.1 Data Pair Construction and Loading

The paired dataset loader (`PairedDehazeDataset`) builds triplets `(hazy, clean, depth)` by matching image stems and a canonical numeric identifier.

Implemented matching logic:
- Candidate keys for matching include full stem (case-sensitive and lowercase), canonical stem prefix before underscore, and zero-padded canonical stem.
- Depth file is required at `depth_dir/<hazy_stem>.png`.
- Samples without matched clean image or missing depth map are excluded.

Data splits used by implemented loaders:
- Training: `archive/RESIDE-6K/train/hazy` + `archive/RESIDE-6K/train/GT` + `artifacts/depth_cache/reside_train_hazy`.
- Validation/Test: `archive/RESIDE-6K/test/hazy` + `archive/RESIDE-6K/test/GT` + `artifacts/depth_cache/reside_test_hazy`.
- O-HAZY evaluation/fine-tuning: `o-haze/O-HAZY/hazy` + `o-haze/O-HAZY/GT` + `artifacts/depth_cache/ohaze_hazy`.

Transform pipeline:
- Training transform:
  - Resize RGB and depth only when `min(H, W) < image_size`.
  - Joint random crop to `image_size x image_size` using one crop window for hazy/clean/depth.
  - Random horizontal flip with probability `0.5` applied consistently to hazy/clean/depth.
  - Tensor conversion for RGB and depth clamp to `[0, 1]`.
- Validation transform:
  - Deterministic resize of hazy/clean to `image_size x image_size`.
  - Bilinear interpolation of depth to the same size.
  - Depth clamp to `[0, 1]`.

### 2.2 Monocular Depth Precomputation

Depth maps are precomputed by `scripts/precompute_depth.py` using MiDaS loaded from `torch.hub`.

Implemented procedure:
- Load MiDaS model (`MiDaS_small` default) and transform set.
- For each hazy input image:
  - Convert RGB image to float array in `[0, 1]`.
  - Run MiDaS forward pass.
  - Resize prediction back to original image size with bicubic interpolation.
  - Min-max normalize depth map per image.
  - Save normalized depth as 16-bit PNG (`I;16`) via `save_depth_png`.

Depth cache directories produced by script:
- `artifacts/depth_cache/reside_train_hazy`
- `artifacts/depth_cache/reside_test_hazy`
- `artifacts/depth_cache/ohaze_hazy`

### 2.3 Network Architecture

The model class is `DepthGuidedAODNet`.

Implemented components:
- `AODNet(in_channels=4)` backbone.
- `BetaEstimator` to infer image-level scattering coefficient from brightness statistics.

#### 2.3.1 RGB-Depth Fusion

Depth is enforced to have channel dimension `B x 1 x H x W` and clamped to `[0, 1]`. Input to backbone is:

$$
X = [I_h, D] \in \mathbb{R}^{B \times 4 \times H \times W}
$$

where $I_h$ is hazy RGB and $D$ is normalized depth.

#### 2.3.2 AOD-Net Forward Structure

Backbone layers are:
- `conv1: 4 -> 3`, kernel $1 \times 1$
- `conv2: 3 -> 3`, kernel $3 \times 3$
- `conv3: 6 -> 3`, kernel $5 \times 5$
- `conv4: 6 -> 3`, kernel $7 \times 7$
- `conv5: 12 -> 3`, kernel $3 \times 3$
- ReLU after each convolution.

Concatenation flow:
- $x_1 = \mathrm{ReLU}(\mathrm{conv1}(X))$
- $x_2 = \mathrm{ReLU}(\mathrm{conv2}(x_1))$
- $x_3 = \mathrm{ReLU}(\mathrm{conv3}([x_1, x_2]))$
- $x_4 = \mathrm{ReLU}(\mathrm{conv4}([x_2, x_3]))$
- $k = \mathrm{ReLU}(\mathrm{conv5}([x_1, x_2, x_3, x_4]))$

Implemented reconstruction formula:

$$
J = k \odot I_h - k + 1
$$

where $J$ is dehazed output and $\odot$ is element-wise multiplication.

### 2.4 Adaptive Beta Estimation and Transmission Prior

`BetaEstimator` computes four per-image statistics from hazy RGB:
- Mean luma
- Luma standard deviation
- Mean dark channel proxy (`min` over RGB channels)
- Mean bright channel proxy (`max` over RGB channels)

Luma is implemented as:

$$
Y = 0.299R + 0.587G + 0.114B
$$

Let feature vector be:

$$
s = [\mu_Y, \sigma_Y, \mu_{\text{dark}}, \mu_{\text{bright}}] \in \mathbb{R}^{4}
$$

An MLP `4 -> 16 -> 1` with ReLU and sigmoid outputs normalized beta $\hat{\beta} \in (0,1)$, then scales to configured range:

$$
\beta = \beta_{\min} + (\beta_{\max} - \beta_{\min})\hat{\beta}
$$

with config values $\beta_{\min}=0.3$, $\beta_{\max}=2.0$.

Transmission prior is computed in model forward pass:

$$
t = \exp(-\beta D)
$$

and clamped to `[0, 1]` for output.

### 2.5 Loss Function and Optimization

Training objective is `CompositeDehazeLoss`:

$$
\mathcal{L}_{\text{total}} = \lambda_1\mathcal{L}_{L1} + \lambda_s\mathcal{L}_{SSIM} + \lambda_e\mathcal{L}_{edge} + \lambda_d\mathcal{L}_{depth} + \lambda_p\mathcal{L}_{phys}
$$

Configured weights:
- $\lambda_1 = 1.0$
- $\lambda_s = 0.3$
- $\lambda_e = 0.2$
- $\lambda_d = 0.1$
- $\lambda_p = 0.2$

Implemented terms:
- $\mathcal{L}_{L1}$: pixelwise L1 between dehazed output and clean target.
- $\mathcal{L}_{SSIM}$: `1 - SSIM` mean.
- $\mathcal{L}_{edge}$: L1 distance between Sobel gradient magnitudes of prediction and target, channel-wise.
- $\mathcal{L}_{depth}$: edge-aware image smoothness weighted by depth gradients:

$$
\mathcal{L}_{depth} = \mathbb{E}(|\partial_x J_g| e^{-\alpha |\partial_x D|}) + \mathbb{E}(|\partial_y J_g| e^{-\alpha |\partial_y D|}), \quad \alpha=10
$$

where $J_g$ is grayscale dehazed output.

- $\mathcal{L}_{phys}$: atmospheric consistency using reconstructed hazy image:

$$
A = \max_{h,w}(I_h), \qquad
\tilde{I}_h = J \odot t + A \odot (1-t), \qquad
\mathcal{L}_{phys} = \|\tilde{I}_h - I_h\|_1
$$

In this loss, transmission is clamped to `[0.05, 1.0]` before reconstruction.

Optimization setup from `train.py` and config:
- Optimizer: AdamW (`lr=2e-4`, `weight_decay=1e-4`).
- Epochs: `25` (baseline training config).
- Gradient clipping: max norm `1.0`.
- AMP: disabled in config (`amp: false`).
- Seed control: Python/NumPy/PyTorch deterministic setup via `seed_everything`.

### 2.6 Inference and Visualization

Single-image inference (`infer.py`) supports:
- External depth (`.png` or `.npy`) or on-the-fly MiDaS depth.
- Checkpoint-configured beta bounds.
- Output generation:
  - `dehazed.png`
  - `depth_map.png`
  - `transmission_map.png`
  - `comparison.png`

During validation and packaging scripts, visualization utilities write:
- Dehazed-vs-hazy-vs-GT panels.
- Depth colormap images.
- Transmission colormap images.

### 2.7 Evaluation and Metric Computation

`evaluate.py` computes dataset-level metrics for `reside`, `ohaze`, or both.

Implemented reference metrics:
- PSNR:

$$
\mathrm{PSNR} = 10\log_{10}\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}+\epsilon}\right), \quad \mathrm{MAX}=1
$$

- SSIM computed by Gaussian-window statistics with:
  - Window size `11`
  - $\sigma=1.5$
  - $C_1=0.01^2$, $C_2=0.03^2$

Implemented no-reference metric utility:
- Attempts NIQE and BRISQUE through PIQ version-dependent APIs.
- Saves only metrics successfully computed at runtime.

## 3. Mathematical Formulations

This section consolidates the equations directly implemented in model, losses, and metrics.

### 3.1 Variables

- $I_h \in [0,1]^{3 \times H \times W}$: hazy RGB input.
- $I_c \in [0,1]^{3 \times H \times W}$: clean ground truth.
- $D \in [0,1]^{1 \times H \times W}$: normalized depth.
- $J \in \mathbb{R}^{3 \times H \times W}$: network dehazed output.
- $k \in \mathbb{R}^{3 \times H \times W}$: AOD intermediate map.
- $\beta \in [\beta_{\min},\beta_{\max}]$: adaptive scattering coefficient.
- $t \in [0,1]^{1 \times H \times W}$: transmission map.
- $A \in \mathbb{R}^{3 \times 1 \times 1}$: estimated atmospheric light.

### 3.2 Implemented Equations

1. AOD reconstruction:

$$
J = k \odot I_h - k + 1
$$

2. Beta scaling from normalized MLP output:

$$
\beta = \beta_{\min} + (\beta_{\max} - \beta_{\min})\hat{\beta}
$$

3. Transmission prior:

$$
t = e^{-\beta D}
$$

4. Atmospheric consistency reconstruction:

$$
\tilde{I}_h = J \odot t + A \odot (1 - t)
$$

5. Composite objective:

$$
\mathcal{L}_{\text{total}} = 1.0\,\mathcal{L}_{L1} + 0.3\,\mathcal{L}_{SSIM} + 0.2\,\mathcal{L}_{edge} + 0.1\,\mathcal{L}_{depth} + 0.2\,\mathcal{L}_{phys}
$$

6. Depth-regularized smoothness:

$$
\mathcal{L}_{depth} = \mathbb{E}(|\partial_x J_g| e^{-10|\partial_x D|}) + \mathbb{E}(|\partial_y J_g| e^{-10|\partial_y D|})
$$

7. PSNR:

$$
\mathrm{PSNR} = 10\log_{10}\left(\frac{1}{\mathrm{MSE}+\epsilon}\right)
$$

8. SSIM loss term:

$$
\mathcal{L}_{SSIM} = 1 - \mathrm{SSIM}(J, I_c)
$$

### 3.3 Code Mapping

- Equations (1), (2), (3): `models/aod_net.py`, `models/beta_estimator.py`, `models/depth_guided_model.py`.
- Equation (4): `losses/physics_loss.py`.
- Equation (5): `losses/composite_loss.py`.
- Equation (6): `losses/depth_regularization_loss.py`.
- Equations (7), (8): `utils/metrics.py`, `losses/ssim_loss.py`.

## 4. Full Pipeline

The implemented workflow from raw hazy images to packaged outputs is:

1. Initialize experiment configuration.
- Load `config/default.yaml`.
- Set deterministic seeds.
- Resolve dataset roots, depth cache root, model and optimization hyperparameters.

2. Precompute depth maps.
- Run `scripts/precompute_depth.py`.
- For each hazy image in RESIDE train/test and O-HAZY hazy folders:
  - Infer depth with MiDaS.
  - Resize to source resolution.
  - Normalize to `[0,1]`.
  - Save 16-bit PNG in cache directory.

3. Build paired training and validation datasets.
- Match `(hazy, GT, depth)` triplets by stem/canonical id.
- Construct DataLoaders:
  - Train loader with random crop and random horizontal flip.
  - Validation loader with deterministic resizing.

4. Construct model and losses.
- Instantiate `DepthGuidedAODNet(beta_min=0.3, beta_max=2.0)`.
- Instantiate `CompositeDehazeLoss` with configured term weights.
- Instantiate AdamW optimizer.

5. Execute supervised training (`train.py`).
- For each epoch:
  - Forward pass on RGB + depth.
  - Compute composite loss in float32.
  - Backpropagate total loss.
  - Clip gradient norm to `1.0`.
  - Update parameters.
- Validate each epoch:
  - Run model on validation loader.
  - Compute validation loss, PSNR, SSIM.
  - Save visual artifacts for first batch when enabled.
- Save checkpoints:
  - `checkpoints/last.pth` every epoch.
  - `checkpoints/best.pth` when validation PSNR improves.
- Append epoch metrics to `checkpoints/train_log.csv`.

6. Evaluate checkpointed model (`scripts/evaluate.py`).
- Load checkpoint and reconstruct model.
- Build evaluation dataset for RESIDE and/or O-HAZY.
- Compute PSNR and SSIM for each split.
- Compute no-reference metrics when enabled and available.
- Save JSON metrics file in outputs directory.

7. Fine-tune on O-HAZY (`scripts/finetune_ohaze.py`).
- Load baseline checkpoint weights.
- Train and evaluate on O-HAZY data for 40 epochs.
- Save:
  - `checkpoints/last_ohaze_finetuned.pth`
  - `checkpoints/best_ohaze_finetuned.pth`
  - `checkpoints/ohaze_finetune_log.csv`

8. Run inference for single image (`infer.py`) and sample packaging (`scripts/package_results.py`).
- Inference writes dehazed result and depth/transmission visual maps.
- Packaging script writes final bundle:
  - quantitative metrics JSON and sample CSV
  - comparison/inference/maps images
  - training curves plot
  - consolidated text report

## 5. Results

This section reports only values present in workspace output artifacts.

### 5.1 Dataset-Level Evaluation Metrics

| Artifact | Split | PSNR | SSIM | BRISQUE |
|---|---|---:|---:|---:|
| `outputs/eval_metrics_full.json` | RESIDE | 20.6789 | 0.8730 | 24.4387 |
| `outputs/eval_metrics_full.json` | O-HAZY | 15.7868 | 0.6812 | 14.2745 |
| `outputs/eval_metrics_ohaze_finetuned.json` | RESIDE | 17.6483 | 0.8201 | 27.9318 |
| `outputs/eval_metrics_ohaze_finetuned.json` | O-HAZY | 18.7762 | 0.7599 | 19.3366 |

### 5.2 Training Log Summary

`checkpoints/train_log.csv` contains three logged training segments (restart pattern based on epoch reset to 1):

| Log File | Segment | Epochs | Best Epoch | Best Val PSNR | Best Val SSIM |
|---|---|---:|---:|---:|---:|
| `checkpoints/train_log.csv` | Segment 1 | 13 | 10 | 4.0174 | 0.3674 |
| `checkpoints/train_log.csv` | Segment 2 | 3 | 3 | 21.0080 | 0.8965 |
| `checkpoints/train_log.csv` | Segment 3 | 25 | 23 | 20.6819 | 0.8735 |
| `checkpoints/ohaze_finetune_log.csv` | O-HAZY Fine-tune | 40 | 40 | 18.7815 | 0.7600 |

### 5.3 Packaged Bundle Summaries

Values reported in generated final reports:

| Bundle | Epochs Reported | Best Validation Point | Sample Mean PSNR | Sample Mean SSIM |
|---|---:|---|---:|---:|
| `outputs/final_results_bundle_20260314_142906/final_report.txt` | 25 | epoch 23, PSNR 20.6819, SSIM 0.8735 | 19.3048 | 0.7453 |
| `outputs/final_results_bundle_20260314_150313/final_report.txt` | 40 | epoch 40, PSNR 18.7815, SSIM 0.7600 | 18.9142 | 0.7603 |

### 5.4 Packaged Sample-Level Metrics

Detailed per-sample metrics are stored in:
- `outputs/final_results_bundle_20260314_142906/quantitative/sample_metrics.csv`
- `outputs/final_results_bundle_20260314_150313/quantitative/sample_metrics.csv`

Each CSV row contains:
- dataset name (`reside` or `ohaze`)
- sample id
- PSNR
- SSIM
- SSIM percentage
- source hazy path
- matched GT path

## End-to-End Technical Explanation

The implemented end-to-end system operates as a depth-guided supervised dehazing pipeline.

1. Hazy images are first converted into paired supervised samples by matching each hazy image with a clean target and a cached depth map.
2. Depth is generated by MiDaS and normalized per image, then serialized as 16-bit PNG for reuse.
3. During model forward pass, RGB and depth are concatenated as a 4-channel tensor.
4. AOD-Net predicts an intermediate map `k`, and dehazed output is computed by the AOD reconstruction equation `J = k*I_h - k + 1`.
5. In parallel, an adaptive beta value is estimated from global brightness statistics; this beta and depth produce transmission `t = exp(-beta*D)`.
6. The network is trained against clean targets with a weighted composite objective that enforces pixel fidelity, structural similarity, edge consistency, depth-aware regularization, and atmospheric image formation consistency.
7. Training logs and checkpoints are written every epoch; best model selection is based on validation PSNR in baseline training and validation SSIM in O-HAZY fine-tuning script.
8. Evaluation computes PSNR/SSIM on dataset splits and stores results as JSON.
9. Packaging scripts run sample inference, save visual panels and maps, and produce final quantitative and textual reports.

This sequence represents the complete implemented workflow from raw hazy input folders to final numerical reports and qualitative comparison artifacts.