# Benchmark Results (1500-Train Run, Multi-Segment Waymo)

## 1. Overview

This benchmark compares three autoencoder variants on synchronized Waymo RGB + LiDAR range data:

1. **Range-only autoencoder**
2. **RGB-only autoencoder**
3. **Fusion autoencoder** (shared latent for both modalities)

The benchmark was scaled to **1,500 training samples** (from 8 Waymo training segments), with held-out validation and test splits.  
At this scale, **single-modality baselines outperform fusion on their own modality**, while fusion remains reasonably close and still provides a unified cross-modal latent representation.

---

## 2. Dataset and Split Protocol

### Source data

- Waymo Open Dataset (training split files)
- Processed from **8 training segments**:
  - `10017090168044687777_6380_000_6400_000`
  - `10023947602400723454_1120_000_1140_000`
  - `1005081002024129653_5313_150_5333_150`
  - `10061305430875486848_1080_000_1100_000`
  - `10072140764565668044_4060_000_4080_000`
  - `10072231702153043603_5725_000_5745_000`
  - `10075870402459732738_1060_000_1080_000`
  - `10082223140073588526_6140_000_6160_000`

### Processed sample count

- Total paired samples extracted: **1,586**
- Logical split (seeded shuffle, seed 42):
  - **Train:** 1,500
  - **Validation:** 43
  - **Test:** 43

### Important split caveat

Validation/test are held-out frames, but sampled from the same pool of 8 training segments (not official Waymo val/test partitions and not unseen-segment evaluation).

---

## 3. Model/Run Configuration

All runs used latent dimension **128** and optimizer **Adam** on backend `burn-flex-autodiff`.

| Model | Run ID | Epochs | Batch Size | Train Samples | Duration |
| --- | --- | ---: | ---: | ---: | ---: |
| Range-only | `range_run_001_006` | 3 | 8 | 1500 | 00:36:13 |
| RGB-only | `rgb_run_001_007` | 3 | 4 | 1500 | 01:41:52 |
| Fusion | `fusion_run_001_006` | 2 | 4 | 1500 | 01:46:38 |

Fusion uses a shared bottleneck across both modalities (reported `z_modality = 256` in summary metadata).

---

## 4. Training-End Losses

From each run’s `summary.json`.

| Model | Final Train Loss | Final Validation Loss | Notes |
| --- | ---: | ---: | --- |
| Range-only | 0.008547 | 0.006402 | Loss is range MSE |
| RGB-only | 0.006043 | 0.005700 | Loss is RGB MSE |
| Fusion | 0.019837 | 0.015962 | Total loss = RGB MSE + Range MSE |

For fusion eval on validation:

- RGB MSE: **0.008859**
- Range MSE: **0.007105**
- Sum: **0.015964** (consistent with summary val loss)

---

## 5. Checkpoint-Backed Evaluation (Held-Out Splits)

Metrics below come from `cargo xtask eval` output files (`eval/val.json`, `eval/test.json`).

### 5.1 Validation (43 samples)

| Model | RGB MSE | RGB PSNR (dB) | RGB SSIM | Range MSE | Range PSNR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Range-only | n/a | n/a | n/a | 0.006403 | 22.243538 |
| RGB-only | 0.005701 | 23.043993 | 0.624985 | n/a | n/a |
| Fusion | 0.008859 | 21.171820 | 0.583736 | 0.007105 | 21.831455 |

### 5.2 Test (43 samples)

| Model | RGB MSE | RGB PSNR (dB) | RGB SSIM | Range MSE | Range PSNR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Range-only | n/a | n/a | n/a | 0.007703 | 21.511272 |
| RGB-only | 0.005012 | 23.515305 | 0.639857 | n/a | n/a |
| Fusion | 0.008752 | 21.281649 | 0.599379 | 0.008766 | 20.952612 |

---

## 6. Comparative Interpretation

### 6.1 Per-modality winners

- **Range reconstruction winner:** Range-only baseline
  - Val PSNR: 22.24 dB (vs Fusion 21.83 dB)
  - Test PSNR: 21.51 dB (vs Fusion 20.95 dB)
- **RGB reconstruction winner:** RGB-only baseline
  - Val PSNR: 23.04 dB (vs Fusion 21.17 dB)
  - Test PSNR: 23.52 dB (vs Fusion 21.28 dB)

### 6.2 What changed versus smaller-scale run

In the earlier small-data run (158 train), fusion improved range reconstruction over range-only.  
In this larger 1,500-train benchmark, that advantage disappears and specialists win on their own modality.

Practical interpretation: with enough data, dedicated single-modality decoders can exploit modality-specific structure better than a shared-latent compromise.

### 6.3 What fusion still provides

Fusion remains viable as a **single shared representation**:

- It reconstructs both modalities from one latent vector.
- It stays within ~0.6–2 dB of specialist baselines.
- It may still be useful when joint embedding simplicity is more important than peak per-modality fidelity.

---

## 7. Qualitative Outputs

Validation triptychs were exported from `fusion_run_001_006` to:

- `artifacts/exports/fusion-val/sample_000088_rgb.png`
- `artifacts/exports/fusion-val/sample_000088_range.png`
- `artifacts/exports/fusion-val/sample_000142_rgb.png`
- `artifacts/exports/fusion-val/sample_000142_range.png`

General qualitative behavior matches numeric metrics: coarse global scene structure is preserved, while fine texture/high-frequency boundaries remain blurred.

---

## 8. Reproducibility Commands

```bash
cargo xtask train --config configs/train.range-only.toml
cargo xtask train --config configs/train.rgb-only.toml
cargo xtask train --config configs/train.fusion.toml

cargo xtask eval --run artifacts/checkpoints/range/range_run_001_006 --split val
cargo xtask eval --run artifacts/checkpoints/range/range_run_001_006 --split test

cargo xtask eval --run artifacts/checkpoints/rgb/rgb_run_001_007 --split val
cargo xtask eval --run artifacts/checkpoints/rgb/rgb_run_001_007 --split test

cargo xtask eval --run artifacts/checkpoints/fusion/fusion_run_001_006 --split val
cargo xtask eval --run artifacts/checkpoints/fusion/fusion_run_001_006 --split test

cargo xtask export --run artifacts/checkpoints/fusion/fusion_run_001_006 --split val --n 6 --out artifacts/exports/fusion-val
```

---

## 9. Benchmark Limitations

1. Validation/test frames are held out but not from unseen segments.
2. Official Waymo validation/test partitions were not used.
3. Training horizon is short (2–3 epochs), so this is a strong prototype benchmark, not a full convergence study.
4. Reconstruction metrics (MSE/PSNR/SSIM) are proxy metrics and do not directly measure downstream perception performance.
