# Sensor Fusion Experiment

A research implementation of a multimodal autoencoder fusing RGB camera and LiDAR range data from the Waymo Open Dataset.

## Overview

- Trains separate range-only, RGB-only, and fusion autoencoders.
- Uses a shared latent space for multimodal fusion with dual-modality reconstruction.
- Produces deterministic processed splits from a fixed dataset config and uses seeded training batches.
- Exposes an `xtask` CLI for setup, Google Cloud access, dataset listing/fetch/prepare/inspect/preview, model training, evaluation, comparison, compression benchmarking, export, and reporting.
- Stores generated data, credentials, and run artifacts outside version control in `.xtask/`, `data/`, and `artifacts/`.

> Note: `cargo xtask eval`, `compress-bench`, `compare`, `export`, and `report` load trained checkpoints through the `sfx-train` inference layer, so metrics, visualizations, comparisons, and reports reflect real `model.bin` reconstructions.

## Prerequisites

- Rust 1.80+ (`rustup install stable`)
- `cargo` (bundled with Rust)
- Google Cloud CLI (`gcloud`) for Waymo access
- Waymo Open Dataset access (apply at <https://waymo.com/open/>)
- Sufficient local disk for raw parquet files, processed tensors, and checkpoints

## Quick Start

### 1. Initialize the workspace

```bash
cargo xtask init
cargo xtask doctor
```

### 2. Authenticate with Google Cloud

```bash
cargo xtask gcloud auth
cargo xtask gcloud check
cargo xtask gcloud whoami
```

### 3. List available Waymo segments

```bash
cargo xtask dataset list --split training --limit 5
```

### 4. Download a small subset

```bash
cargo xtask dataset fetch --splits train,val --train-files 1 --val-files 1
```

### 5. Prepare processed tensors

Use both training and validation splits if you want validation loss during training.

```bash
cargo xtask dataset prepare --splits train,val --inspect
```

### 6. Inspect the processed dataset

```bash
cargo xtask dataset inspect
cargo xtask dataset preview
```

### 7. Train the baseline and fusion models

```bash
# Range-only baseline
cargo xtask train --config configs/train.range-only.toml

# RGB-only baseline
cargo xtask train --config configs/train.rgb-only.toml

# Fusion autoencoder
cargo xtask train --config configs/train.fusion.toml

# Small debug run
cargo xtask train --config configs/train.debug.toml
```

Each run creates a directory under `artifacts/checkpoints/<model-kind>/<run-id>/` containing `config.toml`, `dataset.toml`, `model.toml`, `metrics.jsonl`, `summary.json`, `model.bin`, `optimizer.json`, and `previews/`.

### 8. Evaluate trained runs

```bash
cargo xtask eval --run artifacts/checkpoints/fusion/<run-id> --split all
```

Evaluation loads the run's `model.bin`, runs checkpoint-backed inference over the selected split (`all`, `train`, `val`, or `test`), prints a metric table, and writes `<run_dir>/eval/<split>.{json,csv,md}`. Outputs include normalized proxy metrics (MSE/MAE/RMSE/PSNR/SSIM) plus physical depth reconstruction metrics for range (`depth_mae_m`, `depth_rmse_m`, `delta<1.25`, `delta<1.25^2`, and valid-pixel fraction).

### 9. Compare runs

```bash
cargo xtask compare --runs artifacts/checkpoints/range/<run-a>,artifacts/checkpoints/fusion/<run-b> --out artifacts/reports/compare.md
```

Comparison prints a Markdown table of run summaries and optionally writes it with `--out`. When `<run_dir>/eval/*.json` exists, the table appends evaluation columns for the detected split, including RGB and range proxy metrics plus range physical-depth metrics when available.

### 10. Export visualizations

```bash
cargo xtask export --run artifacts/checkpoints/fusion/<run-id> --split val --n 16 --out artifacts/exports/fusion-val
```

Export loads the checkpoint and writes triptych preview images to `--out`: `<sample>_rgb.ppm` for RGB reconstructions and/or `<sample>_range.pgm` for range reconstructions. Each triptych shows original, reconstructed, and absolute-error panels.

### 11. Generate reports

```bash
cargo xtask report --run artifacts/checkpoints/fusion/<run-id> --out artifacts/reports
```

Report generation writes `artifacts/reports/report.md` by default. The report includes training configuration, curves, final metrics, a `## Evaluation` section populated from `<run_dir>/eval/*.json`, and a `## Previews` section listing images under `<run_dir>/previews`.

### 12. Run compression benchmark (AE vs codec anchors)

```bash
cargo xtask compress-bench --run artifacts/checkpoints/fusion/<run-id> --split val --config configs/bench.compression.toml
```

`compress-bench` writes one JSON + Markdown pair per selected split to `artifacts/bench/compression/` by default: `<split>.json` and `<split>.md`. With `--split all`, it emits `train.*`, `val.*`, and `test.*`.

Key options:

- `--run <path-or-run-id>`: run directory (or run id) to load `model.bin`; if omitted, uses `.xtask/runs/latest.json`.
- `--split <all|train|val|test>`: overrides the split(s) from the config file.
- `--config <path>`: compression benchmark config (default `configs/bench.compression.toml`).
- Important overrides: `--out` (output directory), `--codecs` (codec/encoder list), `--crf` (applied to each selected codec), `--mode` (`independent` or `intra-sequence`), `--sample-cap`.

Methodology + caveats:

- **u8 reference domain**: headline AE and codec tables are computed in the packed 8-bit frame domain (not direct f32 tensor space). f32/depth context is retained in `ae_operating_points.*.notes` inside JSON output.
- **Range handling**: range frames are packed as RGB24 `[depth, channel-2-or-0, 0]`; only the first one/two channels are used for compression-benchmark quality metrics. Physical depth metrics always use channel 0 (depth), with `max_range_meters` from `normalization.json` when available.
- **Validity mask behavior**: depth metrics use a validity channel only when one is configured and still within the retained channels; otherwise valid pixels fall back to `depth_truth > 0`.
- **Per-frame vs sequence mode**: `independent` encodes each sample independently; `intra-sequence` concatenates split samples into one stream per modality (RGB and range are still encoded separately and their bitrates are summed). `intra-sequence` does not inject GOP/all-intra constraints, so temporal prediction is codec-dependent.
- **ffmpeg/encoder availability**: the command requires a working `ffmpeg` + encoder stack. Missing `ffmpeg` or unavailable encoders fail the run (no automatic codec skipping in `xtask compress-bench`).
- **Interpretation boundary**: codec curves are reference anchors in this u8 packed domain, while AE points include model reconstruction effects; combined rows average modality summaries (RGB/range) and matched-quality rows should be treated as directional framing, not as a definitive “best codec” claim.

## Project Structure

The repository is a Rust workspace with focused crates under `crates/`:

- `sfx-core`: shared manifest types, schema constants, and core errors
- `sfx-config`: typed TOML loading and validation for dataset, model, training, and evaluation configs
- `sfx-data`: processed dataset manifests, runtime loading, batching, and split access
- `sfx-waymo`: Waymo-specific discovery, download, and extraction utilities
- `sfx-preprocess`: camera/LiDAR alignment, normalization, resizing, and preview generation
- `sfx-models`: range-only, RGB-only, and fusion autoencoder definitions
- `sfx-train`: training loops, run directory management, checkpointing, inference, metrics, and previews
- `sfx-eval`: evaluation-related building blocks used by the wider workspace
- `sfx-tui`: terminal-oriented dataset exploration components
- `xtask`: the operational CLI entry point for setup, data, auth, training, evaluation, compression benchmarking, export, comparison, and reporting tasks

See `docs/architecture.md` for the high-level crate map.

## Configuration

Configuration lives in `configs/`:

- `dataset.waymo.small.toml`: default mixed protocol dataset profile (raw/processed dirs, tensor sizes, split ratios, and `.xtask/manifests/processed_samples.json`)
- `dataset.waymo.small.holdout.toml`: segment-disjoint holdout profile wired to `.xtask/manifests/processed_samples.segment_holdout.json`
- `model.range-only.tiny.toml`: tiny range autoencoder shape (`kind = "range-only"`, `latent_dim = 128`)
- `model.rgb-only.tiny.toml`: tiny RGB autoencoder shape (`kind = "rgb-only"`, `latent_dim = 128`)
- `model.fusion.tiny.toml`: tiny fusion model shape (`kind = "fusion"`, `latent_dim = 128`, `z_modality = 256`)
- `train.range-only.toml`, `train.rgb-only.toml`, `train.fusion.toml`: baseline mixed-protocol training entry points
- `train.range-only.holdout.toml`, `train.rgb-only.holdout.toml`, `train.fusion.holdout.toml`: segment-holdout retraining entry points
- `train.debug.toml`: small fusion debug profile for fast smoke runs
- `train.nextai.toml`: additional training profile kept alongside the main presets
- `eval.default.toml`: evaluation configuration scaffold kept in the workspace for evaluation workflows
- `bench.compression.toml`: compression benchmark sweep (codec list, CRF sweep, mode, split, quantization, and output directory)

Key training parameters are grouped under `[train]` (`run_name`, `batch_size`, `learning_rate`, `epochs`, `seed`) and then reference dataset/model config files through `[dataset].config` and `[model].config`.

## Data Layout

Local state is intentionally outside version control.

### `.xtask/`

- `.xtask/gcloud/`: local auth state and cached credentials
- `.xtask/manifests/raw_files.json`: discovered Waymo objects from `dataset list`
- `.xtask/manifests/downloaded_files.json`: downloaded raw parquet inventory
- `.xtask/manifests/processed_samples.json`: mixed-protocol processed tensor sample catalog
- `.xtask/manifests/processed_samples.segment_holdout.json`: segment-holdout processed tensor sample catalog
- `.xtask/manifests/extraction_summary.json`: extraction/preparation summary
- `.xtask/manifests/splits.json`: mixed sample-level split assignments
- `.xtask/manifests/splits.segment_holdout.json`: holdout sample-level split assignments
- `.xtask/runs/run_index.json`: run history and statuses
- `.xtask/runs/latest.json`: pointer to the latest completed run

### `data/`

- `data/raw/waymo/training`, `validation`, `testing`: downloaded Waymo parquet files
- `data/processed/waymo-range-rgb-v1/train`, `val`, `test`: processed tensor dataset produced by `dataset prepare`
  - per-sample `labels.json` sidecars (camera-front class counts/presence + validity-mask metadata)
  - root-level `labels_manifest.json` for efficient label lookup
  - root-level `normalization.json` documenting range/intensity/validity normalization semantics
- `data/samples/`: local sample artifacts used by preparation utilities

### `artifacts/`

- `artifacts/checkpoints/<range|rgb|fusion>/<run-id>/`: training summaries, metrics, checkpoints, optimizer metadata, and previews
- `artifacts/previews/dataset/`: dataset grid previews from `dataset preview`
- `artifacts/bench/compression/`: `compress-bench` split summaries (`<split>.json`, `<split>.md`)
- `artifacts/metrics/`, `artifacts/reports/`, `artifacts/tui_exports/`: initialized output roots for additional artifact types

## Running Tests

```bash
cargo test --workspace
```

`cargo xtask doctor` is also useful after setup to validate configs, manifests, and expected local directories.

## Architecture

See `docs/architecture.md`.

## License

MIT
