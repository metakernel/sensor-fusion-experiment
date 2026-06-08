# Sensor Fusion Experiment

A research implementation of a multimodal autoencoder fusing RGB camera and LiDAR range data from the Waymo Open Dataset.

## Overview

- Trains separate range-only, RGB-only, and fusion autoencoders.
- Uses a shared latent space for multimodal fusion with dual-modality reconstruction.
- Produces deterministic processed splits from a fixed dataset config and uses seeded training batches.
- Exposes an `xtask` CLI for setup, Google Cloud access, dataset listing/fetch/prepare/inspect/preview, model training, evaluation, comparison, export, and reporting.
- Stores generated data, credentials, and run artifacts outside version control in `.xtask/`, `data/`, and `artifacts/`.

> Note: `cargo xtask eval`, `compare`, `export`, and `report` load trained checkpoints through the `sfx-train` inference layer, so metrics, visualizations, comparisons, and reports reflect real `model.bin` reconstructions.

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

Evaluation loads the run's `model.bin`, runs checkpoint-backed inference over the selected split (`all`, `train`, `val`, or `test`), prints a metric table, and writes `<run_dir>/eval/<split>.{json,csv,md}`. Metrics include MSE, MAE, RMSE, PSNR, and SSIM for RGB reconstructions.

### 9. Compare runs

```bash
cargo xtask compare --runs artifacts/checkpoints/range/<run-a>,artifacts/checkpoints/fusion/<run-b> --out artifacts/reports/compare.md
```

Comparison prints a Markdown table of run summaries and optionally writes it with `--out`. When `<run_dir>/eval/*.json` exists, the table appends evaluation columns for the detected split, including RGB MSE/PSNR/SSIM and range MSE/PSNR.

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
- `xtask`: the operational CLI entry point for setup, data, auth, training, evaluation, export, comparison, and reporting tasks

See `docs/architecture.md` for the high-level crate map.

## Configuration

Configuration lives in `configs/`:

- `dataset.waymo.small.toml`: dataset name, raw/processed directories, tensor sizes, range channels, and split ratios
- `model.range-only.tiny.toml`: tiny range autoencoder shape (`kind = "range-only"`, `latent_dim = 128`)
- `model.rgb-only.tiny.toml`: tiny RGB autoencoder shape (`kind = "rgb-only"`, `latent_dim = 128`)
- `model.fusion.tiny.toml`: tiny fusion model shape (`kind = "fusion"`, `latent_dim = 128`, `z_modality = 256`)
- `train.range-only.toml`, `train.rgb-only.toml`, `train.fusion.toml`: baseline training entry points
- `train.debug.toml`: small fusion debug profile for fast smoke runs
- `train.nextai.toml`: additional training profile kept alongside the main presets
- `eval.default.toml`: evaluation configuration scaffold kept in the workspace for evaluation workflows

Key training parameters are grouped under `[train]` (`run_name`, `batch_size`, `learning_rate`, `epochs`, `seed`) and then reference dataset/model config files through `[dataset].config` and `[model].config`.

## Data Layout

Local state is intentionally outside version control.

### `.xtask/`

- `.xtask/gcloud/`: local auth state and cached credentials
- `.xtask/manifests/raw_files.json`: discovered Waymo objects from `dataset list`
- `.xtask/manifests/downloaded_files.json`: downloaded raw parquet inventory
- `.xtask/manifests/processed_samples.json`: processed tensor sample catalog
- `.xtask/manifests/extraction_summary.json`: extraction/preparation summary
- `.xtask/manifests/splits.json`: processed split assignments
- `.xtask/runs/run_index.json`: run history and statuses
- `.xtask/runs/latest.json`: pointer to the latest completed run

### `data/`

- `data/raw/waymo/training`, `validation`, `testing`: downloaded Waymo parquet files
- `data/processed/waymo-range-rgb-v1/train`, `val`, `test`: processed tensor dataset produced by `dataset prepare`
- `data/samples/`: local sample artifacts used by preparation utilities

### `artifacts/`

- `artifacts/checkpoints/<range|rgb|fusion>/<run-id>/`: training summaries, metrics, checkpoints, optimizer metadata, and previews
- `artifacts/previews/dataset/`: dataset grid previews from `dataset preview`
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
