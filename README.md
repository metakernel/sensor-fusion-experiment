# Sensor Fusion Experiment

A research implementation of a multimodal autoencoder fusing RGB camera and LiDAR range data from the Waymo Open Dataset.

## Overview

- Trains separate range-only, RGB-only, and fusion autoencoders.
- Uses a shared latent space for multimodal fusion with dual-modality reconstruction.
- Produces deterministic processed splits from a fixed dataset config and uses seeded training batches.
- Exposes an `xtask` CLI for setup, Google Cloud access, dataset listing/fetch/prepare/inspect/preview, and model training.
- Stores generated data, credentials, and run artifacts outside version control in `.xtask/`, `data/`, and `artifacts/`.

> Note: this branch does **not** expose top-level `cargo xtask eval`, `compare`, `export`, or `report` commands in `crates/xtask/src/main.rs`. Training runs still emit machine-readable metrics, summaries, checkpoints, and preview images that support manual evaluation and comparison.

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

There is no standalone `cargo xtask eval` command on this branch yet. Instead, inspect the generated run artifacts:

- `summary.json` for final loss, run metadata, dataset/model config paths, and output locations
- `metrics.jsonl` for per-epoch train/validation losses
- `previews/` for qualitative reconstructions written during training

### 9. Compare runs

There is no standalone `cargo xtask compare` subcommand yet. Compare the `summary.json` and `metrics.jsonl` files from multiple run directories to review loss curves and final metrics side by side.

### 10. Export visualizations

Dataset previews are available today via:

```bash
cargo xtask dataset preview --split train --count 16 --out artifacts/previews/dataset
```

Training also writes per-run preview images under each run directory's `previews/` folder.

### 11. Generate reports

This branch writes structured JSON/JSONL outputs that can be consumed by external notebooks or scripts. The dedicated `cargo xtask report` workflow described in planning documents is not exposed as a CLI command yet.

## Project Structure

The repository is a Rust workspace with focused crates under `crates/`:

- `sfx-core`: shared manifest types, schema constants, and core errors
- `sfx-config`: typed TOML loading and validation for dataset, model, training, and evaluation configs
- `sfx-data`: processed dataset manifests, runtime loading, batching, and split access
- `sfx-waymo`: Waymo-specific discovery, download, and extraction utilities
- `sfx-preprocess`: camera/LiDAR alignment, normalization, resizing, and preview generation
- `sfx-models`: range-only, RGB-only, and fusion autoencoder definitions
- `sfx-train`: training loops, run directory management, checkpointing, metrics, and previews
- `sfx-eval`: evaluation-related building blocks used by the wider workspace
- `sfx-tui`: terminal-oriented dataset exploration components
- `xtask`: the operational CLI entry point for setup, data, auth, and training tasks

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
- `eval.default.toml`: evaluation configuration scaffold kept in the workspace even though `xtask` does not currently expose a top-level `eval` command

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
