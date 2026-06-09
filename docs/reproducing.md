# Reproducing Sensor Fusion Experiment Results

This guide describes the current end-to-end workflow available on this branch. It covers environment setup, data acquisition, tensor preparation, and training the three shipped model presets.

> Note: `crates/xtask/src/main.rs` currently exposes `init`, `doctor`, `gcloud`, `dataset`, and `train`. Standalone `eval`, `compare`, `export`, and `report` commands are planned but are not wired into the CLI on this branch, so evaluation and comparison use the run artifacts emitted by training.

## Requirements

- Rust 1.80+ with `cargo`
- Google Cloud CLI (`gcloud`) installed and on `PATH`
- Approved Waymo Open Dataset access
- Enough disk for raw parquet files, processed tensors, and checkpoints
- A clean workspace checkout with writable `.xtask/`, `data/`, and `artifacts/` directories

## Step 1: Clone and Setup

```bash
git clone https://github.com/metakernel/sensor-fusion-experiment.git
cd sensor-fusion-experiment
cargo xtask init
cargo xtask doctor
```

What this does:

- creates the local workspace directories under `.xtask/`, `data/`, and `artifacts/`
- initializes default manifest files
- validates the checked-in config files

## Step 2: Google Cloud Setup

Authenticate once for the local workspace:

```bash
cargo xtask gcloud auth
cargo xtask gcloud check
cargo xtask gcloud whoami
```

If authentication succeeds, local credentials are written under `.xtask/gcloud/` and `gcloud check` should confirm that an access token can be minted.

## Step 3: Data Download

First inspect available segments:

```bash
cargo xtask dataset list --split training --limit 5
```

Then download a reproducible small subset across all three source splits:

```bash
cargo xtask dataset fetch --splits train,val,test --train-files 1 --val-files 1 --test-files 1
```

Expected side effects:

- raw files appear under `data/raw/waymo/training`, `validation`, and `testing`
- `.xtask/manifests/raw_files.json` records discovered remote objects
- `.xtask/manifests/downloaded_files.json` records downloaded files

## Step 4: Data Preparation

Convert the downloaded raw data into processed tensors and split manifests:

```bash
cargo xtask dataset prepare --splits train,val,test --inspect
```

This step uses the checked-in dataset config (`configs/dataset.waymo.small.toml`) by default and prepares:

- RGB tensors shaped `3x128x256`
- range tensors shaped `2x64x256`
- processed samples under `data/processed/waymo-range-rgb-v1/{train,val,test}`
- `.xtask/manifests/processed_samples.json`
- `.xtask/manifests/extraction_summary.json`
- `.xtask/manifests/splits.json`

Sanity-check the prepared dataset:

```bash
cargo xtask dataset inspect
cargo xtask dataset preview --split train --count 16 --out artifacts/previews/dataset
```

## Step 5: Training All Models

Run the three checked-in training profiles exactly as shipped.

### Range-only baseline

```bash
cargo xtask train --config configs/train.range-only.toml
```

Expected output shape:

```text
ok   run range_run_001
     dir:        artifacts/checkpoints/range/range_run_001
     samples:    <train-sample-count>
     epochs:     3
     batch size: 8
     latent dim: 128
     final loss: <finite number>
     val loss:   <finite number>
ok   artifacts/checkpoints/range/range_run_001/metrics.jsonl
ok   artifacts/checkpoints/range/range_run_001/summary.json
ok   artifacts/checkpoints/range/range_run_001/model.bin
```

### RGB-only baseline

```bash
cargo xtask train --config configs/train.rgb-only.toml
```

Expected output shape:

```text
ok   run rgb_run_001
     dir:        artifacts/checkpoints/rgb/rgb_run_001
     samples:    <train-sample-count>
     epochs:     3
     batch size: 4
     latent dim: 128
     final loss: <finite number>
     val loss:   <finite number>
ok   artifacts/checkpoints/rgb/rgb_run_001/metrics.jsonl
ok   artifacts/checkpoints/rgb/rgb_run_001/summary.json
ok   artifacts/checkpoints/rgb/rgb_run_001/model.bin
```

### Fusion autoencoder

```bash
cargo xtask train --config configs/train.fusion.toml
```

Expected output shape:

```text
ok   run fusion_run_001
     dir:        artifacts/checkpoints/fusion/fusion_run_001
     samples:    <train-sample-count>
     epochs:     2
     batch size: 4
     latent dim: 128
     final loss: <finite number>
     val loss:   <finite number>
ok   artifacts/checkpoints/fusion/fusion_run_001/metrics.jsonl
ok   artifacts/checkpoints/fusion/fusion_run_001/summary.json
ok   artifacts/checkpoints/fusion/fusion_run_001/model.bin
```

Optional fast smoke run:

```bash
cargo xtask train --config configs/train.debug.toml
```

Every run also writes `config.toml`, `dataset.toml`, `model.toml`, `optimizer.json`, and `previews/` inside the run directory.

### Segment-holdout headline retraining (5/1/2)

```bash
cargo xtask train --config configs/train.range-only.holdout.toml
cargo xtask train --config configs/train.rgb-only.holdout.toml
cargo xtask train --config configs/train.fusion.holdout.toml
```

These configs pin `seed = 42`, use the segment assignment in
`artifacts/bench/audit/splits.segment_holdout.json`, and read split labels from
`.xtask/manifests/processed_samples.segment_holdout.json`.

## Step 6: Evaluation

This branch evaluates runs through the training artifacts rather than a dedicated `xtask eval` subcommand.

For each run directory, inspect:

- `summary.json` for `final_train_loss`, `final_val_loss`, `epochs`, `batch_size`, and run metadata
- `metrics.jsonl` for per-epoch loss history
- `previews/` for qualitative reconstruction samples

You can also verify resumability metadata with:

```bash
cargo xtask train resume --run artifacts/checkpoints/fusion/fusion_run_001
```

That command reports whether the run has a checkpoint, metrics, and optimizer metadata. It also reminds you that optimizer-state resume is not implemented yet.

## Step 7: Comparison Table

Build a manual comparison table from each run's `summary.json` and final `metrics.jsonl` record. A useful format is:

| Run | Model kind | Epochs | Batch size | Final train loss | Final val loss | Run directory |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `range_run_001` | range | 3 | 8 | from `summary.json` | from `summary.json` | `artifacts/checkpoints/range/range_run_001` |
| `rgb_run_001` | rgb | 3 | 4 | from `summary.json` | from `summary.json` | `artifacts/checkpoints/rgb/rgb_run_001` |
| `fusion_run_001` | fusion | 2 | 4 | from `summary.json` | from `summary.json` | `artifacts/checkpoints/fusion/fusion_run_001` |

Because the CLI does not yet expose `compare`, the comparison step is intentionally file-based and reproducible.

## Expected Results

The repository does not check in golden metric numbers because the fetched Waymo subset is user-selected. Use the tiny configs as a sanity check rather than a benchmark contract.

| Config | Approximate expectation |
| --- | --- |
| `configs/train.range-only.toml` | `metrics.jsonl` contains 3 rows, `train_loss` and `val_loss` stay finite, and the final loss is typically lower than or comparable to epoch 1. |
| `configs/train.rgb-only.toml` | `metrics.jsonl` contains 3 rows, `train_loss` and `val_loss` stay finite, and the model produces preview images under the run `previews/` directory. |
| `configs/train.fusion.toml` | `metrics.jsonl` contains 2 rows with `train_loss`, `rgb_train_loss`, `range_train_loss`, `val_loss`, `rgb_val_loss`, and `range_val_loss`; all should remain finite. |
| `configs/train.debug.toml` | Produces a small fusion smoke-test run using the same dataset/model defaults but a `debug` run name. |

## Troubleshooting

- **`cargo xtask doctor` fails**: fix the missing file, directory, or invalid config reported by the command before continuing.
- **`cargo xtask gcloud check` fails**: rerun `cargo xtask gcloud auth` and confirm the active account has Waymo dataset access.
- **`dataset list` or `dataset fetch` cannot see objects**: verify Google Cloud authentication and access permissions, then retry with the correct split selection.
- **Training reports missing validation data**: rerun `cargo xtask dataset prepare --splits train,val,test` so `val/` tensors exist alongside `train/`.
- **A run directory name already exists**: the trainer automatically appends a numeric suffix such as `_002`; record the actual printed run directory when comparing results.
- **`train resume` does not continue optimizer state**: this is expected; the current prototype preserves checkpoint and config metadata, but not serialized optimizer state.
