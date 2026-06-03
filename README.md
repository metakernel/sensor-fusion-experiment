# sensor-fusion-experiment

Experimental Rust workspace for learning shared latent representations from RGB camera images and LiDAR range images.

## Project goal

Build a reproducible research workflow for preparing paired RGB + LiDAR range samples, training reconstruction models, evaluating reconstruction quality, and producing report-ready artifacts.

## Data policy

This repository must not contain Waymo Open Dataset files, extracted sensor frames, processed tensors, trained checkpoints derived from restricted data, or cached credentials. Local data and generated outputs live under ignored directories such as `.xtask/`, `data/`, and `artifacts/`.

## Quickstart

First start by installing the [Google Cloud SDK](https://docs.cloud.google.com/sdk/docs/install-sdk?hl=fr) and authenticating with `cargo xtask gcloud auth` to enable access to the Waymo Open Dataset on Google Cloud Storage. The `doctor` command will report any missing credentials or configuration issues.

The workspace foundation supports project setup, environment checks, and local Google Cloud authentication:

```bash
cargo xtask init
cargo xtask doctor
cargo xtask gcloud auth
cargo xtask gcloud check
cargo xtask dataset list --split training --limit 20
cargo xtask dataset fetch --splits train,val --train-files 1 --val-files 1
cargo xtask dataset prepare --max-frames 128 --rgb-size 128x256 --range-size 64x256 --range-channels range,intensity
cargo xtask dataset inspect --sample-count 32
cargo xtask dataset preview --split train --count 16 --out artifacts/previews/dataset
```

## Command overview

- `cargo xtask init`: create local cache, data, artifact directories, and initial manifest files.
- `cargo xtask doctor`: print project environment status and validate configuration plus local manifests.
- `cargo xtask gcloud auth`: start Google Cloud application-default login with read-only storage scope and save local credentials.
- `cargo xtask gcloud check`: verify local credentials can mint an access token.
- `cargo xtask gcloud whoami`: show the active Cloud SDK account when available.
- `cargo xtask gcloud logout`: revoke application-default credentials and remove local credential files.
- `cargo xtask dataset list`: list Waymo Cloud Storage objects for one source split and write `.xtask/manifests/raw_files.json`.
- `cargo xtask dataset fetch`: download selected Waymo objects into split-specific raw data folders and write `.xtask/manifests/downloaded_files.json`.
- `cargo xtask dataset prepare`: extract and align camera/lidar frames from matched parquet files, resize and normalize tensors, then write processed samples plus manifests.
- `cargo xtask dataset inspect`: report split counts, tensor stats, and suspicious sample checks from the processed dataset.
- `cargo xtask dataset preview`: export RGB and range grid previews from processed samples.

## License

MIT.
