# sensor-fusion-experiment

Experimental Rust workspace for learning shared latent representations from RGB camera images and LiDAR range images.

## Project goal

Build a reproducible research workflow for preparing paired RGB + LiDAR range samples, training reconstruction models, evaluating reconstruction quality, and producing report-ready artifacts.

## Data policy

This repository must not contain Waymo Open Dataset files, extracted sensor frames, processed tensors, trained checkpoints derived from restricted data, or cached credentials. Local data and generated outputs live under ignored directories such as `.xtask/`, `data/`, and `artifacts/`.

## Quickstart

The workspace foundation supports project setup, environment checks, and local Google Cloud authentication:

```bash
cargo xtask init
cargo xtask doctor
cargo xtask gcloud auth
cargo xtask gcloud check
```

## Command overview

- `cargo xtask init`: create local cache, data, artifact directories, and initial manifest files.
- `cargo xtask doctor`: print project environment status and validate configuration plus local manifests.
- `cargo xtask gcloud auth`: start Google Cloud application-default login with read-only storage scope and save local credentials.
- `cargo xtask gcloud check`: verify local credentials can mint an access token.
- `cargo xtask gcloud whoami`: show the active Cloud SDK account when available.
- `cargo xtask gcloud logout`: revoke application-default credentials and remove local credential files.

## License

MIT.
