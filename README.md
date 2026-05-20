# sensor-fusion-experiment

Experimental Rust workspace for learning shared latent representations from RGB camera images and LiDAR range images.

## Project goal

Build a reproducible research workflow for preparing paired RGB + LiDAR range samples, training reconstruction models, evaluating reconstruction quality, and producing report-ready artifacts.

## Data policy

This repository must not contain Waymo Open Dataset files, extracted sensor frames, processed tensors, trained checkpoints derived from restricted data, or cached credentials. Local data and generated outputs live under ignored directories such as `.xtask/`, `data/`, and `artifacts/`.

## Quickstart

The workspace foundation supports project setup and environment checks:

```bash
cargo xtask init
cargo xtask doctor
```

## Command overview

- `cargo xtask init`: create local cache, data, and artifact directories.
- `cargo xtask doctor`: print project environment status.

## License

MIT.
