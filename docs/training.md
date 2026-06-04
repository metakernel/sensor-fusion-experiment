# Training (Scaffold)

## Purpose
Define run profiles and expected training controls.

## Profiles
- `train.debug.toml`: CPU-safe smoke and debug run.
- `train.nextai.toml`: multi-device baseline for larger runs.

## Expected Knobs
- Run identity (`name`, `seed`, device, precision).
- Trainer limits (epochs, batching, logging, checkpoints).
- Optimizer and scheduler settings.
- Distributed strategy and optional early stopping.
- Output directory and save behavior.
- Resume path (`--resume`) can point at either a run directory or a checkpoint JSON.
- Deterministic seed override (`--seed`) and debug shortening (`--debug`) are available on `xtask train`.

## Resume and Seeds
- New runs create `config.json`, `metrics.jsonl`, `summary.json`, `previews/`, and `checkpoints/` under the configured output directory.
- `--resume <run-dir>` resolves to `<run-dir>/checkpoints/last.ckpt.json`; `--resume <checkpoint>` resumes that checkpoint directly.
- Resumed runs keep the original run id/output directory, preserve previous metrics, and continue epochs/global steps from the checkpoint.
- Resume fails explicitly when the checkpoint is missing or when the saved seed/model metadata does not match the requested run.
- The applied seed is persisted in `config.json`, `summary.json`, every metric record, and checkpoint metadata.
- `--debug` forces one epoch with one train batch and one validation batch for quick plumbing checks.

## Next Steps
- Add metric retention policy.
