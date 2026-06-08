# Architecture

The project is organized as a Rust workspace with library crates under `crates/` and a single operational binary named `xtask` that uses Rust well used polyfill pattern to orchestrate commands for data preparation, training, evaluation, and reporting. The workspace is designed to be modular and extensible, allowing for clear separation of concerns and easy addition of new functionality as the project evolves.

Current crate layout:

- `sfx-core`: shared types and errors.
- `sfx-config`: configuration loading and validation.
- `sfx-data`: processed dataset loading.
- `sfx-waymo`: Waymo-specific access and extraction utilities.
- `sfx-preprocess`: dataset conversion and normalization.
- `sfx-models`: model definitions.
- `sfx-train`: training workflows and checkpoint-backed inference.
- `sfx-eval`: evaluation and reporting helpers.
- `sfx-tui`: terminal demo UI.
- `xtask`: command orchestration.

Local state is intentionally outside version control:

- `.xtask/` for cache, credentials, manifests, and run indexes.
- `data/` for raw and processed datasets.
- `artifacts/` for checkpoints, metrics, previews, reports, and exports.

Manifest JSON files under `.xtask/` track discovered raw files, downloaded files, processed samples, dataset splits, and run history. They are local state and can be regenerated as the pipeline matures.

The `sfx-train::inference` layer exposes `RunInference`, which loads a run's `model.bin` and dispatches the range-only, RGB-only, or fusion model based on the saved model configuration. The `xtask eval` and `xtask export` commands use that layer for real checkpoint reconstructions; `xtask compare` and `xtask report` then consume `<run_dir>/eval/*.json`, summaries, and preview paths for comparison tables and Markdown reports.
