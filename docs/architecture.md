# Architecture

The project is organized as a Rust workspace with library crates under `crates/` and a single operational binary named `xtask`.

Current crate layout:

- `sfx-core`: shared types and errors.
- `sfx-config`: configuration loading and validation.
- `sfx-data`: processed dataset loading.
- `sfx-waymo`: Waymo-specific access and extraction utilities.
- `sfx-preprocess`: dataset conversion and normalization.
- `sfx-models`: model definitions.
- `sfx-train`: training workflows.
- `sfx-eval`: evaluation and reporting helpers.
- `sfx-tui`: terminal demo UI.
- `xtask`: command orchestration.

Local state is intentionally outside version control:

- `.xtask/` for cache, credentials, manifests, and run indexes.
- `data/` for raw and processed datasets.
- `artifacts/` for checkpoints, metrics, previews, reports, and exports.

Manifest JSON files under `.xtask/` track discovered raw files, downloaded files, processed samples, dataset splits, and run history. They are local state and can be regenerated as the pipeline matures.
