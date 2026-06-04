# Changelog

## [Unreleased]

### Added
- Phase 0: Workspace foundation — crate skeletons (`sfx-core`, `sfx-config`, `sfx-data`, `sfx-models`, `sfx-train`, `sfx-preprocess`, `sfx-waymo`, `sfx-tui`, `sfx-eval`, `xtask`)
- Phase 1: `sfx-config` typed configuration system with TOML loading and validation
- Phase 2: `sfx-core` manifest schema (`ProcessedSampleManifest`, `SplitsManifest`, `RunIndex`, and related types)
- Phase 3: `gcloud auth`, `check`, `whoami`, and `logout` commands
- Phase 4: `dataset list` command with Waymo segment discovery
- Phase 5: `dataset fetch` with resume-safe downloads and progress tracking
- Phase 6: Frame extraction pipeline with camera/LiDAR alignment
- Phase 7: `dataset prepare` CHW `f32` tensor export (RGB `3x128x256`, range `2x64x256`), previews, and split manifests
- Phase 8: `dataset inspect` statistics and `dataset preview` sample-grid export
- Phase 9: `sfx-data` runtime loader with seeded batching and shuffle controls
- Phase 10: `RangeAutoencoder` baseline (encoder, decoder, and training)
- Phase 11: `RgbAutoencoder` baseline (encoder, decoder, and training)
- Phase 12: `FusionAutoencoder` with shared latent space and dual-modality reconstruction
- Phase 13: Training infrastructure with hardened run directories, run indexing, resume inspection, seeds, and backend selection
- Phase 14: `sfx-eval` evaluation primitives and checked-in `configs/eval.default.toml`
- Phase 15: Run indexing and summary artifacts that support repeatable multi-run comparison workflows
- Phase 16: Visual preview export for datasets and per-run reconstructions
- Phase 17: Terminal UI groundwork in `sfx-tui`
- Phase 18: Structured training summaries, metrics, and report-oriented artifacts
- Phase 19: Complete reproducibility documentation (`README.md`, `docs/reproducing.md`, and `CHANGELOG.md`)
