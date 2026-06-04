# Dataset (Scaffold)

## Purpose
Capture assumptions for Waymo small-split ingestion and preprocessing.

## Expected Config Inputs
- `dataset.name`, `dataset.root`, and split names.
- Modality flags (`range_image`, `rgb`, `point_cloud`).
- Sampling knobs (`max_scenes`, `max_frames_per_scene`).
- Preprocessing knobs (resize, normalization, range clipping).

## Current Placeholder Scope
- Assumes local data staging under `data/raw/waymo/small`.
- Uses deterministic split seed and bounded sample sizes for early experiments.

## Phase 6 Intermediate Extraction Contract
- Contract type: `sfx_data::manifest::ExtractedFramesManifest`.
- Manifest file: `data/intermediate/extracted/extracted_frames.json`.
- Artifact directories:
  - `data/intermediate/extracted/rgb/` for synchronized RGB image artifacts.
  - `data/intermediate/extracted/range/` for synchronized range artifacts.
- Each frame record carries stable `sample_id` and `frame_id`, split, frame index,
  timestamp in microseconds, source segment/file references, RGB/range artifact
  references with shape metadata, and extractor provenance.

## Next Steps
- Define canonical train/val/test split generation.
- Document data validation checks and failure modes.
