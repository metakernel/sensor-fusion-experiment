# Model (Scaffold)

## Purpose
Describe model families and the shared configuration surface.

## Planned Variants
- `model.range-only.tiny.toml`: range-only baseline.
- `model.rgb-only.tiny.toml`: camera-only baseline.
- `model.fusion.tiny.toml`: two-branch fusion baseline.

## Shared Knobs
- Input modality toggles.
- Backbone channel and block depth.
- Fusion method/projection (fusion model only).
- Head settings (`num_classes`, hidden dim, loss type).

## Next Steps
- Map config keys to concrete Rust structs.
- Add shape contracts between encoder, fusion, and head.
- Document initialization and checkpoint compatibility rules.
