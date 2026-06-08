use crate::{InnerBackend, TrainingSummary, tensor_values};
use anyhow::{Context, Result, bail};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::Device;
use sfx_config::ModelKind;
use sfx_core::manifest::TensorShape;
use sfx_data::FusionBatch;
use sfx_models::{
    RANGE_CHANNELS, RANGE_HEIGHT, RANGE_WIDTH, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH,
    RangeAutoencoder, RangeAutoencoderConfig, RgbAutoencoder, RgbAutoencoderConfig,
    SharedLatentMultimodalAutoencoder, SharedLatentMultimodalAutoencoderConfig,
};
use std::path::Path;

enum LoadedModel {
    Range(RangeAutoencoder<InnerBackend>),
    Rgb(RgbAutoencoder<InnerBackend>),
    Fusion(Box<SharedLatentMultimodalAutoencoder<InnerBackend>>),
}

pub struct RunInference {
    model: LoadedModel,
    device: Device<InnerBackend>,
    model_kind: ModelKind,
}

/// Contiguous NCHW reconstructions; Some only for modalities the model emits.
pub struct Reconstruction {
    pub sample_ids: Vec<String>,
    pub rgb_hat: Option<Vec<f32>>,
    pub range_hat: Option<Vec<f32>>,
}

impl RunInference {
    pub fn load(run_dir: &Path) -> Result<Self> {
        let device = Device::<InnerBackend>::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let checkpoint_stem = run_dir.join("model");
        let checkpoint_path = run_dir.join("model.bin");
        if !checkpoint_path.exists() {
            bail!("checkpoint not found: {}", checkpoint_path.display());
        }

        let model_toml = run_dir.join("model.toml");
        let (kind, latent_dim, z_modality): (ModelKind, usize, Option<usize>) =
            if model_toml.exists() {
                let mc = sfx_config::load_model_config(run_dir, "model.toml")
                    .with_context(|| format!("loading {}", model_toml.display()))?;
                (mc.kind, mc.latent_dim, Some(mc.effective_z_modality()))
            } else {
                let summary = read_summary(run_dir)?;
                (
                    kind_from_slug(&summary.model_kind)?,
                    summary.latent_dim,
                    summary.z_modality,
                )
            };

        let model = match kind {
            ModelKind::RangeOnly => LoadedModel::Range(
                RangeAutoencoderConfig::new(latent_dim)
                    .init::<InnerBackend>(&device)
                    .load_file(&checkpoint_stem, &recorder, &device)?,
            ),
            ModelKind::RgbOnly => LoadedModel::Rgb(
                RgbAutoencoderConfig::new(latent_dim)
                    .init::<InnerBackend>(&device)
                    .load_file(&checkpoint_stem, &recorder, &device)?,
            ),
            ModelKind::Fusion => {
                let z = z_modality.context(
                    "fusion run needs z_modality but neither model.toml nor summary.json provides it; re-train or restore model.toml",
                )?;
                LoadedModel::Fusion(Box::new(
                    SharedLatentMultimodalAutoencoderConfig::new(latent_dim, z)
                        .init::<InnerBackend>(&device)
                        .load_file(&checkpoint_stem, &recorder, &device)?,
                ))
            }
        };

        Ok(Self {
            model,
            device,
            model_kind: kind,
        })
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Validates shapes, runs forward, returns per-sample-contiguous NCHW reconstructions.
    pub fn reconstruct(&self, batch: &FusionBatch) -> Result<Reconstruction> {
        validate_shape("rgb", &batch.rgb_shape, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH)?;
        validate_shape(
            "range",
            &batch.range_shape,
            RANGE_CHANNELS,
            RANGE_HEIGHT,
            RANGE_WIDTH,
        )?;
        let batch_size = batch.batch_size();
        ensure_len(
            "rgb",
            batch.rgb.len(),
            batch_size * batch.rgb_shape.value_count(),
        )?;
        ensure_len(
            "range",
            batch.range.len(),
            batch_size * batch.range_shape.value_count(),
        )?;

        let burn = batch.clone().into_burn::<InnerBackend>(&self.device);
        let sample_ids = burn.sample_ids.iter().map(|id| id.0.clone()).collect();
        let (rgb_hat, range_hat) = match &self.model {
            LoadedModel::Range(model) => {
                let output = model.forward(burn.range);
                (None, Some(tensor_values(output.range_hat)?))
            }
            LoadedModel::Rgb(model) => {
                let output = model.forward(burn.rgb);
                (Some(tensor_values(output.rgb_hat)?), None)
            }
            LoadedModel::Fusion(model) => {
                let output = model.forward(burn.rgb, burn.range);
                (
                    Some(tensor_values(output.rgb_hat)?),
                    Some(tensor_values(output.range_hat)?),
                )
            }
        };

        Ok(Reconstruction {
            sample_ids,
            rgb_hat,
            range_hat,
        })
    }
}

fn read_summary(run_dir: &Path) -> Result<TrainingSummary> {
    let summary_path = run_dir.join("summary.json");
    let text = std::fs::read_to_string(&summary_path)
        .with_context(|| format!("reading {}", summary_path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", summary_path.display()))
}

fn kind_from_slug(slug: &str) -> Result<ModelKind> {
    match slug {
        "rgb" => Ok(ModelKind::RgbOnly),
        "range" => Ok(ModelKind::RangeOnly),
        "fusion" => Ok(ModelKind::Fusion),
        other => bail!("unknown model_kind slug `{other}` in summary.json"),
    }
}

fn validate_shape(
    label: &str,
    shape: &TensorShape,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    if shape.channels != channels || shape.height != height || shape.width != width {
        bail!(
            "{label} shape is {}x{}x{} (CxHxW); expected {channels}x{height}x{width}",
            shape.channels,
            shape.height,
            shape.width
        );
    }
    Ok(())
}

fn ensure_len(label: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        bail!("{label} tensor has {got} values; expected {want}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;
    use sfx_core::manifest::{
        MANIFEST_SCHEMA_VERSION, MultimodalSampleMeta, ProcessedSampleEntry,
        ProcessedSampleManifest, SampleId, Split, write_manifest,
    };
    use sfx_data::{BatchOptions, FusionDataset};
    use std::path::{Path, PathBuf};

    const LATENT_DIM: usize = 8;
    const Z_MODALITY: usize = 16;

    #[test]
    fn loads_fusion_checkpoint_and_reconstructs_both_modalities() -> Result<()> {
        let root = test_root("fusion");
        let batch = full_size_batch(&root)?;
        let run_dir = create_run(&root, ModelKind::Fusion, LATENT_DIM, Some(Z_MODALITY), true)?;

        let infer = RunInference::load(&run_dir)?;
        assert_eq!(infer.model_kind(), ModelKind::Fusion);

        let recon = infer.reconstruct(&batch)?;

        assert_eq!(recon.sample_ids, sample_id_strings(&batch));
        assert_values(
            recon.rgb_hat.as_deref().expect("fusion emits rgb"),
            batch.batch_size() * batch.rgb_shape.value_count(),
        );
        assert_values(
            recon.range_hat.as_deref().expect("fusion emits range"),
            batch.batch_size() * batch.range_shape.value_count(),
        );
        Ok(())
    }

    #[test]
    fn loads_range_checkpoint_from_summary_and_reconstructs_range_only() -> Result<()> {
        let root = test_root("range");
        let batch = full_size_batch(&root)?;
        let run_dir = create_run(&root, ModelKind::RangeOnly, LATENT_DIM, None, false)?;

        let infer = RunInference::load(&run_dir)?;
        assert_eq!(infer.model_kind(), ModelKind::RangeOnly);

        let recon = infer.reconstruct(&batch)?;

        assert_eq!(recon.sample_ids, sample_id_strings(&batch));
        assert!(recon.rgb_hat.is_none());
        assert_values(
            recon.range_hat.as_deref().expect("range emits range"),
            batch.batch_size() * batch.range_shape.value_count(),
        );
        Ok(())
    }

    fn create_run(
        root: &Path,
        kind: ModelKind,
        latent_dim: usize,
        z_modality: Option<usize>,
        include_model_toml: bool,
    ) -> Result<PathBuf> {
        let run_dir = root.join("run").join(summary_slug(kind));
        std::fs::create_dir_all(&run_dir)
            .with_context(|| format!("creating {}", run_dir.display()))?;
        save_checkpoint(&run_dir, kind, latent_dim, z_modality)?;
        write_summary(&run_dir, kind, latent_dim, z_modality)?;
        if include_model_toml {
            write_model_toml(&run_dir, kind, latent_dim, z_modality)?;
        }
        Ok(run_dir)
    }

    fn save_checkpoint(
        run_dir: &Path,
        kind: ModelKind,
        latent_dim: usize,
        z_modality: Option<usize>,
    ) -> Result<()> {
        let device = Device::<InnerBackend>::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let checkpoint_stem = run_dir.join("model");
        match kind {
            ModelKind::RangeOnly => RangeAutoencoderConfig::new(latent_dim)
                .init::<InnerBackend>(&device)
                .save_file(&checkpoint_stem, &recorder)?,
            ModelKind::RgbOnly => RgbAutoencoderConfig::new(latent_dim)
                .init::<InnerBackend>(&device)
                .save_file(&checkpoint_stem, &recorder)?,
            ModelKind::Fusion => SharedLatentMultimodalAutoencoderConfig::new(
                latent_dim,
                z_modality.context("fusion test run needs z_modality")?,
            )
            .init::<InnerBackend>(&device)
            .save_file(&checkpoint_stem, &recorder)?,
        }
        Ok(())
    }

    fn write_summary(
        run_dir: &Path,
        kind: ModelKind,
        latent_dim: usize,
        z_modality: Option<usize>,
    ) -> Result<()> {
        let summary_path = run_dir.join("summary.json");
        let summary = crate::TrainingSummary {
            run_id: summary_slug(kind).to_string(),
            run_name: summary_slug(kind).to_string(),
            model_kind: summary_slug(kind).to_string(),
            status: "completed".to_string(),
            started_at: "unix:0".to_string(),
            completed_at: Some("unix:0".to_string()),
            run_dir: run_dir.to_path_buf(),
            config_path: run_dir.join("config.toml"),
            dataset_config_path: run_dir.join("dataset.toml"),
            model_config_path: run_dir.join("model.toml"),
            metrics_path: run_dir.join("metrics.jsonl"),
            summary_path: summary_path.clone(),
            checkpoint_path: run_dir.join("model.bin"),
            optimizer_path: run_dir.join("optimizer.json"),
            epochs: 1,
            train_samples: 2,
            batch_size: 2,
            max_batches_per_epoch: Some(1),
            latent_dim,
            z_modality,
            final_train_loss: 0.0,
            final_val_loss: Some(0.0),
            backend: "burn-flex-autodiff".to_string(),
            optimizer: "adam".to_string(),
        };
        std::fs::write(
            &summary_path,
            format!("{}\n", serde_json::to_string_pretty(&summary)?),
        )
        .with_context(|| format!("writing {}", summary_path.display()))
    }

    fn write_model_toml(
        run_dir: &Path,
        kind: ModelKind,
        latent_dim: usize,
        z_modality: Option<usize>,
    ) -> Result<()> {
        let mut text = format!(
            "[model]\nkind = \"{}\"\nlatent_dim = {latent_dim}\n",
            toml_kind(kind)
        );
        if let Some(z_modality) = z_modality {
            text.push_str(&format!("z_modality = {z_modality}\n"));
        }
        let path = run_dir.join("model.toml");
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    }

    fn full_size_batch(root: &Path) -> Result<FusionBatch> {
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = full_size_manifest(&processed)?;
        write_manifest(&manifest_path, &manifest)
            .with_context(|| format!("writing {}", manifest_path.display()))?;
        let dataset = FusionDataset::open_split(&processed, &manifest_path, Some(Split::Train))?;
        let mut batches = dataset.batches(BatchOptions::new(2))?;
        batches
            .next()
            .context("expected a training batch")?
            .context("loading training batch")
    }

    fn full_size_manifest(processed: &Path) -> Result<ProcessedSampleManifest> {
        let rgb_shape = TensorShape {
            channels: RGB_CHANNELS,
            height: RGB_HEIGHT,
            width: RGB_WIDTH,
        };
        let range_shape = TensorShape {
            channels: RANGE_CHANNELS,
            height: RANGE_HEIGHT,
            width: RANGE_WIDTH,
        };
        let mut samples = Vec::new();

        for i in 0..3 {
            let split = if i < 2 { Split::Train } else { Split::Val };
            let split_dir = match &split {
                Split::Train => "train",
                Split::Val => "val",
                Split::Test => "test",
            };
            let id = SampleId(format!("sample_{i:06}"));
            let sample_rel = PathBuf::from(split_dir).join(&id.0);
            let sample_dir = processed.join(&sample_rel);
            std::fs::create_dir_all(&sample_dir)
                .with_context(|| format!("creating {}", sample_dir.display()))?;
            write_zero_f32_file(&sample_dir.join("rgb.f32.bin"), rgb_shape.value_count())?;
            write_zero_f32_file(&sample_dir.join("range.f32.bin"), range_shape.value_count())?;

            samples.push(ProcessedSampleEntry {
                meta: MultimodalSampleMeta {
                    id,
                    split,
                    rgb_path: sample_rel.join("rgb.f32.bin"),
                    range_path: sample_rel.join("range.f32.bin"),
                    timestamp_micros: i as i64,
                    source_segment: "segment".to_string(),
                },
                meta_path: sample_rel.join("meta.json"),
                preview_rgb_path: Some(sample_rel.join("preview_rgb.png")),
                preview_range_path: Some(sample_rel.join("preview_range.png")),
            });
        }

        Ok(ProcessedSampleManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: Some("fixture".to_string()),
            rgb_shape: Some(rgb_shape),
            range_shape: Some(range_shape),
            samples,
        })
    }

    fn write_zero_f32_file(path: &Path, values: usize) -> Result<()> {
        std::fs::write(path, vec![0u8; values * std::mem::size_of::<f32>()])
            .with_context(|| format!("writing {}", path.display()))
    }

    fn assert_values(values: &[f32], expected_len: usize) {
        assert_eq!(values.len(), expected_len);
        assert!(
            values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }

    fn sample_id_strings(batch: &FusionBatch) -> Vec<String> {
        batch.sample_ids.iter().map(|id| id.0.clone()).collect()
    }

    fn summary_slug(kind: ModelKind) -> &'static str {
        match kind {
            ModelKind::RgbOnly => "rgb",
            ModelKind::RangeOnly => "range",
            ModelKind::Fusion => "fusion",
        }
    }

    fn toml_kind(kind: ModelKind) -> &'static str {
        match kind {
            ModelKind::RgbOnly => "rgb-only",
            ModelKind::RangeOnly => "range-only",
            ModelKind::Fusion => "fusion",
        }
    }

    fn test_root(name: &str) -> PathBuf {
        let root = PathBuf::from("target")
            .join("sfx-train-inference-tests")
            .join(format!("{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
