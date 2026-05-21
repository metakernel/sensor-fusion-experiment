use anyhow::{Context, Result, bail};
use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::cast::ToElement;
use burn::tensor::{Device, Tensor, TensorData, backend::Backend};
use serde::Serialize;
use sfx_config::ModelKind;
use sfx_core::manifest::Split;
use sfx_data::{BatchOptions, FusionDataset};
use sfx_models::{RangeAutoencoder, RangeAutoencoderConfig, RgbAutoencoder, RgbAutoencoderConfig};
use std::path::{Path, PathBuf};

pub const CRATE_NAME: &str = "sfx-train";

type InnerBackend = Flex;
type TrainBackend = Autodiff<InnerBackend>;

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingSummary {
    pub run_name: String,
    pub run_dir: PathBuf,
    pub metrics_path: PathBuf,
    pub summary_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub epochs: usize,
    pub train_samples: usize,
    pub batch_size: usize,
    pub latent_dim: usize,
    pub final_train_loss: f64,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize)]
struct EpochMetric {
    epoch: usize,
    train_loss: f64,
    val_loss: Option<f64>,
    sample_count: usize,
    batch_count: usize,
}

pub fn train_range_autoencoder(root: &Path, config_path: &Path) -> Result<TrainingSummary> {
    let config = sfx_config::load_training_config(root, config_path)
        .with_context(|| format!("loading training config {}", config_path.display()))?;
    if config.model.kind != ModelKind::RangeOnly {
        bail!(
            "Phase 10 only supports range-only training; config uses {:?}",
            config.model.kind
        );
    }

    let manifest_path = root.join(".xtask/manifests/processed_samples.json");
    let train_dataset = FusionDataset::open_split(
        &config.dataset.processed_dir,
        &manifest_path,
        Some(Split::Train),
    )
    .with_context(|| format!("opening training dataset from {}", manifest_path.display()))?;
    if train_dataset.is_empty() {
        bail!("training split is empty; run `cargo xtask dataset prepare --splits train` first");
    }
    train_dataset.validate_tensor_files()?;

    let run_dir = root
        .join("artifacts/checkpoints/range")
        .join(&config.run_name);
    std::fs::create_dir_all(&run_dir).with_context(|| format!("creating {}", run_dir.display()))?;
    copy_if_exists(config_path, &run_dir.join("config.toml"))?;
    copy_if_exists(&config.dataset_config_path, &run_dir.join("dataset.toml"))?;
    copy_if_exists(&config.model_config_path, &run_dir.join("model.toml"))?;

    let device = Device::<TrainBackend>::default();
    TrainBackend::seed(&device, config.seed);
    let mut model =
        RangeAutoencoderConfig::new(config.model.latent_dim).init::<TrainBackend>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainBackend, RangeAutoencoder<TrainBackend>>();

    let metrics_path = run_dir.join("metrics.jsonl");
    let mut metric_lines = String::new();
    let mut final_train_loss = f64::INFINITY;

    for epoch in 1..=config.epochs {
        let options = BatchOptions::new(config.batch_size).shuffled(config.seed + epoch as u64);
        let mut total_loss = 0.0f64;
        let mut total_samples = 0usize;
        let mut batch_count = 0usize;

        for batch in train_dataset.batches(options)? {
            let batch = batch?.into_burn::<TrainBackend>(&device);
            let current_batch_size = batch.batch_size();
            let target = batch.range.clone();
            let output = model.forward(batch.range);
            let loss = mse_loss(output.range_hat, target);
            let loss_value = loss.clone().into_scalar().to_f64();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);

            total_loss += loss_value * current_batch_size as f64;
            total_samples += current_batch_size;
            batch_count += 1;
        }

        final_train_loss = total_loss / total_samples.max(1) as f64;
        let metric = EpochMetric {
            epoch,
            train_loss: final_train_loss,
            val_loss: None,
            sample_count: total_samples,
            batch_count,
        };
        metric_lines.push_str(&serde_json::to_string(&metric)?);
        metric_lines.push('\n');
        println!(
            "epoch {epoch:03}/{:03} train_loss={:.6} batches={batch_count}",
            config.epochs, final_train_loss
        );
    }

    std::fs::write(&metrics_path, metric_lines)
        .with_context(|| format!("writing {}", metrics_path.display()))?;

    let valid_model = model.valid();
    let checkpoint_stem = run_dir.join("model");
    valid_model
        .clone()
        .save_file(
            &checkpoint_stem,
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )
        .with_context(|| {
            format!(
                "saving model checkpoint to {}.bin",
                checkpoint_stem.display()
            )
        })?;
    write_range_previews(&valid_model, &train_dataset, &run_dir.join("previews"))?;

    let summary = TrainingSummary {
        run_name: config.run_name,
        run_dir: run_dir.clone(),
        metrics_path: metrics_path.clone(),
        summary_path: run_dir.join("summary.json"),
        checkpoint_path: run_dir.join("model.bin"),
        epochs: config.epochs,
        train_samples: train_dataset.len(),
        batch_size: config.batch_size,
        latent_dim: config.model.latent_dim,
        final_train_loss,
        backend: "burn-flex-autodiff".to_string(),
    };
    std::fs::write(
        &summary.summary_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("writing {}", summary.summary_path.display()))?;

    Ok(summary)
}

pub fn train_rgb_autoencoder(root: &Path, config_path: &Path) -> Result<TrainingSummary> {
    let config = sfx_config::load_training_config(root, config_path)
        .with_context(|| format!("loading training config {}", config_path.display()))?;
    if config.model.kind != ModelKind::RgbOnly {
        bail!(
            "Phase 11 only supports RGB-only training; config uses {:?}",
            config.model.kind
        );
    }

    let manifest_path = root.join(".xtask/manifests/processed_samples.json");
    let train_dataset = FusionDataset::open_split(
        &config.dataset.processed_dir,
        &manifest_path,
        Some(Split::Train),
    )
    .with_context(|| format!("opening training dataset from {}", manifest_path.display()))?;
    if train_dataset.is_empty() {
        bail!("training split is empty; run `cargo xtask dataset prepare --splits train` first");
    }
    train_dataset.validate_tensor_files()?;

    let run_dir = root
        .join("artifacts/checkpoints/rgb")
        .join(&config.run_name);
    std::fs::create_dir_all(&run_dir).with_context(|| format!("creating {}", run_dir.display()))?;
    copy_if_exists(config_path, &run_dir.join("config.toml"))?;
    copy_if_exists(&config.dataset_config_path, &run_dir.join("dataset.toml"))?;
    copy_if_exists(&config.model_config_path, &run_dir.join("model.toml"))?;

    let device = Device::<TrainBackend>::default();
    TrainBackend::seed(&device, config.seed);
    let mut model =
        RgbAutoencoderConfig::new(config.model.latent_dim).init::<TrainBackend>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainBackend, RgbAutoencoder<TrainBackend>>();

    let metrics_path = run_dir.join("metrics.jsonl");
    let mut metric_lines = String::new();
    let mut final_train_loss = f64::INFINITY;

    for epoch in 1..=config.epochs {
        let options = BatchOptions::new(config.batch_size).shuffled(config.seed + epoch as u64);
        let mut total_loss = 0.0f64;
        let mut total_samples = 0usize;
        let mut batch_count = 0usize;

        for batch in train_dataset.batches(options)? {
            let batch = batch?.into_burn::<TrainBackend>(&device);
            let current_batch_size = batch.batch_size();
            let target = batch.rgb.clone();
            let output = model.forward(batch.rgb);
            let loss = mse_loss(output.rgb_hat, target);
            let loss_value = loss.clone().into_scalar().to_f64();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);

            total_loss += loss_value * current_batch_size as f64;
            total_samples += current_batch_size;
            batch_count += 1;
        }

        final_train_loss = total_loss / total_samples.max(1) as f64;
        let metric = EpochMetric {
            epoch,
            train_loss: final_train_loss,
            val_loss: None,
            sample_count: total_samples,
            batch_count,
        };
        metric_lines.push_str(&serde_json::to_string(&metric)?);
        metric_lines.push('\n');
        println!(
            "epoch {epoch:03}/{:03} train_loss={:.6} batches={batch_count}",
            config.epochs, final_train_loss
        );
    }

    std::fs::write(&metrics_path, metric_lines)
        .with_context(|| format!("writing {}", metrics_path.display()))?;

    let valid_model = model.valid();
    let checkpoint_stem = run_dir.join("model");
    valid_model
        .clone()
        .save_file(
            &checkpoint_stem,
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )
        .with_context(|| {
            format!(
                "saving model checkpoint to {}.bin",
                checkpoint_stem.display()
            )
        })?;
    write_rgb_previews(&valid_model, &train_dataset, &run_dir.join("previews"))?;

    let summary = TrainingSummary {
        run_name: config.run_name,
        run_dir: run_dir.clone(),
        metrics_path: metrics_path.clone(),
        summary_path: run_dir.join("summary.json"),
        checkpoint_path: run_dir.join("model.bin"),
        epochs: config.epochs,
        train_samples: train_dataset.len(),
        batch_size: config.batch_size,
        latent_dim: config.model.latent_dim,
        final_train_loss,
        backend: "burn-flex-autodiff".to_string(),
    };
    std::fs::write(
        &summary.summary_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("writing {}", summary.summary_path.display()))?;

    Ok(summary)
}

fn mse_loss<B: Backend>(pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    (pred - target).square().mean()
}

fn copy_if_exists(source: &Path, target: &Path) -> Result<()> {
    if source.exists() {
        std::fs::copy(source, target)
            .with_context(|| format!("copying {} to {}", source.display(), target.display()))?;
    }
    Ok(())
}

fn write_range_previews(
    model: &RangeAutoencoder<InnerBackend>,
    dataset: &FusionDataset,
    out_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let device = Device::<InnerBackend>::default();
    let Some(batch) = dataset.batches(BatchOptions::new(4))?.next() else {
        return Ok(());
    };
    let raw_batch = batch?;
    let batch_size = raw_batch.batch_size();
    let batch = raw_batch.into_burn::<InnerBackend>(&device);
    let target = batch.range.clone();
    let output = model.forward(batch.range);
    let target_values = tensor_values(target)?;
    let recon_values = tensor_values(output.range_hat)?;

    let h = dataset.range_shape().height;
    let w = dataset.range_shape().width;
    let c = dataset.range_shape().channels;
    let sample_values = c * h * w;
    for sample_idx in 0..batch_size {
        let offset = sample_idx * sample_values;
        let path = out_dir.join(format!("range_recon_{sample_idx:03}.png"));
        save_side_by_side_range(
            &target_values[offset..offset + h * w],
            &recon_values[offset..offset + h * w],
            h,
            w,
            &path,
        )?;
    }

    Ok(())
}

fn write_rgb_previews(
    model: &RgbAutoencoder<InnerBackend>,
    dataset: &FusionDataset,
    out_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let device = Device::<InnerBackend>::default();
    let Some(batch) = dataset.batches(BatchOptions::new(4))?.next() else {
        return Ok(());
    };
    let raw_batch = batch?;
    let batch_size = raw_batch.batch_size();
    let batch = raw_batch.into_burn::<InnerBackend>(&device);
    let target = batch.rgb.clone();
    let output = model.forward(batch.rgb);
    let target_values = tensor_values(target)?;
    let recon_values = tensor_values(output.rgb_hat)?;

    let h = dataset.rgb_shape().height;
    let w = dataset.rgb_shape().width;
    let c = dataset.rgb_shape().channels;
    let sample_values = c * h * w;
    for sample_idx in 0..batch_size {
        let offset = sample_idx * sample_values;
        let path = out_dir.join(format!("rgb_recon_{sample_idx:03}.png"));
        save_side_by_side_rgb(
            &target_values[offset..offset + sample_values],
            &recon_values[offset..offset + sample_values],
            h,
            w,
            &path,
        )?;
    }

    Ok(())
}

fn tensor_values<B: Backend>(tensor: Tensor<B, 4>) -> Result<Vec<f32>> {
    TensorData::convert::<f32>(tensor.to_data())
        .to_vec::<f32>()
        .context("converting tensor data to f32 values")
}

fn save_side_by_side_range(
    original: &[f32],
    reconstruction: &[f32],
    h: usize,
    w: usize,
    path: &Path,
) -> Result<()> {
    let mut pixels = vec![0u8; h * w * 2];
    for row in 0..h {
        for col in 0..w {
            let src = row * w + col;
            pixels[row * (2 * w) + col] = (original[src].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[row * (2 * w) + w + col] = (reconstruction[src].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    let img = image::GrayImage::from_raw((2 * w) as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create preview image"))?;
    img.save(path)
        .with_context(|| format!("saving {}", path.display()))?;
    Ok(())
}

fn save_side_by_side_rgb(
    original: &[f32],
    reconstruction: &[f32],
    h: usize,
    w: usize,
    path: &Path,
) -> Result<()> {
    let mut img = image::RgbImage::new((2 * w) as u32, h as u32);
    for row in 0..h {
        for col in 0..w {
            let src = row * w + col;
            let original_pixel = image::Rgb([
                channel_value(original, h, w, 0, src),
                channel_value(original, h, w, 1, src),
                channel_value(original, h, w, 2, src),
            ]);
            let recon_pixel = image::Rgb([
                channel_value(reconstruction, h, w, 0, src),
                channel_value(reconstruction, h, w, 1, src),
                channel_value(reconstruction, h, w, 2, src),
            ]);
            img.put_pixel(col as u32, row as u32, original_pixel);
            img.put_pixel((w + col) as u32, row as u32, recon_pixel);
        }
    }

    img.save(path)
        .with_context(|| format!("saving {}", path.display()))?;
    Ok(())
}

fn channel_value(values: &[f32], h: usize, w: usize, channel: usize, pixel_index: usize) -> u8 {
    let offset = channel * h * w + pixel_index;
    (values[offset].clamp(0.0, 1.0) * 255.0) as u8
}
