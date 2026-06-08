use anyhow::{Context, Result, bail};
use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::cast::ToElement;
use burn::tensor::{Device, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use sfx_config::ModelKind;
use sfx_core::manifest::{
    LatestRun, MANIFEST_SCHEMA_VERSION, RunIndex, RunIndexEntry, RunStatus, Split, read_manifest,
    write_manifest,
};
use sfx_data::{BatchOptions, FusionDataset};
use sfx_models::{
    RangeAutoencoder, RangeAutoencoderConfig, RgbAutoencoder, RgbAutoencoderConfig,
    SharedLatentMultimodalAutoencoder, SharedLatentMultimodalAutoencoderConfig,
};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod inference;
pub use inference::{Reconstruction, RunInference};

pub const CRATE_NAME: &str = "sfx-train";

type InnerBackend = Flex;
type TrainBackend = Autodiff<InnerBackend>;

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    #[serde(default)]
    pub run_id: String,
    pub run_name: String,
    #[serde(default)]
    pub model_kind: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub started_at: String,
    #[serde(default)]
    pub completed_at: Option<String>,
    pub run_dir: PathBuf,
    #[serde(default)]
    pub config_path: PathBuf,
    #[serde(default)]
    pub dataset_config_path: PathBuf,
    #[serde(default)]
    pub model_config_path: PathBuf,
    pub metrics_path: PathBuf,
    pub summary_path: PathBuf,
    pub checkpoint_path: PathBuf,
    #[serde(default)]
    pub optimizer_path: PathBuf,
    pub epochs: usize,
    pub train_samples: usize,
    pub batch_size: usize,
    #[serde(default)]
    pub max_batches_per_epoch: Option<usize>,
    pub latent_dim: usize,
    #[serde(default)]
    pub z_modality: Option<usize>,
    pub final_train_loss: f64,
    #[serde(default)]
    pub final_val_loss: Option<f64>,
    pub backend: String,
    #[serde(default)]
    pub optimizer: String,
}

#[derive(Debug, Clone)]
pub struct ResumeReport {
    pub run_dir: PathBuf,
    pub summary: TrainingSummary,
    pub checkpoint_exists: bool,
    pub metrics_exists: bool,
    pub optimizer_metadata_exists: bool,
    pub can_resume_optimizer_state: bool,
    pub restart_config_path: PathBuf,
}

#[derive(Debug, Clone)]
struct RunContext {
    run_id: String,
    model_kind: String,
    started_at: String,
    run_dir: PathBuf,
    relative_run_dir: PathBuf,
    metrics_path: PathBuf,
    summary_path: PathBuf,
    checkpoint_path: PathBuf,
    optimizer_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct OptimizerMetadata {
    optimizer: String,
    state_checkpoint: Option<PathBuf>,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct EpochMetric {
    epoch: usize,
    train_loss: f64,
    val_loss: Option<f64>,
    rgb_train_loss: Option<f64>,
    range_train_loss: Option<f64>,
    rgb_val_loss: Option<f64>,
    range_val_loss: Option<f64>,
    sample_count: usize,
    batch_count: usize,
}

pub fn train_range_autoencoder(root: &Path, config_path: &Path) -> Result<TrainingSummary> {
    let config = sfx_config::load_training_config(root, config_path)
        .with_context(|| format!("loading training config {}", config_path.display()))?;
    if config.model.kind != ModelKind::RangeOnly {
        bail!(
            "range training requires model kind `range-only`; config uses {:?}",
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
    let val_dataset = FusionDataset::open_split(
        &config.dataset.processed_dir,
        &manifest_path,
        Some(Split::Val),
    )
    .with_context(|| {
        format!(
            "opening validation dataset from {}",
            manifest_path.display()
        )
    })?;
    if !val_dataset.is_empty() {
        val_dataset.validate_tensor_files()?;
    }

    let run = prepare_run(
        root,
        ModelKind::RangeOnly,
        &config.run_name,
        config_path,
        &config,
    )?;

    let device = Device::<TrainBackend>::default();
    TrainBackend::seed(&device, config.seed);
    let mut model =
        RangeAutoencoderConfig::new(config.model.latent_dim).init::<TrainBackend>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainBackend, RangeAutoencoder<TrainBackend>>();

    let metrics_path = run.metrics_path.clone();
    let mut metric_lines = String::new();
    let mut final_train_loss = f64::INFINITY;
    let mut final_val_loss = None;
    let total_batches_per_epoch = train_dataset.len().div_ceil(config.batch_size);
    let batches_per_epoch = config
        .max_batches_per_epoch
        .unwrap_or(total_batches_per_epoch)
        .min(total_batches_per_epoch);
    println!(
        "range training: samples={} val_samples={} batch_size={} batches/epoch={} epochs={}",
        train_dataset.len(),
        val_dataset.len(),
        config.batch_size,
        batches_per_epoch,
        config.epochs
    );

    for epoch in 1..=config.epochs {
        let options = BatchOptions::new(config.batch_size).shuffled(config.seed + epoch as u64);
        let mut total_loss = 0.0f64;
        let mut total_samples = 0usize;
        let mut batch_count = 0usize;

        for batch in train_dataset.batches(options)? {
            let batch = batch?.into_burn::<TrainBackend>(&device);
            let current_batch_size = batch.batch_size();
            batch_count += 1;
            let target = batch.range.clone();
            let output = model.forward(batch.range);
            let loss = mse_loss(output.range_hat, target);
            let loss_value = loss.clone().into_scalar().to_f64();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);

            total_loss += loss_value * current_batch_size as f64;
            total_samples += current_batch_size;
            if batch_count >= batches_per_epoch {
                break;
            }
        }

        final_train_loss = total_loss / total_samples.max(1) as f64;
        let val_loss = evaluate_range_autoencoder(&model.valid(), &val_dataset, config.batch_size)?;
        final_val_loss = val_loss;
        let metric = EpochMetric {
            epoch,
            train_loss: final_train_loss,
            val_loss,
            rgb_train_loss: None,
            range_train_loss: None,
            rgb_val_loss: None,
            range_val_loss: None,
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
    let checkpoint_stem = run.run_dir.join("model");
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
    write_optimizer_metadata(&run.optimizer_path, "adam")?;
    write_range_previews(&valid_model, &train_dataset, &run.run_dir.join("previews"))?;

    let completed_at = now_timestamp();
    let summary = TrainingSummary {
        run_id: run.run_id.clone(),
        run_name: run.run_id.clone(),
        model_kind: run.model_kind.clone(),
        status: "completed".to_string(),
        started_at: run.started_at.clone(),
        completed_at: Some(completed_at.clone()),
        run_dir: run.run_dir.clone(),
        config_path: sfx_config::resolve_from_root(root, config_path),
        dataset_config_path: config.dataset_config_path.clone(),
        model_config_path: config.model_config_path.clone(),
        metrics_path: metrics_path.clone(),
        summary_path: run.summary_path.clone(),
        checkpoint_path: run.checkpoint_path.clone(),
        optimizer_path: run.optimizer_path.clone(),
        epochs: config.epochs,
        train_samples: train_dataset.len(),
        batch_size: config.batch_size,
        max_batches_per_epoch: config.max_batches_per_epoch,
        latent_dim: config.model.latent_dim,
        z_modality: None,
        final_train_loss,
        final_val_loss,
        backend: "burn-flex-autodiff".to_string(),
        optimizer: "adam".to_string(),
    };
    std::fs::write(
        &summary.summary_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("writing {}", summary.summary_path.display()))?;
    mark_run_completed(root, &run, &completed_at)?;

    Ok(summary)
}

fn evaluate_range_autoencoder(
    model: &RangeAutoencoder<InnerBackend>,
    dataset: &FusionDataset,
    batch_size: usize,
) -> Result<Option<f64>> {
    if dataset.is_empty() {
        return Ok(None);
    }

    let device = Device::<InnerBackend>::default();
    let mut total_loss = 0.0f64;
    let mut total_samples = 0usize;

    for batch in dataset.batches(BatchOptions::new(batch_size))? {
        let batch = batch?.into_burn::<InnerBackend>(&device);
        let current_batch_size = batch.batch_size();
        let target = batch.range.clone();
        let output = model.forward(batch.range);
        let loss = mse_loss(output.range_hat, target);
        let loss_value = loss.into_scalar().to_f64();
        total_loss += loss_value * current_batch_size as f64;
        total_samples += current_batch_size;
    }

    Ok(Some(total_loss / total_samples.max(1) as f64))
}

fn evaluate_rgb_autoencoder(
    model: &RgbAutoencoder<InnerBackend>,
    dataset: &FusionDataset,
    batch_size: usize,
) -> Result<Option<f64>> {
    if dataset.is_empty() {
        return Ok(None);
    }

    let device = Device::<InnerBackend>::default();
    let mut total_loss = 0.0f64;
    let mut total_samples = 0usize;

    for batch in dataset.batches(BatchOptions::new(batch_size))? {
        let batch = batch?.into_burn::<InnerBackend>(&device);
        let current_batch_size = batch.batch_size();
        let target = batch.rgb.clone();
        let output = model.forward(batch.rgb);
        let loss = mse_loss(output.rgb_hat, target);
        let loss_value = loss.into_scalar().to_f64();
        total_loss += loss_value * current_batch_size as f64;
        total_samples += current_batch_size;
    }

    Ok(Some(total_loss / total_samples.max(1) as f64))
}

fn evaluate_fusion_autoencoder(
    model: &SharedLatentMultimodalAutoencoder<InnerBackend>,
    dataset: &FusionDataset,
    batch_size: usize,
) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
    if dataset.is_empty() {
        return Ok((None, None, None));
    }

    let device = Device::<InnerBackend>::default();
    let mut total_loss = 0.0f64;
    let mut total_rgb_loss = 0.0f64;
    let mut total_range_loss = 0.0f64;
    let mut total_samples = 0usize;

    for batch in dataset.batches(BatchOptions::new(batch_size))? {
        let batch = batch?.into_burn::<InnerBackend>(&device);
        let current_batch_size = batch.batch_size();
        let rgb_target = batch.rgb.clone();
        let range_target = batch.range.clone();
        let output = model.forward(batch.rgb, batch.range);
        let rgb_loss = mse_loss(output.rgb_hat, rgb_target);
        let range_loss = mse_loss(output.range_hat, range_target);
        let rgb_loss_value = rgb_loss.into_scalar().to_f64();
        let range_loss_value = range_loss.into_scalar().to_f64();
        total_rgb_loss += rgb_loss_value * current_batch_size as f64;
        total_range_loss += range_loss_value * current_batch_size as f64;
        total_loss += (rgb_loss_value + range_loss_value) * current_batch_size as f64;
        total_samples += current_batch_size;
    }

    let denom = total_samples.max(1) as f64;
    Ok((
        Some(total_loss / denom),
        Some(total_rgb_loss / denom),
        Some(total_range_loss / denom),
    ))
}

pub fn train_rgb_autoencoder(root: &Path, config_path: &Path) -> Result<TrainingSummary> {
    let config = sfx_config::load_training_config(root, config_path)
        .with_context(|| format!("loading training config {}", config_path.display()))?;
    if config.model.kind != ModelKind::RgbOnly {
        bail!(
            "rgb training requires model kind `rgb-only`; config uses {:?}",
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
    let val_dataset = FusionDataset::open_split(
        &config.dataset.processed_dir,
        &manifest_path,
        Some(Split::Val),
    )
    .with_context(|| {
        format!(
            "opening validation dataset from {}",
            manifest_path.display()
        )
    })?;
    if !val_dataset.is_empty() {
        val_dataset.validate_tensor_files()?;
    }

    let run = prepare_run(
        root,
        ModelKind::RgbOnly,
        &config.run_name,
        config_path,
        &config,
    )?;

    let device = Device::<TrainBackend>::default();
    TrainBackend::seed(&device, config.seed);
    let mut model =
        RgbAutoencoderConfig::new(config.model.latent_dim).init::<TrainBackend>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainBackend, RgbAutoencoder<TrainBackend>>();

    let metrics_path = run.metrics_path.clone();
    let mut metric_lines = String::new();
    let mut final_train_loss = f64::INFINITY;
    let mut final_val_loss = None;
    let total_batches_per_epoch = train_dataset.len().div_ceil(config.batch_size);
    let batches_per_epoch = config
        .max_batches_per_epoch
        .unwrap_or(total_batches_per_epoch)
        .min(total_batches_per_epoch);
    println!(
        "rgb training: samples={} val_samples={} batch_size={} batches/epoch={} epochs={}",
        train_dataset.len(),
        val_dataset.len(),
        config.batch_size,
        batches_per_epoch,
        config.epochs
    );

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
            if batch_count >= batches_per_epoch {
                break;
            }
        }

        final_train_loss = total_loss / total_samples.max(1) as f64;
        let val_loss = evaluate_rgb_autoencoder(&model.valid(), &val_dataset, config.batch_size)?;
        final_val_loss = val_loss;
        let metric = EpochMetric {
            epoch,
            train_loss: final_train_loss,
            val_loss,
            rgb_train_loss: None,
            range_train_loss: None,
            rgb_val_loss: None,
            range_val_loss: None,
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
    let checkpoint_stem = run.run_dir.join("model");
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
    write_optimizer_metadata(&run.optimizer_path, "adam")?;
    write_rgb_previews(&valid_model, &train_dataset, &run.run_dir.join("previews"))?;

    let completed_at = now_timestamp();
    let summary = TrainingSummary {
        run_id: run.run_id.clone(),
        run_name: run.run_id.clone(),
        model_kind: run.model_kind.clone(),
        status: "completed".to_string(),
        started_at: run.started_at.clone(),
        completed_at: Some(completed_at.clone()),
        run_dir: run.run_dir.clone(),
        config_path: sfx_config::resolve_from_root(root, config_path),
        dataset_config_path: config.dataset_config_path.clone(),
        model_config_path: config.model_config_path.clone(),
        metrics_path: metrics_path.clone(),
        summary_path: run.summary_path.clone(),
        checkpoint_path: run.checkpoint_path.clone(),
        optimizer_path: run.optimizer_path.clone(),
        epochs: config.epochs,
        train_samples: train_dataset.len(),
        batch_size: config.batch_size,
        max_batches_per_epoch: config.max_batches_per_epoch,
        latent_dim: config.model.latent_dim,
        z_modality: None,
        final_train_loss,
        final_val_loss,
        backend: "burn-flex-autodiff".to_string(),
        optimizer: "adam".to_string(),
    };
    std::fs::write(
        &summary.summary_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("writing {}", summary.summary_path.display()))?;
    mark_run_completed(root, &run, &completed_at)?;

    Ok(summary)
}

pub fn train_fusion_autoencoder(root: &Path, config_path: &Path) -> Result<TrainingSummary> {
    let config = sfx_config::load_training_config(root, config_path)
        .with_context(|| format!("loading training config {}", config_path.display()))?;
    if config.model.kind != ModelKind::Fusion {
        bail!(
            "fusion training requires model kind `fusion`; config uses {:?}",
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
    let val_dataset = FusionDataset::open_split(
        &config.dataset.processed_dir,
        &manifest_path,
        Some(Split::Val),
    )
    .with_context(|| {
        format!(
            "opening validation dataset from {}",
            manifest_path.display()
        )
    })?;
    if !val_dataset.is_empty() {
        val_dataset.validate_tensor_files()?;
    }

    let run = prepare_run(
        root,
        ModelKind::Fusion,
        &config.run_name,
        config_path,
        &config,
    )?;

    let device = Device::<TrainBackend>::default();
    TrainBackend::seed(&device, config.seed);
    let mut model = SharedLatentMultimodalAutoencoderConfig::new(
        config.model.latent_dim,
        config.model.effective_z_modality(),
    )
    .init::<TrainBackend>(&device);
    let mut optimizer =
        AdamConfig::new().init::<TrainBackend, SharedLatentMultimodalAutoencoder<TrainBackend>>();

    let metrics_path = run.metrics_path.clone();
    let mut metric_lines = String::new();
    let mut final_train_loss = f64::INFINITY;
    let mut final_val_loss = None;
    let total_batches_per_epoch = train_dataset.len().div_ceil(config.batch_size);
    let batches_per_epoch = config
        .max_batches_per_epoch
        .unwrap_or(total_batches_per_epoch)
        .min(total_batches_per_epoch);
    println!(
        "fusion training: samples={} val_samples={} batch_size={} batches/epoch={} epochs={}",
        train_dataset.len(),
        val_dataset.len(),
        config.batch_size,
        batches_per_epoch,
        config.epochs
    );

    for epoch in 1..=config.epochs {
        let options = BatchOptions::new(config.batch_size).shuffled(config.seed + epoch as u64);
        let mut total_loss = 0.0f64;
        let mut total_rgb_loss = 0.0f64;
        let mut total_range_loss = 0.0f64;
        let mut total_samples = 0usize;
        let mut batch_count = 0usize;

        for batch in train_dataset.batches(options)? {
            let batch = batch?.into_burn::<TrainBackend>(&device);
            let current_batch_size = batch.batch_size();
            batch_count += 1;
            if batch_count == 1 || batch_count.is_multiple_of(5) || batch_count == batches_per_epoch
            {
                println!(
                    "epoch {epoch:03}/{:03} batch {batch_count}/{batches_per_epoch}",
                    config.epochs
                );
            }
            let rgb_target = batch.rgb.clone();
            let range_target = batch.range.clone();
            let output = model.forward(batch.rgb, batch.range);
            let rgb_loss = mse_loss(output.rgb_hat, rgb_target);
            let range_loss = mse_loss(output.range_hat, range_target);
            let rgb_loss_value = rgb_loss.clone().into_scalar().to_f64();
            let range_loss_value = range_loss.clone().into_scalar().to_f64();
            let loss = rgb_loss + range_loss;
            let loss_value = loss.clone().into_scalar().to_f64();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);

            total_loss += loss_value * current_batch_size as f64;
            total_rgb_loss += rgb_loss_value * current_batch_size as f64;
            total_range_loss += range_loss_value * current_batch_size as f64;
            total_samples += current_batch_size;
            if batch_count >= batches_per_epoch {
                break;
            }
        }

        final_train_loss = total_loss / total_samples.max(1) as f64;
        let rgb_train_loss = total_rgb_loss / total_samples.max(1) as f64;
        let range_train_loss = total_range_loss / total_samples.max(1) as f64;
        let (val_loss, rgb_val_loss, range_val_loss) =
            evaluate_fusion_autoencoder(&model.valid(), &val_dataset, config.batch_size)?;
        final_val_loss = val_loss;
        let metric = EpochMetric {
            epoch,
            train_loss: final_train_loss,
            val_loss,
            rgb_train_loss: Some(rgb_train_loss),
            range_train_loss: Some(range_train_loss),
            rgb_val_loss,
            range_val_loss,
            sample_count: total_samples,
            batch_count,
        };
        metric_lines.push_str(&serde_json::to_string(&metric)?);
        metric_lines.push('\n');
        println!(
            "epoch {epoch:03}/{:03} train_loss={:.6} rgb={:.6} range={:.6} batches={batch_count}",
            config.epochs, final_train_loss, rgb_train_loss, range_train_loss
        );
    }

    std::fs::write(&metrics_path, metric_lines)
        .with_context(|| format!("writing {}", metrics_path.display()))?;

    let valid_model = model.valid();
    let checkpoint_stem = run.run_dir.join("model");
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
    write_optimizer_metadata(&run.optimizer_path, "adam")?;
    write_fusion_previews(&valid_model, &train_dataset, &run.run_dir.join("previews"))?;

    let completed_at = now_timestamp();
    let summary = TrainingSummary {
        run_id: run.run_id.clone(),
        run_name: run.run_id.clone(),
        model_kind: run.model_kind.clone(),
        status: "completed".to_string(),
        started_at: run.started_at.clone(),
        completed_at: Some(completed_at.clone()),
        run_dir: run.run_dir.clone(),
        config_path: sfx_config::resolve_from_root(root, config_path),
        dataset_config_path: config.dataset_config_path.clone(),
        model_config_path: config.model_config_path.clone(),
        metrics_path: metrics_path.clone(),
        summary_path: run.summary_path.clone(),
        checkpoint_path: run.checkpoint_path.clone(),
        optimizer_path: run.optimizer_path.clone(),
        epochs: config.epochs,
        train_samples: train_dataset.len(),
        batch_size: config.batch_size,
        max_batches_per_epoch: config.max_batches_per_epoch,
        latent_dim: config.model.latent_dim,
        z_modality: Some(config.model.effective_z_modality()),
        final_train_loss,
        final_val_loss,
        backend: "burn-flex-autodiff".to_string(),
        optimizer: "adam".to_string(),
    };
    std::fs::write(
        &summary.summary_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("writing {}", summary.summary_path.display()))?;
    mark_run_completed(root, &run, &completed_at)?;

    Ok(summary)
}

pub fn inspect_resume_run(root: &Path, run_path: &Path) -> Result<ResumeReport> {
    let run_dir = sfx_config::resolve_from_root(root, run_path);
    let summary_path = run_dir.join("summary.json");
    let summary_text = std::fs::read_to_string(&summary_path)
        .with_context(|| format!("reading {}", summary_path.display()))?;
    let mut summary: TrainingSummary = serde_json::from_str(&summary_text)
        .with_context(|| format!("parsing {}", summary_path.display()))?;
    if summary.run_id.is_empty() {
        summary.run_id = run_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown-run")
            .to_string();
    }
    if summary.run_name.is_empty() {
        summary.run_name = summary.run_id.clone();
    }

    let checkpoint_path = if summary.checkpoint_path.as_os_str().is_empty() {
        run_dir.join("model.bin")
    } else {
        summary.checkpoint_path.clone()
    };
    let metrics_path = if summary.metrics_path.as_os_str().is_empty() {
        run_dir.join("metrics.jsonl")
    } else {
        summary.metrics_path.clone()
    };
    let optimizer_path = if summary.optimizer_path.as_os_str().is_empty() {
        run_dir.join("optimizer.json")
    } else {
        summary.optimizer_path.clone()
    };
    let run_config_path = run_dir.join("config.toml");
    let restart_config_path = if run_config_path.exists() {
        run_config_path
    } else if summary.config_path.as_os_str().is_empty() {
        run_dir.join("config.toml")
    } else {
        summary.config_path.clone()
    };

    Ok(ResumeReport {
        run_dir,
        summary,
        checkpoint_exists: checkpoint_path.exists(),
        metrics_exists: metrics_path.exists(),
        optimizer_metadata_exists: optimizer_path.exists(),
        can_resume_optimizer_state: false,
        restart_config_path,
    })
}

fn mse_loss<B: Backend>(pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    (pred - target).square().mean()
}

fn prepare_run(
    root: &Path,
    kind: ModelKind,
    requested_run_name: &str,
    config_path: &Path,
    config: &sfx_config::TrainingConfig,
) -> Result<RunContext> {
    let model_kind = model_kind_slug(kind).to_string();
    let run_id = allocate_run_id(root, &model_kind, requested_run_name)?;
    let relative_run_dir = PathBuf::from("artifacts/checkpoints")
        .join(&model_kind)
        .join(&run_id);
    let run_dir = root.join(&relative_run_dir);
    std::fs::create_dir_all(&run_dir).with_context(|| format!("creating {}", run_dir.display()))?;
    copy_if_exists(config_path, &run_dir.join("config.toml"))?;
    copy_if_exists(&config.dataset_config_path, &run_dir.join("dataset.toml"))?;
    copy_if_exists(&config.model_config_path, &run_dir.join("model.toml"))?;

    let run = RunContext {
        run_id,
        model_kind,
        started_at: now_timestamp(),
        metrics_path: run_dir.join("metrics.jsonl"),
        summary_path: run_dir.join("summary.json"),
        checkpoint_path: run_dir.join("model.bin"),
        optimizer_path: run_dir.join("optimizer.json"),
        run_dir,
        relative_run_dir,
    };
    upsert_run_status(root, &run, RunStatus::Running, None)?;
    Ok(run)
}

fn allocate_run_id(root: &Path, model_kind: &str, requested_run_name: &str) -> Result<String> {
    let base_dir = root.join("artifacts/checkpoints").join(model_kind);
    std::fs::create_dir_all(&base_dir)
        .with_context(|| format!("creating {}", base_dir.display()))?;
    if run_slot_available(&base_dir.join(requested_run_name))? {
        return Ok(requested_run_name.to_string());
    }

    for index in 2..10_000 {
        let candidate = format!("{requested_run_name}_{index:03}");
        if run_slot_available(&base_dir.join(&candidate))? {
            return Ok(candidate);
        }
    }

    bail!("could not allocate a free run directory for {requested_run_name}")
}

fn run_slot_available(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(true);
    }
    if !path.is_dir() {
        return Ok(false);
    }
    let mut entries = std::fs::read_dir(path)
        .with_context(|| format!("checking whether {} is empty", path.display()))?;
    Ok(entries.next().is_none())
}

fn mark_run_completed(root: &Path, run: &RunContext, completed_at: &str) -> Result<()> {
    upsert_run_status(
        root,
        run,
        RunStatus::Completed,
        Some(completed_at.to_string()),
    )?;
    let latest = LatestRun {
        schema_version: MANIFEST_SCHEMA_VERSION,
        run_id: Some(run.run_id.clone()),
        path: Some(run.relative_run_dir.clone()),
    };
    write_manifest(root.join(".xtask/runs/latest.json"), &latest).context("writing latest run")
}

fn upsert_run_status(
    root: &Path,
    run: &RunContext,
    status: RunStatus,
    completed_at: Option<String>,
) -> Result<()> {
    let path = root.join(".xtask/runs/run_index.json");
    let mut index = if path.exists() {
        read_manifest::<RunIndex>(&path).with_context(|| format!("reading {}", path.display()))?
    } else {
        RunIndex::default()
    };
    index.runs.retain(|entry| entry.run_id != run.run_id);
    index.runs.push(RunIndexEntry {
        run_id: run.run_id.clone(),
        model_kind: run.model_kind.clone(),
        path: run.relative_run_dir.clone(),
        status,
        started_at: Some(run.started_at.clone()),
        completed_at,
    });
    write_manifest(&path, &index).with_context(|| format!("writing {}", path.display()))
}

fn write_optimizer_metadata(path: &Path, optimizer: &str) -> Result<()> {
    let metadata = OptimizerMetadata {
        optimizer: optimizer.to_string(),
        state_checkpoint: None,
        note: "Optimizer state serialization is not yet supported by this prototype; restart from the saved config and model checkpoint instead.".to_string(),
    };
    let text = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(path, format!("{text}\n")).with_context(|| format!("writing {}", path.display()))
}

fn now_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{seconds}")
}

fn model_kind_slug(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::RgbOnly => "rgb",
        ModelKind::RangeOnly => "range",
        ModelKind::Fusion => "fusion",
    }
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

fn write_fusion_previews(
    model: &SharedLatentMultimodalAutoencoder<InnerBackend>,
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
    let rgb_target = batch.rgb.clone();
    let range_target = batch.range.clone();
    let output = model.forward(batch.rgb, batch.range);
    let rgb_target_values = tensor_values(rgb_target)?;
    let rgb_recon_values = tensor_values(output.rgb_hat)?;
    let range_target_values = tensor_values(range_target)?;
    let range_recon_values = tensor_values(output.range_hat)?;

    let rgb_h = dataset.rgb_shape().height;
    let rgb_w = dataset.rgb_shape().width;
    let rgb_c = dataset.rgb_shape().channels;
    let rgb_sample_values = rgb_c * rgb_h * rgb_w;
    let range_h = dataset.range_shape().height;
    let range_w = dataset.range_shape().width;
    let range_c = dataset.range_shape().channels;
    let range_sample_values = range_c * range_h * range_w;

    for sample_idx in 0..batch_size {
        let rgb_offset = sample_idx * rgb_sample_values;
        save_side_by_side_rgb(
            &rgb_target_values[rgb_offset..rgb_offset + rgb_sample_values],
            &rgb_recon_values[rgb_offset..rgb_offset + rgb_sample_values],
            rgb_h,
            rgb_w,
            &out_dir.join(format!("rgb_recon_{sample_idx:03}.png")),
        )?;

        let range_offset = sample_idx * range_sample_values;
        save_side_by_side_range(
            &range_target_values[range_offset..range_offset + range_h * range_w],
            &range_recon_values[range_offset..range_offset + range_h * range_w],
            range_h,
            range_w,
            &out_dir.join(format!("range_recon_{sample_idx:03}.png")),
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
