use crate::{EvalArgs, ProjectPaths, display_from_root, resolve_run_dir};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use sfx_core::manifest::{ProcessedSampleManifest, Split, read_manifest};
use sfx_data::{BatchOptions, FusionDataset};
use sfx_eval::{
    SampleEvalResult, compute_modality_metrics, compute_range_depth_metrics, compute_rgb_ssim,
    summarize_split, write_eval_csv, write_eval_json, write_eval_markdown,
};
use sfx_train::{RunInference, TrainingSummary};
use std::fs;
use std::path::{Path, PathBuf};

const EVAL_BATCH: usize = 8;

pub(crate) fn run(args: EvalArgs, paths: &ProjectPaths) -> Result<()> {
    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let (manifest_path, split_protocol, dataset_config_path) =
        resolve_manifest_path(paths, &run_dir)?;
    if !manifest_path.exists() {
        println!(
            "warn processed sample manifest not found at {}",
            display_from_root(&paths.root, &manifest_path)
        );
        return Ok(());
    }

    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let inferer = RunInference::load(&run_dir).context("loading run checkpoint")?;
    let processed_dir = resolve_processed_dir(paths, &manifest);
    let range_eval = resolve_range_eval_config(
        paths,
        &run_dir,
        &processed_dir,
        dataset_config_path.as_deref(),
    )?;
    let selected_splits = parse_splits(&args.split)?;

    let eval_dir = run_dir.join("eval");
    fs::create_dir_all(&eval_dir).with_context(|| format!("creating {}", eval_dir.display()))?;
    println!(
        "using split protocol `{split_protocol}` with manifest {}",
        display_from_root(&paths.root, &manifest_path)
    );
    println!(
        "range depth metrics: max_range_meters={:.3}, validity_mask={}",
        range_eval.max_range_meters,
        range_eval.validity_mask_label()
    );

    println!(
        "| split | samples | rgb mse | rgb proxy-psnr | rgb proxy-ssim | range mse | range proxy-psnr | range depth-mae(m) | range depth-rmse(m) | range δ<1.25 | range valid-frac |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    for split in selected_splits {
        let name = split_name(&split);
        let dataset = FusionDataset::open_split(&processed_dir, &manifest_path, Some(split))
            .with_context(|| format!("opening {name} dataset"))?;
        let mut results = Vec::new();

        for batch in dataset.batches(BatchOptions::new(EVAL_BATCH))? {
            let batch = batch?;
            let recon = inferer.reconstruct(&batch)?;
            let rgb_values_per_sample = batch.rgb_shape.value_count();
            let range_values_per_sample = batch.range_shape.value_count();
            let (rgb_height, rgb_width) = (batch.rgb_shape.height, batch.rgb_shape.width);
            let range_pixels = batch.range_shape.height * batch.range_shape.width;
            if range_pixels == 0 {
                return Err(anyhow!("range tensor has zero pixels"));
            }
            if range_eval
                .validity_channel_index
                .is_some_and(|index| index >= batch.range_shape.channels)
            {
                return Err(anyhow!(
                    "validity channel index {} out of bounds for range tensor with {} channels",
                    range_eval
                        .validity_channel_index
                        .expect("validated is_some above"),
                    batch.range_shape.channels
                ));
            }

            for (index, id) in batch.sample_ids.iter().enumerate() {
                let rgb_metrics = recon.rgb_hat.as_ref().map(|hat| {
                    let start = index * rgb_values_per_sample;
                    let end = start + rgb_values_per_sample;
                    let truth = &batch.rgb[start..end];
                    let predicted = &hat[start..end];
                    let mut metrics = compute_modality_metrics(predicted, truth, 1.0);
                    metrics.ssim = Some(compute_rgb_ssim(predicted, truth, rgb_height, rgb_width));
                    metrics
                });
                let (range_metrics, range_depth_metrics) = recon
                    .range_hat
                    .as_ref()
                    .map(|hat| {
                        let start = index * range_values_per_sample;
                        let end = start + range_values_per_sample;
                        let truth = &batch.range[start..end];
                        let predicted = &hat[start..end];
                        let depth_truth = &truth[..range_pixels];
                        let depth_predicted = &predicted[..range_pixels];
                        let range_metrics =
                            compute_modality_metrics(depth_predicted, depth_truth, 1.0);
                        let range_depth_metrics = compute_range_depth_metrics(
                            predicted,
                            truth,
                            batch.range_shape.channels,
                            batch.range_shape.height,
                            batch.range_shape.width,
                            range_eval.validity_channel_index,
                            range_eval.max_range_meters,
                        );
                        (Some(range_metrics), Some(range_depth_metrics))
                    })
                    .unwrap_or((None, None));

                results.push(SampleEvalResult {
                    sample_id: id.0.clone(),
                    rgb_metrics,
                    range_metrics,
                    range_depth_metrics,
                });
            }
        }

        let summary = summarize_split(&results, name);
        write_eval_json(&summary, &eval_dir.join(format!("{name}.json")))?;
        write_eval_csv(&results, &eval_dir.join(format!("{name}.csv")))?;
        write_eval_markdown(&summary, &eval_dir.join(format!("{name}.md")))?;

        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            summary.split,
            summary.n_samples,
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_mse)),
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_psnr_db)),
            format_metric(summary.rgb.as_ref().and_then(|metrics| metrics.mean_ssim)),
            format_metric(summary.range.as_ref().map(|metrics| metrics.mean_mse)),
            format_metric(summary.range.as_ref().map(|metrics| metrics.mean_psnr_db)),
            format_metric(
                summary
                    .range_depth
                    .as_ref()
                    .and_then(|metrics| metrics.mean_depth_mae_m)
            ),
            format_metric(
                summary
                    .range_depth
                    .as_ref()
                    .and_then(|metrics| metrics.mean_depth_rmse_m)
            ),
            format_metric(
                summary
                    .range_depth
                    .as_ref()
                    .and_then(|metrics| metrics.mean_delta_lt_1_25)
            ),
            format_metric(
                summary
                    .range_depth
                    .as_ref()
                    .map(|metrics| metrics.mean_valid_pixel_fraction)
            ),
        );
    }

    println!("ok   {}", display_from_root(&paths.root, &eval_dir));
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RangeEvalConfig {
    max_range_meters: f32,
    validity_channel_index: Option<usize>,
}

impl RangeEvalConfig {
    fn validity_mask_label(&self) -> String {
        self.validity_channel_index
            .map(|index| format!("channel[{index}]>0.5 (raw-return semantics)"))
            .unwrap_or_else(|| "truth depth > 0 fallback".to_string())
    }
}

#[derive(Debug, Deserialize)]
struct NormalizationManifest {
    range_normalization: RangeNormalizationDetails,
}

#[derive(Debug, Deserialize)]
struct RangeNormalizationDetails {
    max_range_meters: f32,
    #[serde(default)]
    configured_output_channels: Vec<String>,
}

fn resolve_manifest_path(
    paths: &ProjectPaths,
    run_dir: &std::path::Path,
) -> Result<(PathBuf, String, Option<PathBuf>)> {
    let default_manifest = paths.processed_sample_manifest_path();
    let summary_path = run_dir.join("summary.json");
    if !summary_path.exists() {
        return Ok((default_manifest, "mixed".to_string(), None));
    }

    let summary_text = fs::read_to_string(&summary_path)
        .with_context(|| format!("reading {}", summary_path.display()))?;
    let summary: TrainingSummary = serde_json::from_str(&summary_text)
        .with_context(|| format!("parsing {}", summary_path.display()))?;
    let split_protocol = if summary.split_protocol.trim().is_empty() {
        "mixed".to_string()
    } else {
        summary.split_protocol
    };
    let manifest_path = if summary.dataset_manifest_path.as_os_str().is_empty() {
        default_manifest
    } else {
        sfx_config::resolve_from_root(&paths.root, summary.dataset_manifest_path)
    };
    let dataset_config_path = (!summary.dataset_config_path.as_os_str().is_empty())
        .then(|| sfx_config::resolve_from_root(&paths.root, summary.dataset_config_path));

    Ok((manifest_path, split_protocol, dataset_config_path))
}

fn resolve_range_eval_config(
    paths: &ProjectPaths,
    run_dir: &Path,
    processed_dir: &Path,
    dataset_config_path: Option<&Path>,
) -> Result<RangeEvalConfig> {
    let mut max_range_meters = sfx_preprocess::MAX_RANGE_METERS;
    let mut validity_channel_index = None;
    let normalization_path = processed_dir.join("normalization.json");

    if normalization_path.exists() {
        let raw = fs::read_to_string(&normalization_path)
            .with_context(|| format!("reading {}", normalization_path.display()))?;
        let normalization: NormalizationManifest = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", normalization_path.display()))?;
        if normalization.range_normalization.max_range_meters > 0.0 {
            max_range_meters = normalization.range_normalization.max_range_meters;
        }
        validity_channel_index = validity_channel_index.or_else(|| {
            validity_channel_index_from_names(
                &normalization.range_normalization.configured_output_channels,
            )
        });
    }

    let config_candidates = [
        dataset_config_path.map(Path::to_path_buf),
        Some(run_dir.join("dataset.toml")),
    ];
    for candidate in config_candidates.into_iter().flatten() {
        if !candidate.exists() {
            continue;
        }
        let dataset_config = sfx_config::load_dataset_config(&paths.root, &candidate)
            .with_context(|| format!("loading dataset config {}", candidate.display()))?;
        validity_channel_index = validity_channel_index
            .or_else(|| validity_channel_index_from_names(&dataset_config.range_channels));
        break;
    }

    Ok(RangeEvalConfig {
        max_range_meters,
        validity_channel_index,
    })
}

fn validity_channel_index_from_names(channels: &[String]) -> Option<usize> {
    channels
        .iter()
        .position(|channel| is_validity_channel_name(channel))
}

fn is_validity_channel_name(channel: &str) -> bool {
    matches!(
        channel.trim().to_ascii_lowercase().as_str(),
        "valid" | "validity" | "validity-mask" | "mask"
    )
}

fn parse_splits(value: &str) -> Result<Vec<Split>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" => Ok(vec![Split::Train, Split::Val, Split::Test]),
        "train" => Ok(vec![Split::Train]),
        "val" | "validation" => Ok(vec![Split::Val]),
        "test" => Ok(vec![Split::Test]),
        other => Err(anyhow!(
            "unsupported split {other}; expected all, train, val, or test"
        )),
    }
}

fn split_name(split: &Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

fn resolve_processed_dir(paths: &ProjectPaths, manifest: &ProcessedSampleManifest) -> PathBuf {
    if let Some(sample) = manifest.samples.first() {
        let root_relative = paths.root.join(&sample.meta.rgb_path);
        if root_relative.exists() {
            return paths.root.clone();
        }
    }

    if let Some(dataset) = manifest.dataset.as_deref() {
        let candidate = paths.root.join("data").join("processed").join(dataset);
        if manifest
            .samples
            .first()
            .map(|sample| candidate.join(&sample.meta.rgb_path).exists())
            .unwrap_or(true)
        {
            return candidate;
        }
    }

    paths.root.join("data").join("processed")
}

fn format_metric(value: Option<f32>) -> String {
    value
        .map(|metric| {
            if metric.is_infinite() {
                "inf".to_string()
            } else {
                format!("{metric:.6}")
            }
        })
        .unwrap_or_else(|| "-".to_string())
}
