use crate::{EvalArgs, ProjectPaths, display_from_root, resolve_run_dir};
use anyhow::{Context, Result, anyhow};
use sfx_core::manifest::{ProcessedSampleManifest, Split, read_manifest};
use sfx_data::{BatchOptions, FusionDataset};
use sfx_eval::{
    SampleEvalResult, compute_modality_metrics, compute_rgb_ssim, summarize_split, write_eval_csv,
    write_eval_json, write_eval_markdown,
};
use sfx_train::RunInference;
use std::fs;
use std::path::PathBuf;

const EVAL_BATCH: usize = 8;

pub(crate) fn run(args: EvalArgs, paths: &ProjectPaths) -> Result<()> {
    let manifest_path = paths.processed_sample_manifest_path();
    if !manifest_path.exists() {
        println!(
            "warn processed sample manifest not found at {}",
            display_from_root(&paths.root, &manifest_path)
        );
        return Ok(());
    }

    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;

    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let inferer = RunInference::load(&run_dir).context("loading run checkpoint")?;
    let processed_dir = resolve_processed_dir(paths, &manifest);
    let selected_splits = parse_splits(&args.split)?;

    let eval_dir = run_dir.join("eval");
    fs::create_dir_all(&eval_dir).with_context(|| format!("creating {}", eval_dir.display()))?;

    println!("| split | samples | rgb mse | rgb psnr | rgb ssim | range mse | range psnr |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");

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
                let range_metrics = recon.range_hat.as_ref().map(|hat| {
                    let start = index * range_values_per_sample;
                    let end = start + range_values_per_sample;
                    let truth = &batch.range[start..end];
                    let predicted = &hat[start..end];
                    compute_modality_metrics(predicted, truth, 1.0)
                });

                results.push(SampleEvalResult {
                    sample_id: id.0.clone(),
                    rgb_metrics,
                    range_metrics,
                });
            }
        }

        let summary = summarize_split(&results, name);
        write_eval_json(&summary, &eval_dir.join(format!("{name}.json")))?;
        write_eval_csv(&results, &eval_dir.join(format!("{name}.csv")))?;
        write_eval_markdown(&summary, &eval_dir.join(format!("{name}.md")))?;

        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            summary.split,
            summary.n_samples,
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_mse)),
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_psnr_db)),
            format_metric(summary.rgb.as_ref().and_then(|metrics| metrics.mean_ssim)),
            format_metric(summary.range.as_ref().map(|metrics| metrics.mean_mse)),
            format_metric(summary.range.as_ref().map(|metrics| metrics.mean_psnr_db)),
        );
    }

    println!("ok   {}", display_from_root(&paths.root, &eval_dir));
    Ok(())
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
