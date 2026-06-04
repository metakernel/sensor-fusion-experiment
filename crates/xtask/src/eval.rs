use crate::{EvalArgs, ProjectPaths, display_from_root, resolve_run_dir};
use anyhow::{Context, Result, anyhow, ensure};
use sfx_core::manifest::{ProcessedSampleManifest, Split, SplitsManifest, read_manifest};
use sfx_eval::{
    SampleEvalResult, compute_modality_metrics, summarize_split, write_eval_csv, write_eval_json,
    write_eval_markdown,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

pub(crate) fn run(args: EvalArgs, paths: &ProjectPaths) -> Result<()> {
    let manifest_path = paths.processed_sample_manifest_path();
    if !manifest_path.exists() {
        println!(
            "warn processed sample manifest not found at {}",
            display_from_root(&paths.root, &manifest_path)
        );
        return Ok(());
    }

    let splits_path = paths.splits_manifest_path();
    if !splits_path.exists() {
        println!(
            "warn splits manifest not found at {}",
            display_from_root(&paths.root, &splits_path)
        );
        return Ok(());
    }

    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let splits: SplitsManifest = read_manifest(&splits_path)
        .with_context(|| format!("reading {}", splits_path.display()))?;

    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let processed_dir = resolve_processed_dir(paths, &manifest);
    let selected_splits = parse_splits(&args.split)?;
    let rgb_shape = manifest.rgb_shape.as_ref();
    let range_shape = manifest.range_shape.as_ref();

    let eval_dir = run_dir.join("eval");
    fs::create_dir_all(&eval_dir).with_context(|| format!("creating {}", eval_dir.display()))?;

    println!("| split | samples | rgb mse | rgb psnr | range mse | range psnr |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: |");

    for split in selected_splits {
        let split_name = split_name(&split);
        let split_ids: BTreeSet<String> = split_ids_for(&splits, &split)
            .iter()
            .map(|sample_id| sample_id.0.clone())
            .collect();
        let mut results = Vec::new();
        for sample in manifest
            .samples
            .iter()
            .filter(|entry| split_ids.contains(&entry.meta.id.0))
        {
            let rgb = match rgb_shape {
                Some(shape) => Some(load_tensor(
                    &processed_dir.join(&sample.meta.rgb_path),
                    shape.value_count(),
                )?),
                None => None,
            };
            let range = match range_shape {
                Some(shape) => Some(load_tensor(
                    &processed_dir.join(&sample.meta.range_path),
                    shape.value_count(),
                )?),
                None => None,
            };

            // TODO: replace zero reconstructions with Burn checkpoint loading from run_dir.
            let rgb_metrics = rgb.as_ref().map(|truth| {
                let predicted = vec![0.0; truth.len()];
                compute_modality_metrics(&predicted, truth, 1.0)
            });
            let range_metrics = range.as_ref().map(|truth| {
                let predicted = vec![0.0; truth.len()];
                compute_modality_metrics(&predicted, truth, 1.0)
            });

            results.push(SampleEvalResult {
                sample_id: sample.meta.id.0.clone(),
                rgb_metrics,
                range_metrics,
            });
        }

        let summary = summarize_split(&results, split_name);
        write_eval_json(&summary, &eval_dir.join(format!("{split_name}.json")))?;
        write_eval_csv(&results, &eval_dir.join(format!("{split_name}.csv")))?;
        write_eval_markdown(&summary, &eval_dir.join(format!("{split_name}.md")))?;

        println!(
            "| {} | {} | {} | {} | {} | {} |",
            summary.split,
            summary.n_samples,
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_mse)),
            format_metric(summary.rgb.as_ref().map(|metrics| metrics.mean_psnr_db)),
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

fn split_ids_for<'a>(
    splits: &'a SplitsManifest,
    split: &Split,
) -> &'a [sfx_core::manifest::SampleId] {
    match split {
        Split::Train => &splits.train,
        Split::Val => &splits.val,
        Split::Test => &splits.test,
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

fn load_tensor(path: &std::path::Path, expected_len: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "{} has {} trailing bytes; expected f32-aligned data",
        path.display(),
        bytes.len() % 4
    );
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    ensure!(
        floats.len() == expected_len,
        "{} has {} values; expected {}",
        path.display(),
        floats.len(),
        expected_len
    );
    Ok(floats)
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
