use crate::{ProjectPaths, ReportArgs, display_from_root, resolve_run_dir};
use anyhow::{Context, Result};
use sfx_eval::{AggregatedMetrics, SplitEvalSummary};
use sfx_train::TrainingSummary;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn run(args: ReportArgs, paths: &ProjectPaths) -> Result<()> {
    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let summary_path = run_dir.join("summary.json");
    if !summary_path.exists() {
        println!(
            "warn summary not found at {}",
            display_from_root(&paths.root, &summary_path)
        );
        return Ok(());
    }

    let summary_file = fs::File::open(&summary_path)
        .with_context(|| format!("opening {}", summary_path.display()))?;
    let summary: TrainingSummary = serde_json::from_reader(summary_file)
        .with_context(|| format!("parsing {}", summary_path.display()))?;

    let metrics_path = run_dir.join("metrics.jsonl");
    let metrics_text = fs::read_to_string(&metrics_path)
        .with_context(|| format!("reading {}", metrics_path.display()))?;
    let metrics: Vec<EpochMetric> = metrics_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<EpochMetric>(line).context("parsing metrics.jsonl line"))
        .collect::<Result<_>>()?;

    let train_curve: Vec<f64> = metrics.iter().map(|metric| metric.train_loss).collect();
    let report = build_report(&summary, &train_curve, &paths.root, &run_dir)?;

    let out_dir = sfx_config::resolve_from_root(&paths.root, &args.out);
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join("report.md");
    fs::write(&out_path, &report).with_context(|| format!("writing {}", out_path.display()))?;

    print!("{report}");
    println!("\nok   {}", display_from_root(&paths.root, &out_path));

    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct EpochMetric {
    train_loss: f64,
}

fn build_report(
    summary: &TrainingSummary,
    train_curve: &[f64],
    root: &Path,
    run_dir: &Path,
) -> Result<String> {
    let mut markdown = String::new();
    markdown.push_str(&format!("# {}\n\n", summary.run_name));
    markdown.push_str("## Config\n\n");
    markdown.push_str("| Key | Value |\n");
    markdown.push_str("| --- | --- |\n");
    markdown.push_str(&format!("| Model | {} |\n", summary.model_kind));
    markdown.push_str(&format!("| Backend | {} |\n", summary.backend));
    markdown.push_str(&format!("| Epochs | {} |\n", summary.epochs));
    markdown.push_str(&format!("| Batch size | {} |\n", summary.batch_size));
    markdown.push_str(&format!("| Latent dim | {} |\n\n", summary.latent_dim));
    markdown.push_str("## Training Curve\n\n");
    markdown.push_str(&format!("- train_loss: {}\n\n", sparkline(train_curve)));
    markdown.push_str("## Final Metrics\n\n");
    markdown.push_str("| Metric | Value |\n");
    markdown.push_str("| --- | ---: |\n");
    markdown.push_str(&format!(
        "| Final train loss | {:.6} |\n",
        summary.final_train_loss
    ));
    markdown.push_str(&format!(
        "| Final val loss | {} |\n",
        summary
            .final_val_loss
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string())
    ));
    markdown.push_str(&format!("| Status | {} |\n\n", summary.status));

    append_evaluation_section(&mut markdown, run_dir)?;
    append_previews_section(&mut markdown, root, run_dir)?;

    Ok(markdown)
}

fn append_evaluation_section(markdown: &mut String, run_dir: &Path) -> Result<()> {
    markdown.push_str("## Evaluation\n\n");

    let eval_dir = run_dir.join("eval");
    let mut found_eval = false;
    for split in ["train", "val", "test"] {
        let eval_path = eval_dir.join(format!("{split}.json"));
        if !eval_path.exists() {
            continue;
        }

        if !found_eval {
            markdown
                .push_str("| Split | Modality | Mean MSE | Mean PSNR | Mean SSIM | Samples |\n");
            markdown.push_str("| --- | --- | ---: | ---: | ---: | ---: |\n");
            found_eval = true;
        }

        let eval_file = fs::File::open(&eval_path)
            .with_context(|| format!("opening {}", eval_path.display()))?;
        let summary: SplitEvalSummary = serde_json::from_reader(eval_file)
            .with_context(|| format!("parsing {}", eval_path.display()))?;
        let split_label = if summary.split.is_empty() {
            split
        } else {
            &summary.split
        };

        if let Some(rgb) = &summary.rgb {
            markdown.push_str(&evaluation_row(split_label, "RGB", rgb, summary.n_samples));
        }
        if let Some(range) = &summary.range {
            markdown.push_str(&evaluation_row(
                split_label,
                "Range",
                range,
                summary.n_samples,
            ));
        }
    }

    if !found_eval {
        markdown.push_str("_No evaluation metrics found. Run `cargo xtask eval`._\n");
    }
    markdown.push('\n');

    Ok(())
}

fn evaluation_row(
    split: &str,
    modality: &str,
    metrics: &AggregatedMetrics,
    n_samples: usize,
) -> String {
    format!(
        "| {} | {} | {} | {} | {} | {} |\n",
        split,
        modality,
        format_metric(metrics.mean_mse),
        format_metric(metrics.mean_psnr_db),
        format_optional_metric(metrics.mean_ssim),
        n_samples
    )
}

fn append_previews_section(markdown: &mut String, root: &Path, run_dir: &Path) -> Result<()> {
    markdown.push_str("## Previews\n\n");

    let previews_dir = run_dir.join("previews");
    let preview_paths = collect_preview_images(&previews_dir)?;
    if preview_paths.is_empty() {
        markdown.push_str("_No preview images found. Run `cargo xtask export`._\n");
    } else {
        for path in preview_paths {
            let display_path = display_from_root(root, &path);
            markdown.push_str(&format!("- [{display_path}]({display_path})\n"));
        }
    }

    Ok(())
}

fn collect_preview_images(previews_dir: &Path) -> Result<Vec<PathBuf>> {
    if !previews_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for entry in
        fs::read_dir(previews_dir).with_context(|| format!("reading {}", previews_dir.display()))?
    {
        let entry = entry.with_context(|| format!("reading {} entry", previews_dir.display()))?;
        let path = entry.path();
        if path.is_file() && is_image_path(&path) {
            paths.push(path);
        }
    }

    paths.sort_by(|left, right| {
        let left_name = left
            .file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let right_name = right
            .file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        left_name.cmp(&right_name)
    });
    Ok(paths)
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "bmp" | "gif" | "jpeg" | "jpg" | "pgm" | "png" | "ppm" | "tif" | "tiff" | "webp"
            )
        })
        .unwrap_or(false)
}

fn format_metric(value: f32) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        "inf".to_string()
    } else {
        format!("{value:.6}")
    }
}

fn format_optional_metric(value: Option<f32>) -> String {
    value.map(format_metric).unwrap_or_else(|| "-".to_string())
}

fn sparkline(values: &[f64]) -> String {
    const SYMBOLS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    const WIDTH: usize = 10;

    if values.is_empty() {
        return "-".to_string();
    }

    let sampled: Vec<f64> = if values.len() == 1 {
        vec![values[0]; WIDTH]
    } else {
        (0..WIDTH)
            .map(|index| {
                let position = index as f64 * (values.len() - 1) as f64 / (WIDTH - 1) as f64;
                values[position.round() as usize]
            })
            .collect()
    };

    let min = sampled.iter().copied().fold(f64::INFINITY, f64::min);
    let max = sampled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return std::iter::repeat_n(SYMBOLS[0], WIDTH).collect();
    }

    sampled
        .iter()
        .map(|value| {
            let normalized = (value - min) / (max - min);
            let bucket = (normalized * (SYMBOLS.len() as f64 - 1.0)).round() as usize;
            SYMBOLS[bucket]
        })
        .collect()
}
