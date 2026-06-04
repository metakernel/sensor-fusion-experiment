use crate::{ProjectPaths, ReportArgs, display_from_root, resolve_run_dir};
use anyhow::{Context, Result};
use sfx_train::TrainingSummary;
use std::fs;

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
    let report = build_report(&summary, &train_curve);

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

fn build_report(summary: &TrainingSummary, train_curve: &[f64]) -> String {
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
    markdown.push_str(&format!("| Status | {} |\n", summary.status));
    markdown
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
