use crate::{CompareArgs, ProjectPaths, display_from_root};
use anyhow::{Context, Result};
use sfx_eval::SplitEvalSummary;
use sfx_train::TrainingSummary;
use std::fs;
use std::path::{Path, PathBuf};

const EVAL_SPLITS: [&str; 3] = ["val", "test", "train"];

struct RunComparison {
    training: TrainingSummary,
    eval: Option<DetectedEvalSummary>,
}

struct DetectedEvalSummary {
    split: String,
    summary: SplitEvalSummary,
}

pub(crate) fn run(args: CompareArgs, paths: &ProjectPaths) -> Result<()> {
    if args.runs.is_empty() {
        println!("warn no runs provided; pass --runs <run_dir_a,run_dir_b>");
        return Ok(());
    }

    let comparisons: Vec<_> = args
        .runs
        .iter()
        .map(|run| load_run_comparison(paths, run))
        .collect::<Result<_>>()?;
    let has_eval = comparisons
        .iter()
        .any(|comparison| comparison.eval.is_some());

    let mut table = String::new();
    if has_eval {
        table.push_str("| run_name | model | epochs | final_train_loss | final_val_loss | backend | eval_split | rgb_mse | rgb_proxy_psnr | rgb_proxy_ssim | range_mse | range_proxy_psnr | range_depth_mae_m | range_valid_frac |\n");
        table.push_str(
            "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        );
        for comparison in &comparisons {
            let summary = &comparison.training;
            let eval = comparison.eval.as_ref();
            table.push_str(&format!(
                "| {} | {} | {} | {:.6} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                escape_md(&summary.run_name),
                escape_md(&summary.model_kind),
                summary.epochs,
                summary.final_train_loss,
                summary
                    .final_val_loss
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string()),
                escape_md(&summary.backend),
                eval.map(|eval| escape_md(&eval.split))
                    .unwrap_or_else(|| "-".to_string()),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary.rgb.as_ref().map(|metrics| metrics.mean_mse)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary
                        .rgb
                        .as_ref()
                        .map(|metrics| metrics.mean_psnr_db)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary
                        .rgb
                        .as_ref()
                        .and_then(|metrics| metrics.mean_ssim)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary.range.as_ref().map(|metrics| metrics.mean_mse)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary
                        .range
                        .as_ref()
                        .map(|metrics| metrics.mean_psnr_db)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary
                        .range_depth
                        .as_ref()
                        .and_then(|metrics| metrics.mean_depth_mae_m)
                })),
                format_optional_metric(eval.and_then(|eval| {
                    eval.summary
                        .range_depth
                        .as_ref()
                        .map(|metrics| metrics.mean_valid_pixel_fraction)
                })),
            ));
        }
    } else {
        table.push_str(
            "| run_name | model | epochs | final_train_loss | final_val_loss | backend |\n",
        );
        table.push_str("| --- | --- | ---: | ---: | ---: | --- |\n");
        for comparison in &comparisons {
            let summary = &comparison.training;
            table.push_str(&format!(
                "| {} | {} | {} | {:.6} | {} | {} |\n",
                escape_md(&summary.run_name),
                escape_md(&summary.model_kind),
                summary.epochs,
                summary.final_train_loss,
                summary
                    .final_val_loss
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string()),
                escape_md(&summary.backend)
            ));
        }
    }

    print!("{table}");

    if let Some(out) = args.out {
        let out_path = sfx_config::resolve_from_root(&paths.root, out);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&out_path, &table).with_context(|| format!("writing {}", out_path.display()))?;
        println!("ok   {}", display_from_root(&paths.root, &out_path));
    }

    Ok(())
}

fn load_run_comparison(paths: &ProjectPaths, run: &Path) -> Result<RunComparison> {
    let run_path = sfx_config::resolve_from_root(&paths.root, run);
    let summary_path = summary_path_for_run(&run_path);
    let training = load_training_summary(&summary_path)?;
    let run_dir = summary_path
        .parent()
        .with_context(|| format!("finding run directory for {}", summary_path.display()))?;
    let eval = load_eval_summary(run_dir)?;

    Ok(RunComparison { training, eval })
}

fn load_training_summary(summary_path: &Path) -> Result<TrainingSummary> {
    let file = fs::File::open(summary_path)
        .with_context(|| format!("opening {}", summary_path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parsing {}", summary_path.display()))
}

fn load_eval_summary(run_dir: &Path) -> Result<Option<DetectedEvalSummary>> {
    for split in EVAL_SPLITS {
        let eval_path = run_dir.join("eval").join(format!("{split}.json"));
        if eval_path.exists() {
            let file = fs::File::open(&eval_path)
                .with_context(|| format!("opening {}", eval_path.display()))?;
            let summary: SplitEvalSummary = serde_json::from_reader(file)
                .with_context(|| format!("parsing {}", eval_path.display()))?;
            let split = if summary.split.is_empty() {
                split.to_string()
            } else {
                summary.split.clone()
            };
            return Ok(Some(DetectedEvalSummary { split, summary }));
        }
    }

    Ok(None)
}

fn summary_path_for_run(run_path: &Path) -> PathBuf {
    if run_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("summary.json"))
    {
        run_path.to_path_buf()
    } else {
        run_path.join("summary.json")
    }
}

fn format_optional_metric(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "-".to_string())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|")
}
