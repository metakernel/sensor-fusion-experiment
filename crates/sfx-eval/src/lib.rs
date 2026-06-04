use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModalityMetrics {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub psnr_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleEvalResult {
    pub sample_id: String,
    pub rgb_metrics: Option<ModalityMetrics>,
    pub range_metrics: Option<ModalityMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitEvalSummary {
    pub split: String,
    pub n_samples: usize,
    pub rgb: Option<AggregatedMetrics>,
    pub range: Option<AggregatedMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedMetrics {
    pub mean_mse: f32,
    pub mean_mae: f32,
    pub mean_rmse: f32,
    pub mean_psnr_db: f32,
    pub std_mse: f32,
}

pub fn compute_modality_metrics(
    predicted: &[f32],
    ground_truth: &[f32],
    max_val: f32,
) -> ModalityMetrics {
    assert_eq!(
        predicted.len(),
        ground_truth.len(),
        "tensor lengths must match"
    );
    assert!(!predicted.is_empty(), "tensors must not be empty");
    assert!(max_val > 0.0, "max_val must be positive");

    let n = predicted.len() as f32;
    let (sum_squared_error, sum_absolute_error) = predicted.iter().zip(ground_truth.iter()).fold(
        (0.0f32, 0.0f32),
        |(sse, sae), (pred, truth)| {
            let diff = pred - truth;
            (sse + diff * diff, sae + diff.abs())
        },
    );

    let mse = sum_squared_error / n;
    let mae = sum_absolute_error / n;
    let rmse = mse.sqrt();
    let psnr_db = if mse == 0.0 {
        f32::INFINITY
    } else {
        10.0 * ((max_val * max_val) / mse).log10()
    };

    ModalityMetrics {
        mse,
        mae,
        rmse,
        psnr_db,
    }
}

pub fn aggregate_metrics(samples: &[ModalityMetrics]) -> AggregatedMetrics {
    assert!(!samples.is_empty(), "samples must not be empty");

    let n = samples.len() as f32;
    let mean_mse = samples.iter().map(|metrics| metrics.mse).sum::<f32>() / n;
    let mean_mae = samples.iter().map(|metrics| metrics.mae).sum::<f32>() / n;
    let mean_rmse = samples.iter().map(|metrics| metrics.rmse).sum::<f32>() / n;
    let mean_psnr_db = samples.iter().map(|metrics| metrics.psnr_db).sum::<f32>() / n;
    let std_mse = (samples
        .iter()
        .map(|metrics| {
            let delta = metrics.mse - mean_mse;
            delta * delta
        })
        .sum::<f32>()
        / n)
        .sqrt();

    AggregatedMetrics {
        mean_mse,
        mean_mae,
        mean_rmse,
        mean_psnr_db,
        std_mse,
    }
}

pub fn summarize_split(results: &[SampleEvalResult], split: &str) -> SplitEvalSummary {
    let rgb_metrics: Vec<_> = results
        .iter()
        .filter_map(|result| result.rgb_metrics.clone())
        .collect();
    let range_metrics: Vec<_> = results
        .iter()
        .filter_map(|result| result.range_metrics.clone())
        .collect();

    SplitEvalSummary {
        split: split.to_owned(),
        n_samples: results.len(),
        rgb: (!rgb_metrics.is_empty()).then(|| aggregate_metrics(&rgb_metrics)),
        range: (!range_metrics.is_empty()).then(|| aggregate_metrics(&range_metrics)),
    }
}

pub fn write_eval_json(summary: &SplitEvalSummary, path: &Path) -> Result<()> {
    create_parent_dir(path)?;
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, summary)?;
    Ok(())
}

pub fn write_eval_csv(results: &[SampleEvalResult], path: &Path) -> Result<()> {
    create_parent_dir(path)?;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "sample_id,rgb_mse,rgb_mae,rgb_rmse,rgb_psnr_db,range_mse,range_mae,range_rmse,range_psnr_db"
    )?;

    for result in results {
        let rgb = result.rgb_metrics.as_ref();
        let range = result.range_metrics.as_ref();
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{}",
            escape_csv_field(&result.sample_id),
            format_optional_metric(rgb.map(|m| m.mse)),
            format_optional_metric(rgb.map(|m| m.mae)),
            format_optional_metric(rgb.map(|m| m.rmse)),
            format_optional_metric(rgb.map(|m| m.psnr_db)),
            format_optional_metric(range.map(|m| m.mse)),
            format_optional_metric(range.map(|m| m.mae)),
            format_optional_metric(range.map(|m| m.rmse)),
            format_optional_metric(range.map(|m| m.psnr_db)),
        )?;
    }

    Ok(())
}

pub fn write_eval_markdown(summary: &SplitEvalSummary, path: &Path) -> Result<()> {
    create_parent_dir(path)?;
    let mut markdown = String::new();
    markdown.push_str(&format!("# Evaluation Summary: {}\n\n", summary.split));
    markdown.push_str(&format!("- Samples: {}\n\n", summary.n_samples));
    markdown
        .push_str("| Modality | Mean MSE | Mean MAE | Mean RMSE | Mean PSNR (dB) | Std MSE |\n");
    markdown.push_str("| --- | ---: | ---: | ---: | ---: | ---: |\n");

    if let Some(rgb) = &summary.rgb {
        markdown.push_str(&metrics_row("RGB", rgb));
    }
    if let Some(range) = &summary.range {
        markdown.push_str(&metrics_row("Range", range));
    }

    if summary.rgb.is_none() && summary.range.is_none() {
        markdown.push_str("| None | - | - | - | - | - |\n");
    }

    fs::write(path, markdown)?;
    Ok(())
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn format_optional_metric(value: Option<f32>) -> String {
    match value {
        Some(value) if value.is_infinite() && value.is_sign_positive() => "inf".to_owned(),
        Some(value) => value.to_string(),
        None => String::new(),
    }
}

fn escape_csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn metrics_row(label: &str, metrics: &AggregatedMetrics) -> String {
    format!(
        "| {} | {:.6} | {:.6} | {:.6} | {} | {:.6} |\n",
        label,
        metrics.mean_mse,
        metrics.mean_mae,
        metrics.mean_rmse,
        format_optional_metric(Some(metrics.mean_psnr_db)),
        metrics.std_mse,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn approx_eq(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "left={left}, right={right}");
    }

    fn unique_test_path(name: &str, extension: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "sfx_eval_{name}_{}_{}.{}",
            process::id(),
            nanos,
            extension
        ))
    }

    #[test]
    fn compute_metrics_for_identical_tensors() {
        let predicted = [0.25, 0.5, 0.75];
        let truth = [0.25, 0.5, 0.75];

        let metrics = compute_modality_metrics(&predicted, &truth, 1.0);

        approx_eq(metrics.mse, 0.0);
        approx_eq(metrics.mae, 0.0);
        approx_eq(metrics.rmse, 0.0);
        assert!(metrics.psnr_db.is_infinite());
    }

    #[test]
    fn compute_metrics_for_all_zeros_vs_ones() {
        let predicted = [0.0, 0.0, 0.0, 0.0];
        let truth = [1.0, 1.0, 1.0, 1.0];

        let metrics = compute_modality_metrics(&predicted, &truth, 1.0);

        approx_eq(metrics.mse, 1.0);
        approx_eq(metrics.mae, 1.0);
        approx_eq(metrics.rmse, 1.0);
        approx_eq(metrics.psnr_db, 0.0);
    }

    #[test]
    fn compute_metrics_uses_psnr_formula_for_small_constant_error() {
        let predicted = [0.6, 0.6, 0.6, 0.6];
        let truth = [0.5, 0.5, 0.5, 0.5];

        let metrics = compute_modality_metrics(&predicted, &truth, 1.0);
        let expected_mse = 0.01_f32;
        let expected_psnr = 10.0_f32 * (1.0_f32 / expected_mse).log10();

        approx_eq(metrics.mse, expected_mse);
        approx_eq(metrics.psnr_db, expected_psnr);
    }

    #[test]
    fn compute_metrics_distinguishes_mae_mse_and_rmse() {
        let predicted = [2.0, 2.0, 0.0, 0.0];
        let truth = [0.0, 0.0, 0.0, 0.0];

        let metrics = compute_modality_metrics(&predicted, &truth, 2.0);

        approx_eq(metrics.mae, 1.0);
        approx_eq(metrics.mse, 2.0);
        approx_eq(metrics.rmse, metrics.mse.sqrt());
        approx_eq(metrics.rmse, 2.0_f32.sqrt());
    }

    #[test]
    fn aggregate_metrics_computes_means_and_std_dev() {
        let samples = [
            ModalityMetrics {
                mse: 1.0,
                mae: 2.0,
                rmse: 1.0,
                psnr_db: 10.0,
            },
            ModalityMetrics {
                mse: 3.0,
                mae: 4.0,
                rmse: 2.0,
                psnr_db: 20.0,
            },
        ];

        let aggregate = aggregate_metrics(&samples);

        approx_eq(aggregate.mean_mse, 2.0);
        approx_eq(aggregate.mean_mae, 3.0);
        approx_eq(aggregate.mean_rmse, 1.5);
        approx_eq(aggregate.mean_psnr_db, 15.0);
        approx_eq(aggregate.std_mse, 1.0);
    }

    #[test]
    fn aggregate_metrics_computes_multiple_sample_statistics() {
        let samples = [
            ModalityMetrics {
                mse: 1.0,
                mae: 0.5,
                rmse: 1.0,
                psnr_db: 30.0,
            },
            ModalityMetrics {
                mse: 2.0,
                mae: 1.5,
                rmse: 2.0_f32.sqrt(),
                psnr_db: 20.0,
            },
            ModalityMetrics {
                mse: 5.0,
                mae: 2.5,
                rmse: 5.0_f32.sqrt(),
                psnr_db: 10.0,
            },
        ];

        let aggregate = aggregate_metrics(&samples);
        let expected_mean_mse = (1.0_f32 + 2.0_f32 + 5.0_f32) / 3.0_f32;
        let expected_std_mse = (((1.0_f32 - expected_mean_mse).powi(2)
            + (2.0_f32 - expected_mean_mse).powi(2)
            + (5.0_f32 - expected_mean_mse).powi(2))
            / 3.0_f32)
            .sqrt();

        approx_eq(aggregate.mean_mse, expected_mean_mse);
        approx_eq(aggregate.std_mse, expected_std_mse);
    }

    #[test]
    fn summarize_split_populates_rgb_and_range_metrics() {
        let results = vec![
            SampleEvalResult {
                sample_id: "sample-a".to_owned(),
                rgb_metrics: Some(ModalityMetrics {
                    mse: 1.0,
                    mae: 0.5,
                    rmse: 1.0,
                    psnr_db: 30.0,
                }),
                range_metrics: Some(ModalityMetrics {
                    mse: 4.0,
                    mae: 2.0,
                    rmse: 2.0,
                    psnr_db: 12.0,
                }),
            },
            SampleEvalResult {
                sample_id: "sample-b".to_owned(),
                rgb_metrics: Some(ModalityMetrics {
                    mse: 9.0,
                    mae: 3.0,
                    rmse: 3.0,
                    psnr_db: 8.0,
                }),
                range_metrics: Some(ModalityMetrics {
                    mse: 16.0,
                    mae: 4.0,
                    rmse: 4.0,
                    psnr_db: 6.0,
                }),
            },
        ];

        let summary = summarize_split(&results, "validation");

        assert_eq!(summary.split, "validation");
        assert_eq!(summary.n_samples, 2);

        let rgb = summary.rgb.expect("rgb metrics should be present");
        approx_eq(rgb.mean_mse, 5.0);
        approx_eq(rgb.mean_mae, 1.75);
        approx_eq(rgb.mean_rmse, 2.0);
        approx_eq(rgb.mean_psnr_db, 19.0);
        approx_eq(rgb.std_mse, 4.0);

        let range = summary.range.expect("range metrics should be present");
        approx_eq(range.mean_mse, 10.0);
        approx_eq(range.mean_mae, 3.0);
        approx_eq(range.mean_rmse, 3.0);
        approx_eq(range.mean_psnr_db, 9.0);
        approx_eq(range.std_mse, 6.0);
    }

    #[test]
    fn write_eval_csv_round_trips_expected_headers_and_values() {
        let path = unique_test_path("eval_csv", "csv");
        let results = vec![SampleEvalResult {
            sample_id: "sample-1".to_owned(),
            rgb_metrics: Some(ModalityMetrics {
                mse: 0.25,
                mae: 0.5,
                rmse: 0.5,
                psnr_db: f32::INFINITY,
            }),
            range_metrics: Some(ModalityMetrics {
                mse: 1.5,
                mae: 1.25,
                rmse: 1.2247449,
                psnr_db: 7.5,
            }),
        }];

        write_eval_csv(&results, &path).expect("csv should be written");
        let content = fs::read_to_string(&path).expect("csv should be readable");
        let _ = fs::remove_file(&path);

        assert!(content.contains(
            "sample_id,rgb_mse,rgb_mae,rgb_rmse,rgb_psnr_db,range_mse,range_mae,range_rmse,range_psnr_db"
        ));
        assert!(content.contains("sample-1,0.25,0.5,0.5,inf,1.5,1.25,1.2247449,7.5"));
    }

    #[test]
    fn write_eval_markdown_includes_split_and_metric_values() {
        let path = unique_test_path("eval_markdown", "md");
        let summary = SplitEvalSummary {
            split: "test".to_owned(),
            n_samples: 3,
            rgb: Some(AggregatedMetrics {
                mean_mse: 1.5,
                mean_mae: 0.75,
                mean_rmse: 1.2247449,
                mean_psnr_db: 18.0,
                std_mse: 0.25,
            }),
            range: None,
        };

        write_eval_markdown(&summary, &path).expect("markdown should be written");
        let content = fs::read_to_string(&path).expect("markdown should be readable");
        let _ = fs::remove_file(&path);

        assert!(content.contains("# Evaluation Summary: test"));
        assert!(content.contains("- Samples: 3"));
        assert!(content.contains("| RGB | 1.500000 | 0.750000 | 1.224745 | 18 | 0.250000 |"));
    }
}
