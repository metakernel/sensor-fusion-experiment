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
    #[serde(default)]
    pub ssim: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleEvalResult {
    pub sample_id: String,
    pub rgb_metrics: Option<ModalityMetrics>,
    pub range_metrics: Option<ModalityMetrics>,
    #[serde(default)]
    pub range_depth_metrics: Option<RangeDepthMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitEvalSummary {
    pub split: String,
    pub n_samples: usize,
    pub rgb: Option<AggregatedMetrics>,
    pub range: Option<AggregatedMetrics>,
    #[serde(default)]
    pub range_depth: Option<AggregatedRangeDepthMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedMetrics {
    pub mean_mse: f32,
    pub mean_mae: f32,
    pub mean_rmse: f32,
    pub mean_psnr_db: f32,
    #[serde(default)]
    pub mean_ssim: Option<f32>,
    pub std_mse: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RangeDepthMetrics {
    #[serde(default)]
    pub depth_mae_m: Option<f32>,
    #[serde(default)]
    pub depth_rmse_m: Option<f32>,
    #[serde(default)]
    pub delta_lt_1_25: Option<f32>,
    #[serde(default)]
    pub delta_lt_1_25_sq: Option<f32>,
    pub valid_pixel_fraction: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedRangeDepthMetrics {
    #[serde(default)]
    pub mean_depth_mae_m: Option<f32>,
    #[serde(default)]
    pub mean_depth_rmse_m: Option<f32>,
    #[serde(default)]
    pub mean_delta_lt_1_25: Option<f32>,
    #[serde(default)]
    pub mean_delta_lt_1_25_sq: Option<f32>,
    pub mean_valid_pixel_fraction: f32,
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
        ssim: None,
    }
}

pub fn compute_range_depth_metrics(
    predicted: &[f32],
    ground_truth: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    validity_channel: Option<usize>,
    max_range_meters: f32,
) -> RangeDepthMetrics {
    assert_eq!(
        predicted.len(),
        ground_truth.len(),
        "tensor lengths must match"
    );
    assert!(channels > 0, "channels must be positive");
    assert!(height > 0, "height must be positive");
    assert!(width > 0, "width must be positive");
    assert!(max_range_meters > 0.0, "max_range_meters must be positive");

    let pixels = height
        .checked_mul(width)
        .expect("image dimensions overflow");
    let expected_len = channels
        .checked_mul(pixels)
        .expect("image dimensions overflow");
    assert_eq!(
        predicted.len(),
        expected_len,
        "tensors must be CHW with declared dimensions"
    );
    if let Some(channel) = validity_channel {
        assert!(
            channel < channels,
            "validity channel index {channel} out of bounds for {channels} channel tensor"
        );
    }

    let depth_truth = &ground_truth[..pixels];
    let depth_pred = &predicted[..pixels];
    let mask = validity_channel.map(|channel| {
        let start = channel * pixels;
        &ground_truth[start..start + pixels]
    });
    let mut valid = 0usize;
    let mut sum_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut inlier_125 = 0usize;
    let mut inlier_125_sq = 0usize;
    let delta_125 = 1.25f32;
    let delta_125_sq = delta_125 * delta_125;

    for idx in 0..pixels {
        let is_valid = match mask {
            Some(mask) => mask[idx] > 0.5,
            None => depth_truth[idx] > 0.0,
        };
        if !is_valid {
            continue;
        }

        let truth_m = depth_truth[idx].clamp(0.0, 1.0) * max_range_meters;
        if truth_m <= 0.0 {
            continue;
        }

        let pred_m = depth_pred[idx].clamp(0.0, 1.0) * max_range_meters;
        let diff = pred_m - truth_m;
        sum_abs += diff.abs();
        sum_sq += diff * diff;
        valid += 1;

        let denom = pred_m.abs().max(1e-6);
        let ratio = (pred_m / truth_m).abs().max((truth_m / denom).abs());
        if ratio < delta_125 {
            inlier_125 += 1;
        }
        if ratio < delta_125_sq {
            inlier_125_sq += 1;
        }
    }

    let valid_pixel_fraction = valid as f32 / pixels.max(1) as f32;
    if valid == 0 {
        return RangeDepthMetrics {
            depth_mae_m: None,
            depth_rmse_m: None,
            delta_lt_1_25: None,
            delta_lt_1_25_sq: None,
            valid_pixel_fraction,
        };
    }

    let denom = valid as f32;
    RangeDepthMetrics {
        depth_mae_m: Some(sum_abs / denom),
        depth_rmse_m: Some((sum_sq / denom).sqrt()),
        delta_lt_1_25: Some(inlier_125 as f32 / denom),
        delta_lt_1_25_sq: Some(inlier_125_sq as f32 / denom),
        valid_pixel_fraction,
    }
}

/// Windowed SSIM on RGB luma. `pred`/`truth` are CHW with 3 channels (len == 3*h*w), values ~[0,1].
pub fn compute_rgb_ssim(pred: &[f32], truth: &[f32], height: usize, width: usize) -> f32 {
    assert!(height > 0, "height must be positive");
    assert!(width > 0, "width must be positive");
    let pixels = height
        .checked_mul(width)
        .expect("image dimensions overflow");
    let expected_len = pixels.checked_mul(3).expect("image dimensions overflow");
    assert_eq!(pred.len(), expected_len, "pred must be CHW RGB");
    assert_eq!(truth.len(), expected_len, "truth must be CHW RGB");

    const WIN: usize = 8;
    const STEP: usize = 4;
    const C1: f32 = 1e-4;
    const C2: f32 = 9e-4;

    let pred_luma = rgb_luma(pred, pixels);
    let truth_luma = rgb_luma(truth, pixels);
    let (window_height, window_width, row_starts, col_starts) = if height <= WIN && width <= WIN {
        (height, width, vec![0], vec![0])
    } else {
        let window = WIN.min(height).min(width);
        (
            window,
            window,
            window_starts(height, window, STEP),
            window_starts(width, window, STEP),
        )
    };

    let mut total_ssim = 0.0f32;
    let mut window_count = 0usize;

    for row_start in row_starts {
        for &col_start in &col_starts {
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;

            for row in row_start..row_start + window_height {
                let offset = row * width;
                for col in col_start..col_start + window_width {
                    let idx = offset + col;
                    sum_x += pred_luma[idx];
                    sum_y += truth_luma[idx];
                }
            }

            let n = (window_height * window_width) as f32;
            let mu_x = sum_x / n;
            let mu_y = sum_y / n;
            let mut var_x = 0.0f32;
            let mut var_y = 0.0f32;
            let mut cov_xy = 0.0f32;

            for row in row_start..row_start + window_height {
                let offset = row * width;
                for col in col_start..col_start + window_width {
                    let idx = offset + col;
                    let dx = pred_luma[idx] - mu_x;
                    let dy = truth_luma[idx] - mu_y;
                    var_x += dx * dx;
                    var_y += dy * dy;
                    cov_xy += dx * dy;
                }
            }

            var_x /= n;
            var_y /= n;
            cov_xy /= n;

            total_ssim += ((2.0 * mu_x * mu_y + C1) * (2.0 * cov_xy + C2))
                / ((mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2));
            window_count += 1;
        }
    }

    total_ssim / window_count as f32
}

fn rgb_luma(values: &[f32], pixels: usize) -> Vec<f32> {
    (0..pixels)
        .map(|idx| {
            0.299 * values[idx] + 0.587 * values[pixels + idx] + 0.114 * values[2 * pixels + idx]
        })
        .collect()
}

fn window_starts(len: usize, window: usize, step: usize) -> Vec<usize> {
    if len <= window {
        return vec![0];
    }

    let mut starts = vec![0];
    let mut start = step;
    while start + window < len {
        starts.push(start);
        start += step;
    }

    let edge_start = len - window;
    if starts.last().copied() != Some(edge_start) {
        starts.push(edge_start);
    }
    starts
}

pub fn aggregate_metrics(samples: &[ModalityMetrics]) -> AggregatedMetrics {
    assert!(!samples.is_empty(), "samples must not be empty");

    let n = samples.len() as f32;
    let mean_mse = samples.iter().map(|metrics| metrics.mse).sum::<f32>() / n;
    let mean_mae = samples.iter().map(|metrics| metrics.mae).sum::<f32>() / n;
    let mean_rmse = samples.iter().map(|metrics| metrics.rmse).sum::<f32>() / n;
    let mean_psnr_db = samples.iter().map(|metrics| metrics.psnr_db).sum::<f32>() / n;
    let ssim_values: Vec<_> = samples.iter().filter_map(|metrics| metrics.ssim).collect();
    let mean_ssim = (!ssim_values.is_empty())
        .then(|| ssim_values.iter().sum::<f32>() / ssim_values.len() as f32);
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
        mean_ssim,
        std_mse,
    }
}

pub fn aggregate_range_depth_metrics(samples: &[RangeDepthMetrics]) -> AggregatedRangeDepthMetrics {
    assert!(!samples.is_empty(), "samples must not be empty");

    let n = samples.len() as f32;
    let mean_valid_pixel_fraction = samples
        .iter()
        .map(|metrics| metrics.valid_pixel_fraction)
        .sum::<f32>()
        / n;
    let depth_mae_values: Vec<_> = samples
        .iter()
        .filter_map(|metrics| metrics.depth_mae_m)
        .collect();
    let depth_rmse_values: Vec<_> = samples
        .iter()
        .filter_map(|metrics| metrics.depth_rmse_m)
        .collect();
    let delta_125_values: Vec<_> = samples
        .iter()
        .filter_map(|metrics| metrics.delta_lt_1_25)
        .collect();
    let delta_125_sq_values: Vec<_> = samples
        .iter()
        .filter_map(|metrics| metrics.delta_lt_1_25_sq)
        .collect();

    AggregatedRangeDepthMetrics {
        mean_depth_mae_m: (!depth_mae_values.is_empty())
            .then(|| depth_mae_values.iter().sum::<f32>() / depth_mae_values.len() as f32),
        mean_depth_rmse_m: (!depth_rmse_values.is_empty())
            .then(|| depth_rmse_values.iter().sum::<f32>() / depth_rmse_values.len() as f32),
        mean_delta_lt_1_25: (!delta_125_values.is_empty())
            .then(|| delta_125_values.iter().sum::<f32>() / delta_125_values.len() as f32),
        mean_delta_lt_1_25_sq: (!delta_125_sq_values.is_empty())
            .then(|| delta_125_sq_values.iter().sum::<f32>() / delta_125_sq_values.len() as f32),
        mean_valid_pixel_fraction,
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
    let range_depth_metrics: Vec<_> = results
        .iter()
        .filter_map(|result| result.range_depth_metrics.clone())
        .collect();

    SplitEvalSummary {
        split: split.to_owned(),
        n_samples: results.len(),
        rgb: (!rgb_metrics.is_empty()).then(|| aggregate_metrics(&rgb_metrics)),
        range: (!range_metrics.is_empty()).then(|| aggregate_metrics(&range_metrics)),
        range_depth: (!range_depth_metrics.is_empty())
            .then(|| aggregate_range_depth_metrics(&range_depth_metrics)),
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
        "sample_id,rgb_mse,rgb_mae,rgb_rmse,rgb_psnr_db,rgb_ssim,range_mse,range_mae,range_rmse,range_psnr_db,range_ssim,range_proxy_psnr_db,range_proxy_ssim,range_depth_mae_m,range_depth_rmse_m,range_delta_lt_1_25,range_delta_lt_1_25_sq,range_valid_pixel_fraction"
    )?;

    for result in results {
        let rgb = result.rgb_metrics.as_ref();
        let range = result.range_metrics.as_ref();
        let range_depth = result.range_depth_metrics.as_ref();
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            escape_csv_field(&result.sample_id),
            format_optional_metric(rgb.map(|m| m.mse)),
            format_optional_metric(rgb.map(|m| m.mae)),
            format_optional_metric(rgb.map(|m| m.rmse)),
            format_optional_metric(rgb.map(|m| m.psnr_db)),
            format_optional_metric(rgb.and_then(|m| m.ssim)),
            format_optional_metric(range.map(|m| m.mse)),
            format_optional_metric(range.map(|m| m.mae)),
            format_optional_metric(range.map(|m| m.rmse)),
            format_optional_metric(range.map(|m| m.psnr_db)),
            format_optional_metric(range.and_then(|m| m.ssim)),
            format_optional_metric(range.map(|m| m.psnr_db)),
            format_optional_metric(range.and_then(|m| m.ssim)),
            format_optional_metric(range_depth.and_then(|m| m.depth_mae_m)),
            format_optional_metric(range_depth.and_then(|m| m.depth_rmse_m)),
            format_optional_metric(range_depth.and_then(|m| m.delta_lt_1_25)),
            format_optional_metric(range_depth.and_then(|m| m.delta_lt_1_25_sq)),
            format_optional_metric(range_depth.map(|m| m.valid_pixel_fraction)),
        )?;
    }

    Ok(())
}

pub fn write_eval_markdown(summary: &SplitEvalSummary, path: &Path) -> Result<()> {
    create_parent_dir(path)?;
    let mut markdown = String::new();
    markdown.push_str(&format!("# Evaluation Summary: {}\n\n", summary.split));
    markdown.push_str(&format!("- Samples: {}\n\n", summary.n_samples));
    markdown.push_str(
        "| Modality | Mean MSE | Mean MAE | Mean RMSE | Mean Proxy PSNR (dB) | Mean Proxy SSIM | Mean Depth MAE (m) | Mean Depth RMSE (m) | Mean δ<1.25 | Mean δ<1.25² | Mean Valid Fraction | Std MSE |\n",
    );
    markdown.push_str(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );

    if let Some(rgb) = &summary.rgb {
        markdown.push_str(&metrics_row("RGB", rgb, None));
    }
    if let Some(range) = &summary.range {
        markdown.push_str(&metrics_row("Range", range, summary.range_depth.as_ref()));
    }

    if summary.rgb.is_none() && summary.range.is_none() {
        markdown.push_str("| None | - | - | - | - | - | - | - | - | - | - | - |\n");
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

fn metrics_row(
    label: &str,
    metrics: &AggregatedMetrics,
    range_depth: Option<&AggregatedRangeDepthMetrics>,
) -> String {
    let mean_ssim = metrics.mean_ssim.map_or_else(
        || "-".to_owned(),
        |value| format_optional_metric(Some(value)),
    );
    let depth_mae = range_depth
        .and_then(|metrics| metrics.mean_depth_mae_m)
        .map_or_else(
            || "-".to_owned(),
            |value| format_optional_metric(Some(value)),
        );
    let depth_rmse = range_depth
        .and_then(|metrics| metrics.mean_depth_rmse_m)
        .map_or_else(
            || "-".to_owned(),
            |value| format_optional_metric(Some(value)),
        );
    let delta_125 = range_depth
        .and_then(|metrics| metrics.mean_delta_lt_1_25)
        .map_or_else(
            || "-".to_owned(),
            |value| format_optional_metric(Some(value)),
        );
    let delta_125_sq = range_depth
        .and_then(|metrics| metrics.mean_delta_lt_1_25_sq)
        .map_or_else(
            || "-".to_owned(),
            |value| format_optional_metric(Some(value)),
        );
    let valid_fraction = range_depth
        .map(|metrics| format_optional_metric(Some(metrics.mean_valid_pixel_fraction)))
        .unwrap_or_else(|| "-".to_owned());
    format!(
        "| {} | {:.6} | {:.6} | {:.6} | {} | {} | {} | {} | {} | {} | {} | {:.6} |\n",
        label,
        metrics.mean_mse,
        metrics.mean_mae,
        metrics.mean_rmse,
        format_optional_metric(Some(metrics.mean_psnr_db)),
        mean_ssim,
        depth_mae,
        depth_rmse,
        delta_125,
        delta_125_sq,
        valid_fraction,
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
    fn compute_range_depth_metrics_respects_validity_channel_and_scale() {
        let predicted = [0.1, 0.5, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0];
        let truth = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 1.0, 0.0];

        let metrics = compute_range_depth_metrics(&predicted, &truth, 2, 1, 4, Some(1), 75.0);

        approx_eq(metrics.valid_pixel_fraction, 0.5);
        approx_eq(metrics.depth_mae_m.expect("depth_mae_m"), 3.75);
        approx_eq(metrics.depth_rmse_m.expect("depth_rmse_m"), 5.303301);
        approx_eq(metrics.delta_lt_1_25.expect("delta_lt_1_25"), 0.5);
        approx_eq(metrics.delta_lt_1_25_sq.expect("delta_lt_1_25_sq"), 0.5);
    }

    #[test]
    fn compute_range_depth_metrics_handles_empty_valid_mask() {
        let predicted = [0.1, 0.2, 0.0, 0.0];
        let truth = [0.2, 0.3, 0.0, 0.0];

        let metrics = compute_range_depth_metrics(&predicted, &truth, 2, 1, 2, Some(1), 75.0);

        approx_eq(metrics.valid_pixel_fraction, 0.0);
        assert_eq!(metrics.depth_mae_m, None);
        assert_eq!(metrics.depth_rmse_m, None);
        assert_eq!(metrics.delta_lt_1_25, None);
        assert_eq!(metrics.delta_lt_1_25_sq, None);
    }

    #[test]
    fn compute_range_depth_metrics_falls_back_to_nonzero_truth_when_mask_missing() {
        let predicted = [0.0, 0.1, 0.5];
        let truth = [0.0, 0.2, 0.4];

        let metrics = compute_range_depth_metrics(&predicted, &truth, 1, 1, 3, None, 75.0);

        approx_eq(metrics.valid_pixel_fraction, 2.0 / 3.0);
        approx_eq(metrics.depth_mae_m.expect("depth_mae_m"), 7.5);
        approx_eq(metrics.depth_rmse_m.expect("depth_rmse_m"), 7.5);
        approx_eq(metrics.delta_lt_1_25.expect("delta_lt_1_25"), 0.0);
        approx_eq(metrics.delta_lt_1_25_sq.expect("delta_lt_1_25_sq"), 0.5);
    }

    #[test]
    fn compute_rgb_ssim_returns_one_for_identical_images() {
        let height = 4;
        let width = 4;
        let pixels = height * width;
        let image: Vec<_> = (0..3 * pixels)
            .map(|idx| (idx as f32 % 11.0) / 10.0)
            .collect();

        let ssim = compute_rgb_ssim(&image, &image, height, width);

        assert!((ssim - 1.0).abs() < 1e-3, "ssim={ssim}");
    }

    #[test]
    fn compute_rgb_ssim_is_low_for_black_vs_white_images() {
        let height = 8;
        let width = 8;
        let pixels = height * width;
        let predicted = vec![0.0; 3 * pixels];
        let truth = vec![1.0; 3 * pixels];

        let ssim = compute_rgb_ssim(&predicted, &truth, height, width);

        assert!(ssim < 0.2, "ssim={ssim}");
    }

    #[test]
    fn serde_defaults_missing_ssim_fields() {
        let metrics: ModalityMetrics =
            serde_json::from_str(r#"{"mse":1.0,"mae":0.5,"rmse":1.0,"psnr_db":30.0}"#)
                .expect("old modality metrics json should deserialize");
        assert_eq!(metrics.ssim, None);

        let aggregate: AggregatedMetrics = serde_json::from_str(
            r#"{"mean_mse":1.0,"mean_mae":0.5,"mean_rmse":1.0,"mean_psnr_db":30.0,"std_mse":0.0}"#,
        )
        .expect("old aggregate metrics json should deserialize");
        assert_eq!(aggregate.mean_ssim, None);
    }

    #[test]
    fn aggregate_metrics_computes_means_and_std_dev() {
        let samples = [
            ModalityMetrics {
                mse: 1.0,
                mae: 2.0,
                rmse: 1.0,
                psnr_db: 10.0,
                ssim: Some(0.4),
            },
            ModalityMetrics {
                mse: 3.0,
                mae: 4.0,
                rmse: 2.0,
                psnr_db: 20.0,
                ssim: Some(0.8),
            },
        ];

        let aggregate = aggregate_metrics(&samples);

        approx_eq(aggregate.mean_mse, 2.0);
        approx_eq(aggregate.mean_mae, 3.0);
        approx_eq(aggregate.mean_rmse, 1.5);
        approx_eq(aggregate.mean_psnr_db, 15.0);
        approx_eq(
            aggregate.mean_ssim.expect("mean ssim should be present"),
            0.6,
        );
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
                ssim: None,
            },
            ModalityMetrics {
                mse: 2.0,
                mae: 1.5,
                rmse: 2.0_f32.sqrt(),
                psnr_db: 20.0,
                ssim: Some(0.6),
            },
            ModalityMetrics {
                mse: 5.0,
                mae: 2.5,
                rmse: 5.0_f32.sqrt(),
                psnr_db: 10.0,
                ssim: Some(0.9),
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
        approx_eq(
            aggregate.mean_ssim.expect("mean ssim should be present"),
            0.75,
        );
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
                    ssim: None,
                }),
                range_metrics: Some(ModalityMetrics {
                    mse: 4.0,
                    mae: 2.0,
                    rmse: 2.0,
                    psnr_db: 12.0,
                    ssim: None,
                }),
                range_depth_metrics: None,
            },
            SampleEvalResult {
                sample_id: "sample-b".to_owned(),
                rgb_metrics: Some(ModalityMetrics {
                    mse: 9.0,
                    mae: 3.0,
                    rmse: 3.0,
                    psnr_db: 8.0,
                    ssim: None,
                }),
                range_metrics: Some(ModalityMetrics {
                    mse: 16.0,
                    mae: 4.0,
                    rmse: 4.0,
                    psnr_db: 6.0,
                    ssim: None,
                }),
                range_depth_metrics: None,
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
        assert_eq!(rgb.mean_ssim, None);
        approx_eq(rgb.std_mse, 4.0);

        let range = summary.range.expect("range metrics should be present");
        approx_eq(range.mean_mse, 10.0);
        approx_eq(range.mean_mae, 3.0);
        approx_eq(range.mean_rmse, 3.0);
        approx_eq(range.mean_psnr_db, 9.0);
        assert_eq!(range.mean_ssim, None);
        approx_eq(range.std_mse, 6.0);
        assert_eq!(summary.range_depth, None);
    }

    #[test]
    fn summarize_split_aggregates_range_depth_metrics() {
        let results = vec![
            SampleEvalResult {
                sample_id: "sample-a".to_owned(),
                rgb_metrics: None,
                range_metrics: None,
                range_depth_metrics: Some(RangeDepthMetrics {
                    depth_mae_m: Some(1.0),
                    depth_rmse_m: Some(2.0),
                    delta_lt_1_25: Some(0.5),
                    delta_lt_1_25_sq: Some(0.75),
                    valid_pixel_fraction: 0.4,
                }),
            },
            SampleEvalResult {
                sample_id: "sample-b".to_owned(),
                rgb_metrics: None,
                range_metrics: None,
                range_depth_metrics: Some(RangeDepthMetrics {
                    depth_mae_m: Some(3.0),
                    depth_rmse_m: Some(4.0),
                    delta_lt_1_25: Some(0.25),
                    delta_lt_1_25_sq: Some(0.5),
                    valid_pixel_fraction: 0.8,
                }),
            },
        ];

        let summary = summarize_split(&results, "test");
        let depth = summary.range_depth.expect("range depth should be present");

        approx_eq(depth.mean_depth_mae_m.expect("mean_depth_mae_m"), 2.0);
        approx_eq(depth.mean_depth_rmse_m.expect("mean_depth_rmse_m"), 3.0);
        approx_eq(depth.mean_delta_lt_1_25.expect("mean_delta_lt_1_25"), 0.375);
        approx_eq(
            depth.mean_delta_lt_1_25_sq.expect("mean_delta_lt_1_25_sq"),
            0.625,
        );
        approx_eq(depth.mean_valid_pixel_fraction, 0.6);
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
                ssim: Some(0.9876543),
            }),
            range_metrics: Some(ModalityMetrics {
                mse: 1.5,
                mae: 1.25,
                rmse: 1.2247449,
                psnr_db: 7.5,
                ssim: None,
            }),
            range_depth_metrics: Some(RangeDepthMetrics {
                depth_mae_m: Some(0.5),
                depth_rmse_m: Some(0.75),
                delta_lt_1_25: Some(0.9),
                delta_lt_1_25_sq: Some(1.0),
                valid_pixel_fraction: 0.5,
            }),
        }];

        write_eval_csv(&results, &path).expect("csv should be written");
        let content = fs::read_to_string(&path).expect("csv should be readable");
        let _ = fs::remove_file(&path);

        assert!(content.contains(
            "sample_id,rgb_mse,rgb_mae,rgb_rmse,rgb_psnr_db,rgb_ssim,range_mse,range_mae,range_rmse,range_psnr_db,range_ssim,range_proxy_psnr_db,range_proxy_ssim,range_depth_mae_m,range_depth_rmse_m,range_delta_lt_1_25,range_delta_lt_1_25_sq,range_valid_pixel_fraction"
        ));
        assert!(content.contains(
            "sample-1,0.25,0.5,0.5,inf,0.9876543,1.5,1.25,1.2247449,7.5,,7.5,,0.5,0.75,0.9,1,0.5"
        ));
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
                mean_ssim: Some(0.875),
                std_mse: 0.25,
            }),
            range: None,
            range_depth: None,
        };

        write_eval_markdown(&summary, &path).expect("markdown should be written");
        let content = fs::read_to_string(&path).expect("markdown should be readable");
        let _ = fs::remove_file(&path);

        assert!(content.contains("# Evaluation Summary: test"));
        assert!(content.contains("- Samples: 3"));
        assert!(content.contains(
            "| Modality | Mean MSE | Mean MAE | Mean RMSE | Mean Proxy PSNR (dB) | Mean Proxy SSIM | Mean Depth MAE (m) | Mean Depth RMSE (m) | Mean δ<1.25 | Mean δ<1.25² | Mean Valid Fraction | Std MSE |"
        ));
        assert!(content.contains(
            "| RGB | 1.500000 | 0.750000 | 1.224745 | 18 | 0.875 | - | - | - | - | - | 0.250000 |"
        ));
    }
}
