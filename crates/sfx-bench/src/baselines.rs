use crate::{
    ClassificationProbeResult, MetricValue, ProbeConfig, ProbeDataset, ProbeError, ProbeRunResult,
    RegressionLossKind, RegressionProbeResult, classification_metrics_from_scores,
    regression_metrics_from_log_predictions, run_linear_probe,
};
use serde::{Deserialize, Serialize};

pub const CLASS_PRIOR_BASELINE_NAME: &str = "class-prior";
pub const COUNT_MEAN_BASELINE_NAME: &str = "count-mean";
pub const LOG1P_MEAN_BASELINE_NAME: &str = "log1p-mean";
pub const RANDOM_PROJECTION_BASELINE_NAME: &str = "random-projection";
pub const PCA_ON_RAW_BASELINE_NAME: &str = "pca-on-raw";
pub const RAW_LINEAR_VAL_REGULARIZED_BASELINE_NAME: &str = "raw-linear-val-regularized";

const DEFAULT_L2_CANDIDATES: &[f32] = &[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0];
const EPS: f32 = 1e-6;

pub type Result<T> = std::result::Result<T, ProbeError>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CountBaselineKind {
    CountMean,
    Log1pMean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationCandidateScore {
    pub l2: f32,
    pub validation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawLinearValRegularizedResult {
    pub selected_l2: f32,
    #[serde(default)]
    pub candidate_scores: Vec<RegularizationCandidateScore>,
    pub probe_result: ProbeRunResult,
}

pub fn run_class_prior_baseline(dataset: &ProbeDataset, seed: u64) -> Result<ProbeRunResult> {
    let feature_dim = validated_feature_dim(&dataset.train_features, "train_features")?;
    validate_feature_matrix(
        &dataset.eval_features,
        "eval_features",
        Some(feature_dim),
        false,
    )?;
    let (train_presence, eval_presence, class_names) = classification_targets(dataset)?;
    let priors = fit_class_priors(train_presence);
    let train_probs = repeated_row(train_presence.len(), &priors);
    let eval_probs = repeated_row(eval_presence.len(), &priors);
    let train_loss = binary_cross_entropy(train_presence, &train_probs);
    let eval_loss = binary_cross_entropy(eval_presence, &eval_probs);
    let metrics = classification_metrics_from_scores(eval_presence, &eval_probs, &class_names)?;

    Ok(ProbeRunResult {
        seed,
        feature_dim,
        standardized: false,
        standardizer: None,
        classification: Some(ClassificationProbeResult {
            train_loss,
            eval_loss,
            metrics,
        }),
        regression: None,
    })
}

pub fn run_count_mean_baseline(
    dataset: &ProbeDataset,
    kind: CountBaselineKind,
    config: &ProbeConfig,
    seed: u64,
) -> Result<ProbeRunResult> {
    let feature_dim = validated_feature_dim(&dataset.train_features, "train_features")?;
    validate_feature_matrix(
        &dataset.eval_features,
        "eval_features",
        Some(feature_dim),
        false,
    )?;
    let (train_counts, eval_counts, target_names) = count_targets(dataset)?;
    let train_log_targets = log1p_targets(train_counts);
    let eval_log_targets = log1p_targets(eval_counts);
    let log_baseline = fit_count_baseline_logs(train_counts, kind);
    let train_pred_logs = repeated_row(train_counts.len(), &log_baseline);
    let eval_pred_logs = repeated_row(eval_counts.len(), &log_baseline);
    let train_loss = regression_loss(
        &train_log_targets,
        &train_pred_logs,
        &config.regression_loss,
        config.huber_delta,
    );
    let eval_loss = regression_loss(
        &eval_log_targets,
        &eval_pred_logs,
        &config.regression_loss,
        config.huber_delta,
    );
    let metrics =
        regression_metrics_from_log_predictions(eval_counts, &eval_pred_logs, &target_names)?;

    Ok(ProbeRunResult {
        seed,
        feature_dim,
        standardized: false,
        standardizer: None,
        classification: None,
        regression: Some(RegressionProbeResult {
            train_loss,
            eval_loss,
            metrics,
        }),
    })
}

pub fn run_random_projection_baseline(
    dataset: &ProbeDataset,
    config: &ProbeConfig,
    latent_dim: usize,
    seed: u64,
) -> Result<ProbeRunResult> {
    if latent_dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: "latent_dim must be greater than zero".to_string(),
        });
    }
    let input_dim = validated_feature_dim(&dataset.train_features, "train_features")?;
    validate_feature_matrix(
        &dataset.eval_features,
        "eval_features",
        Some(input_dim),
        false,
    )?;

    let projection = random_projection_matrix(input_dim, latent_dim, seed ^ 0x7261_6e64_7072_6f6a);
    let train_features = project_features(&dataset.train_features, &projection);
    let eval_features = project_features(&dataset.eval_features, &projection);
    let projected = dataset_with_features(dataset, train_features, eval_features);
    run_linear_probe(&projected, config, seed ^ 0x7072_6f62_655f_7261)
}

pub fn run_pca_on_raw_baseline(
    dataset: &ProbeDataset,
    config: &ProbeConfig,
    latent_dim: usize,
    seed: u64,
) -> Result<ProbeRunResult> {
    if latent_dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: "latent_dim must be greater than zero".to_string(),
        });
    }
    let input_dim = validated_feature_dim(&dataset.train_features, "train_features")?;
    validate_feature_matrix(
        &dataset.eval_features,
        "eval_features",
        Some(input_dim),
        false,
    )?;

    let projector = PcaProjector::fit(
        &dataset.train_features,
        latent_dim,
        seed ^ 0x7063_615f_6261_7365,
    )?;
    let train_features = projector.transform(&dataset.train_features)?;
    let eval_features = projector.transform(&dataset.eval_features)?;
    let projected = dataset_with_features(dataset, train_features, eval_features);
    run_linear_probe(&projected, config, seed ^ 0x7063_615f_7072_6f62)
}

pub fn run_raw_linear_val_regularized_baseline(
    train_eval_dataset: &ProbeDataset,
    train_val_dataset: &ProbeDataset,
    config: &ProbeConfig,
    seed: u64,
    l2_candidates: &[f32],
) -> Result<RawLinearValRegularizedResult> {
    let candidates = if l2_candidates.is_empty() {
        DEFAULT_L2_CANDIDATES
    } else {
        l2_candidates
    };
    if candidates.is_empty() {
        return Err(ProbeError::InvalidDataset {
            message: "at least one L2 regularization candidate is required".to_string(),
        });
    }
    for &candidate in candidates {
        if !candidate.is_finite() || candidate < 0.0 {
            return Err(ProbeError::InvalidDataset {
                message: format!("invalid L2 candidate: {candidate}"),
            });
        }
    }

    let mut candidate_scores = Vec::with_capacity(candidates.len());
    let mut best_index = None;
    for (idx, &l2) in candidates.iter().enumerate() {
        let mut candidate_cfg = config.clone();
        candidate_cfg.l2 = l2;
        let val_result = run_linear_probe(
            train_val_dataset,
            &candidate_cfg,
            seed ^ ((idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        )?;
        let score = selection_score(&val_result)?;
        candidate_scores.push(RegularizationCandidateScore {
            l2,
            validation_score: score,
        });

        match best_index {
            None => best_index = Some(idx),
            Some(current_idx) => {
                let current = &candidate_scores[current_idx];
                if score > current.validation_score + f64::EPSILON
                    || ((score - current.validation_score).abs() <= f64::EPSILON && l2 < current.l2)
                {
                    best_index = Some(idx);
                }
            }
        }
    }

    let selected_index = best_index.ok_or_else(|| ProbeError::InvalidDataset {
        message: "failed to score L2 candidates".to_string(),
    })?;
    let selected_l2 = candidate_scores[selected_index].l2;
    let mut selected_cfg = config.clone();
    selected_cfg.l2 = selected_l2;
    let probe_result = run_linear_probe(
        train_eval_dataset,
        &selected_cfg,
        seed ^ 0x7261_775f_6c69_6e65,
    )?;

    Ok(RawLinearValRegularizedResult {
        selected_l2,
        candidate_scores,
        probe_result,
    })
}

fn validated_feature_dim(features: &[Vec<f32>], name: &str) -> Result<usize> {
    validate_feature_matrix(features, name, None, true)
}

fn validate_feature_matrix(
    features: &[Vec<f32>],
    name: &str,
    expected_dim: Option<usize>,
    require_non_empty: bool,
) -> Result<usize> {
    if features.is_empty() {
        if require_non_empty {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} must not be empty"),
            });
        }
        return Ok(expected_dim.unwrap_or(0));
    }

    let dim = expected_dim.unwrap_or(features[0].len());
    if dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} rows must include at least one feature"),
        });
    }
    for (row_idx, row) in features.iter().enumerate() {
        if row.len() != dim {
            return Err(ProbeError::InvalidDataset {
                message: format!(
                    "{name} row {row_idx} has width {}, expected {dim}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} row {row_idx} contains non-finite values"),
            });
        }
    }
    Ok(dim)
}

fn classification_targets(dataset: &ProbeDataset) -> Result<(&[Vec<u8>], &[Vec<u8>], Vec<String>)> {
    let train_presence =
        dataset
            .train_presence
            .as_ref()
            .ok_or_else(|| ProbeError::InvalidDataset {
                message: "class-prior baseline requires train_presence".to_string(),
            })?;
    let eval_presence =
        dataset
            .eval_presence
            .as_ref()
            .ok_or_else(|| ProbeError::InvalidDataset {
                message: "class-prior baseline requires eval_presence".to_string(),
            })?;
    if train_presence.len() != dataset.train_features.len() {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "train_presence row count {} does not match train_features rows {}",
                train_presence.len(),
                dataset.train_features.len()
            ),
        });
    }
    if eval_presence.len() != dataset.eval_features.len() {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "eval_presence row count {} does not match eval_features rows {}",
                eval_presence.len(),
                dataset.eval_features.len()
            ),
        });
    }
    let class_dim = validate_binary_matrix(train_presence, "train_presence", true)?;
    let eval_dim = validate_binary_matrix(eval_presence, "eval_presence", true)?;
    if class_dim != eval_dim {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "class dimension mismatch between train_presence ({class_dim}) and eval_presence ({eval_dim})"
            ),
        });
    }
    let class_names = resolve_names(&dataset.class_names, class_dim, "class")?;
    Ok((train_presence, eval_presence, class_names))
}

fn count_targets(dataset: &ProbeDataset) -> Result<(&[Vec<f32>], &[Vec<f32>], Vec<String>)> {
    let train_counts = dataset
        .train_counts
        .as_ref()
        .ok_or_else(|| ProbeError::InvalidDataset {
            message: "count baseline requires train_counts".to_string(),
        })?;
    let eval_counts = dataset
        .eval_counts
        .as_ref()
        .ok_or_else(|| ProbeError::InvalidDataset {
            message: "count baseline requires eval_counts".to_string(),
        })?;
    if train_counts.len() != dataset.train_features.len() {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "train_counts row count {} does not match train_features rows {}",
                train_counts.len(),
                dataset.train_features.len()
            ),
        });
    }
    if eval_counts.len() != dataset.eval_features.len() {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "eval_counts row count {} does not match eval_features rows {}",
                eval_counts.len(),
                dataset.eval_features.len()
            ),
        });
    }
    let target_dim = validate_count_matrix(train_counts, "train_counts", true)?;
    let eval_dim = validate_count_matrix(eval_counts, "eval_counts", true)?;
    if target_dim != eval_dim {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "count target mismatch between train_counts ({target_dim}) and eval_counts ({eval_dim})"
            ),
        });
    }
    let target_names = resolve_names(&dataset.count_target_names, target_dim, "count")?;
    Ok((train_counts, eval_counts, target_names))
}

fn validate_binary_matrix(
    targets: &[Vec<u8>],
    name: &str,
    require_non_empty: bool,
) -> Result<usize> {
    if targets.is_empty() {
        if require_non_empty {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} must not be empty"),
            });
        }
        return Ok(0);
    }
    let dim = targets[0].len();
    if dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} rows must include at least one target"),
        });
    }
    for (row_idx, row) in targets.iter().enumerate() {
        if row.len() != dim {
            return Err(ProbeError::InvalidDataset {
                message: format!(
                    "{name} row {row_idx} has width {}, expected {dim}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|&value| value > 1) {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} row {row_idx} contains values outside [0, 1]"),
            });
        }
    }
    Ok(dim)
}

fn validate_count_matrix(
    targets: &[Vec<f32>],
    name: &str,
    require_non_empty: bool,
) -> Result<usize> {
    if targets.is_empty() {
        if require_non_empty {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} must not be empty"),
            });
        }
        return Ok(0);
    }
    let dim = targets[0].len();
    if dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} rows must include at least one target"),
        });
    }
    for (row_idx, row) in targets.iter().enumerate() {
        if row.len() != dim {
            return Err(ProbeError::InvalidDataset {
                message: format!(
                    "{name} row {row_idx} has width {}, expected {dim}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(ProbeError::InvalidDataset {
                message: format!("{name} row {row_idx} contains non-finite values"),
            });
        }
    }
    Ok(dim)
}

fn resolve_names(raw_names: &[String], expected: usize, prefix: &str) -> Result<Vec<String>> {
    if raw_names.is_empty() {
        return Ok((0..expected).map(|idx| format!("{prefix}_{idx}")).collect());
    }
    if raw_names.len() != expected {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "{prefix} name count {} does not match target dimension {expected}",
                raw_names.len()
            ),
        });
    }
    if raw_names.iter().any(|name| name.trim().is_empty()) {
        return Err(ProbeError::InvalidDataset {
            message: format!("{prefix} names must be non-empty"),
        });
    }
    Ok(raw_names.to_vec())
}

fn fit_class_priors(train_presence: &[Vec<u8>]) -> Vec<f32> {
    let n = train_presence.len() as f32;
    let class_dim = train_presence[0].len();
    (0..class_dim)
        .map(|class_idx| {
            train_presence
                .iter()
                .map(|row| row[class_idx] as f32)
                .sum::<f32>()
                / n
        })
        .collect()
}

fn fit_count_baseline_logs(train_counts: &[Vec<f32>], kind: CountBaselineKind) -> Vec<f32> {
    let n = train_counts.len() as f32;
    let target_dim = train_counts[0].len();
    (0..target_dim)
        .map(|target_idx| match kind {
            CountBaselineKind::CountMean => {
                let mean_count = train_counts
                    .iter()
                    .map(|row| row[target_idx].max(0.0))
                    .sum::<f32>()
                    / n;
                mean_count.ln_1p()
            }
            CountBaselineKind::Log1pMean => {
                train_counts
                    .iter()
                    .map(|row| row[target_idx].max(0.0).ln_1p())
                    .sum::<f32>()
                    / n
            }
        })
        .collect()
}

fn repeated_row(rows: usize, values: &[f32]) -> Vec<Vec<f32>> {
    (0..rows).map(|_| values.to_vec()).collect()
}

fn binary_cross_entropy(labels: &[Vec<u8>], probs: &[Vec<f32>]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (target_row, prob_row) in labels.iter().zip(probs.iter()) {
        for (&target, &prob) in target_row.iter().zip(prob_row.iter()) {
            let p = prob.clamp(EPS, 1.0 - EPS) as f64;
            let y = target as f64;
            sum += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

fn regression_loss(
    truth_log: &[Vec<f32>],
    pred_log: &[Vec<f32>],
    loss_kind: &RegressionLossKind,
    huber_delta: f32,
) -> f64 {
    let delta = huber_delta.max(0.0) as f64;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (truth_row, pred_row) in truth_log.iter().zip(pred_log.iter()) {
        for (&truth, &pred) in truth_row.iter().zip(pred_row.iter()) {
            let diff = pred as f64 - truth as f64;
            let value = match loss_kind {
                RegressionLossKind::Huber => {
                    let abs = diff.abs();
                    if abs <= delta {
                        0.5 * diff * diff
                    } else {
                        delta * (abs - 0.5 * delta)
                    }
                }
                RegressionLossKind::Mse => diff * diff,
            };
            sum += value;
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

fn log1p_targets(counts: &[Vec<f32>]) -> Vec<Vec<f32>> {
    counts
        .iter()
        .map(|row| row.iter().map(|value| value.max(0.0).ln_1p()).collect())
        .collect()
}

fn dataset_with_features(
    dataset: &ProbeDataset,
    train_features: Vec<Vec<f32>>,
    eval_features: Vec<Vec<f32>>,
) -> ProbeDataset {
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: dataset.train_presence.clone(),
        eval_presence: dataset.eval_presence.clone(),
        train_counts: dataset.train_counts.clone(),
        eval_counts: dataset.eval_counts.clone(),
        class_names: dataset.class_names.clone(),
        count_target_names: dataset.count_target_names.clone(),
    }
}

fn random_projection_matrix(input_dim: usize, output_dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(seed);
    let scale = 1.0f32 / (output_dim as f32).sqrt();
    (0..input_dim)
        .map(|_| {
            (0..output_dim)
                .map(|_| {
                    if (rng.next_u64() & 1) == 0 {
                        -scale
                    } else {
                        scale
                    }
                })
                .collect()
        })
        .collect()
}

fn project_features(features: &[Vec<f32>], projection: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let output_dim = projection.first().map_or(0, Vec::len);
    features
        .iter()
        .map(|row| {
            let mut out = vec![0.0f32; output_dim];
            for (input_idx, &value) in row.iter().enumerate() {
                let weights = &projection[input_idx];
                for out_idx in 0..output_dim {
                    out[out_idx] += value * weights[out_idx];
                }
            }
            out
        })
        .collect()
}

fn selection_score(result: &ProbeRunResult) -> Result<f64> {
    let mut values = Vec::new();
    if let Some(classification) = &result.classification {
        if let Some(value) = metric_value(&classification.metrics.map) {
            values.push(value);
        }
        if let Some(value) = metric_value(&classification.metrics.balanced_accuracy) {
            values.push(value);
        }
    }
    if let Some(regression) = &result.regression {
        if let Some(value) = metric_value(&regression.metrics.rmse) {
            values.push(-value);
        }
        if let Some(value) = metric_value(&regression.metrics.mae) {
            values.push(-value);
        }
    }
    if values.is_empty() {
        return Err(ProbeError::InvalidDataset {
            message: "cannot score validation result without defined metrics".to_string(),
        });
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn metric_value(metric: &MetricValue) -> Option<f64> {
    match metric {
        MetricValue::Value { value } => Some(*value),
        MetricValue::Na { .. } => None,
    }
}

#[derive(Debug, Clone)]
struct PcaProjector {
    means: Vec<f32>,
    components: Vec<Vec<f32>>,
    output_dim: usize,
}

impl PcaProjector {
    fn fit(train_features: &[Vec<f32>], output_dim: usize, seed: u64) -> Result<Self> {
        let feature_dim = validated_feature_dim(train_features, "train_features")?;
        let means = column_means(train_features);
        let mut centered = center_rows(train_features, &means);
        let component_count = output_dim.min(feature_dim);
        let mut components = Vec::with_capacity(component_count);

        for component_idx in 0..component_count {
            let mut vector = seeded_unit_vector(
                feature_dim,
                seed ^ ((component_idx as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9)),
            );
            orthogonalize(&mut vector, &components);
            if normalize(&mut vector) <= EPS {
                vector = fallback_basis(feature_dim, component_idx);
            }
            for _ in 0..40 {
                let mut next = covariance_times_vector(&centered, &vector);
                orthogonalize(&mut next, &components);
                if normalize(&mut next) <= EPS {
                    break;
                }
                vector = next;
            }
            orient_sign(&mut vector);
            components.push(vector.clone());
            deflate(&mut centered, &vector);
        }

        Ok(Self {
            means,
            components,
            output_dim,
        })
    }

    fn transform(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let expected_dim = self.means.len();
        validate_feature_matrix(features, "features", Some(expected_dim), false)?;

        let mut projected = Vec::with_capacity(features.len());
        for row in features {
            let mut out = vec![0.0f32; self.output_dim];
            for (component_idx, component) in self.components.iter().enumerate() {
                let mut dot_value = 0.0f32;
                for (value_idx, &value) in row.iter().enumerate() {
                    dot_value += (value - self.means[value_idx]) * component[value_idx];
                }
                out[component_idx] = dot_value;
            }
            projected.push(out);
        }
        Ok(projected)
    }
}

fn column_means(features: &[Vec<f32>]) -> Vec<f32> {
    let mut means = vec![0.0f32; features[0].len()];
    for row in features {
        for (idx, value) in row.iter().enumerate() {
            means[idx] += *value;
        }
    }
    for mean in &mut means {
        *mean /= features.len() as f32;
    }
    means
}

fn center_rows(features: &[Vec<f32>], means: &[f32]) -> Vec<Vec<f32>> {
    features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| *value - means[idx])
                .collect()
        })
        .collect()
}

fn seeded_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut vector = (0..dim)
        .map(|_| {
            let value = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
            value * 2.0 - 1.0
        })
        .collect::<Vec<_>>();
    if normalize(&mut vector) <= EPS {
        vector = fallback_basis(dim, 0);
    }
    vector
}

fn covariance_times_vector(centered: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let mut sample_scores = vec![0.0f32; centered.len()];
    for (row_idx, row) in centered.iter().enumerate() {
        sample_scores[row_idx] = dot(row, vector);
    }

    let mut result = vec![0.0f32; vector.len()];
    for (row, &score) in centered.iter().zip(sample_scores.iter()) {
        for feature_idx in 0..vector.len() {
            result[feature_idx] += row[feature_idx] * score;
        }
    }
    if !centered.is_empty() {
        let scale = centered.len() as f32;
        for value in &mut result {
            *value /= scale;
        }
    }
    result
}

fn orthogonalize(vector: &mut [f32], basis: &[Vec<f32>]) {
    for component in basis {
        let projection = dot(vector, component);
        for idx in 0..vector.len() {
            vector[idx] -= projection * component[idx];
        }
    }
}

fn deflate(centered: &mut [Vec<f32>], component: &[f32]) {
    for row in centered {
        let projection = dot(row, component);
        for idx in 0..row.len() {
            row[idx] -= projection * component[idx];
        }
    }
}

fn normalize(vector: &mut [f32]) -> f32 {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > EPS {
        for value in vector {
            *value /= norm;
        }
    }
    norm
}

fn orient_sign(vector: &mut [f32]) {
    if let Some((pivot_idx, _)) = vector
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        && vector[pivot_idx] < 0.0
    {
        for value in vector {
            *value = -*value;
        }
    }
}

fn fallback_basis(dim: usize, component_idx: usize) -> Vec<f32> {
    let mut basis = vec![0.0f32; dim];
    basis[component_idx % dim] = 1.0;
    basis
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum::<f32>()
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
