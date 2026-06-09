use crate::{ProbeConfig, RegressionLossKind};
use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{BinaryCrossEntropyLossConfig, HuberLossConfig, MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::sigmoid;
use burn::tensor::cast::ToElement;
use burn::tensor::{Device, Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

type InnerBackend = Flex;
type TrainBackend = Autodiff<InnerBackend>;

pub type Result<T> = std::result::Result<T, ProbeError>;

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("invalid probe dataset: {message}")]
    InvalidDataset { message: String },
    #[error("failed to extract tensor data for {context}: {message}")]
    TensorData {
        context: &'static str,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeDataset {
    pub train_features: Vec<Vec<f32>>,
    pub eval_features: Vec<Vec<f32>>,
    #[serde(default)]
    pub train_presence: Option<Vec<Vec<u8>>>,
    #[serde(default)]
    pub eval_presence: Option<Vec<Vec<u8>>>,
    #[serde(default)]
    pub train_counts: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub eval_counts: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub class_names: Vec<String>,
    #[serde(default)]
    pub count_target_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeRunResult {
    pub seed: u64,
    pub feature_dim: usize,
    pub standardized: bool,
    #[serde(default)]
    pub standardizer: Option<FeatureStandardizer>,
    #[serde(default)]
    pub classification: Option<ClassificationProbeResult>,
    #[serde(default)]
    pub regression: Option<RegressionProbeResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStandardizer {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl FeatureStandardizer {
    fn fit(features: &[Vec<f32>]) -> Result<Self> {
        let feature_dim = feature_dim(features, "train_features")?;
        let n = features.len() as f32;
        let mut means = vec![0.0f32; feature_dim];
        for row in features {
            for (idx, value) in row.iter().enumerate() {
                means[idx] += *value;
            }
        }
        for mean in &mut means {
            *mean /= n;
        }

        let mut variances = vec![0.0f32; feature_dim];
        for row in features {
            for (idx, value) in row.iter().enumerate() {
                let diff = *value - means[idx];
                variances[idx] += diff * diff;
            }
        }
        let stds = variances
            .into_iter()
            .map(|variance| {
                let std = (variance / n).sqrt();
                if std <= 1e-6 { 1.0 } else { std }
            })
            .collect();

        Ok(Self { means, stds })
    }

    fn transform(&self, features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(idx, value)| (*value - self.means[idx]) / self.stds[idx])
                    .collect()
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationProbeResult {
    pub train_loss: f64,
    pub eval_loss: f64,
    pub metrics: ClassificationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionProbeResult {
    pub train_loss: f64,
    pub eval_loss: f64,
    pub metrics: RegressionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub map: MetricValue,
    pub f1: MetricValue,
    pub auroc: MetricValue,
    pub balanced_accuracy: MetricValue,
    #[serde(default)]
    pub per_class: BTreeMap<String, BinaryClassMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryClassMetrics {
    pub average_precision: MetricValue,
    pub f1: MetricValue,
    pub auroc: MetricValue,
    pub balanced_accuracy: MetricValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionMetrics {
    pub mae: MetricValue,
    pub rmse: MetricValue,
    pub r2: MetricValue,
    #[serde(default)]
    pub per_target: BTreeMap<String, RegressionTargetMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTargetMetrics {
    pub mae: MetricValue,
    pub rmse: MetricValue,
    pub r2: MetricValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum MetricValue {
    Value { value: f64 },
    Na { reason: String },
}

impl MetricValue {
    fn value(value: f64) -> Self {
        Self::Value { value }
    }

    fn na(reason: impl Into<String>) -> Self {
        Self::Na {
            reason: reason.into(),
        }
    }

    fn as_value(&self) -> Option<f64> {
        match self {
            MetricValue::Value { value } => Some(*value),
            MetricValue::Na { .. } => None,
        }
    }
}

#[derive(Module, Debug)]
struct LinearProbeHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearProbeHead<B> {
    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(features)
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearProbeHeadConfig {
    in_features: usize,
    out_features: usize,
}

impl LinearProbeHeadConfig {
    fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> LinearProbeHead<B> {
        LinearProbeHead {
            linear: LinearConfig::new(self.in_features, self.out_features).init(device),
        }
    }
}

pub fn run_linear_probe(
    dataset: &ProbeDataset,
    config: &ProbeConfig,
    seed: u64,
) -> Result<ProbeRunResult> {
    let train_feature_dim = feature_dim(&dataset.train_features, "train_features")?;
    let eval_feature_dim = feature_dim(&dataset.eval_features, "eval_features")?;
    if train_feature_dim != eval_feature_dim {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "feature dimensions must match between train ({train_feature_dim}) and eval ({eval_feature_dim})"
            ),
        });
    }
    if !config.kind.trim().eq_ignore_ascii_case("linear") {
        return Err(ProbeError::InvalidDataset {
            message: format!("probe.kind must be `linear`, got {}", config.kind.trim()),
        });
    }

    let classification_data = validate_classification_data(dataset)?;
    let regression_data = validate_regression_data(dataset)?;
    if classification_data.is_none() && regression_data.is_none() {
        return Err(ProbeError::InvalidDataset {
            message: "at least one probe task must be provided".to_string(),
        });
    }

    let standardizer = if config.standardize_features {
        Some(FeatureStandardizer::fit(&dataset.train_features)?)
    } else {
        None
    };
    let train_features = standardizer.as_ref().map_or_else(
        || dataset.train_features.clone(),
        |s| s.transform(&dataset.train_features),
    );
    let eval_features = standardizer.as_ref().map_or_else(
        || dataset.eval_features.clone(),
        |s| s.transform(&dataset.eval_features),
    );

    let classification = if let Some(data) = classification_data {
        Some(train_classification_probe(
            &train_features,
            &eval_features,
            &data.train_presence,
            &data.eval_presence,
            &data.class_names,
            config,
            seed,
        )?)
    } else {
        None
    };

    let regression = if let Some(data) = regression_data {
        Some(train_regression_probe(
            &train_features,
            &eval_features,
            &data.train_counts,
            &data.eval_counts,
            &data.target_names,
            config,
            seed ^ 0x9e37_79b9_7f4a_7c15,
        )?)
    } else {
        None
    };

    Ok(ProbeRunResult {
        seed,
        feature_dim: train_feature_dim,
        standardized: config.standardize_features,
        standardizer,
        classification,
        regression,
    })
}

pub fn classification_metrics_from_scores(
    labels: &[Vec<u8>],
    scores: &[Vec<f32>],
    class_names: &[String],
) -> Result<ClassificationMetrics> {
    let target_dim = binary_target_dim(labels, "labels", labels.len())?;
    validate_prediction_matrix(scores, "scores", labels.len(), target_dim)?;
    for (row_idx, row) in scores.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ProbeError::InvalidDataset {
                    message: format!("scores row {row_idx} col {col_idx} is not finite"),
                });
            }
            if !(0.0..=1.0).contains(value) {
                return Err(ProbeError::InvalidDataset {
                    message: format!(
                        "scores row {row_idx} col {col_idx} is outside [0, 1]: {value}"
                    ),
                });
            }
        }
    }
    let class_names = resolve_names(class_names, target_dim, "class")?;
    Ok(compute_classification_metrics(labels, scores, &class_names))
}

pub fn regression_metrics_from_log_predictions(
    truth_counts: &[Vec<f32>],
    pred_log_counts: &[Vec<f32>],
    target_names: &[String],
) -> Result<RegressionMetrics> {
    let target_dim = count_target_dim(truth_counts, "truth_counts", truth_counts.len())?;
    validate_prediction_matrix(
        pred_log_counts,
        "pred_log_counts",
        truth_counts.len(),
        target_dim,
    )?;
    for (row_idx, row) in pred_log_counts.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ProbeError::InvalidDataset {
                    message: format!("pred_log_counts row {row_idx} col {col_idx} is not finite"),
                });
            }
        }
    }
    let target_names = resolve_names(target_names, target_dim, "count")?;
    Ok(compute_regression_metrics(
        truth_counts,
        pred_log_counts,
        &target_names,
    ))
}

#[derive(Debug, Clone)]
struct ClassificationData {
    train_presence: Vec<Vec<u8>>,
    eval_presence: Vec<Vec<u8>>,
    class_names: Vec<String>,
}

#[derive(Debug, Clone)]
struct RegressionData {
    train_counts: Vec<Vec<f32>>,
    eval_counts: Vec<Vec<f32>>,
    target_names: Vec<String>,
}

fn validate_classification_data(dataset: &ProbeDataset) -> Result<Option<ClassificationData>> {
    match (&dataset.train_presence, &dataset.eval_presence) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(ProbeError::InvalidDataset {
            message: "classification requires both train_presence and eval_presence".to_string(),
        }),
        (Some(train_presence), Some(eval_presence)) => {
            let target_dim = binary_target_dim(
                train_presence,
                "train_presence",
                dataset.train_features.len(),
            )?;
            let eval_dim =
                binary_target_dim(eval_presence, "eval_presence", dataset.eval_features.len())?;
            if target_dim != eval_dim {
                return Err(ProbeError::InvalidDataset {
                    message: format!(
                        "classification target dimensions must match between train ({target_dim}) and eval ({eval_dim})"
                    ),
                });
            }

            let class_names = resolve_names(&dataset.class_names, target_dim, "class")?;
            Ok(Some(ClassificationData {
                train_presence: train_presence.clone(),
                eval_presence: eval_presence.clone(),
                class_names,
            }))
        }
    }
}

fn validate_regression_data(dataset: &ProbeDataset) -> Result<Option<RegressionData>> {
    match (&dataset.train_counts, &dataset.eval_counts) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(ProbeError::InvalidDataset {
            message: "regression requires both train_counts and eval_counts".to_string(),
        }),
        (Some(train_counts), Some(eval_counts)) => {
            let target_dim =
                count_target_dim(train_counts, "train_counts", dataset.train_features.len())?;
            let eval_dim =
                count_target_dim(eval_counts, "eval_counts", dataset.eval_features.len())?;
            if target_dim != eval_dim {
                return Err(ProbeError::InvalidDataset {
                    message: format!(
                        "regression target dimensions must match between train ({target_dim}) and eval ({eval_dim})"
                    ),
                });
            }

            let target_names = resolve_names(&dataset.count_target_names, target_dim, "count")?;
            Ok(Some(RegressionData {
                train_counts: train_counts.clone(),
                eval_counts: eval_counts.clone(),
                target_names,
            }))
        }
    }
}

fn validate_prediction_matrix(
    matrix: &[Vec<f32>],
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    if matrix.len() != expected_rows {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "{name} row count {} does not match expected rows {expected_rows}",
                matrix.len()
            ),
        });
    }
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != expected_cols {
            return Err(ProbeError::InvalidDataset {
                message: format!(
                    "{name} row {row_idx} has width {}, expected {expected_cols}",
                    row.len()
                ),
            });
        }
    }
    Ok(())
}

fn train_classification_probe(
    train_features: &[Vec<f32>],
    eval_features: &[Vec<f32>],
    train_targets: &[Vec<u8>],
    eval_targets: &[Vec<u8>],
    class_names: &[String],
    config: &ProbeConfig,
    seed: u64,
) -> Result<ClassificationProbeResult> {
    let device = Device::<TrainBackend>::default();
    if config.deterministic {
        TrainBackend::seed(&device, seed);
    }

    let input_dim = train_features[0].len();
    let output_dim = train_targets[0].len();
    let mut model = LinearProbeHeadConfig::new(input_dim, output_dim).init::<TrainBackend>(&device);
    let mut optimizer_cfg = AdamConfig::new();
    if config.l2 > 0.0 {
        optimizer_cfg = optimizer_cfg.with_weight_decay(Some(WeightDecayConfig::new(config.l2)));
    }
    let mut optimizer = optimizer_cfg.init::<TrainBackend, LinearProbeHead<TrainBackend>>();
    let class_weights = config
        .balance_classes
        .then(|| compute_class_weights(train_targets));
    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .with_weights(class_weights.clone())
        .init(&device);

    let batch_size = config.batch_size.max(1).min(train_features.len());
    let mut train_loss = f64::INFINITY;

    for step in 0..config.max_iters {
        let mut order: Vec<usize> = (0..train_features.len()).collect();
        if config.deterministic {
            seeded_shuffle(
                &mut order,
                seed.wrapping_add(step as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
        }

        let mut loss_sum = 0.0f64;
        let mut seen = 0usize;
        for indices in order.chunks(batch_size) {
            let features =
                feature_tensor_from_indices::<TrainBackend>(&device, train_features, indices);
            let targets =
                binary_tensor_from_indices::<TrainBackend>(&device, train_targets, indices);
            let logits = model.forward(features);
            let loss = loss_fn.forward(logits, targets);
            let loss_value = loss.clone().into_scalar().to_f64();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);
            loss_sum += loss_value * indices.len() as f64;
            seen += indices.len();
        }
        train_loss = loss_sum / seen.max(1) as f64;
    }

    let valid_model = model.valid();
    let eval_device = Device::<InnerBackend>::default();
    let eval_features_tensor = feature_tensor_full::<InnerBackend>(&eval_device, eval_features);
    let eval_targets_tensor = binary_tensor_full::<InnerBackend>(&eval_device, eval_targets);
    let eval_loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .with_weights(class_weights)
        .init(&eval_device);
    let logits = valid_model.forward(eval_features_tensor);
    let eval_loss = eval_loss_fn
        .forward(logits.clone(), eval_targets_tensor)
        .into_scalar()
        .to_f64();
    let probs = sigmoid(logits);
    let probs_flat = tensor_to_vec_f32("classification probabilities", probs)?;
    let prob_rows = reshape_matrix(probs_flat, eval_targets.len(), output_dim)?;
    let metrics = compute_classification_metrics(eval_targets, &prob_rows, class_names);

    Ok(ClassificationProbeResult {
        train_loss,
        eval_loss,
        metrics,
    })
}

fn train_regression_probe(
    train_features: &[Vec<f32>],
    eval_features: &[Vec<f32>],
    train_counts: &[Vec<f32>],
    eval_counts: &[Vec<f32>],
    target_names: &[String],
    config: &ProbeConfig,
    seed: u64,
) -> Result<RegressionProbeResult> {
    let train_targets = log1p_targets(train_counts);
    let eval_targets = log1p_targets(eval_counts);
    let device = Device::<TrainBackend>::default();
    if config.deterministic {
        TrainBackend::seed(&device, seed);
    }

    let input_dim = train_features[0].len();
    let output_dim = train_targets[0].len();
    let mut model = LinearProbeHeadConfig::new(input_dim, output_dim).init::<TrainBackend>(&device);
    let mut optimizer_cfg = AdamConfig::new();
    if config.l2 > 0.0 {
        optimizer_cfg = optimizer_cfg.with_weight_decay(Some(WeightDecayConfig::new(config.l2)));
    }
    let mut optimizer = optimizer_cfg.init::<TrainBackend, LinearProbeHead<TrainBackend>>();
    let huber = HuberLossConfig::new(config.huber_delta).init();
    let mse = MseLoss::new();
    let batch_size = config.batch_size.max(1).min(train_features.len());
    let mut train_loss = f64::INFINITY;

    for step in 0..config.max_iters {
        let mut order: Vec<usize> = (0..train_features.len()).collect();
        if config.deterministic {
            seeded_shuffle(
                &mut order,
                seed.wrapping_add(step as u64)
                    .wrapping_mul(0xbf58_476d_1ce4_e5b9),
            );
        }

        let mut loss_sum = 0.0f64;
        let mut seen = 0usize;
        for indices in order.chunks(batch_size) {
            let features =
                feature_tensor_from_indices::<TrainBackend>(&device, train_features, indices);
            let targets =
                float_tensor_from_indices::<TrainBackend>(&device, &train_targets, indices);
            let predictions = model.forward(features);
            let loss = match config.regression_loss {
                RegressionLossKind::Huber => huber.forward(predictions, targets, Reduction::Mean),
                RegressionLossKind::Mse => mse.forward(predictions, targets, Reduction::Mean),
            };
            let loss_value = loss.clone().into_scalar().to_f64();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate as f64, model, grads);
            loss_sum += loss_value * indices.len() as f64;
            seen += indices.len();
        }
        train_loss = loss_sum / seen.max(1) as f64;
    }

    let valid_model = model.valid();
    let eval_device = Device::<InnerBackend>::default();
    let eval_features_tensor = feature_tensor_full::<InnerBackend>(&eval_device, eval_features);
    let eval_targets_tensor = float_tensor_full::<InnerBackend>(&eval_device, &eval_targets);
    let predictions = valid_model.forward(eval_features_tensor);
    let eval_loss = match config.regression_loss {
        RegressionLossKind::Huber => huber
            .forward(predictions.clone(), eval_targets_tensor, Reduction::Mean)
            .into_scalar()
            .to_f64(),
        RegressionLossKind::Mse => mse
            .forward(predictions.clone(), eval_targets_tensor, Reduction::Mean)
            .into_scalar()
            .to_f64(),
    };

    let pred_flat = tensor_to_vec_f32("regression predictions", predictions)?;
    let pred_log = reshape_matrix(pred_flat, eval_targets.len(), output_dim)?;
    let metrics = compute_regression_metrics(eval_counts, &pred_log, target_names);

    Ok(RegressionProbeResult {
        train_loss,
        eval_loss,
        metrics,
    })
}

fn compute_class_weights(targets: &[Vec<u8>]) -> Vec<f32> {
    let sample_count = targets.len() as f32;
    let class_count = targets[0].len();
    let mut weights = vec![1.0f32; class_count];
    for class_idx in 0..class_count {
        let positives = targets.iter().map(|row| row[class_idx] as f32).sum::<f32>();
        let prevalence = (positives / sample_count).clamp(0.05, 0.95);
        weights[class_idx] = 1.0 / prevalence;
    }
    let mean = weights.iter().sum::<f32>() / class_count as f32;
    for weight in &mut weights {
        *weight /= mean;
    }
    weights
}

fn compute_classification_metrics(
    labels: &[Vec<u8>],
    scores: &[Vec<f32>],
    class_names: &[String],
) -> ClassificationMetrics {
    let mut per_class = BTreeMap::new();
    for (class_idx, class_name) in class_names.iter().enumerate() {
        let label_col: Vec<u8> = labels.iter().map(|row| row[class_idx]).collect();
        let score_col: Vec<f32> = scores.iter().map(|row| row[class_idx]).collect();
        per_class.insert(
            class_name.clone(),
            compute_binary_class_metrics(&label_col, &score_col),
        );
    }

    let map = mean_defined(
        per_class.values().map(|metrics| &metrics.average_precision),
        "average precision",
    );
    let f1 = mean_defined(per_class.values().map(|metrics| &metrics.f1), "f1");
    let auroc = mean_defined(per_class.values().map(|metrics| &metrics.auroc), "auroc");
    let balanced_accuracy = mean_defined(
        per_class.values().map(|metrics| &metrics.balanced_accuracy),
        "balanced accuracy",
    );
    ClassificationMetrics {
        map,
        f1,
        auroc,
        balanced_accuracy,
        per_class,
    }
}

fn compute_binary_class_metrics(labels: &[u8], scores: &[f32]) -> BinaryClassMetrics {
    let average_precision = average_precision(labels, scores);
    let (tp, tn, fp, fn_) = confusion(labels, scores, 0.5);
    let f1 = if 2 * tp + fp + fn_ == 0 {
        MetricValue::na("no positive labels or predictions")
    } else {
        MetricValue::value((2 * tp) as f64 / (2 * tp + fp + fn_) as f64)
    };
    let auroc = auroc(labels, scores);
    let positives = tp + fn_;
    let negatives = tn + fp;
    let balanced_accuracy = if positives == 0 || negatives == 0 {
        MetricValue::na("requires both positive and negative labels")
    } else {
        let tpr = tp as f64 / positives as f64;
        let tnr = tn as f64 / negatives as f64;
        MetricValue::value((tpr + tnr) / 2.0)
    };

    BinaryClassMetrics {
        average_precision,
        f1,
        auroc,
        balanced_accuracy,
    }
}

fn average_precision(labels: &[u8], scores: &[f32]) -> MetricValue {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    if positives == 0 {
        return MetricValue::na("requires at least one positive label");
    }

    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .partial_cmp(&scores[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });

    let mut true_positives = 0usize;
    let mut precision_sum = 0.0f64;
    for (rank, sample_idx) in order.into_iter().enumerate() {
        if labels[sample_idx] == 1 {
            true_positives += 1;
            precision_sum += true_positives as f64 / (rank + 1) as f64;
        }
    }

    MetricValue::value(precision_sum / positives as f64)
}

fn auroc(labels: &[u8], scores: &[f32]) -> MetricValue {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let negatives = labels.len().saturating_sub(positives);
    if positives == 0 || negatives == 0 {
        return MetricValue::na("requires both positive and negative labels");
    }

    let mut paired: Vec<(f32, u8)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    paired.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

    let mut rank = 1usize;
    let mut idx = 0usize;
    let mut rank_sum_positive = 0.0f64;
    while idx < paired.len() {
        let mut end = idx + 1;
        while end < paired.len() && paired[end].0 == paired[idx].0 {
            end += 1;
        }

        let tie_len = end - idx;
        let avg_rank = (rank as f64 + (rank + tie_len - 1) as f64) / 2.0;
        let positives_in_tie = paired[idx..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        rank_sum_positive += avg_rank * positives_in_tie as f64;

        rank += tie_len;
        idx = end;
    }

    let positives = positives as f64;
    let negatives = negatives as f64;
    let auc = (rank_sum_positive - positives * (positives + 1.0) / 2.0) / (positives * negatives);
    MetricValue::value(auc)
}

fn compute_regression_metrics(
    truth_counts: &[Vec<f32>],
    pred_log_counts: &[Vec<f32>],
    target_names: &[String],
) -> RegressionMetrics {
    let target_dim = truth_counts[0].len();
    let mut per_target = BTreeMap::new();
    for target_idx in 0..target_dim {
        let truth: Vec<f64> = truth_counts
            .iter()
            .map(|row| row[target_idx].max(0.0) as f64)
            .collect();
        let preds: Vec<f64> = pred_log_counts
            .iter()
            .map(|row| log_to_count(row[target_idx]))
            .collect();

        let mae = mae(&truth, &preds);
        let rmse = rmse(&truth, &preds);
        let r2 = r2(&truth, &preds);
        per_target.insert(
            target_names[target_idx].clone(),
            RegressionTargetMetrics { mae, rmse, r2 },
        );
    }

    RegressionMetrics {
        mae: mean_defined(per_target.values().map(|metrics| &metrics.mae), "mae"),
        rmse: mean_defined(per_target.values().map(|metrics| &metrics.rmse), "rmse"),
        r2: mean_defined(per_target.values().map(|metrics| &metrics.r2), "r2"),
        per_target,
    }
}

fn mae(truth: &[f64], preds: &[f64]) -> MetricValue {
    if truth.is_empty() {
        return MetricValue::na("requires at least one sample");
    }
    let sum = truth
        .iter()
        .zip(preds.iter())
        .map(|(target, pred)| (target - pred).abs())
        .sum::<f64>();
    MetricValue::value(sum / truth.len() as f64)
}

fn rmse(truth: &[f64], preds: &[f64]) -> MetricValue {
    if truth.is_empty() {
        return MetricValue::na("requires at least one sample");
    }
    let sum = truth
        .iter()
        .zip(preds.iter())
        .map(|(target, pred)| {
            let diff = target - pred;
            diff * diff
        })
        .sum::<f64>();
    MetricValue::value((sum / truth.len() as f64).sqrt())
}

fn r2(truth: &[f64], preds: &[f64]) -> MetricValue {
    if truth.is_empty() {
        return MetricValue::na("requires at least one sample");
    }
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_res = truth
        .iter()
        .zip(preds.iter())
        .map(|(target, pred)| {
            let diff = target - pred;
            diff * diff
        })
        .sum::<f64>();
    let ss_tot = truth
        .iter()
        .map(|target| {
            let diff = target - mean;
            diff * diff
        })
        .sum::<f64>();
    if ss_tot <= f64::EPSILON {
        MetricValue::na("target variance is zero")
    } else {
        MetricValue::value(1.0 - ss_res / ss_tot)
    }
}

fn mean_defined<'a>(values: impl Iterator<Item = &'a MetricValue>, label: &str) -> MetricValue {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    for value in values {
        if let Some(metric) = value.as_value() {
            count += 1;
            sum += metric;
        }
    }
    if count == 0 {
        MetricValue::na(format!("all {label} values are undefined"))
    } else {
        MetricValue::value(sum / count as f64)
    }
}

fn confusion(labels: &[u8], scores: &[f32], threshold: f32) -> (usize, usize, usize, usize) {
    labels.iter().zip(scores.iter()).fold(
        (0usize, 0usize, 0usize, 0usize),
        |(tp, tn, fp, fn_), (label, score)| {
            let prediction = (*score >= threshold) as u8;
            match (*label, prediction) {
                (1, 1) => (tp + 1, tn, fp, fn_),
                (0, 0) => (tp, tn + 1, fp, fn_),
                (0, 1) => (tp, tn, fp + 1, fn_),
                (1, 0) => (tp, tn, fp, fn_ + 1),
                _ => (tp, tn, fp, fn_),
            }
        },
    )
}

fn log1p_targets(targets: &[Vec<f32>]) -> Vec<Vec<f32>> {
    targets
        .iter()
        .map(|row| row.iter().map(|value| value.max(0.0).ln_1p()).collect())
        .collect()
}

fn log_to_count(value: f32) -> f64 {
    let count = (value as f64).exp_m1();
    if count.is_finite() {
        count.max(0.0)
    } else if count.is_sign_negative() {
        0.0
    } else {
        f64::MAX
    }
}

fn seeded_shuffle(values: &mut [usize], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for idx in (1..values.len()).rev() {
        let swap_idx = (rng.next_u64() as usize) % (idx + 1);
        values.swap(idx, swap_idx);
    }
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

fn feature_dim(features: &[Vec<f32>], name: &str) -> Result<usize> {
    if features.is_empty() {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} must not be empty"),
        });
    }
    let dim = features[0].len();
    if dim == 0 {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} rows must have at least one feature"),
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
    }
    Ok(dim)
}

fn binary_target_dim(targets: &[Vec<u8>], name: &str, expected_rows: usize) -> Result<usize> {
    if targets.len() != expected_rows {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "{name} row count {} does not match feature rows {expected_rows}",
                targets.len()
            ),
        });
    }
    if targets.is_empty() {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} must not be empty"),
        });
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

fn count_target_dim(targets: &[Vec<f32>], name: &str, expected_rows: usize) -> Result<usize> {
    if targets.len() != expected_rows {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "{name} row count {} does not match feature rows {expected_rows}",
                targets.len()
            ),
        });
    }
    if targets.is_empty() {
        return Err(ProbeError::InvalidDataset {
            message: format!("{name} must not be empty"),
        });
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

fn feature_tensor_from_indices<B: Backend>(
    device: &B::Device,
    features: &[Vec<f32>],
    indices: &[usize],
) -> Tensor<B, 2> {
    let feature_dim = features[0].len();
    let mut flat = Vec::with_capacity(indices.len() * feature_dim);
    for &idx in indices {
        flat.extend_from_slice(&features[idx]);
    }
    Tensor::<B, 2>::from_data(TensorData::new(flat, [indices.len(), feature_dim]), device)
}

fn feature_tensor_full<B: Backend>(device: &B::Device, features: &[Vec<f32>]) -> Tensor<B, 2> {
    let indices: Vec<usize> = (0..features.len()).collect();
    feature_tensor_from_indices(device, features, &indices)
}

fn binary_tensor_from_indices<B: Backend>(
    device: &B::Device,
    targets: &[Vec<u8>],
    indices: &[usize],
) -> Tensor<B, 2, Int> {
    let target_dim = targets[0].len();
    let mut flat = Vec::with_capacity(indices.len() * target_dim);
    for &idx in indices {
        for &target in &targets[idx] {
            flat.push(i64::from(target));
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [indices.len(), target_dim]), device)
}

fn binary_tensor_full<B: Backend>(device: &B::Device, targets: &[Vec<u8>]) -> Tensor<B, 2, Int> {
    let indices: Vec<usize> = (0..targets.len()).collect();
    binary_tensor_from_indices(device, targets, &indices)
}

fn float_tensor_from_indices<B: Backend>(
    device: &B::Device,
    targets: &[Vec<f32>],
    indices: &[usize],
) -> Tensor<B, 2> {
    let target_dim = targets[0].len();
    let mut flat = Vec::with_capacity(indices.len() * target_dim);
    for &idx in indices {
        flat.extend_from_slice(&targets[idx]);
    }
    Tensor::<B, 2>::from_data(TensorData::new(flat, [indices.len(), target_dim]), device)
}

fn float_tensor_full<B: Backend>(device: &B::Device, targets: &[Vec<f32>]) -> Tensor<B, 2> {
    let indices: Vec<usize> = (0..targets.len()).collect();
    float_tensor_from_indices(device, targets, &indices)
}

fn tensor_to_vec_f32<B: Backend, const D: usize>(
    context: &'static str,
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>> {
    TensorData::convert::<f32>(tensor.to_data())
        .to_vec::<f32>()
        .map_err(|err| ProbeError::TensorData {
            context,
            message: err.to_string(),
        })
}

fn reshape_matrix(values: Vec<f32>, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    if values.len() != rows * cols {
        return Err(ProbeError::InvalidDataset {
            message: format!(
                "tensor had {} values, expected {} for shape [{rows}, {cols}]",
                values.len(),
                rows * cols
            ),
        });
    }
    let mut matrix = Vec::with_capacity(rows);
    for chunk in values.chunks(cols) {
        matrix.push(chunk.to_vec());
    }
    Ok(matrix)
}
