use sfx_bench::{MetricValue, ProbeConfig, ProbeDataset, RegressionLossKind, run_linear_probe};

#[test]
fn separable_fixture_produces_strong_classification_metrics() {
    let dataset = separable_classification_fixture(512, 256, 3);
    let config = ProbeConfig {
        max_iters: 180,
        learning_rate: 0.05,
        batch_size: 64,
        l2: 1e-4,
        balance_classes: true,
        standardize_features: true,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let result = run_linear_probe(&dataset, &config, 7).expect("probe run should succeed");
    let classification = result
        .classification
        .expect("classification output should be present");

    assert_metric_gte(&classification.metrics.map, 0.90);
    assert_metric_gte(&classification.metrics.f1, 0.85);
    assert_metric_gte(&classification.metrics.auroc, 0.90);
    assert_metric_gte(&classification.metrics.balanced_accuracy, 0.85);
    for class_metrics in classification.metrics.per_class.values() {
        assert_metric_gte(&class_metrics.average_precision, 0.88);
    }
}

#[test]
fn noisy_fixture_stays_near_prior_classification_performance() {
    let dataset = noisy_classification_fixture(900, 450, &[0.2, 0.5, 0.8]);
    let config = ProbeConfig {
        max_iters: 220,
        learning_rate: 0.04,
        batch_size: 96,
        l2: 1e-3,
        balance_classes: true,
        standardize_features: true,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let result = run_linear_probe(&dataset, &config, 42).expect("probe run should succeed");
    let classification = result
        .classification
        .expect("classification output should be present");

    assert_metric_between(&classification.metrics.auroc, 0.35, 0.65);
    assert_metric_between(&classification.metrics.balanced_accuracy, 0.40, 0.60);
    assert_metric_between(&classification.metrics.map, 0.30, 0.70);
}

#[test]
fn regression_fixture_has_sane_error_and_explained_variance() {
    let dataset = regression_fixture(480, 220);
    let config = ProbeConfig {
        max_iters: 240,
        learning_rate: 0.05,
        batch_size: 64,
        l2: 5e-4,
        regression_loss: RegressionLossKind::Huber,
        huber_delta: 1.0,
        standardize_features: true,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let result = run_linear_probe(&dataset, &config, 1337).expect("probe run should succeed");
    let regression = result
        .regression
        .expect("regression output should be present");

    assert_metric_lte(&regression.metrics.mae, 1.2);
    assert_metric_lte(&regression.metrics.rmse, 1.6);
    assert_metric_gte(&regression.metrics.r2, 0.75);
}

#[test]
fn undefined_auroc_is_reported_as_na() {
    let mut dataset = separable_classification_fixture(300, 120, 2);
    let eval_presence = dataset
        .eval_presence
        .as_mut()
        .expect("fixture should include eval labels");
    for labels in eval_presence {
        labels[1] = 1;
    }
    let config = ProbeConfig {
        max_iters: 160,
        learning_rate: 0.05,
        batch_size: 64,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let result = run_linear_probe(&dataset, &config, 2024).expect("probe run should succeed");
    let class_metrics = &result
        .classification
        .expect("classification output should be present")
        .metrics
        .per_class["class_1"];

    assert!(
        matches!(class_metrics.auroc, MetricValue::Na { .. }),
        "auroc should be N/A when only one class is present"
    );
}

#[test]
fn seeded_runs_are_deterministic() {
    let dataset = separable_classification_fixture(320, 160, 3);
    let config = ProbeConfig {
        max_iters: 140,
        learning_rate: 0.05,
        batch_size: 64,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let first = run_linear_probe(&dataset, &config, 99).expect("first probe run should succeed");
    let second = run_linear_probe(&dataset, &config, 99).expect("second probe run should succeed");

    let first_cls = first
        .classification
        .expect("classification output should be present");
    let second_cls = second
        .classification
        .expect("classification output should be present");
    assert_metric_close(&first_cls.metrics.map, &second_cls.metrics.map, 5e-4);
    assert_metric_close(&first_cls.metrics.f1, &second_cls.metrics.f1, 5e-4);
    assert_metric_close(&first_cls.metrics.auroc, &second_cls.metrics.auroc, 5e-4);
    assert_metric_close(
        &first_cls.metrics.balanced_accuracy,
        &second_cls.metrics.balanced_accuracy,
        5e-4,
    );
}

fn separable_classification_fixture(
    train_samples: usize,
    eval_samples: usize,
    class_count: usize,
) -> ProbeDataset {
    let (train_features, train_presence) =
        build_separable_classification_split(train_samples, class_count, 11);
    let (eval_features, eval_presence) =
        build_separable_classification_split(eval_samples, class_count, 311);
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: Some(train_presence),
        eval_presence: Some(eval_presence),
        train_counts: None,
        eval_counts: None,
        class_names: (0..class_count).map(|idx| format!("class_{idx}")).collect(),
        count_target_names: Vec::new(),
    }
}

fn noisy_classification_fixture(
    train_samples: usize,
    eval_samples: usize,
    priors: &[f32],
) -> ProbeDataset {
    let (train_features, train_presence) =
        build_noisy_classification_split(train_samples, priors, 91);
    let (eval_features, eval_presence) =
        build_noisy_classification_split(eval_samples, priors, 5091);
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: Some(train_presence),
        eval_presence: Some(eval_presence),
        train_counts: None,
        eval_counts: None,
        class_names: (0..priors.len())
            .map(|idx| format!("class_{idx}"))
            .collect(),
        count_target_names: Vec::new(),
    }
}

fn regression_fixture(train_samples: usize, eval_samples: usize) -> ProbeDataset {
    let (train_features, train_counts) = build_regression_split(train_samples, 77);
    let (eval_features, eval_counts) = build_regression_split(eval_samples, 9077);
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: None,
        eval_presence: None,
        train_counts: Some(train_counts),
        eval_counts: Some(eval_counts),
        class_names: Vec::new(),
        count_target_names: vec!["objects".to_string(), "moving-objects".to_string()],
    }
}

fn build_separable_classification_split(
    samples: usize,
    class_count: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut features = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let x0 = signal(sample_idx, 0, seed);
        let x1 = signal(sample_idx, 1, seed);
        let x2 = signal(sample_idx, 2, seed);
        let x3 = signal(sample_idx, 3, seed);
        features.push(vec![x0, x1, x2, x3]);

        let mut labels = vec![0u8; class_count];
        if class_count > 0 {
            labels[0] = u8::from(x0 + 0.25 * x1 > 0.0);
        }
        if class_count > 1 {
            labels[1] = u8::from(x2 - 0.2 * x3 > 0.0);
        }
        if class_count > 2 {
            labels[2] = u8::from(x0 + x2 + 0.15 * x3 > 0.2);
        }
        for class_idx in 3..class_count {
            labels[class_idx] = u8::from(
                x0 * (class_idx as f32 + 1.0) + x1 - x2 * 0.5 + x3 * 0.25 > class_idx as f32 * 0.1,
            );
        }
        targets.push(labels);
    }
    (features, targets)
}

fn build_noisy_classification_split(
    samples: usize,
    priors: &[f32],
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut features = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let x0 = signal(sample_idx, 0, seed);
        let x1 = signal(sample_idx, 1, seed);
        let x2 = signal(sample_idx, 2, seed);
        let x3 = signal(sample_idx, 3, seed);
        features.push(vec![x0, x1, x2, x3]);

        let labels = priors
            .iter()
            .enumerate()
            .map(|(class_idx, prior)| {
                let draw = hashed_unit(
                    sample_idx as u64,
                    class_idx as u64,
                    seed ^ 0xa5a5_a5a5_a5a5_a5a5,
                );
                u8::from(draw < *prior)
            })
            .collect();
        targets.push(labels);
    }
    (features, targets)
}

fn build_regression_split(samples: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut features = Vec::with_capacity(samples);
    let mut counts = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let x0 = signal(sample_idx, 0, seed);
        let x1 = signal(sample_idx, 1, seed);
        let x2 = signal(sample_idx, 2, seed);
        let x3 = signal(sample_idx, 3, seed);
        let x4 = signal(sample_idx, 4, seed);
        features.push(vec![x0, x1, x2, x3, x4]);

        let noise_a = hashed_unit(sample_idx as u64, 0, seed ^ 0x1234_5678_9abc_def0) - 0.5;
        let noise_b = hashed_unit(sample_idx as u64, 1, seed ^ 0x0fed_cba9_8765_4321) - 0.5;
        let target_a = (7.5 + 2.3 * x0 - 1.7 * x1 + 0.9 * x2 + 0.5 * noise_a).max(0.0);
        let target_b = (4.0 + 1.2 * x3 + 0.8 * x0 - 1.4 * x4 + 0.6 * noise_b).max(0.0);
        counts.push(vec![target_a.round(), target_b.round()]);
    }
    (features, counts)
}

fn signal(sample_idx: usize, dim_idx: usize, seed: u64) -> f32 {
    let i = sample_idx as f64;
    let d = dim_idx as f64;
    let s = seed as f64;
    let a = (i * (0.37 + d * 0.13) + s * 0.071).sin();
    let b = (i * (0.19 + d * 0.29) + s * 0.037).cos();
    ((a + b) * 0.75) as f32
}

fn hashed_unit(sample_idx: u64, class_idx: u64, seed: u64) -> f32 {
    let mut x = sample_idx
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(class_idx.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(seed);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x as f64 / u64::MAX as f64) as f32
}

fn assert_metric_gte(metric: &MetricValue, threshold: f64) {
    let value = metric_value(metric);
    assert!(
        value >= threshold,
        "expected metric >= {threshold}, got {value}"
    );
}

fn assert_metric_lte(metric: &MetricValue, threshold: f64) {
    let value = metric_value(metric);
    assert!(
        value <= threshold,
        "expected metric <= {threshold}, got {value}"
    );
}

fn assert_metric_between(metric: &MetricValue, min: f64, max: f64) {
    let value = metric_value(metric);
    assert!(
        (min..=max).contains(&value),
        "expected metric in [{min}, {max}], got {value}"
    );
}

fn metric_value(metric: &MetricValue) -> f64 {
    match metric {
        MetricValue::Value { value } => *value,
        MetricValue::Na { reason } => panic!("expected numeric metric, got N/A: {reason}"),
    }
}

fn assert_metric_close(left: &MetricValue, right: &MetricValue, tolerance: f64) {
    let left = metric_value(left);
    let right = metric_value(right);
    assert!(
        (left - right).abs() <= tolerance,
        "expected metrics to match within {tolerance}, left={left}, right={right}"
    );
}
