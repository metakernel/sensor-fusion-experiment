use sfx_bench::{
    CountBaselineKind, MetricValue, ProbeConfig, ProbeDataset, RegressionLossKind,
    run_class_prior_baseline, run_count_mean_baseline, run_pca_on_raw_baseline,
    run_random_projection_baseline, run_raw_linear_val_regularized_baseline,
};

#[test]
fn class_prior_baseline_is_train_only_and_deterministic() {
    let dataset = prior_shift_fixture(400, 240);
    let first = run_class_prior_baseline(&dataset, 7).expect("class-prior baseline should run");
    let second = run_class_prior_baseline(&dataset, 77).expect("class-prior baseline should run");

    let first_cls = first
        .classification
        .expect("classification metrics should be present");
    let second_cls = second
        .classification
        .expect("classification metrics should be present");

    assert_metric_close(&first_cls.metrics.map, &second_cls.metrics.map, 1e-12);
    assert_metric_close(&first_cls.metrics.f1, &second_cls.metrics.f1, 1e-12);
    assert_metric_close(&first_cls.metrics.auroc, &second_cls.metrics.auroc, 1e-12);
    assert_metric_close(
        &first_cls.metrics.balanced_accuracy,
        &second_cls.metrics.balanced_accuracy,
        1e-12,
    );

    assert!(
        first_cls.eval_loss > first_cls.train_loss + 0.25,
        "expected train-only fit to suffer on shifted eval prior, train_loss={}, eval_loss={}",
        first_cls.train_loss,
        first_cls.eval_loss
    );
    assert_metric_close(
        &first_cls.metrics.auroc,
        &MetricValue::Value { value: 0.5 },
        1e-12,
    );
}

#[test]
fn count_baselines_are_train_only_and_deterministic() {
    let dataset = count_shift_fixture(360, 220);
    let config = ProbeConfig {
        regression_loss: RegressionLossKind::Mse,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let first = run_count_mean_baseline(&dataset, CountBaselineKind::CountMean, &config, 1)
        .expect("count-mean baseline should run");
    let second = run_count_mean_baseline(&dataset, CountBaselineKind::CountMean, &config, 11)
        .expect("count-mean baseline should run");
    let log1p = run_count_mean_baseline(&dataset, CountBaselineKind::Log1pMean, &config, 1)
        .expect("log1p-mean baseline should run");

    let first_reg = first
        .regression
        .expect("regression metrics should be present for count baseline");
    let second_reg = second
        .regression
        .expect("regression metrics should be present for count baseline");
    let log1p_reg = log1p
        .regression
        .expect("regression metrics should be present for count baseline");

    assert_metric_close(&first_reg.metrics.rmse, &second_reg.metrics.rmse, 1e-12);
    assert_metric_close(&first_reg.metrics.mae, &second_reg.metrics.mae, 1e-12);
    assert!(
        first_reg.eval_loss > first_reg.train_loss + 0.2,
        "expected train-only fit to incur larger eval loss under count shift, train_loss={}, eval_loss={}",
        first_reg.train_loss,
        first_reg.eval_loss
    );

    let count_mean_rmse = metric_value(&first_reg.metrics.rmse);
    let log1p_mean_rmse = metric_value(&log1p_reg.metrics.rmse);
    assert!(
        (count_mean_rmse - log1p_mean_rmse).abs() > 1e-3,
        "count-mean and log1p-mean should produce different RMSE on this skewed fixture"
    );
}

#[test]
fn representation_baselines_are_deterministic_and_beat_class_prior() {
    let dataset = signal_classification_fixture(420, 220, 24);
    let config = ProbeConfig {
        max_iters: 180,
        learning_rate: 0.05,
        batch_size: 64,
        l2: 1e-3,
        balance_classes: true,
        standardize_features: true,
        deterministic: true,
        ..ProbeConfig::default()
    };

    let class_prior = run_class_prior_baseline(&dataset, 5).expect("class-prior should run");
    let random_first = run_random_projection_baseline(&dataset, &config, 6, 99)
        .expect("random-projection baseline should run");
    let random_second = run_random_projection_baseline(&dataset, &config, 6, 99)
        .expect("random-projection baseline should run");
    let pca_first =
        run_pca_on_raw_baseline(&dataset, &config, 6, 99).expect("pca-on-raw baseline should run");
    let pca_second =
        run_pca_on_raw_baseline(&dataset, &config, 6, 99).expect("pca-on-raw baseline should run");

    let prior_map = metric_value(
        &class_prior
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );
    let random_first_map = metric_value(
        &random_first
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );
    let random_second_map = metric_value(
        &random_second
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );
    let pca_first_map = metric_value(
        &pca_first
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );
    let pca_second_map = metric_value(
        &pca_second
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );

    assert_close_f64(random_first_map, random_second_map, 1e-9);
    assert_close_f64(pca_first_map, pca_second_map, 1e-9);
    assert!(
        random_first_map > prior_map + 0.05,
        "random projection baseline should beat class prior on informative features, random_map={random_first_map}, prior_map={prior_map}"
    );
    assert!(
        pca_first_map > prior_map + 0.05,
        "pca baseline should beat class prior on informative features, pca_map={pca_first_map}, prior_map={prior_map}"
    );
}

#[test]
fn raw_linear_val_regularized_baseline_is_deterministic() {
    let (train_eval_dataset, train_val_dataset) = raw_linear_regularization_fixture();
    let config = ProbeConfig {
        max_iters: 150,
        learning_rate: 0.04,
        batch_size: 64,
        l2: 0.0,
        balance_classes: true,
        standardize_features: true,
        deterministic: true,
        ..ProbeConfig::default()
    };
    let candidates = [0.0, 1e-4, 1e-3, 1e-2];

    let first = run_raw_linear_val_regularized_baseline(
        &train_eval_dataset,
        &train_val_dataset,
        &config,
        123,
        &candidates,
    )
    .expect("raw linear val-regularized baseline should run");
    let second = run_raw_linear_val_regularized_baseline(
        &train_eval_dataset,
        &train_val_dataset,
        &config,
        123,
        &candidates,
    )
    .expect("raw linear val-regularized baseline should run");

    assert_eq!(
        first.selected_l2, second.selected_l2,
        "selected l2 should be deterministic"
    );
    assert_eq!(first.candidate_scores.len(), candidates.len());
    assert!(
        candidates.contains(&first.selected_l2),
        "selected l2 must come from candidate grid"
    );

    let first_map = metric_value(
        &first
            .probe_result
            .classification
            .as_ref()
            .expect("classification should be present")
            .metrics
            .map,
    );
    let second_map = metric_value(
        &second
            .probe_result
            .classification
            .as_ref()
            .expect("classification should be present")
            .metrics
            .map,
    );
    assert_close_f64(first_map, second_map, 1e-9);

    let best_candidate_score = first
        .candidate_scores
        .iter()
        .map(|candidate| candidate.validation_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let selected_score = first
        .candidate_scores
        .iter()
        .find(|candidate| candidate.l2 == first.selected_l2)
        .map(|candidate| candidate.validation_score)
        .expect("selected l2 should appear in candidate score table");
    assert_close_f64(selected_score, best_candidate_score, 1e-12);

    let class_prior = run_class_prior_baseline(&train_eval_dataset, 5)
        .expect("class-prior baseline should run for comparison");
    let prior_map = metric_value(
        &class_prior
            .classification
            .expect("classification should be present")
            .metrics
            .map,
    );
    assert!(
        first_map > prior_map + 0.05,
        "raw linear val-regularized baseline should beat class prior on informative data, raw_map={first_map}, prior_map={prior_map}"
    );
}

fn prior_shift_fixture(train_samples: usize, eval_samples: usize) -> ProbeDataset {
    let train_features = build_features(train_samples, 8, 13);
    let eval_features = build_features(eval_samples, 8, 1113);
    let train_presence = (0..train_samples)
        .map(|idx| vec![u8::from(idx % 10 != 0), u8::from(idx % 5 == 0)])
        .collect();
    let eval_presence = (0..eval_samples)
        .map(|idx| vec![u8::from(idx % 10 == 0), u8::from(idx % 5 != 0)])
        .collect();
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: Some(train_presence),
        eval_presence: Some(eval_presence),
        train_counts: None,
        eval_counts: None,
        class_names: vec!["vehicle".to_string(), "pedestrian".to_string()],
        count_target_names: Vec::new(),
    }
}

fn count_shift_fixture(train_samples: usize, eval_samples: usize) -> ProbeDataset {
    let train_features = build_features(train_samples, 6, 29);
    let eval_features = build_features(eval_samples, 6, 2029);
    let train_counts = (0..train_samples)
        .map(|idx| {
            vec![
                18.0 + (idx % 9) as f32,
                10.0 + ((idx * 3) % 7) as f32,
                6.0 + ((idx * 5) % 6) as f32,
            ]
        })
        .collect();
    let eval_counts = (0..eval_samples)
        .map(|idx| {
            vec![
                1.0 + (idx % 3) as f32,
                2.0 + ((idx * 2) % 4) as f32,
                0.5 + ((idx * 7) % 3) as f32,
            ]
        })
        .collect();
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: None,
        eval_presence: None,
        train_counts: Some(train_counts),
        eval_counts: Some(eval_counts),
        class_names: Vec::new(),
        count_target_names: vec![
            "objects".to_string(),
            "moving-objects".to_string(),
            "small-objects".to_string(),
        ],
    }
}

fn signal_classification_fixture(
    train_samples: usize,
    eval_samples: usize,
    feature_dim: usize,
) -> ProbeDataset {
    let (train_features, train_presence) = build_signal_split(train_samples, feature_dim, 91);
    let (eval_features, eval_presence) = build_signal_split(eval_samples, feature_dim, 4091);
    ProbeDataset {
        train_features,
        eval_features,
        train_presence: Some(train_presence),
        eval_presence: Some(eval_presence),
        train_counts: None,
        eval_counts: None,
        class_names: vec!["vehicle".to_string(), "cyclist".to_string()],
        count_target_names: Vec::new(),
    }
}

fn raw_linear_regularization_fixture() -> (ProbeDataset, ProbeDataset) {
    let feature_dim = 64;
    let (train_features, train_presence) = build_signal_split(360, feature_dim, 777);
    let (val_features, val_presence) = build_signal_split(180, feature_dim, 1777);
    let (eval_features, eval_presence) = build_signal_split(180, feature_dim, 2777);
    let class_names = vec!["vehicle".to_string(), "cyclist".to_string()];

    let train_eval_dataset = ProbeDataset {
        train_features: train_features.clone(),
        eval_features,
        train_presence: Some(train_presence.clone()),
        eval_presence: Some(eval_presence),
        train_counts: None,
        eval_counts: None,
        class_names: class_names.clone(),
        count_target_names: Vec::new(),
    };
    let train_val_dataset = ProbeDataset {
        train_features,
        eval_features: val_features,
        train_presence: Some(train_presence),
        eval_presence: Some(val_presence),
        train_counts: None,
        eval_counts: None,
        class_names,
        count_target_names: Vec::new(),
    };

    (train_eval_dataset, train_val_dataset)
}

fn build_signal_split(
    samples: usize,
    feature_dim: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut features = Vec::with_capacity(samples);
    let mut labels = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let x0 = signal(sample_idx, 0, seed);
        let x1 = signal(sample_idx, 1, seed);
        let x2 = signal(sample_idx, 2, seed);
        let x3 = signal(sample_idx, 3, seed);
        let noise0 = hashed_unit(sample_idx as u64, 0, seed ^ 0xa1) - 0.5;
        let noise1 = hashed_unit(sample_idx as u64, 1, seed ^ 0xb2) - 0.5;

        let mut row = Vec::with_capacity(feature_dim);
        row.push(1.8 * x0 + 0.5 * x1 + 0.2 * noise0);
        row.push(-0.9 * x1 + 0.4 * x2 + 0.2 * noise1);
        row.push(1.5 * x2 - 0.2 * x3 + 0.1 * noise0);
        row.push(1.2 * x3 + 0.3 * x0 + 0.1 * noise1);
        for dim_idx in 4..feature_dim {
            let jitter = hashed_unit(
                sample_idx as u64,
                dim_idx as u64,
                seed ^ 0x52f9_26db_0f17_1d3b,
            ) - 0.5;
            let value =
                0.35 * signal(sample_idx, dim_idx + 7, seed ^ 0x9e37_79b9_7f4a_7c15) + 0.5 * jitter;
            row.push(value);
        }
        let label_vehicle = u8::from(row[0] - 0.6 * row[1] + 0.25 * row[2] > 0.0);
        let label_cyclist = u8::from(row[2] + 0.7 * row[3] - 0.2 * row[0] > 0.1);
        labels.push(vec![label_vehicle, label_cyclist]);
        features.push(row);
    }

    (features, labels)
}

fn build_features(samples: usize, feature_dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..samples)
        .map(|sample_idx| {
            (0..feature_dim)
                .map(|dim_idx| signal(sample_idx, dim_idx, seed))
                .collect()
        })
        .collect()
}

fn signal(sample_idx: usize, dim_idx: usize, seed: u64) -> f32 {
    let i = sample_idx as f64;
    let d = dim_idx as f64;
    let s = seed as f64;
    let a = (i * (0.31 + d * 0.17) + s * 0.047).sin();
    let b = (i * (0.19 + d * 0.23) + s * 0.079).cos();
    ((a + b) * 0.75) as f32
}

fn hashed_unit(sample_idx: u64, dim_idx: u64, seed: u64) -> f32 {
    let mut x = sample_idx
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(dim_idx.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(seed);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x as f64 / u64::MAX as f64) as f32
}

fn metric_value(metric: &MetricValue) -> f64 {
    match metric {
        MetricValue::Value { value } => *value,
        MetricValue::Na { reason } => panic!("expected numeric metric, got N/A: {reason}"),
    }
}

fn assert_metric_close(left: &MetricValue, right: &MetricValue, tolerance: f64) {
    assert_close_f64(metric_value(left), metric_value(right), tolerance);
}

fn assert_close_f64(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "expected values within {tolerance}, left={left}, right={right}"
    );
}
