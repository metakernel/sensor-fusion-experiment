use crate::{BenchArgs, ProjectPaths, display_from_root};
use anyhow::{Context, Result};
use sfx_bench::{
    BaselineResult, BenchResult, LeakageReport, MetricCI, ProtocolResult, ReconResult, TaskResult,
};
use std::collections::BTreeMap;
use std::fs;

pub(crate) fn run(args: BenchArgs, paths: &ProjectPaths) -> Result<()> {
    let config_path = sfx_bench::resolve_from_root(&paths.root, &args.config);
    let config = sfx_bench::load_bench_config(&paths.root, &args.config)
        .with_context(|| format!("loading {}", config_path.display()))?;

    let out_dir = args
        .out
        .as_ref()
        .map(|path| sfx_config::resolve_from_root(&paths.root, path))
        .unwrap_or_else(|| config.bench.artifacts_dir.clone());
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let protocol_results = config
        .protocols
        .iter()
        .map(|protocol| ProtocolResult {
            protocol: protocol.name.clone(),
            split: protocol.split.clone(),
            baseline_results: protocol
                .baselines
                .iter()
                .map(|baseline| {
                    build_placeholder_baseline(
                        baseline,
                        &protocol.tasks,
                        config.seeds.len(),
                        &config.recon.metrics,
                    )
                })
                .collect(),
            leakage: Some(LeakageReport {
                leakage_detected: false,
                overlap_fraction: 0.0,
                overlapping_groups: 0,
                notes: vec!["TODO(ws-h): implement leakage checks".to_string()],
            }),
        })
        .collect();

    let mut metadata = BTreeMap::new();
    metadata.insert("status".to_string(), "placeholder".to_string());
    metadata.insert(
        "todo".to_string(),
        "TODO(ws-h): run configured benchmark protocols".to_string(),
    );

    let result = BenchResult {
        bench_name: config.bench.name,
        config_path: display_from_root(&paths.root, &config_path),
        seeds: config.seeds,
        classes: config.classes,
        protocol_results,
        leakage: Some(LeakageReport {
            leakage_detected: false,
            overlap_fraction: 0.0,
            overlapping_groups: 0,
            notes: vec!["TODO(ws-h): aggregate leakage report across protocols".to_string()],
        }),
        metadata,
    };

    let result_path = out_dir.join("bench.placeholder.json");
    let report_path = out_dir.join("bench.placeholder.txt");
    let raw = serde_json::to_string_pretty(&result).context("serializing benchmark placeholder")?;
    fs::write(&result_path, raw).with_context(|| format!("writing {}", result_path.display()))?;
    fs::write(
        &report_path,
        "TODO(ws-h): implement benchmark execution and replace placeholder outputs.\n",
    )
    .with_context(|| format!("writing {}", report_path.display()))?;

    println!("ok   {}", display_from_root(&paths.root, &result_path));
    println!("ok   {}", display_from_root(&paths.root, &report_path));
    println!("note benchmark execution is currently stubbed");

    Ok(())
}

fn build_placeholder_baseline(
    baseline: &str,
    tasks: &[String],
    n_seeds: usize,
    recon_metrics: &[String],
) -> BaselineResult {
    let task_results = tasks
        .iter()
        .filter_map(|task| placeholder_task_result(task, n_seeds))
        .collect();
    let recon_results = if tasks
        .iter()
        .any(|task| task.eq_ignore_ascii_case("reconstruction"))
    {
        vec![
            placeholder_recon_result("rgb", recon_metrics, n_seeds),
            placeholder_recon_result("range", recon_metrics, n_seeds),
        ]
    } else {
        Vec::new()
    };

    BaselineResult {
        baseline: baseline.to_string(),
        task_results,
        recon_results,
        metrics: BTreeMap::new(),
        notes: vec!["TODO(ws-h): populate metrics from benchmark runs".to_string()],
    }
}

fn placeholder_task_result(task: &str, n_seeds: usize) -> Option<TaskResult> {
    let score = MetricCI::point_estimate(0.0, n_seeds);
    match task.trim().to_ascii_lowercase().as_str() {
        "classification" => Some(TaskResult::Classification {
            target: "class".to_string(),
            primary_metric: "macro_f1".to_string(),
            score,
            per_class: BTreeMap::new(),
        }),
        "regression" => Some(TaskResult::Regression {
            target: "value".to_string(),
            primary_metric: "rmse".to_string(),
            score,
            per_target: BTreeMap::new(),
        }),
        _ => None,
    }
}

fn placeholder_recon_result(modality: &str, metrics: &[String], n_seeds: usize) -> ReconResult {
    let includes = |names: &[&str]| {
        metrics.iter().any(|metric| {
            names
                .iter()
                .any(|name| metric.trim().eq_ignore_ascii_case(name))
        })
    };
    ReconResult {
        modality: modality.to_string(),
        mse: includes(&["mse"]).then(|| MetricCI::point_estimate(0.0, n_seeds)),
        mae: includes(&["mae"]).then(|| MetricCI::point_estimate(0.0, n_seeds)),
        rmse: includes(&["rmse"]).then(|| MetricCI::point_estimate(0.0, n_seeds)),
        psnr_db: includes(&["psnr", "proxy-psnr", "proxy_psnr_db"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        ssim: includes(&["ssim", "proxy-ssim", "proxy_ssim"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        proxy_psnr_db: includes(&["psnr", "proxy-psnr", "proxy_psnr_db"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        proxy_ssim: includes(&["ssim", "proxy-ssim", "proxy_ssim"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        depth_mae_m: includes(&["depth-mae-m", "depth_mae_m"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        depth_rmse_m: includes(&["depth-rmse-m", "depth_rmse_m"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        depth_delta_lt_1_25: includes(&["delta-1.25", "depth-delta-1.25", "depth_delta_lt_1_25"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        depth_delta_lt_1_25_sq: includes(&[
            "delta-1.25^2",
            "delta-1.25-sq",
            "depth-delta-1.25-sq",
            "depth_delta_lt_1_25_sq",
        ])
        .then(|| MetricCI::point_estimate(0.0, n_seeds)),
        valid_pixel_fraction: includes(&["valid-pixel-fraction", "valid_pixel_fraction"])
            .then(|| MetricCI::point_estimate(0.0, n_seeds)),
    }
}
