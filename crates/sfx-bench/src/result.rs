use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub bench_name: String,
    #[serde(default)]
    pub config_path: String,
    #[serde(default)]
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default)]
    pub protocol_results: Vec<ProtocolResult>,
    #[serde(default)]
    pub leakage: Option<LeakageReport>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResult {
    pub protocol: String,
    pub split: String,
    #[serde(default)]
    pub baseline_results: Vec<BaselineResult>,
    #[serde(default)]
    pub leakage: Option<LeakageReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    pub baseline: String,
    #[serde(default)]
    pub task_results: Vec<TaskResult>,
    #[serde(default)]
    pub recon_results: Vec<ReconResult>,
    #[serde(default)]
    pub metrics: BTreeMap<String, MetricCI>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task_type", rename_all = "kebab-case")]
pub enum TaskResult {
    Classification {
        target: String,
        primary_metric: String,
        score: MetricCI,
        #[serde(default)]
        per_class: BTreeMap<String, MetricCI>,
    },
    Regression {
        target: String,
        primary_metric: String,
        score: MetricCI,
        #[serde(default)]
        per_target: BTreeMap<String, MetricCI>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconResult {
    pub modality: String,
    #[serde(default)]
    pub mse: Option<MetricCI>,
    #[serde(default)]
    pub mae: Option<MetricCI>,
    #[serde(default)]
    pub rmse: Option<MetricCI>,
    #[serde(default)]
    pub psnr_db: Option<MetricCI>,
    #[serde(default)]
    pub ssim: Option<MetricCI>,
    #[serde(default)]
    pub proxy_psnr_db: Option<MetricCI>,
    #[serde(default)]
    pub proxy_ssim: Option<MetricCI>,
    #[serde(default)]
    pub depth_mae_m: Option<MetricCI>,
    #[serde(default)]
    pub depth_rmse_m: Option<MetricCI>,
    #[serde(default)]
    pub depth_delta_lt_1_25: Option<MetricCI>,
    #[serde(default)]
    pub depth_delta_lt_1_25_sq: Option<MetricCI>,
    #[serde(default)]
    pub valid_pixel_fraction: Option<MetricCI>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricCI {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub n_seeds: usize,
}

impl MetricCI {
    pub fn point_estimate(mean: f64, n_seeds: usize) -> Self {
        Self {
            mean,
            lo: mean,
            hi: mean,
            n_seeds,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageReport {
    pub leakage_detected: bool,
    pub overlap_fraction: f64,
    #[serde(default)]
    pub overlapping_groups: usize,
    #[serde(default)]
    pub notes: Vec<String>,
}
