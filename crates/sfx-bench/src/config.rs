use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, ConfigError>;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid {path}: {message}")]
    Invalid { path: PathBuf, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    #[serde(default)]
    pub bench: BenchMetadata,
    #[serde(default = "default_seeds")]
    pub seeds: Vec<u64>,
    #[serde(default = "default_classes")]
    pub classes: Vec<String>,
    #[serde(default = "default_protocols")]
    pub protocols: Vec<ProtocolConfig>,
    #[serde(default)]
    pub probe: ProbeConfig,
    #[serde(default)]
    pub recon: ReconConfig,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            bench: BenchMetadata::default(),
            seeds: default_seeds(),
            classes: default_classes(),
            protocols: default_protocols(),
            probe: ProbeConfig::default(),
            recon: ReconConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetadata {
    #[serde(default = "default_bench_name")]
    pub name: String,
    #[serde(default = "default_artifacts_dir")]
    pub artifacts_dir: PathBuf,
}

impl Default for BenchMetadata {
    fn default() -> Self {
        Self {
            name: default_bench_name(),
            artifacts_dir: default_artifacts_dir(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    pub name: String,
    #[serde(default = "default_protocol_split")]
    pub split: String,
    #[serde(default = "default_protocol_baselines")]
    pub baselines: Vec<String>,
    #[serde(default = "default_protocol_tasks")]
    pub tasks: Vec<String>,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            name: "linear-probe-val".to_string(),
            split: default_protocol_split(),
            baselines: default_protocol_baselines(),
            tasks: default_protocol_tasks(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegressionLossKind {
    Huber,
    Mse,
}

impl Default for RegressionLossKind {
    fn default() -> Self {
        Self::Huber
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    #[serde(default = "default_probe_kind")]
    pub kind: String,
    #[serde(default = "default_probe_max_iters")]
    pub max_iters: usize,
    #[serde(default = "default_probe_learning_rate")]
    pub learning_rate: f32,
    #[serde(default = "default_probe_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_probe_l2")]
    pub l2: f32,
    #[serde(default = "default_probe_balance_classes")]
    pub balance_classes: bool,
    #[serde(default = "default_probe_standardize_features")]
    pub standardize_features: bool,
    #[serde(default)]
    pub regression_loss: RegressionLossKind,
    #[serde(default = "default_probe_huber_delta")]
    pub huber_delta: f32,
    #[serde(default = "default_probe_deterministic")]
    pub deterministic: bool,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            kind: default_probe_kind(),
            max_iters: default_probe_max_iters(),
            learning_rate: default_probe_learning_rate(),
            batch_size: default_probe_batch_size(),
            l2: default_probe_l2(),
            balance_classes: default_probe_balance_classes(),
            standardize_features: default_probe_standardize_features(),
            regression_loss: RegressionLossKind::default(),
            huber_delta: default_probe_huber_delta(),
            deterministic: default_probe_deterministic(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconConfig {
    #[serde(default = "default_recon_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_recon_metrics")]
    pub metrics: Vec<String>,
    #[serde(default)]
    pub save_previews: bool,
}

impl Default for ReconConfig {
    fn default() -> Self {
        Self {
            batch_size: default_recon_batch_size(),
            metrics: default_recon_metrics(),
            save_previews: false,
        }
    }
}

pub fn resolve_from_root(root: impl AsRef<Path>, path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.as_ref().join(path)
    }
}

pub fn load_bench_config(root: impl AsRef<Path>, path: impl AsRef<Path>) -> Result<BenchConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let raw = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;

    let mut config: BenchConfig = toml::from_str(&raw).map_err(|source| ConfigError::Parse {
        path: path.clone(),
        source,
    })?;

    validate_config(&path, &config)?;
    config.bench.artifacts_dir = resolve_from_root(root, &config.bench.artifacts_dir);
    Ok(config)
}

fn validate_config(path: &Path, config: &BenchConfig) -> Result<()> {
    ensure(
        path,
        !config.bench.name.trim().is_empty(),
        "bench name is required",
    )?;
    ensure(
        path,
        !config.seeds.is_empty(),
        "at least one seed is required",
    )?;
    ensure(
        path,
        !config.classes.is_empty(),
        "at least one class label is required",
    )?;
    for class in &config.classes {
        ensure(
            path,
            !class.trim().is_empty(),
            "class labels cannot be empty",
        )?;
    }
    ensure(
        path,
        !config.protocols.is_empty(),
        "at least one protocol is required",
    )?;
    for protocol in &config.protocols {
        ensure(
            path,
            !protocol.name.trim().is_empty(),
            "protocol name is required",
        )?;
        ensure(
            path,
            matches!(
                protocol.split.trim().to_ascii_lowercase().as_str(),
                "train" | "val" | "test" | "all"
            ),
            format!(
                "protocol split must be train, val, test, or all, got {}",
                protocol.split
            ),
        )?;
        ensure(
            path,
            !protocol.baselines.is_empty(),
            format!("protocol {} requires at least one baseline", protocol.name),
        )?;
        for baseline in &protocol.baselines {
            ensure(
                path,
                !baseline.trim().is_empty(),
                format!("protocol {} contains an empty baseline name", protocol.name),
            )?;
        }
        ensure(
            path,
            !protocol.tasks.is_empty(),
            format!("protocol {} requires at least one task", protocol.name),
        )?;
        for task in &protocol.tasks {
            ensure(
                path,
                !task.trim().is_empty(),
                format!("protocol {} contains an empty task name", protocol.name),
            )?;
        }
    }
    ensure(
        path,
        config.probe.kind.trim().eq_ignore_ascii_case("linear"),
        format!(
            "probe.kind must be `linear`, got {}",
            config.probe.kind.trim()
        ),
    )?;
    ensure(
        path,
        config.probe.max_iters > 0,
        "probe.max_iters must be greater than zero",
    )?;
    ensure(
        path,
        config.probe.learning_rate > 0.0,
        "probe.learning_rate must be greater than zero",
    )?;
    ensure(
        path,
        config.probe.batch_size > 0,
        "probe.batch_size must be greater than zero",
    )?;
    ensure(path, config.probe.l2 >= 0.0, "probe.l2 cannot be negative")?;
    ensure(
        path,
        config.probe.huber_delta >= 0.0,
        "probe.huber_delta cannot be negative",
    )?;
    ensure(
        path,
        config.recon.batch_size > 0,
        "recon.batch_size must be greater than zero",
    )?;
    ensure(
        path,
        !config.recon.metrics.is_empty(),
        "recon.metrics must include at least one metric",
    )?;
    for metric in &config.recon.metrics {
        ensure(
            path,
            !metric.trim().is_empty(),
            "recon metric names cannot be empty",
        )?;
    }

    Ok(())
}

fn ensure(path: &Path, condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ConfigError::Invalid {
            path: path.to_path_buf(),
            message: message.into(),
        })
    }
}

fn default_bench_name() -> String {
    "waymo-benchmark-v0".to_string()
}

fn default_artifacts_dir() -> PathBuf {
    PathBuf::from("artifacts/bench")
}

fn default_seeds() -> Vec<u64> {
    vec![7, 42, 1337]
}

fn default_classes() -> Vec<String> {
    vec![
        "vehicle".to_string(),
        "pedestrian".to_string(),
        "cyclist".to_string(),
    ]
}

fn default_protocols() -> Vec<ProtocolConfig> {
    vec![ProtocolConfig::default()]
}

fn default_protocol_split() -> String {
    "val".to_string()
}

fn default_protocol_baselines() -> Vec<String> {
    vec![
        "range-only".to_string(),
        "rgb-only".to_string(),
        "fusion".to_string(),
    ]
}

fn default_protocol_tasks() -> Vec<String> {
    vec![
        "classification".to_string(),
        "regression".to_string(),
        "reconstruction".to_string(),
    ]
}

fn default_probe_kind() -> String {
    "linear".to_string()
}

fn default_probe_max_iters() -> usize {
    200
}

fn default_probe_learning_rate() -> f32 {
    5e-2
}

fn default_probe_batch_size() -> usize {
    64
}

fn default_probe_l2() -> f32 {
    1e-3
}

fn default_probe_balance_classes() -> bool {
    true
}

fn default_probe_standardize_features() -> bool {
    true
}

fn default_probe_huber_delta() -> f32 {
    1.0
}

fn default_probe_deterministic() -> bool {
    true
}

fn default_recon_batch_size() -> usize {
    8
}

fn default_recon_metrics() -> Vec<String> {
    vec![
        "mse".to_string(),
        "mae".to_string(),
        "rmse".to_string(),
        "psnr".to_string(),
        "ssim".to_string(),
        "depth-mae-m".to_string(),
        "depth-rmse-m".to_string(),
        "delta-1.25".to_string(),
        "delta-1.25^2".to_string(),
        "valid-pixel-fraction".to_string(),
    ]
}
