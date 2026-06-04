use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, ValidationIssue, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub run: RunConfig,
    pub trainer: TrainerConfig,
    pub distributed: Option<DistributedConfig>,
    pub optimizer: OptimizerConfig,
    pub scheduler: Option<SchedulerConfig>,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub outputs: OutputConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            run: RunConfig::default(),
            trainer: TrainerConfig::default(),
            distributed: None,
            optimizer: OptimizerConfig::default(),
            scheduler: None,
            early_stopping: None,
            outputs: OutputConfig::default(),
        }
    }
}

impl TrainingConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: TrainingToml = parse_toml_str(raw)?;
        let config = match parsed {
            TrainingToml::Wrapped(value) => value.training,
            TrainingToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: TrainingToml = parse_toml_file(path)?;
        let config = match parsed {
            TrainingToml::Wrapped(value) => value.training,
            TrainingToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file_with_repo_root(
        path: impl AsRef<Path>,
        repo_root: impl AsRef<Path>,
    ) -> Result<Self, ConfigError> {
        let mut config = Self::load_from_file(path)?;
        config.resolve_paths(repo_root);
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn resolve_paths(&mut self, repo_root: impl AsRef<Path>) {
        self.outputs.resolve_paths(repo_root.as_ref());
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        self.run.validate_into("run", &mut issues);
        self.trainer.validate_into("trainer", &mut issues);
        self.optimizer.validate_into("optimizer", &mut issues);
        self.outputs.validate_into("outputs", &mut issues);

        if let Some(distributed) = &self.distributed {
            distributed.validate_into("distributed", &mut issues);
        }
        if let Some(scheduler) = &self.scheduler {
            scheduler.validate_into("scheduler", &mut issues);
        }
        if let Some(early_stopping) = &self.early_stopping {
            early_stopping.validate_into("early_stopping", &mut issues);
        }

        finish_validation(issues)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    pub name: String,
    pub seed: u64,
    pub device: String,
    pub backend: String,
    pub precision: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            name: "run".to_string(),
            seed: 42,
            device: "cpu".to_string(),
            backend: "cpu".to_string(),
            precision: "fp32".to_string(),
        }
    }
}

impl RunConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.name.trim().is_empty() {
            issues.push(issue(format!("{prefix}.name"), "must not be empty"));
        }
        if self.device.trim().is_empty() {
            issues.push(issue(format!("{prefix}.device"), "must not be empty"));
        }
        let backend = self.backend.trim().to_ascii_lowercase();
        if backend.is_empty() {
            issues.push(issue(
                format!("{prefix}.backend"),
                "must not be empty; supported values: cpu, cuda, wgpu, gpu",
            ));
        } else if !matches!(backend.as_str(), "cpu" | "cuda" | "wgpu" | "gpu") {
            issues.push(issue(
                format!("{prefix}.backend"),
                format!(
                    "unsupported backend `{}`; supported values: cpu, cuda, wgpu, gpu",
                    self.backend.trim()
                ),
            ));
        }
        if self.precision.trim().is_empty() {
            issues.push(issue(format!("{prefix}.precision"), "must not be empty"));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainerConfig {
    pub max_epochs: u32,
    pub batch_size: u32,
    pub num_workers: u32,
    pub gradient_accumulation_steps: u32,
    pub gradient_clip_norm: f32,
    pub log_every_n_steps: u32,
    pub limit_train_batches: f64,
    pub limit_val_batches: f64,
    pub checkpoint_every_n_epochs: u32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1,
            batch_size: 1,
            num_workers: 0,
            gradient_accumulation_steps: 1,
            gradient_clip_norm: 1.0,
            log_every_n_steps: 1,
            limit_train_batches: 1.0,
            limit_val_batches: 1.0,
            checkpoint_every_n_epochs: 1,
        }
    }
}

impl TrainerConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.max_epochs == 0 {
            issues.push(issue(format!("{prefix}.max_epochs"), "must be > 0"));
        }
        if self.batch_size == 0 {
            issues.push(issue(format!("{prefix}.batch_size"), "must be > 0"));
        }
        if self.gradient_accumulation_steps == 0 {
            issues.push(issue(
                format!("{prefix}.gradient_accumulation_steps"),
                "must be > 0",
            ));
        }
        if self.log_every_n_steps == 0 {
            issues.push(issue(format!("{prefix}.log_every_n_steps"), "must be > 0"));
        }
        if self.checkpoint_every_n_epochs == 0 {
            issues.push(issue(
                format!("{prefix}.checkpoint_every_n_epochs"),
                "must be > 0",
            ));
        }
        if !self.gradient_clip_norm.is_finite() || self.gradient_clip_norm < 0.0 {
            issues.push(issue(
                format!("{prefix}.gradient_clip_norm"),
                "must be a finite number >= 0.0",
            ));
        }
        if !self.limit_train_batches.is_finite() || self.limit_train_batches <= 0.0 {
            issues.push(issue(
                format!("{prefix}.limit_train_batches"),
                "must be a finite number > 0.0",
            ));
        }
        if !self.limit_val_batches.is_finite() || self.limit_val_batches <= 0.0 {
            issues.push(issue(
                format!("{prefix}.limit_val_batches"),
                "must be a finite number > 0.0",
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DistributedConfig {
    pub strategy: String,
    pub num_nodes: u32,
    pub devices: u32,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            strategy: "ddp".to_string(),
            num_nodes: 1,
            devices: 1,
        }
    }
}

impl DistributedConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.strategy.trim().is_empty() {
            issues.push(issue(format!("{prefix}.strategy"), "must not be empty"));
        }
        if self.num_nodes == 0 {
            issues.push(issue(format!("{prefix}.num_nodes"), "must be > 0"));
        }
        if self.devices == 0 {
            issues.push(issue(format!("{prefix}.devices"), "must be > 0"));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OptimizerConfig {
    pub name: String,
    pub lr: f64,
    pub weight_decay: f64,
    pub betas: Option<[f64; 2]>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            name: "adamw".to_string(),
            lr: 1e-3,
            weight_decay: 0.0,
            betas: None,
        }
    }
}

impl OptimizerConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.name.trim().is_empty() {
            issues.push(issue(format!("{prefix}.name"), "must not be empty"));
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            issues.push(issue(
                format!("{prefix}.lr"),
                "must be a finite number > 0.0",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            issues.push(issue(
                format!("{prefix}.weight_decay"),
                "must be a finite number >= 0.0",
            ));
        }
        if let Some([beta1, beta2]) = self.betas {
            if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
                issues.push(issue(
                    format!("{prefix}.betas[0]"),
                    "must be between 0.0 and 1.0",
                ));
            }
            if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
                issues.push(issue(
                    format!("{prefix}.betas[1]"),
                    "must be between 0.0 and 1.0",
                ));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub name: String,
    pub warmup_steps: u32,
    pub min_lr: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            name: "cosine".to_string(),
            warmup_steps: 0,
            min_lr: 0.0,
        }
    }
}

impl SchedulerConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.name.trim().is_empty() {
            issues.push(issue(format!("{prefix}.name"), "must not be empty"));
        }
        if !self.min_lr.is_finite() || self.min_lr < 0.0 {
            issues.push(issue(
                format!("{prefix}.min_lr"),
                "must be a finite number >= 0.0",
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: u32,
    pub monitor: String,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            patience: 5,
            monitor: "val/loss".to_string(),
        }
    }
}

impl EarlyStoppingConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.enabled {
            if self.patience == 0 {
                issues.push(issue(format!("{prefix}.patience"), "must be > 0"));
            }
            if self.monitor.trim().is_empty() {
                issues.push(issue(format!("{prefix}.monitor"), "must not be empty"));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    pub dir: PathBuf,
    pub save_last: bool,
    pub save_best: bool,
    pub best_monitor: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("artifacts/checkpoints"),
            save_last: true,
            save_best: false,
            best_monitor: None,
        }
    }
}

impl OutputConfig {
    fn resolve_paths(&mut self, repo_root: &Path) {
        self.dir = resolve_if_present(repo_root, &self.dir);
    }

    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.dir.as_os_str().is_empty() {
            issues.push(issue(format!("{prefix}.dir"), "must not be empty"));
        }
        if self.save_best {
            match &self.best_monitor {
                Some(value) if !value.trim().is_empty() => {}
                _ => issues.push(issue(
                    format!("{prefix}.best_monitor"),
                    "must not be empty when save_best is true",
                )),
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TrainingToml {
    Wrapped(TrainingFile),
    Direct(TrainingConfig),
}

#[derive(Debug, Deserialize)]
struct TrainingFile {
    training: TrainingConfig,
}
