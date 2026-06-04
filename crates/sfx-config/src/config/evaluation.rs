use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, ValidationIssue, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    pub run: EvaluationRunConfig,
    pub inputs: EvaluationInputsConfig,
    pub metrics: EvaluationMetricsConfig,
    pub output: EvaluationOutputConfig,
}

impl EvaluationConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: EvaluationToml = parse_toml_str(raw)?;
        let config = match parsed {
            EvaluationToml::Wrapped(value) => value.evaluation,
            EvaluationToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: EvaluationToml = parse_toml_file(path)?;
        let config = match parsed {
            EvaluationToml::Wrapped(value) => value.evaluation,
            EvaluationToml::Direct(value) => value,
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
        self.inputs.resolve_paths(repo_root.as_ref());
        self.output.resolve_paths(repo_root.as_ref());
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        self.run.validate_into("run", &mut issues);
        self.inputs.validate_into("inputs", &mut issues);
        self.metrics.validate_into("metrics", &mut issues);
        self.output.validate_into("output", &mut issues);
        finish_validation(issues)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationRunConfig {
    pub name: String,
    pub device: String,
    pub batch_size: u32,
    pub num_workers: u32,
}

impl Default for EvaluationRunConfig {
    fn default() -> Self {
        Self {
            name: "evaluation".to_string(),
            device: "cpu".to_string(),
            batch_size: 1,
            num_workers: 0,
        }
    }
}

impl EvaluationRunConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.name.trim().is_empty() {
            issues.push(issue(format!("{prefix}.name"), "must not be empty"));
        }
        if self.device.trim().is_empty() {
            issues.push(issue(format!("{prefix}.device"), "must not be empty"));
        }
        if self.batch_size == 0 {
            issues.push(issue(format!("{prefix}.batch_size"), "must be > 0"));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationInputsConfig {
    pub dataset_config: PathBuf,
    pub model_config: PathBuf,
    pub checkpoint_path: PathBuf,
}

impl Default for EvaluationInputsConfig {
    fn default() -> Self {
        Self {
            dataset_config: PathBuf::new(),
            model_config: PathBuf::new(),
            checkpoint_path: PathBuf::new(),
        }
    }
}

impl EvaluationInputsConfig {
    fn resolve_paths(&mut self, repo_root: &Path) {
        self.dataset_config = resolve_if_present(repo_root, &self.dataset_config);
        self.model_config = resolve_if_present(repo_root, &self.model_config);
        self.checkpoint_path = resolve_if_present(repo_root, &self.checkpoint_path);
    }

    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.dataset_config.as_os_str().is_empty() {
            issues.push(issue(
                format!("{prefix}.dataset_config"),
                "must not be empty",
            ));
        }
        if self.model_config.as_os_str().is_empty() {
            issues.push(issue(format!("{prefix}.model_config"), "must not be empty"));
        }
        if self.checkpoint_path.as_os_str().is_empty() {
            issues.push(issue(
                format!("{prefix}.checkpoint_path"),
                "must not be empty",
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationMetricsConfig {
    pub compute_map: bool,
    pub compute_latency: bool,
    pub iou_thresholds: Vec<f32>,
    pub max_detections: u32,
}

impl Default for EvaluationMetricsConfig {
    fn default() -> Self {
        Self {
            compute_map: true,
            compute_latency: false,
            iou_thresholds: vec![0.5],
            max_detections: 100,
        }
    }
}

impl EvaluationMetricsConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.max_detections == 0 {
            issues.push(issue(format!("{prefix}.max_detections"), "must be > 0"));
        }
        if self.iou_thresholds.is_empty() {
            issues.push(issue(
                format!("{prefix}.iou_thresholds"),
                "must contain at least one value",
            ));
        }
        for (idx, value) in self.iou_thresholds.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                issues.push(issue(
                    format!("{prefix}.iou_thresholds[{idx}]"),
                    "must be between 0.0 and 1.0",
                ));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationOutputConfig {
    pub dir: PathBuf,
    pub save_predictions: bool,
    pub save_per_scene: bool,
}

impl Default for EvaluationOutputConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("artifacts/reports/eval"),
            save_predictions: true,
            save_per_scene: false,
        }
    }
}

impl EvaluationOutputConfig {
    fn resolve_paths(&mut self, repo_root: &Path) {
        self.dir = resolve_if_present(repo_root, &self.dir);
    }

    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.dir.as_os_str().is_empty() {
            issues.push(issue(format!("{prefix}.dir"), "must not be empty"));
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EvaluationToml {
    Wrapped(EvaluationFile),
    Direct(EvaluationConfig),
}

#[derive(Debug, Deserialize)]
struct EvaluationFile {
    evaluation: EvaluationConfig,
}
