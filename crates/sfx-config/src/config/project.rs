use serde::{Deserialize, Serialize};
use std::path::Path;

use super::{
    DatasetConfig, EvaluationConfig, GoogleCloudConfig, ModelConfig, TrainingConfig, TuiConfig,
    WaymoConfig,
};
use crate::error::{ConfigError, ValidationErrors, finish_validation};
use crate::io::{parse_toml_file, parse_toml_str};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    pub dataset: DatasetConfig,
    #[serde(default)]
    pub waymo: WaymoConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub evaluation: EvaluationConfig,
    #[serde(default)]
    pub tui: TuiConfig,
    #[serde(default)]
    pub google_cloud: GoogleCloudConfig,
}

impl ProjectConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let config: Self = parse_toml_str(raw)?;
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config: Self = parse_toml_file(path.as_ref())?;
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
        let repo_root = repo_root.as_ref();
        self.dataset.resolve_paths(repo_root);
        self.waymo.resolve_paths(repo_root);
        self.training.resolve_paths(repo_root);
        self.evaluation.resolve_paths(repo_root);
        self.tui.resolve_paths(repo_root);
        self.google_cloud.resolve_paths(repo_root);
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();

        if let Err(err) = self.dataset.validate() {
            issues.extend(err.with_prefix("dataset").into_issues());
        }
        if let Err(err) = self.waymo.validate() {
            issues.extend(err.with_prefix("waymo").into_issues());
        }
        if let Err(err) = self.model.validate() {
            issues.extend(err.with_prefix("model").into_issues());
        }
        if let Err(err) = self.training.validate() {
            issues.extend(err.with_prefix("training").into_issues());
        }
        if let Err(err) = self.evaluation.validate() {
            issues.extend(err.with_prefix("evaluation").into_issues());
        }
        if let Err(err) = self.tui.validate() {
            issues.extend(err.with_prefix("tui").into_issues());
        }
        if let Err(err) = self.google_cloud.validate() {
            issues.extend(err.with_prefix("google_cloud").into_issues());
        }

        finish_validation(issues)
    }
}
