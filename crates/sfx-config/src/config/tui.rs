use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TuiConfig {
    pub enabled: bool,
    pub refresh_rate_ms: u64,
    pub theme: String,
    pub history_path: PathBuf,
    pub max_events: usize,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            refresh_rate_ms: 250,
            theme: "default".to_string(),
            history_path: PathBuf::from("artifacts/reports/tui/history.log"),
            max_events: 2_000,
        }
    }
}

impl TuiConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: TuiToml = parse_toml_str(raw)?;
        let config = match parsed {
            TuiToml::Wrapped(value) => value.tui,
            TuiToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: TuiToml = parse_toml_file(path)?;
        let config = match parsed {
            TuiToml::Wrapped(value) => value.tui,
            TuiToml::Direct(value) => value,
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
        self.history_path = resolve_if_present(repo_root.as_ref(), &self.history_path);
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        if self.enabled {
            if self.refresh_rate_ms == 0 {
                issues.push(issue("refresh_rate_ms", "must be > 0 when enabled"));
            }
            if self.theme.trim().is_empty() {
                issues.push(issue("theme", "must not be empty when enabled"));
            }
            if self.history_path.as_os_str().is_empty() {
                issues.push(issue("history_path", "must not be empty when enabled"));
            }
            if self.max_events == 0 {
                issues.push(issue("max_events", "must be > 0 when enabled"));
            }
        }
        finish_validation(issues)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TuiToml {
    Wrapped(TuiFile),
    Direct(TuiConfig),
}

#[derive(Debug, Deserialize)]
struct TuiFile {
    tui: TuiConfig,
}
