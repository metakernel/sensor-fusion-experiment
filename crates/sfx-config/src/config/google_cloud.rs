use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoogleCloudConfig {
    pub enabled: bool,
    pub project_id: String,
    pub region: String,
    pub bucket: String,
    pub credentials_path: PathBuf,
}

impl Default for GoogleCloudConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            project_id: String::new(),
            region: "us-central1".to_string(),
            bucket: String::new(),
            credentials_path: PathBuf::from(".xtask/gcloud/credentials.json"),
        }
    }
}

impl GoogleCloudConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: GoogleCloudToml = parse_toml_str(raw)?;
        let config = match parsed {
            GoogleCloudToml::Wrapped(value) => value.google_cloud,
            GoogleCloudToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: GoogleCloudToml = parse_toml_file(path)?;
        let config = match parsed {
            GoogleCloudToml::Wrapped(value) => value.google_cloud,
            GoogleCloudToml::Direct(value) => value,
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
        self.credentials_path = resolve_if_present(repo_root.as_ref(), &self.credentials_path);
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        if self.enabled {
            if self.project_id.trim().is_empty() {
                issues.push(issue("project_id", "must not be empty when enabled"));
            }
            if self.region.trim().is_empty() {
                issues.push(issue("region", "must not be empty when enabled"));
            }
            if self.bucket.trim().is_empty() {
                issues.push(issue("bucket", "must not be empty when enabled"));
            }
            if self.credentials_path.as_os_str().is_empty() {
                issues.push(issue("credentials_path", "must not be empty when enabled"));
            }
        }
        finish_validation(issues)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GoogleCloudToml {
    Wrapped(GoogleCloudFile),
    Direct(GoogleCloudConfig),
}

#[derive(Debug, Deserialize)]
struct GoogleCloudFile {
    google_cloud: GoogleCloudConfig,
}
