use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WaymoConfig {
    pub enabled: bool,
    pub records_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub scene_index_path: PathBuf,
    pub max_frames_per_scene: u32,
}

impl Default for WaymoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            records_dir: PathBuf::from("data/raw/waymo"),
            cache_dir: PathBuf::from("data/processed/waymo"),
            scene_index_path: PathBuf::from(".xtask/manifests/waymo-scenes.json"),
            max_frames_per_scene: 200,
        }
    }
}

impl WaymoConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: WaymoToml = parse_toml_str(raw)?;
        let config = match parsed {
            WaymoToml::Wrapped(value) => value.waymo,
            WaymoToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: WaymoToml = parse_toml_file(path)?;
        let config = match parsed {
            WaymoToml::Wrapped(value) => value.waymo,
            WaymoToml::Direct(value) => value,
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
        let repo_root = repo_root.as_ref();
        self.records_dir = resolve_if_present(repo_root, &self.records_dir);
        self.cache_dir = resolve_if_present(repo_root, &self.cache_dir);
        self.scene_index_path = resolve_if_present(repo_root, &self.scene_index_path);
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        if self.enabled {
            if self.records_dir.as_os_str().is_empty() {
                issues.push(issue("records_dir", "must not be empty when enabled"));
            }
            if self.cache_dir.as_os_str().is_empty() {
                issues.push(issue("cache_dir", "must not be empty when enabled"));
            }
            if self.scene_index_path.as_os_str().is_empty() {
                issues.push(issue("scene_index_path", "must not be empty when enabled"));
            }
            if self.max_frames_per_scene == 0 {
                issues.push(issue("max_frames_per_scene", "must be > 0 when enabled"));
            }
        }
        finish_validation(issues)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WaymoToml {
    Wrapped(WaymoFile),
    Direct(WaymoConfig),
}

#[derive(Debug, Deserialize)]
struct WaymoFile {
    waymo: WaymoConfig,
}
