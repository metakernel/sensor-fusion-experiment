use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, ValidationErrors, ValidationIssue, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};
use crate::path::resolve_if_present;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetConfig {
    pub name: String,
    pub root: PathBuf,
    pub train_split: String,
    pub val_split: String,
    pub test_split: String,
    pub split_seed: u64,
    pub split_ratios: SplitRatios,
    pub modalities: DatasetModalities,
    pub sampling: DatasetSampling,
    pub preprocessing: DatasetPreprocessing,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            root: PathBuf::new(),
            train_split: "train".to_string(),
            val_split: "val".to_string(),
            test_split: "test".to_string(),
            split_seed: 42,
            split_ratios: SplitRatios::default(),
            modalities: DatasetModalities::default(),
            sampling: DatasetSampling::default(),
            preprocessing: DatasetPreprocessing::default(),
        }
    }
}

impl DatasetConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: DatasetToml = parse_toml_str(raw)?;
        let config = match parsed {
            DatasetToml::Wrapped(value) => value.dataset,
            DatasetToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: DatasetToml = parse_toml_file(path)?;
        let config = match parsed {
            DatasetToml::Wrapped(value) => value.dataset,
            DatasetToml::Direct(value) => value,
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
        self.root = resolve_if_present(repo_root.as_ref(), &self.root);
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();

        if self.name.trim().is_empty() {
            issues.push(issue("name", "must not be empty"));
        }
        if self.root.as_os_str().is_empty() {
            issues.push(issue("root", "must not be empty"));
        }

        if self.train_split.trim().is_empty() {
            issues.push(issue("train_split", "must not be empty"));
        }
        if self.val_split.trim().is_empty() {
            issues.push(issue("val_split", "must not be empty"));
        }
        if self.test_split.trim().is_empty() {
            issues.push(issue("test_split", "must not be empty"));
        }

        if self.train_split == self.val_split
            || self.train_split == self.test_split
            || self.val_split == self.test_split
        {
            issues.push(issue(
                "train_split/val_split/test_split",
                "must use distinct split names",
            ));
        }

        self.split_ratios.validate_into("split_ratios", &mut issues);
        self.modalities.validate_into("modalities", &mut issues);
        self.sampling.validate_into("sampling", &mut issues);
        self.preprocessing
            .validate_into("preprocessing", &mut issues);

        finish_validation(issues)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplitRatios {
    pub train: f32,
    pub val: f32,
    pub test: f32,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.8,
            val: 0.1,
            test: 0.1,
        }
    }
}

impl SplitRatios {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        for (name, value) in [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ] {
            if !value.is_finite() {
                issues.push(issue(format!("{prefix}.{name}"), "must be a finite number"));
            } else if value <= 0.0 || value >= 1.0 {
                issues.push(issue(format!("{prefix}.{name}"), "must be > 0.0 and < 1.0"));
            }
        }

        let sum = self.train + self.val + self.test;
        if (sum - 1.0).abs() > 1e-3 {
            issues.push(issue(prefix, format!("must sum to 1.0 (found {sum:.6})")));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetModalities {
    pub range_image: bool,
    pub rgb: bool,
    pub point_cloud: bool,
}

impl Default for DatasetModalities {
    fn default() -> Self {
        Self {
            range_image: true,
            rgb: true,
            point_cloud: true,
        }
    }
}

impl DatasetModalities {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if !self.range_image && !self.rgb && !self.point_cloud {
            issues.push(issue(prefix, "at least one modality must be enabled"));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetSampling {
    pub max_scenes: u32,
    pub max_frames_per_scene: u32,
    pub shuffle: bool,
}

impl Default for DatasetSampling {
    fn default() -> Self {
        Self {
            max_scenes: 64,
            max_frames_per_scene: 200,
            shuffle: true,
        }
    }
}

impl DatasetSampling {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.max_scenes == 0 {
            issues.push(issue(format!("{prefix}.max_scenes"), "must be > 0"));
        }
        if self.max_frames_per_scene == 0 {
            issues.push(issue(
                format!("{prefix}.max_frames_per_scene"),
                "must be > 0",
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetPreprocessing {
    pub resize_width: u32,
    pub resize_height: u32,
    pub normalize_rgb: bool,
    pub range_clip_min_m: f32,
    pub range_clip_max_m: f32,
}

impl Default for DatasetPreprocessing {
    fn default() -> Self {
        Self {
            resize_width: 640,
            resize_height: 384,
            normalize_rgb: true,
            range_clip_min_m: 0.0,
            range_clip_max_m: 75.0,
        }
    }
}

impl DatasetPreprocessing {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.resize_width == 0 {
            issues.push(issue(format!("{prefix}.resize_width"), "must be > 0"));
        }
        if self.resize_height == 0 {
            issues.push(issue(format!("{prefix}.resize_height"), "must be > 0"));
        }

        if !self.range_clip_min_m.is_finite() {
            issues.push(issue(
                format!("{prefix}.range_clip_min_m"),
                "must be a finite number",
            ));
        }
        if !self.range_clip_max_m.is_finite() {
            issues.push(issue(
                format!("{prefix}.range_clip_max_m"),
                "must be a finite number",
            ));
        }
        if self.range_clip_min_m < 0.0 {
            issues.push(issue(
                format!("{prefix}.range_clip_min_m"),
                "must be >= 0.0",
            ));
        }
        if self.range_clip_max_m <= self.range_clip_min_m {
            issues.push(issue(
                format!("{prefix}.range_clip_max_m"),
                "must be greater than range_clip_min_m",
            ));
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DatasetToml {
    Wrapped(DatasetFile),
    Direct(DatasetConfig),
}

#[derive(Debug, Deserialize)]
struct DatasetFile {
    dataset: DatasetConfig,
}
