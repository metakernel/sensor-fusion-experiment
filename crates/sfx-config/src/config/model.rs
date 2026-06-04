use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{ConfigError, ValidationErrors, ValidationIssue, finish_validation, issue};
use crate::io::{parse_toml_file, parse_toml_str};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub input: ModelInputConfig,
    pub backbone: Option<ModelBackboneConfig>,
    pub range_branch: Option<ModelBackboneConfig>,
    pub rgb_branch: Option<ModelBackboneConfig>,
    pub fusion: Option<ModelFusionConfig>,
    pub head: ModelHeadConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            model_type: String::new(),
            input: ModelInputConfig::default(),
            backbone: None,
            range_branch: None,
            rgb_branch: None,
            fusion: None,
            head: ModelHeadConfig::default(),
        }
    }
}

impl ModelConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let parsed: ModelToml = parse_toml_str(raw)?;
        let config = match parsed {
            ModelToml::Wrapped(value) => value.model,
            ModelToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let parsed: ModelToml = parse_toml_file(path)?;
        let config = match parsed {
            ModelToml::Wrapped(value) => value.model,
            ModelToml::Direct(value) => value,
        };
        config.validate().map_err(ConfigError::from)?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();

        if self.name.trim().is_empty() {
            issues.push(issue("name", "must not be empty"));
        }
        if self.model_type.trim().is_empty() {
            issues.push(issue("type", "must not be empty"));
        }

        self.input.validate_into("input", &mut issues);
        self.head.validate_into("head", &mut issues);

        let use_range = self.input.use_range_image;
        let use_rgb = self.input.use_rgb;

        if use_range && self.backbone.is_none() && self.range_branch.is_none() {
            issues.push(issue(
                "range_branch",
                "must be set when range input is enabled and shared backbone is absent",
            ));
        }
        if use_rgb && self.backbone.is_none() && self.rgb_branch.is_none() {
            issues.push(issue(
                "rgb_branch",
                "must be set when rgb input is enabled and shared backbone is absent",
            ));
        }
        if !use_range && self.range_branch.is_some() {
            issues.push(issue(
                "range_branch",
                "must be omitted when input.use_range_image is false",
            ));
        }
        if !use_rgb && self.rgb_branch.is_some() {
            issues.push(issue(
                "rgb_branch",
                "must be omitted when input.use_rgb is false",
            ));
        }

        if use_range && use_rgb && self.fusion.is_none() {
            issues.push(issue(
                "fusion",
                "must be set when both range and rgb inputs are enabled",
            ));
        }
        if self.fusion.is_some() && !(use_range && use_rgb) {
            issues.push(issue(
                "fusion",
                "must be omitted unless both range and rgb inputs are enabled",
            ));
        }

        if let Some(backbone) = &self.backbone {
            backbone.validate_into("backbone", &mut issues);
        }
        if let Some(range_branch) = &self.range_branch {
            range_branch.validate_into("range_branch", &mut issues);
        }
        if let Some(rgb_branch) = &self.rgb_branch {
            rgb_branch.validate_into("rgb_branch", &mut issues);
        }
        if let Some(fusion) = &self.fusion {
            fusion.validate_into("fusion", &mut issues);
        }

        finish_validation(issues)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelInputConfig {
    pub use_range_image: bool,
    pub use_rgb: bool,
    pub range_channels: Option<u32>,
    pub image_channels: Option<u32>,
}

impl Default for ModelInputConfig {
    fn default() -> Self {
        Self {
            use_range_image: true,
            use_rgb: true,
            range_channels: None,
            image_channels: None,
        }
    }
}

impl ModelInputConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if !self.use_range_image && !self.use_rgb {
            issues.push(issue(prefix, "at least one modality must be enabled"));
        }

        if let Some(value) = self.range_channels {
            if value == 0 {
                issues.push(issue(format!("{prefix}.range_channels"), "must be > 0"));
            }
            if !self.use_range_image {
                issues.push(issue(
                    format!("{prefix}.range_channels"),
                    "must be omitted when use_range_image is false",
                ));
            }
        }

        if let Some(value) = self.image_channels {
            if value == 0 {
                issues.push(issue(format!("{prefix}.image_channels"), "must be > 0"));
            }
            if !self.use_rgb {
                issues.push(issue(
                    format!("{prefix}.image_channels"),
                    "must be omitted when use_rgb is false",
                ));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelBackboneConfig {
    pub channels: Vec<u32>,
    pub blocks: Vec<u32>,
    pub dropout: Option<f32>,
}

impl Default for ModelBackboneConfig {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            blocks: Vec::new(),
            dropout: Some(0.1),
        }
    }
}

impl ModelBackboneConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.channels.is_empty() {
            issues.push(issue(format!("{prefix}.channels"), "must not be empty"));
        }
        if self.blocks.is_empty() {
            issues.push(issue(format!("{prefix}.blocks"), "must not be empty"));
        }
        if self.channels.len() != self.blocks.len() {
            issues.push(issue(
                format!("{prefix}.channels/blocks"),
                "must have matching lengths",
            ));
        }
        for (idx, value) in self.channels.iter().enumerate() {
            if *value == 0 {
                issues.push(issue(format!("{prefix}.channels[{idx}]"), "must be > 0"));
            }
        }
        for (idx, value) in self.blocks.iter().enumerate() {
            if *value == 0 {
                issues.push(issue(format!("{prefix}.blocks[{idx}]"), "must be > 0"));
            }
        }

        if let Some(value) = self.dropout {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                issues.push(issue(
                    format!("{prefix}.dropout"),
                    "must be between 0.0 and 1.0",
                ));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelFusionConfig {
    pub method: String,
    pub projection_dim: u32,
    pub rgb_loss_weight: f64,
    pub range_loss_weight: f64,
}

impl Default for ModelFusionConfig {
    fn default() -> Self {
        Self {
            method: "concat".to_string(),
            projection_dim: 64,
            rgb_loss_weight: 1.0,
            range_loss_weight: 1.0,
        }
    }
}

impl ModelFusionConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.method.trim().is_empty() {
            issues.push(issue(format!("{prefix}.method"), "must not be empty"));
        }
        if self.projection_dim == 0 {
            issues.push(issue(format!("{prefix}.projection_dim"), "must be > 0"));
        }
        if !self.rgb_loss_weight.is_finite() || self.rgb_loss_weight < 0.0 {
            issues.push(issue(
                format!("{prefix}.rgb_loss_weight"),
                "must be a finite number >= 0.0",
            ));
        }
        if !self.range_loss_weight.is_finite() || self.range_loss_weight < 0.0 {
            issues.push(issue(
                format!("{prefix}.range_loss_weight"),
                "must be a finite number >= 0.0",
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelHeadConfig {
    pub num_classes: u32,
    pub hidden_dim: u32,
    pub loss: String,
}

impl Default for ModelHeadConfig {
    fn default() -> Self {
        Self {
            num_classes: 1,
            hidden_dim: 64,
            loss: "cross_entropy".to_string(),
        }
    }
}

impl ModelHeadConfig {
    fn validate_into(&self, prefix: &str, issues: &mut Vec<ValidationIssue>) {
        if self.num_classes == 0 {
            issues.push(issue(format!("{prefix}.num_classes"), "must be > 0"));
        }
        if self.hidden_dim == 0 {
            issues.push(issue(format!("{prefix}.hidden_dim"), "must be > 0"));
        }
        if self.loss.trim().is_empty() {
            issues.push(issue(format!("{prefix}.loss"), "must not be empty"));
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelToml {
    Wrapped(ModelFile),
    Direct(ModelConfig),
}

#[derive(Debug, Deserialize)]
struct ModelFile {
    model: ModelConfig,
}
