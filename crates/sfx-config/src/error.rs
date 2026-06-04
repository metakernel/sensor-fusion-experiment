use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse TOML from `{path}`: {source}")]
    ParseTomlFile {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("failed to parse TOML: {source}")]
    ParseToml {
        #[source]
        source: toml::de::Error,
    },
    #[error(transparent)]
    Validation(#[from] ValidationErrors),
}

impl ConfigError {
    pub(crate) fn read_file(path: &Path, source: io::Error) -> Self {
        Self::ReadFile {
            path: path.to_path_buf(),
            source,
        }
    }

    pub(crate) fn parse_toml_file(path: &Path, source: toml::de::Error) -> Self {
        Self::ParseTomlFile {
            path: path.to_path_buf(),
            source,
        }
    }

    pub(crate) fn parse_toml(source: toml::de::Error) -> Self {
        Self::ParseToml { source }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub field: String,
    pub message: String,
}

impl ValidationIssue {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.field = if self.field.is_empty() {
            prefix.to_string()
        } else {
            format!("{prefix}.{}", self.field)
        };
        self
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationErrors {
    issues: Vec<ValidationIssue>,
}

impl ValidationErrors {
    pub fn new(issues: Vec<ValidationIssue>) -> Self {
        Self { issues }
    }

    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    pub fn into_issues(self) -> Vec<ValidationIssue> {
        self.issues
    }

    pub fn with_prefix(self, prefix: &str) -> Self {
        Self {
            issues: self
                .issues
                .into_iter()
                .map(|issue| issue.with_prefix(prefix))
                .collect(),
        }
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "configuration validation failed")?;
        for issue in &self.issues {
            write!(f, "\n- {issue}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

pub(crate) fn issue(field: impl Into<String>, message: impl Into<String>) -> ValidationIssue {
    ValidationIssue::new(field, message)
}

pub(crate) fn finish_validation(
    issues: Vec<ValidationIssue>,
) -> std::result::Result<(), ValidationErrors> {
    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidationErrors::new(issues))
    }
}
