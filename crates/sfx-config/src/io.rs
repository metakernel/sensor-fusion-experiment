use serde::de::DeserializeOwned;
use std::fs;
use std::path::Path;

use crate::error::ConfigError;

pub(crate) fn read_toml_file(path: &Path) -> Result<String, ConfigError> {
    fs::read_to_string(path).map_err(|source| ConfigError::read_file(path, source))
}

pub(crate) fn parse_toml_str<T>(raw: &str) -> Result<T, ConfigError>
where
    T: DeserializeOwned,
{
    toml::from_str(raw).map_err(ConfigError::parse_toml)
}

pub(crate) fn parse_toml_str_with_path<T>(raw: &str, path: &Path) -> Result<T, ConfigError>
where
    T: DeserializeOwned,
{
    toml::from_str(raw).map_err(|source| ConfigError::parse_toml_file(path, source))
}

pub(crate) fn parse_toml_file<T>(path: &Path) -> Result<T, ConfigError>
where
    T: DeserializeOwned,
{
    let raw = read_toml_file(path)?;
    parse_toml_str_with_path(&raw, path)
}
