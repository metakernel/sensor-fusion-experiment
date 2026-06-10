use crate::config::{ConfigError, resolve_from_root};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub type CompressionConfigError = ConfigError;
pub type Result<T> = std::result::Result<T, CompressionConfigError>;

const MAX_CRF: u8 = 63;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressionBenchConfig {
    #[serde(default = "default_codecs")]
    pub codecs: Vec<String>,
    #[serde(default = "default_crf_sweep")]
    pub crf_sweep: BTreeMap<String, Vec<u8>>,
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default = "default_split")]
    pub split: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default)]
    pub mode: CompressionMode,
    #[serde(default)]
    pub quantization: QuantizationConfig,
    #[serde(default)]
    pub sample_cap: Option<usize>,
}

impl Default for CompressionBenchConfig {
    fn default() -> Self {
        Self {
            codecs: default_codecs(),
            crf_sweep: default_crf_sweep(),
            preset: default_preset(),
            split: default_split(),
            output_dir: default_output_dir(),
            mode: CompressionMode::default(),
            quantization: QuantizationConfig::default(),
            sample_cap: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CompressionMode {
    Independent,
    IntraSequence,
}

impl Default for CompressionMode {
    fn default() -> Self {
        Self::Independent
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantizationConfig {
    #[serde(default = "default_quantization_enabled")]
    pub enabled: bool,
    #[serde(default = "default_quantization_strategy")]
    pub strategy: QuantizationStrategy,
    #[serde(default = "default_quantization_bits")]
    pub bits: u8,
    #[serde(default = "default_quantization_per_channel")]
    pub per_channel: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: default_quantization_enabled(),
            strategy: default_quantization_strategy(),
            bits: default_quantization_bits(),
            per_channel: default_quantization_per_channel(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum QuantizationStrategy {
    Uniform,
    Logarithmic,
}

impl Default for QuantizationStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CompressionBenchFile {
    #[serde(default)]
    compression: CompressionBenchConfig,
}

pub fn load_compression_bench_config(
    root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<CompressionBenchConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let raw = std::fs::read_to_string(&path).map_err(|source| CompressionConfigError::Read {
        path: path.clone(),
        source,
    })?;

    let mut config = parse_compression_bench_config(&path, &raw)?;
    config.output_dir = resolve_from_root(root, &config.output_dir);
    Ok(config)
}

fn parse_compression_bench_config(path: &Path, raw: &str) -> Result<CompressionBenchConfig> {
    let file: CompressionBenchFile =
        toml::from_str(raw).map_err(|source| CompressionConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    validate_config(path, &file.compression)?;
    Ok(file.compression)
}

fn validate_config(path: &Path, config: &CompressionBenchConfig) -> Result<()> {
    ensure(
        path,
        !config.codecs.is_empty(),
        "at least one codec is required",
    )?;
    ensure(
        path,
        !config.crf_sweep.is_empty(),
        "crf_sweep cannot be empty",
    )?;
    ensure(
        path,
        !config.preset.trim().is_empty(),
        "preset cannot be empty",
    )?;
    ensure(
        path,
        matches!(
            config.split.trim().to_ascii_lowercase().as_str(),
            "train" | "val" | "test" | "all"
        ),
        format!(
            "split must be train, val, test, or all, got {}",
            config.split
        ),
    )?;
    ensure(
        path,
        !config.output_dir.as_os_str().is_empty(),
        "output_dir cannot be empty",
    )?;
    ensure(
        path,
        config.quantization.bits > 0,
        "quantization.bits must be greater than zero",
    )?;
    ensure(
        path,
        config.quantization.bits <= 16,
        "quantization.bits must be <= 16",
    )?;

    if let Some(sample_cap) = config.sample_cap {
        ensure(path, sample_cap > 0, "sample_cap must be greater than zero")?;
    }

    let mut codec_names = BTreeSet::new();
    for codec in &config.codecs {
        let normalized = normalize_codec(codec);
        ensure(path, !normalized.is_empty(), "codec names cannot be empty")?;
        ensure(
            path,
            codec_names.insert(normalized.clone()),
            format!("duplicate codec `{}`", codec.trim()),
        )?;
    }

    let mut sweep_codecs = BTreeMap::<String, &Vec<u8>>::new();
    for (codec, sweep) in &config.crf_sweep {
        let normalized = normalize_codec(codec);
        ensure(
            path,
            !normalized.is_empty(),
            "crf_sweep codec names cannot be empty",
        )?;
        ensure(
            path,
            sweep_codecs.insert(normalized.clone(), sweep).is_none(),
            format!("duplicate crf_sweep entry for codec `{}`", codec.trim()),
        )?;
        ensure(
            path,
            !sweep.is_empty(),
            format!(
                "codec `{}` must include at least one CRF value",
                codec.trim()
            ),
        )?;

        let mut seen = BTreeSet::new();
        for value in sweep {
            ensure(
                path,
                *value <= MAX_CRF,
                format!(
                    "codec `{}` CRF {value} is outside supported range 0..={MAX_CRF}",
                    codec.trim()
                ),
            )?;
            ensure(
                path,
                seen.insert(*value),
                format!("codec `{}` has duplicate CRF value {value}", codec.trim()),
            )?;
        }
    }

    for codec in &config.codecs {
        let normalized = normalize_codec(codec);
        ensure(
            path,
            sweep_codecs.contains_key(&normalized),
            format!("missing crf_sweep entry for codec `{}`", codec.trim()),
        )?;
    }

    for codec in sweep_codecs.keys() {
        ensure(
            path,
            codec_names.contains(codec),
            format!("crf_sweep includes unknown codec `{codec}`"),
        )?;
    }

    Ok(())
}

fn normalize_codec(codec: &str) -> String {
    codec.trim().to_ascii_lowercase()
}

fn ensure(path: &Path, condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(CompressionConfigError::Invalid {
            path: path.to_path_buf(),
            message: message.into(),
        })
    }
}

fn default_codecs() -> Vec<String> {
    vec![
        "libx264".to_string(),
        "libx265".to_string(),
        "libaom-av1".to_string(),
    ]
}

fn default_crf_sweep() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("libx264".to_string(), vec![18, 23, 28, 33]),
        ("libx265".to_string(), vec![20, 26, 32, 38]),
        ("libaom-av1".to_string(), vec![24, 30, 36, 42]),
    ])
}

fn default_preset() -> String {
    "medium".to_string()
}

fn default_split() -> String {
    "val".to_string()
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("artifacts/bench/compression")
}

fn default_quantization_enabled() -> bool {
    true
}

fn default_quantization_strategy() -> QuantizationStrategy {
    QuantizationStrategy::Uniform
}

fn default_quantization_bits() -> u8 {
    8
}

fn default_quantization_per_channel() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses_from_empty_section() {
        let cfg = parse_compression_bench_config(
            Path::new("configs/bench.compression.toml"),
            "[compression]\n",
        )
        .unwrap();

        assert_eq!(cfg, CompressionBenchConfig::default());
    }

    #[test]
    fn rejects_unknown_codec_in_sweep() {
        let raw = r#"
[compression]
codecs = ["libx264"]

[compression.crf_sweep]
libx264 = [18, 24]
libx265 = [20]
"#;

        let err = parse_compression_bench_config(Path::new("configs/bench.compression.toml"), raw)
            .unwrap_err();

        assert!(err.to_string().contains("unknown codec"));
    }

    #[test]
    fn rejects_zero_sample_cap() {
        let raw = r#"
[compression]
sample_cap = 0
"#;

        let err = parse_compression_bench_config(Path::new("configs/bench.compression.toml"), raw)
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("sample_cap must be greater than zero")
        );
    }

    #[test]
    fn accepts_intra_sequence_mode() {
        let raw = r#"
[compression]
mode = "intra-sequence"
"#;

        let cfg = parse_compression_bench_config(Path::new("configs/bench.compression.toml"), raw)
            .unwrap();

        assert_eq!(cfg.mode, CompressionMode::IntraSequence);
    }
}
