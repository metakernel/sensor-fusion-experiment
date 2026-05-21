use serde::Deserialize;
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, ConfigError>;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not find workspace root from {start}")]
    RootNotFound { start: PathBuf },
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid {path}: {message}")]
    Invalid { path: PathBuf, message: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub raw_dir: PathBuf,
    pub processed_dir: PathBuf,
    pub rgb_size: [usize; 2],
    pub range_size: [usize; 2],
    pub range_channels: Vec<String>,
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WaymoConfig {
    pub bucket: Option<String>,
    pub prefix: Option<String>,
    pub training_prefix: Option<String>,
    pub validation_prefix: Option<String>,
    pub testing_prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub kind: ModelKind,
    pub latent_dim: usize,
    #[serde(default)]
    pub z_modality: Option<usize>,
}

impl ModelConfig {
    pub fn effective_z_modality(&self) -> usize {
        self.z_modality.unwrap_or(self.latent_dim)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    RgbOnly,
    RangeOnly,
    Fusion,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub run_name: String,
    pub batch_size: usize,
    pub max_batches_per_epoch: Option<usize>,
    pub learning_rate: f32,
    pub epochs: usize,
    pub seed: u64,
    pub dataset_config_path: PathBuf,
    pub model_config_path: PathBuf,
    pub dataset: DatasetConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationConfig {
    pub split: String,
    pub batch_size: usize,
    pub preview_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TuiConfig {
    #[serde(default = "default_tui_split")]
    pub split: String,
    #[serde(default)]
    pub ascii_fallback: bool,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            split: default_tui_split(),
            ascii_fallback: true,
        }
    }
}

fn default_tui_split() -> String {
    "test".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoogleCloudConfig {
    pub credentials_path: PathBuf,
    #[serde(default)]
    pub token_cache_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct DatasetFile {
    dataset: DatasetConfig,
    #[serde(default)]
    waymo: WaymoConfig,
}

#[derive(Debug, Deserialize)]
struct ModelFile {
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
struct TrainingFile {
    train: TrainingSection,
    dataset: ConfigRef,
    model: ConfigRef,
}

#[derive(Debug, Deserialize)]
struct TrainingSection {
    run_name: String,
    batch_size: usize,
    #[serde(default)]
    max_batches_per_epoch: Option<usize>,
    learning_rate: f32,
    epochs: usize,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct ConfigRef {
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct EvalFile {
    eval: EvaluationConfig,
}

#[derive(Debug, Deserialize)]
struct TuiFile {
    tui: TuiConfig,
}

pub fn discover_workspace_root(start: impl AsRef<Path>) -> Result<PathBuf> {
    let start = start.as_ref();
    let mut dir = if start.is_file() {
        start.parent().unwrap_or(start).to_path_buf()
    } else {
        start.to_path_buf()
    };

    loop {
        let cargo = dir.join("Cargo.toml");
        if cargo.exists() {
            let text = std::fs::read_to_string(&cargo).map_err(|source| ConfigError::Read {
                path: cargo.clone(),
                source,
            })?;
            if text.contains("[workspace]") {
                return Ok(dir);
            }
        }
        if !dir.pop() {
            return Err(ConfigError::RootNotFound {
                start: start.to_path_buf(),
            });
        }
    }
}

pub fn resolve_from_root(root: impl AsRef<Path>, path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.as_ref().join(path)
    }
}

pub fn load_dataset_config(
    root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<DatasetConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let mut file: DatasetFile = load_toml(&path)?;
    let _ = file.waymo;
    resolve_dataset_paths(root, &mut file.dataset);
    validate_dataset(&path, &file.dataset)?;
    Ok(file.dataset)
}

pub fn load_waymo_config(root: impl AsRef<Path>, path: impl AsRef<Path>) -> Result<WaymoConfig> {
    let path = resolve_from_root(root, path);
    let file: DatasetFile = load_toml(&path)?;
    Ok(file.waymo)
}

pub fn load_model_config(root: impl AsRef<Path>, path: impl AsRef<Path>) -> Result<ModelConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let file: ModelFile = load_toml(&path)?;
    validate_model(&path, &file.model)?;
    Ok(file.model)
}

pub fn load_training_config(
    root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<TrainingConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let file: TrainingFile = load_toml(&path)?;
    validate_training_section(&path, &file.train)?;

    let dataset_config_path = resolve_from_root(root, &file.dataset.config);
    let model_config_path = resolve_from_root(root, &file.model.config);
    let dataset = load_dataset_config(root, &dataset_config_path)?;
    let model = load_model_config(root, &model_config_path)?;

    Ok(TrainingConfig {
        run_name: file.train.run_name,
        batch_size: file.train.batch_size,
        max_batches_per_epoch: file.train.max_batches_per_epoch,
        learning_rate: file.train.learning_rate,
        epochs: file.train.epochs,
        seed: file.train.seed,
        dataset_config_path,
        model_config_path,
        dataset,
        model,
    })
}

pub fn load_evaluation_config(
    root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<EvaluationConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let file: EvalFile = load_toml(&path)?;
    validate_evaluation(&path, &file.eval)?;
    Ok(file.eval)
}

pub fn load_tui_config(root: impl AsRef<Path>, path: impl AsRef<Path>) -> Result<TuiConfig> {
    let root = root.as_ref();
    let path = resolve_from_root(root, path);
    let file: TuiFile = load_toml(&path)?;
    validate_split_name(&path, &file.tui.split)?;
    Ok(file.tui)
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

fn resolve_dataset_paths(root: &Path, dataset: &mut DatasetConfig) {
    dataset.raw_dir = resolve_from_root(root, &dataset.raw_dir);
    dataset.processed_dir = resolve_from_root(root, &dataset.processed_dir);
}

fn validate_dataset(path: &Path, config: &DatasetConfig) -> Result<()> {
    ensure(
        path,
        !config.name.trim().is_empty(),
        "dataset name is required",
    )?;
    ensure(
        path,
        config.rgb_size[0] > 0,
        "rgb height must be greater than zero",
    )?;
    ensure(
        path,
        config.rgb_size[1] > 0,
        "rgb width must be greater than zero",
    )?;
    ensure(
        path,
        config.range_size[0] > 0,
        "range height must be greater than zero",
    )?;
    ensure(
        path,
        config.range_size[1] > 0,
        "range width must be greater than zero",
    )?;
    ensure(
        path,
        !config.range_channels.is_empty(),
        "at least one range channel is required",
    )?;
    for channel in &config.range_channels {
        ensure(
            path,
            !channel.trim().is_empty(),
            "range channel names cannot be empty",
        )?;
    }
    ensure(
        path,
        config.train_ratio >= 0.0,
        "train_ratio cannot be negative",
    )?;
    ensure(
        path,
        config.val_ratio >= 0.0,
        "val_ratio cannot be negative",
    )?;
    ensure(
        path,
        config.test_ratio >= 0.0,
        "test_ratio cannot be negative",
    )?;
    let sum = config.train_ratio + config.val_ratio + config.test_ratio;
    ensure(
        path,
        (sum - 1.0).abs() <= 0.001,
        format!("split ratios must sum to 1.0, got {sum:.4}"),
    )
}

fn validate_model(path: &Path, config: &ModelConfig) -> Result<()> {
    ensure(
        path,
        config.latent_dim > 0,
        "latent_dim must be greater than zero",
    )?;
    if let Some(z_modality) = config.z_modality {
        ensure(path, z_modality > 0, "z_modality must be greater than zero")?;
    }
    Ok(())
}

fn validate_training_section(path: &Path, config: &TrainingSection) -> Result<()> {
    ensure(
        path,
        !config.run_name.trim().is_empty(),
        "run_name is required",
    )?;
    ensure(
        path,
        config.batch_size > 0,
        "batch_size must be greater than zero",
    )?;
    if let Some(max_batches_per_epoch) = config.max_batches_per_epoch {
        ensure(
            path,
            max_batches_per_epoch > 0,
            "max_batches_per_epoch must be greater than zero",
        )?;
    }
    ensure(
        path,
        config.learning_rate > 0.0,
        "learning_rate must be greater than zero",
    )?;
    ensure(path, config.epochs > 0, "epochs must be greater than zero")?;
    Ok(())
}

fn validate_evaluation(path: &Path, config: &EvaluationConfig) -> Result<()> {
    validate_split_name(path, &config.split)?;
    ensure(
        path,
        config.batch_size > 0,
        "batch_size must be greater than zero",
    )?;
    Ok(())
}

fn validate_split_name(path: &Path, split: &str) -> Result<()> {
    ensure(
        path,
        matches!(split, "train" | "val" | "test"),
        format!("split must be train, val, or test, got {split}"),
    )
}

fn ensure(path: &Path, condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ConfigError::Invalid {
            path: path.to_path_buf(),
            message: message.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_training_config_with_resolved_paths() {
        let root = temp_root("loads_training_config_with_resolved_paths");
        write_configs(&root, "0.8", "0.1", "0.1");

        let cfg = load_training_config(&root, "configs/train.debug.toml").unwrap();

        assert_eq!(cfg.run_name, "debug");
        assert_eq!(cfg.model.kind, ModelKind::Fusion);
        assert_eq!(cfg.dataset.raw_dir, root.join("data/raw/waymo"));
        assert_eq!(
            cfg.dataset.processed_dir,
            root.join("data/processed/waymo-range-rgb-v1")
        );
    }

    #[test]
    fn rejects_bad_dataset_ratios() {
        let root = temp_root("rejects_bad_dataset_ratios");
        write_configs(&root, "0.8", "0.2", "0.2");

        let err = load_training_config(&root, "configs/train.debug.toml").unwrap_err();

        assert!(err.to_string().contains("split ratios"));
    }

    #[test]
    fn reports_missing_fields() {
        let root = temp_root("reports_missing_fields");
        std::fs::create_dir_all(root.join("configs")).unwrap();
        std::fs::write(
            root.join("configs/model.toml"),
            "[model]\nkind = \"fusion\"\n",
        )
        .unwrap();

        let err = load_model_config(&root, "configs/model.toml").unwrap_err();

        assert!(err.to_string().contains("latent_dim"));
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("sfx-config-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        root
    }

    fn write_configs(root: &Path, train: &str, val: &str, test: &str) {
        let dir = root.join("configs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("dataset.toml"),
            format!(
                "[dataset]\nname = \"waymo-range-rgb-v1\"\nraw_dir = \"data/raw/waymo\"\nprocessed_dir = \"data/processed/waymo-range-rgb-v1\"\nrgb_size = [128, 256]\nrange_size = [64, 256]\nrange_channels = [\"range\", \"intensity\"]\ntrain_ratio = {train}\nval_ratio = {val}\ntest_ratio = {test}\n"
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("model.toml"),
            "[model]\nkind = \"fusion\"\nlatent_dim = 128\nz_modality = 256\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("train.debug.toml"),
            "[train]\nrun_name = \"debug\"\nbatch_size = 4\nlearning_rate = 0.001\nepochs = 3\nseed = 42\n\n[dataset]\nconfig = \"configs/dataset.toml\"\n\n[model]\nconfig = \"configs/model.toml\"\n",
        )
        .unwrap();
    }
}
