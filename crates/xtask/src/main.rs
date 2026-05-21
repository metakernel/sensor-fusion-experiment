mod dataset;
mod gcloud;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use sfx_core::manifest::{
    DownloadedFileManifest, LatestRun, ProcessedSampleManifest, RawFileManifest, RunIndex,
    SourceSplit, SplitsManifest, read_manifest, write_manifest,
};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "xtask", about = "Project task runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Init,
    Doctor,
    Gcloud {
        #[command(subcommand)]
        command: GcloudCommand,
    },
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum GcloudCommand {
    Auth,
    Check,
    Whoami,
    Logout,
}

#[derive(Subcommand, Debug)]
pub(crate) enum DatasetCommand {
    List(DatasetListArgs),
    Fetch(DatasetFetchArgs),
    Prepare(DatasetPrepareArgs),
    Inspect(DatasetInspectArgs),
    Preview(DatasetPreviewArgs),
}

#[derive(Args, Debug)]
pub(crate) struct DatasetListArgs {
    #[arg(long, default_value = "waymo")]
    pub(crate) dataset: String,
    #[arg(long, value_enum, default_value_t = DatasetSourceSplit::Training)]
    pub(crate) split: DatasetSourceSplit,
    #[arg(long, default_value_t = 20)]
    pub(crate) limit: usize,
    #[arg(long, default_value = "configs/dataset.waymo.small.toml")]
    pub(crate) config: PathBuf,
    #[arg(long)]
    pub(crate) bucket: Option<String>,
    #[arg(long)]
    pub(crate) prefix: Option<String>,
    #[arg(long)]
    pub(crate) split_prefix: Option<String>,
    #[arg(long)]
    pub(crate) component: Option<String>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Args, Debug)]
pub(crate) struct DatasetFetchArgs {
    #[arg(long, default_value = "waymo")]
    pub(crate) dataset: String,
    #[arg(long, value_delimiter = ',', value_parser = parse_dataset_source_split, default_value = "train,val")]
    pub(crate) splits: Vec<DatasetSourceSplit>,
    #[arg(long, default_value_t = 1)]
    pub(crate) train_files: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) val_files: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) test_files: usize,
    #[arg(long, default_value = "configs/dataset.waymo.small.toml")]
    pub(crate) config: PathBuf,
    #[arg(long)]
    pub(crate) manifest: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    pub(crate) extra_components: Vec<String>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Args, Debug)]
pub(crate) struct DatasetPrepareArgs {
    #[arg(long, default_value = "waymo")]
    pub(crate) dataset: String,
    #[arg(long, value_delimiter = ',', value_parser = parse_dataset_source_split, default_value = "train")]
    pub(crate) splits: Vec<DatasetSourceSplit>,
    #[arg(long, default_value = "configs/dataset.waymo.small.toml")]
    pub(crate) config: PathBuf,
    #[arg(long)]
    pub(crate) input: Option<PathBuf>,
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    #[arg(long)]
    pub(crate) max_frames: Option<usize>,
    #[arg(long)]
    pub(crate) inspect: bool,
}

#[derive(Args, Debug)]
pub(crate) struct DatasetInspectArgs {
    #[arg(long, default_value = "waymo")]
    pub(crate) dataset: String,
    #[arg(long, default_value = "configs/dataset.waymo.small.toml")]
    pub(crate) config: PathBuf,
    /// Number of sample tensor files to load when computing statistics.
    #[arg(long, default_value_t = 16)]
    pub(crate) sample_count: usize,
}

#[derive(Args, Debug)]
pub(crate) struct DatasetPreviewArgs {
    #[arg(long, default_value = "waymo")]
    pub(crate) dataset: String,
    #[arg(long, default_value = "configs/dataset.waymo.small.toml")]
    pub(crate) config: PathBuf,
    /// Split to source samples from.
    #[arg(long, default_value = "train")]
    pub(crate) split: String,
    /// Number of samples to tile in the preview grid.
    #[arg(long, default_value_t = 16)]
    pub(crate) count: usize,
    /// Output directory for grid images.
    #[arg(long, default_value = "artifacts/previews/dataset")]
    pub(crate) out: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum DatasetSourceSplit {
    Training,
    Validation,
    Testing,
}

impl From<DatasetSourceSplit> for SourceSplit {
    fn from(value: DatasetSourceSplit) -> Self {
        match value {
            DatasetSourceSplit::Training => Self::Training,
            DatasetSourceSplit::Validation => Self::Validation,
            DatasetSourceSplit::Testing => Self::Testing,
        }
    }
}

pub(crate) fn gcloud_exe() -> &'static str {
    if cfg!(windows) { "gcloud.cmd" } else { "gcloud" }
}

fn parse_dataset_source_split(value: &str) -> std::result::Result<DatasetSourceSplit, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "train" | "training" => Ok(DatasetSourceSplit::Training),
        "val" | "valid" | "validation" => Ok(DatasetSourceSplit::Validation),
        "test" | "testing" => Ok(DatasetSourceSplit::Testing),
        other => Err(format!(
            "unsupported split {other}; expected train, val, or test"
        )),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = ProjectPaths::discover()?;

    match cli.command {
        Command::Init => init(&paths),
        Command::Doctor => doctor(&paths),
        Command::Gcloud { command } => gcloud::run(command, &paths),
        Command::Dataset { command } => dataset::run(command, &paths),
    }
}

pub(crate) struct ProjectPaths {
    pub(crate) root: PathBuf,
}

impl ProjectPaths {
    fn discover() -> Result<Self> {
        let mut dir = std::env::current_dir()?;
        loop {
            let cargo = dir.join("Cargo.toml");
            if cargo.exists() && std::fs::read_to_string(&cargo)?.contains("[workspace]") {
                return Ok(Self { root: dir });
            }
            if !dir.pop() {
                anyhow::bail!("could not find workspace root");
            }
        }
    }

    fn local_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.root.join(".xtask"),
            self.root.join(".xtask/gcloud"),
            self.root.join(".xtask/manifests"),
            self.root.join(".xtask/runs"),
            self.root.join("data/raw/waymo/training"),
            self.root.join("data/raw/waymo/validation"),
            self.root.join("data/raw/waymo/testing"),
            self.root.join("data/processed/waymo-range-rgb-v1/train"),
            self.root.join("data/processed/waymo-range-rgb-v1/val"),
            self.root.join("data/processed/waymo-range-rgb-v1/test"),
            self.root.join("data/samples"),
            self.root.join("artifacts/checkpoints"),
            self.root.join("artifacts/metrics"),
            self.root.join("artifacts/previews"),
            self.root.join("artifacts/reports"),
            self.root.join("artifacts/tui_exports"),
        ]
    }

    fn manifest_files(&self) -> Vec<ManifestFile> {
        vec![
            ManifestFile::new(".xtask/manifests/raw_files.json", ManifestKind::RawFiles),
            ManifestFile::new(
                ".xtask/manifests/downloaded_files.json",
                ManifestKind::DownloadedFiles,
            ),
            ManifestFile::new(
                ".xtask/manifests/processed_samples.json",
                ManifestKind::ProcessedSamples,
            ),
            ManifestFile::new(".xtask/manifests/splits.json", ManifestKind::Splits),
            ManifestFile::new(".xtask/runs/run_index.json", ManifestKind::RunIndex),
            ManifestFile::new(".xtask/runs/latest.json", ManifestKind::LatestRun),
        ]
    }

    fn gcloud_dir(&self) -> PathBuf {
        self.root.join(".xtask/gcloud")
    }

    fn gcloud_credentials_path(&self) -> PathBuf {
        self.gcloud_dir().join("credentials.json")
    }

    fn gcloud_adc_path(&self) -> PathBuf {
        self.gcloud_dir()
            .join("application_default_credentials.json")
    }

    fn gcloud_auth_state_path(&self) -> PathBuf {
        self.gcloud_dir().join("auth_state.json")
    }

    fn gcloud_token_cache_path(&self) -> PathBuf {
        self.gcloud_dir().join("token_cache.json")
    }

    fn gcloud_credential_files(&self) -> Vec<PathBuf> {
        vec![
            self.gcloud_credentials_path(),
            self.gcloud_adc_path(),
            self.gcloud_auth_state_path(),
            self.gcloud_token_cache_path(),
        ]
    }

    fn raw_file_manifest_path(&self) -> PathBuf {
        self.root.join(".xtask/manifests/raw_files.json")
    }

    fn downloaded_file_manifest_path(&self) -> PathBuf {
        self.root.join(".xtask/manifests/downloaded_files.json")
    }

    pub(crate) fn processed_sample_manifest_path(&self) -> PathBuf {
        self.root.join(".xtask/manifests/processed_samples.json")
    }

    pub(crate) fn splits_manifest_path(&self) -> PathBuf {
        self.root.join(".xtask/manifests/splits.json")
    }
}

struct ManifestFile {
    relative: &'static str,
    kind: ManifestKind,
}

impl ManifestFile {
    fn new(relative: &'static str, kind: ManifestKind) -> Self {
        Self { relative, kind }
    }
}

#[derive(Clone, Copy)]
enum ManifestKind {
    RawFiles,
    DownloadedFiles,
    ProcessedSamples,
    Splits,
    RunIndex,
    LatestRun,
}

impl ManifestKind {
    fn write_default(self, path: &Path) -> sfx_core::manifest::Result<()> {
        match self {
            Self::RawFiles => write_manifest(path, &RawFileManifest::default()),
            Self::DownloadedFiles => write_manifest(path, &DownloadedFileManifest::default()),
            Self::ProcessedSamples => write_manifest(path, &ProcessedSampleManifest::default()),
            Self::Splits => write_manifest(path, &SplitsManifest::default()),
            Self::RunIndex => write_manifest(path, &RunIndex::default()),
            Self::LatestRun => write_manifest(path, &LatestRun::default()),
        }
    }

    fn validate(self, path: &Path) -> sfx_core::manifest::Result<()> {
        match self {
            Self::RawFiles => read_manifest::<RawFileManifest>(path)?.validate(),
            Self::DownloadedFiles => read_manifest::<DownloadedFileManifest>(path)?.validate(),
            Self::ProcessedSamples => read_manifest::<ProcessedSampleManifest>(path)?.validate(),
            Self::Splits => read_manifest::<SplitsManifest>(path)?.validate(),
            Self::RunIndex => read_manifest::<RunIndex>(path)?.validate(),
            Self::LatestRun => read_manifest::<LatestRun>(path)?.validate(),
        }
    }
}

fn init(paths: &ProjectPaths) -> Result<()> {
    for dir in paths.local_dirs() {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        println!("ok   {}", display_from_root(&paths.root, &dir));
    }

    for manifest in paths.manifest_files() {
        let path = paths.root.join(manifest.relative);
        if !path.exists() {
            manifest
                .kind
                .write_default(&path)
                .with_context(|| format!("creating {}", path.display()))?;
        }
        println!("ok   {}", manifest.relative);
    }
    Ok(())
}

fn doctor(paths: &ProjectPaths) -> Result<()> {
    println!("workspace: {}", paths.root.display());
    check_file(&paths.root, "Cargo.toml");
    check_file(&paths.root, "README.md");
    check_file(&paths.root, ".gitignore");
    check_dir(&paths.root, "configs");
    check_dir(&paths.root, "docs");

    for dir in paths.local_dirs() {
        check_path(&paths.root, &dir);
    }

    let errors = validate_configs(paths) + validate_manifests(paths);
    if errors > 0 {
        anyhow::bail!("{errors} file(s) failed validation");
    }

    Ok(())
}

fn validate_configs(paths: &ProjectPaths) -> usize {
    let root = &paths.root;
    let mut errors = 0;

    errors += check_config("configs/dataset.waymo.small.toml", || {
        sfx_config::load_dataset_config(root, "configs/dataset.waymo.small.toml").map(|_| ())
    });
    errors += check_config("configs/model.range-only.tiny.toml", || {
        sfx_config::load_model_config(root, "configs/model.range-only.tiny.toml").map(|_| ())
    });
    errors += check_config("configs/model.rgb-only.tiny.toml", || {
        sfx_config::load_model_config(root, "configs/model.rgb-only.tiny.toml").map(|_| ())
    });
    errors += check_config("configs/model.fusion.tiny.toml", || {
        sfx_config::load_model_config(root, "configs/model.fusion.tiny.toml").map(|_| ())
    });
    errors += check_config("configs/train.debug.toml", || {
        sfx_config::load_training_config(root, "configs/train.debug.toml").map(|_| ())
    });
    errors += check_config("configs/train.nextai.toml", || {
        sfx_config::load_training_config(root, "configs/train.nextai.toml").map(|_| ())
    });
    errors += check_config("configs/eval.default.toml", || {
        sfx_config::load_evaluation_config(root, "configs/eval.default.toml").map(|_| ())
    });

    errors
}

fn validate_manifests(paths: &ProjectPaths) -> usize {
    let mut errors = 0;

    for manifest in paths.manifest_files() {
        let path = paths.root.join(manifest.relative);
        if !path.exists() {
            println!("miss {}", manifest.relative);
            continue;
        }

        match manifest.kind.validate(&path) {
            Ok(()) => println!("ok   {}", manifest.relative),
            Err(err) => {
                println!("err  {}: {err}", manifest.relative);
                errors += 1;
            }
        }
    }

    errors
}

fn check_config<F>(relative: &str, validate: F) -> usize
where
    F: FnOnce() -> sfx_config::Result<()>,
{
    match validate() {
        Ok(()) => {
            println!("ok   {relative}");
            0
        }
        Err(err) => {
            println!("err  {relative}: {err}");
            1
        }
    }
}

fn check_file(root: &Path, relative: &str) {
    check_path(root, &root.join(relative));
}

fn check_dir(root: &Path, relative: &str) {
    check_path(root, &root.join(relative));
}

fn check_path(root: &Path, path: &Path) {
    let label = display_from_root(root, path);
    if path.exists() {
        println!("ok   {label}");
    } else {
        println!("miss {label}");
    }
}

pub(crate) fn display_from_root(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
