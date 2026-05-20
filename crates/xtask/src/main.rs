mod gcloud;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sfx_core::manifest::{
    DownloadedFileManifest, LatestRun, ProcessedSampleManifest, RawFileManifest, RunIndex,
    SplitsManifest, read_manifest, write_manifest,
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
}

#[derive(Subcommand, Debug)]
pub(crate) enum GcloudCommand {
    Auth,
    Check,
    Whoami,
    Logout,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = ProjectPaths::discover()?;

    match cli.command {
        Command::Init => init(&paths),
        Command::Doctor => doctor(&paths),
        Command::Gcloud { command } => gcloud::run(command, &paths),
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
