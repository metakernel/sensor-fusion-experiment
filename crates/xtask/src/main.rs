use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = ProjectPaths::discover()?;

    match cli.command {
        Command::Init => init(&paths),
        Command::Doctor => doctor(&paths),
    }
}

struct ProjectPaths {
    root: PathBuf,
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
}

fn init(paths: &ProjectPaths) -> Result<()> {
    for dir in paths.local_dirs() {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        println!("ok   {}", display_from_root(&paths.root, &dir));
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

    let config_errors = validate_configs(paths);
    if config_errors > 0 {
        anyhow::bail!("{config_errors} config file(s) failed validation");
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

fn display_from_root(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
