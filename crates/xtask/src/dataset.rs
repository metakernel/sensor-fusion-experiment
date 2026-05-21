use crate::{DatasetCommand, DatasetListArgs, ProjectPaths, display_from_root};
use anyhow::{Context, Result};
use sfx_core::manifest::{MANIFEST_SCHEMA_VERSION, RawFileManifest, SourceSplit, write_manifest};
use sfx_waymo::{DiscoveryConfig, parse_gcloud_storage_listing, split_label};
use std::process::{Command as ProcessCommand, Output};

pub(crate) fn run(command: DatasetCommand, paths: &ProjectPaths) -> Result<()> {
    match command {
        DatasetCommand::List(args) => list(args, paths),
    }
}

fn list(args: DatasetListArgs, paths: &ProjectPaths) -> Result<()> {
    if args.dataset != "waymo" {
        anyhow::bail!("unsupported dataset {}; expected waymo", args.dataset);
    }

    let dataset_config = sfx_config::load_dataset_config(&paths.root, &args.config)
        .with_context(|| format!("loading dataset config {}", args.config.display()))?;
    let waymo_config = sfx_config::load_waymo_config(&paths.root, &args.config)
        .with_context(|| format!("loading Waymo config {}", args.config.display()))?;
    let split = SourceSplit::from(args.split);
    let discovery = discovery_config(&args, waymo_config);
    let uri = discovery.split_uri(&split);
    let limit = (args.limit > 0).then_some(args.limit);

    if args.dry_run {
        println!("dataset: {}", dataset_config.name);
        println!("split: {}", split_label(&split));
        println!("url: {uri}");
        println!("limit: {}", limit_label(limit));
        println!("dry-run: gcloud storage ls --recursive --long {uri}");
        return Ok(());
    }

    let output = storage_ls(paths, &uri)?;
    let files = parse_gcloud_storage_listing(&output, split.clone(), limit);
    let manifest = RawFileManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name),
        files,
    };
    manifest.validate()?;

    let manifest_path = paths.raw_file_manifest_path();
    write_manifest(&manifest_path, &manifest)?;

    println!("Waymo files discovered:");
    println!("[{}]", split_label(&split));
    for file in &manifest.files {
        println!("  {}", file.file_name);
    }
    if manifest.files.is_empty() {
        println!("warn no files returned for {uri}");
    }
    println!("count {}", manifest.files.len());
    println!("ok   {}", display_from_root(&paths.root, &manifest_path));

    Ok(())
}

fn discovery_config(args: &DatasetListArgs, waymo: sfx_config::WaymoConfig) -> DiscoveryConfig {
    let mut config = DiscoveryConfig::default();

    if let Some(bucket) = non_empty(waymo.bucket) {
        config.bucket = bucket;
    }
    if let Some(prefix) = non_empty(waymo.prefix) {
        config.prefix = Some(prefix);
    }
    if let Some(prefix) = non_empty(waymo.training_prefix) {
        config.training_prefix = prefix;
    }
    if let Some(prefix) = non_empty(waymo.validation_prefix) {
        config.validation_prefix = prefix;
    }
    if let Some(prefix) = non_empty(waymo.testing_prefix) {
        config.testing_prefix = prefix;
    }

    if let Some(bucket) = non_empty(args.bucket.clone()) {
        config.bucket = bucket;
    }
    if let Some(prefix) = non_empty(args.prefix.clone()) {
        config.prefix = Some(prefix);
    }
    if let Some(prefix) = non_empty(args.split_prefix.clone()) {
        match SourceSplit::from(args.split) {
            SourceSplit::Training => config.training_prefix = prefix,
            SourceSplit::Validation => config.validation_prefix = prefix,
            SourceSplit::Testing => config.testing_prefix = prefix,
        }
    }

    config
}

fn storage_ls(paths: &ProjectPaths, uri: &str) -> Result<String> {
    let mut command = ProcessCommand::new("gcloud");
    command
        .env("CLOUDSDK_CONFIG", paths.gcloud_dir())
        .args(["storage", "ls", "--recursive", "--long"])
        .arg(uri);

    let credentials = paths.gcloud_credentials_path();
    if credentials.exists() {
        command.env("GOOGLE_APPLICATION_CREDENTIALS", credentials);
    }

    let output = command
        .output()
        .with_context(|| "running `gcloud storage ls`; install the Google Cloud CLI if needed")?;
    if !output.status.success() {
        anyhow::bail!(
            "listing Waymo objects failed: {}\nhint: run `cargo xtask gcloud auth` or use `--dry-run` to inspect the target URL",
            command_error(&output)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn limit_label(limit: Option<usize>) -> String {
    limit
        .map(|limit| limit.to_string())
        .unwrap_or_else(|| "unlimited".to_string())
}

fn command_error(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        output.status.to_string()
    } else {
        stdout
    }
}
