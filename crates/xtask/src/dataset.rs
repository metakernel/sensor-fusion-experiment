use crate::{
    DatasetCommand, DatasetFetchArgs, DatasetListArgs, DatasetPrepareArgs, DatasetSourceSplit,
    ProjectPaths, display_from_root,
};
use anyhow::{Context, Result};
use sfx_core::manifest::{
    DownloadStatus, DownloadedFileEntry, DownloadedFileManifest, MANIFEST_SCHEMA_VERSION,
    ProcessedSampleManifest, RawFileEntry, RawFileManifest, SplitsManifest, SourceSplit, Split,
    read_manifest, write_manifest,
};
use sfx_waymo::{DiscoveryConfig, parse_gcloud_storage_listing, split_label};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output};

pub(crate) fn run(command: DatasetCommand, paths: &ProjectPaths) -> Result<()> {
    match command {
        DatasetCommand::List(args) => list(args, paths),
        DatasetCommand::Fetch(args) => fetch(args, paths),
        DatasetCommand::Prepare(args) => prepare(args, paths),
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
    let base_uri = discovery.split_uri(&split);
    let uri = match &args.component {
        Some(comp) => format!("{base_uri}/{comp}"),
        None => base_uri,
    };
    let limit = (args.limit > 0).then_some(args.limit);

    if args.dry_run {
        println!("dataset: {}", dataset_config.name);
        println!("split: {}", split_label(&split));
        println!("url: {uri}");
        if let Some(c) = &args.component {
            println!("component: {c}");
        }
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

fn fetch(args: DatasetFetchArgs, paths: &ProjectPaths) -> Result<()> {
    if args.dataset != "waymo" {
        anyhow::bail!("unsupported dataset {}; expected waymo", args.dataset);
    }

    let dataset_config = sfx_config::load_dataset_config(&paths.root, &args.config)
        .with_context(|| format!("loading dataset config {}", args.config.display()))?;
    let waymo_config = sfx_config::load_waymo_config(&paths.root, &args.config)
        .with_context(|| format!("loading Waymo config {}", args.config.display()))?;
    let raw_manifest_path = args
        .manifest
        .as_ref()
        .map(|path| sfx_config::resolve_from_root(&paths.root, path))
        .unwrap_or_else(|| paths.raw_file_manifest_path());
    let out_dir = args
        .out
        .as_ref()
        .map(|path| sfx_config::resolve_from_root(&paths.root, path))
        .unwrap_or(dataset_config.raw_dir.clone());

    let raw_manifest = read_raw_manifest_if_exists(&raw_manifest_path)?;
    let raw_manifest = if args.dry_run {
        raw_manifest
    } else {
        ensure_raw_files_for_fetch(
            raw_manifest,
            &raw_manifest_path,
            &args,
            paths,
            &dataset_config.name,
            waymo_config,
        )?
    };
    let selected = select_raw_files(&raw_manifest.files, &args);
    print_selection_warnings(&args, &selected);

    if selected.is_empty() {
        anyhow::bail!(
            "no raw files selected; run `cargo xtask dataset list` first or allow fetch to discover files"
        );
    }

    // Build the full list: primary files + companion component files
    let mut all_files: Vec<RawFileEntry> = selected.clone();
    for comp in &args.extra_components {
        for file in &selected {
            if let Some(companion) = companion_component_entry(file, comp) {
                if !all_files.iter().any(|f| f.uri == companion.uri) {
                    all_files.push(companion);
                }
            }
        }
    }

    println!("Waymo files selected for download:");
    for file in &all_files {
        let target = target_path_for_raw(&out_dir, file);
        println!(
            "  [{}] {} -> {}",
            split_label(&file.split),
            file.file_name,
            display_from_root(&paths.root, &target)
        );
    }

    if args.dry_run {
        println!("dry-run: no files downloaded");
        return Ok(());
    }

    let mut entries = Vec::new();
    let mut failed = 0usize;
    let total = all_files.len();

    for (index, file) in all_files.iter().enumerate() {
        let target = target_path_for_raw(&out_dir, file);
        println!(
            "[{}/{}] {}",
            index + 1,
            total,
            display_from_root(&paths.root, &target)
        );

        match download_raw_file(file, &target, paths) {
            Ok(status) => {
                println!("ok   {}", file.file_name);
                entries.push(downloaded_entry(file, &target, status, None, paths));
            }
            Err(err) => {
                failed += 1;
                println!("err  {}: {err}", file.file_name);
                entries.push(downloaded_entry(
                    file,
                    &target,
                    DownloadStatus::Failed,
                    Some(err.to_string()),
                    paths,
                ));
            }
        }
    }

    let manifest = DownloadedFileManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name),
        files: entries,
    };
    manifest.validate()?;

    let manifest_path = paths.downloaded_file_manifest_path();
    write_manifest(&manifest_path, &manifest)?;
    println!("ok   {}", display_from_root(&paths.root, &manifest_path));

    if failed > 0 {
        anyhow::bail!("{failed} download(s) failed");
    }

    Ok(())
}

fn companion_component_entry(file: &RawFileEntry, component: &str) -> Option<RawFileEntry> {
    // URI: gs://bucket/split/current_component/file.parquet
    // Replace current_component with the given component.
    let parts: Vec<&str> = file.uri.splitn(6, '/').collect();
    if parts.len() == 6 {
        let companion_uri = format!(
            "gs://{}/{}/{}/{}",
            parts[2], parts[3], component, parts[5]
        );
        Some(RawFileEntry {
            uri: companion_uri,
            split: file.split.clone(),
            file_name: file.file_name.clone(),
            size_bytes: None,
            checksum: None,
        })
    } else {
        None
    }
}

fn prepare(args: DatasetPrepareArgs, paths: &ProjectPaths) -> Result<()> {
    if args.dataset != "waymo" {
        anyhow::bail!("unsupported dataset {}; expected waymo", args.dataset);
    }

    let dataset_config = sfx_config::load_dataset_config(&paths.root, &args.config)
        .with_context(|| format!("loading dataset config {}", args.config.display()))?;
    let out_dir = args
        .output
        .as_ref()
        .map(|p| sfx_config::resolve_from_root(&paths.root, p))
        .unwrap_or(dataset_config.processed_dir.clone());
    let in_dir = args
        .input
        .as_ref()
        .map(|p| sfx_config::resolve_from_root(&paths.root, p))
        .unwrap_or(dataset_config.raw_dir.clone());

    let mut all_entries: Vec<sfx_core::manifest::ProcessedSampleEntry> = Vec::new();
    let mut train_ids = Vec::new();
    let mut val_ids = Vec::new();
    let mut test_ids = Vec::new();
    let mut sample_counter = 0usize;

    for split_arg in &args.splits {
        let (split_dir_name, split) = match split_arg {
            DatasetSourceSplit::Training => ("training", Split::Train),
            DatasetSourceSplit::Validation => ("validation", Split::Val),
            DatasetSourceSplit::Testing => ("testing", Split::Test),
        };

        let camera_dir = in_dir.join(split_dir_name).join("camera_image");
        let lidar_dir = in_dir.join(split_dir_name).join("lidar");

        if !camera_dir.exists() {
            println!("warn no camera_image dir for {split_dir_name}; skipping");
            continue;
        }
        if !lidar_dir.exists() {
            println!("warn no lidar dir for {split_dir_name}; skipping");
            continue;
        }

        let camera_files = parquet_files_in(&camera_dir)?;
        let lidar_files = parquet_files_in(&lidar_dir)?;

        if args.inspect {
            println!("\n--- camera_image schema ({split_dir_name}) ---");
            if let Some(cam_path) = camera_files.first() {
                for f in sfx_waymo::extract::read_schema(cam_path)? {
                    println!("  {}: {}", f.name, f.data_type);
                }
            }
            println!("\n--- lidar schema ({split_dir_name}) ---");
            if let Some(lid_path) = lidar_files.first() {
                for f in sfx_waymo::extract::read_schema(lid_path)? {
                    println!("  {}: {}", f.name, f.data_type);
                }
            }
            continue;
        }

        for (cam_path, lid_path) in camera_files.iter().zip(lidar_files.iter()) {
            println!(
                "extracting {} + {}",
                cam_path.file_name().unwrap_or_default().to_string_lossy(),
                lid_path.file_name().unwrap_or_default().to_string_lossy()
            );

            let camera_frames =
                sfx_waymo::extract::read_camera_frames(cam_path, args.max_frames)?;
            let lidar_frames = sfx_waymo::extract::read_lidar_frames(lid_path, args.max_frames)?;

            println!(
                "  camera frames: {}, lidar frames: {}",
                camera_frames.len(),
                lidar_frames.len()
            );

            let pairs = sfx_preprocess::align_frames(camera_frames, lidar_frames);
            println!("  aligned pairs: {}", pairs.len());

            for pair in &pairs {
                let entry =
                    sfx_preprocess::write_sample(&out_dir, pair, &split, sample_counter)?;
                let id = entry.meta.id.clone();
                all_entries.push(entry);
                match &split {
                    Split::Train => train_ids.push(id),
                    Split::Val => val_ids.push(id),
                    Split::Test => test_ids.push(id),
                }
                sample_counter += 1;
            }

            println!("  ok   wrote {} samples", pairs.len());
        }
    }

    if args.inspect {
        return Ok(());
    }

    let sample_manifest = ProcessedSampleManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name.clone()),
        rgb_shape: None,
        range_shape: None,
        samples: all_entries,
    };
    sample_manifest.validate()?;
    let sample_manifest_path = paths.processed_sample_manifest_path();
    write_manifest(&sample_manifest_path, &sample_manifest)?;
    println!("ok   {}", display_from_root(&paths.root, &sample_manifest_path));

    let splits_manifest = SplitsManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name),
        train: train_ids,
        val: val_ids,
        test: test_ids,
    };
    splits_manifest.validate()?;
    let splits_path = paths.splits_manifest_path();
    write_manifest(&splits_path, &splits_manifest)?;
    println!("ok   {}", display_from_root(&paths.root, &splits_path));

    println!("total samples extracted: {sample_counter}");
    Ok(())
}

fn parquet_files_in(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading directory {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    files.sort();
    Ok(files)
}

fn discovery_config(args: &DatasetListArgs, waymo: sfx_config::WaymoConfig) -> DiscoveryConfig {
    let mut config = discovery_config_from_waymo(waymo);

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

fn discovery_config_from_waymo(waymo: sfx_config::WaymoConfig) -> DiscoveryConfig {
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

    config
}

fn read_raw_manifest_if_exists(path: &Path) -> Result<RawFileManifest> {
    if !path.exists() {
        return Ok(RawFileManifest::default());
    }

    let manifest: RawFileManifest = read_manifest(path)?;
    manifest.validate()?;
    Ok(manifest)
}

fn ensure_raw_files_for_fetch(
    mut manifest: RawFileManifest,
    manifest_path: &Path,
    args: &DatasetFetchArgs,
    paths: &ProjectPaths,
    dataset_name: &str,
    waymo: sfx_config::WaymoConfig,
) -> Result<RawFileManifest> {
    let discovery = discovery_config_from_waymo(waymo);
    let mut changed = false;

    if manifest.dataset.is_none() {
        manifest.dataset = Some(dataset_name.to_string());
        changed = true;
    }

    for split_arg in selected_split_args(args) {
        let wanted = requested_count(args, split_arg);
        if wanted == 0 {
            continue;
        }

        let split = SourceSplit::from(split_arg);
        let current = count_files_for_split(&manifest.files, &split);
        if current >= wanted {
            continue;
        }

        let uri = discovery.split_uri(&split);
        println!("discovering {} files from {uri}", split_label(&split));
        let output = storage_ls(paths, &uri)?;
        let files = parse_gcloud_storage_listing(&output, split, Some(wanted));
        changed |= append_unique_raw_files(&mut manifest.files, files);
    }

    if changed {
        manifest.validate()?;
        write_manifest(manifest_path, &manifest)?;
        println!("ok   {}", display_from_root(&paths.root, manifest_path));
    }

    Ok(manifest)
}

fn append_unique_raw_files(target: &mut Vec<RawFileEntry>, files: Vec<RawFileEntry>) -> bool {
    let mut known = target
        .iter()
        .map(|file| file.uri.clone())
        .collect::<BTreeSet<_>>();
    let mut changed = false;

    for file in files {
        if known.insert(file.uri.clone()) {
            target.push(file);
            changed = true;
        }
    }

    changed
}

fn select_raw_files(files: &[RawFileEntry], args: &DatasetFetchArgs) -> Vec<RawFileEntry> {
    let mut selected = Vec::new();

    for split_arg in selected_split_args(args) {
        let split = SourceSplit::from(split_arg);
        let count = requested_count(args, split_arg);
        selected.extend(
            files
                .iter()
                .filter(|file| file.split == split)
                .take(count)
                .cloned(),
        );
    }

    selected
}

fn selected_split_args(args: &DatasetFetchArgs) -> Vec<DatasetSourceSplit> {
    let mut splits = Vec::new();
    for split in &args.splits {
        if !splits.contains(split) {
            splits.push(*split);
        }
    }
    splits
}

fn requested_count(args: &DatasetFetchArgs, split: DatasetSourceSplit) -> usize {
    match split {
        DatasetSourceSplit::Training => args.train_files,
        DatasetSourceSplit::Validation => args.val_files,
        DatasetSourceSplit::Testing => args.test_files,
    }
}

fn count_files_for_split(files: &[RawFileEntry], split: &SourceSplit) -> usize {
    files.iter().filter(|file| &file.split == split).count()
}

fn print_selection_warnings(args: &DatasetFetchArgs, selected: &[RawFileEntry]) {
    for split_arg in selected_split_args(args) {
        let wanted = requested_count(args, split_arg);
        if wanted == 0 {
            continue;
        }

        let split = SourceSplit::from(split_arg);
        let actual = count_files_for_split(selected, &split);
        if actual < wanted {
            println!(
                "warn requested {wanted} {} file(s), selected {actual}",
                split_label(&split)
            );
        }
    }
}

fn target_path_for_raw(out_dir: &Path, file: &RawFileEntry) -> PathBuf {
    out_dir.join(split_label(&file.split)).join(&file.file_name)
}

fn download_raw_file(
    file: &RawFileEntry,
    target: &Path,
    paths: &ProjectPaths,
) -> Result<DownloadStatus> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    if target.exists() {
        if let Some(expected) = file.size_bytes {
            let actual = file_size(target)?;
            if actual == expected {
                return Ok(DownloadStatus::Verified);
            }
            std::fs::remove_file(target)
                .with_context(|| format!("removing incomplete file {}", target.display()))?;
        } else {
            return Ok(DownloadStatus::Downloaded);
        }
    }

    let temp = temp_download_path(target)?;
    remove_file_if_exists(&temp)?;

    if let Some(source) = local_source_path(&file.uri) {
        std::fs::copy(&source, &temp)
            .with_context(|| format!("copying {} to {}", source.display(), temp.display()))?;
    } else {
        storage_cp(paths, &file.uri, &temp)?;
    }

    if let Some(expected) = file.size_bytes {
        let actual = file_size(&temp)?;
        if actual != expected {
            remove_file_if_exists(&temp)?;
            anyhow::bail!(
                "downloaded size mismatch for {}: expected {expected} bytes, got {actual}",
                file.file_name
            );
        }
    }

    if target.exists() {
        std::fs::remove_file(target)
            .with_context(|| format!("removing previous file {}", target.display()))?;
    }
    std::fs::rename(&temp, target)
        .with_context(|| format!("moving {} to {}", temp.display(), target.display()))?;

    Ok(if file.size_bytes.is_some() {
        DownloadStatus::Verified
    } else {
        DownloadStatus::Downloaded
    })
}

fn downloaded_entry(
    file: &RawFileEntry,
    target: &Path,
    status: DownloadStatus,
    error: Option<String>,
    paths: &ProjectPaths,
) -> DownloadedFileEntry {
    DownloadedFileEntry {
        source_uri: file.uri.clone(),
        local_path: manifest_local_path(&paths.root, target),
        split: file.split.clone(),
        size_bytes: file.size_bytes,
        status,
        error,
    }
}

fn manifest_local_path(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

fn temp_download_path(target: &Path) -> Result<PathBuf> {
    let name = target
        .file_name()
        .with_context(|| format!("download target has no file name: {}", target.display()))?;
    Ok(target.with_file_name(format!("{}.download", name.to_string_lossy())))
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len())
}

fn local_source_path(uri: &str) -> Option<PathBuf> {
    let raw = uri.strip_prefix("file://")?;
    #[cfg(windows)]
    let raw = raw
        .as_bytes()
        .get(0..3)
        .is_some_and(|prefix| prefix[0] == b'/' && prefix[2] == b':')
        .then_some(&raw[1..])
        .unwrap_or(raw);
    Some(PathBuf::from(raw))
}

fn storage_ls(paths: &ProjectPaths, uri: &str) -> Result<String> {
    let mut command = storage_command(paths)?;
    command.args(["storage", "ls", "--recursive", "--long"]);
    command.arg(uri);

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

fn storage_cp(paths: &ProjectPaths, source: &str, target: &Path) -> Result<()> {
    let mut command = storage_command(paths)?;
    command.args(["storage", "cp", source]);
    command.arg(target);

    let output = command
        .output()
        .with_context(|| "running `gcloud storage cp`; install the Google Cloud CLI if needed")?;
    if !output.status.success() {
        anyhow::bail!(
            "downloading {source} failed: {}\nhint: run `cargo xtask gcloud auth` and confirm Waymo dataset access",
            command_error(&output)
        );
    }

    Ok(())
}

fn storage_command(paths: &ProjectPaths) -> Result<ProcessCommand> {
    let token = crate::gcloud::access_token(paths)?;
    let mut command = ProcessCommand::new(crate::gcloud_exe());
    command.env("CLOUDSDK_AUTH_ACCESS_TOKEN", token);
    Ok(command)
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("removing {}", path.display())),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_raw_files_respects_requested_splits_and_counts() {
        let args = fetch_args(vec![
            DatasetSourceSplit::Training,
            DatasetSourceSplit::Validation,
        ]);
        let files = vec![
            raw("gs://bucket/training/a.tfrecord", SourceSplit::Training),
            raw("gs://bucket/training/b.tfrecord", SourceSplit::Training),
            raw("gs://bucket/training/c.tfrecord", SourceSplit::Training),
            raw("gs://bucket/validation/d.tfrecord", SourceSplit::Validation),
            raw("gs://bucket/testing/e.tfrecord", SourceSplit::Testing),
        ];

        let selected = select_raw_files(&files, &args);
        let names = selected
            .iter()
            .map(|file| file.file_name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["a.tfrecord", "b.tfrecord", "d.tfrecord"]);
    }

    #[test]
    fn target_path_uses_source_split_directory() {
        let out = PathBuf::from("data/raw/waymo");
        let file = raw("gs://bucket/validation/a.tfrecord", SourceSplit::Validation);

        let target = target_path_for_raw(&out, &file);

        assert_eq!(
            target,
            PathBuf::from("data/raw/waymo/validation/a.tfrecord")
        );
    }

    #[test]
    fn download_file_uri_writes_and_verifies_target() {
        let root = temp_root("download_file_uri_writes_and_verifies_target");
        let source = root.join("source.tfrecord");
        let target = root.join("data/raw/waymo/training/source.tfrecord");
        std::fs::write(&source, b"abcde").unwrap();
        let file = RawFileEntry {
            uri: file_uri(&source),
            split: SourceSplit::Training,
            file_name: "source.tfrecord".to_string(),
            size_bytes: Some(5),
            checksum: None,
        };
        let paths = ProjectPaths { root };

        let status = download_raw_file(&file, &target, &paths).unwrap();
        let second_status = download_raw_file(&file, &target, &paths).unwrap();

        assert_eq!(status, DownloadStatus::Verified);
        assert_eq!(second_status, DownloadStatus::Verified);
        assert_eq!(std::fs::read(target).unwrap(), b"abcde");
    }

    #[test]
    fn download_file_uri_rejects_size_mismatch() {
        let root = temp_root("download_file_uri_rejects_size_mismatch");
        let source = root.join("source.tfrecord");
        let target = root.join("data/raw/waymo/training/source.tfrecord");
        std::fs::write(&source, b"abc").unwrap();
        let file = RawFileEntry {
            uri: file_uri(&source),
            split: SourceSplit::Training,
            file_name: "source.tfrecord".to_string(),
            size_bytes: Some(5),
            checksum: None,
        };
        let paths = ProjectPaths { root };

        let err = download_raw_file(&file, &target, &paths).unwrap_err();

        assert!(err.to_string().contains("size mismatch"));
        assert!(!target.exists());
    }

    fn fetch_args(splits: Vec<DatasetSourceSplit>) -> DatasetFetchArgs {
        DatasetFetchArgs {
            dataset: "waymo".to_string(),
            splits,
            train_files: 2,
            val_files: 1,
            test_files: 1,
            config: PathBuf::from("configs/dataset.waymo.small.toml"),
            manifest: None,
            out: None,
            extra_components: Vec::new(),
            dry_run: false,
        }
    }

    fn raw(uri: &str, split: SourceSplit) -> RawFileEntry {
        RawFileEntry {
            uri: uri.to_string(),
            split,
            file_name: uri.rsplit('/').next().unwrap().to_string(),
            size_bytes: Some(10),
            checksum: None,
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("sfx-xtask-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn file_uri(path: &Path) -> String {
        format!("file://{}", path.display().to_string().replace('\\', "/"))
    }
}
