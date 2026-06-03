use crate::{
    DatasetCommand, DatasetFetchArgs, DatasetInspectArgs, DatasetListArgs, DatasetPrepareArgs,
    DatasetPreviewArgs, DatasetSourceSplit, ProjectPaths, display_from_root,
};
use anyhow::{Context, Result};
use sfx_core::manifest::{
    DownloadStatus, DownloadedFileEntry, DownloadedFileManifest, MANIFEST_SCHEMA_VERSION,
    ExtractionSummaryEntry, ExtractionSummaryManifest, ProcessedSampleManifest, RawFileEntry,
    RawFileManifest, SourceSplit, Split, SplitsManifest, read_manifest, write_manifest,
};
use sfx_waymo::{DiscoveryConfig, parse_gcloud_storage_listing, split_label};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output};

pub(crate) fn run(command: DatasetCommand, paths: &ProjectPaths) -> Result<()> {
    match command {
        DatasetCommand::List(args) => list(args, paths),
        DatasetCommand::Fetch(args) => fetch(args, paths),
        DatasetCommand::Prepare(args) => prepare(args, paths),
        DatasetCommand::Inspect(args) => inspect(args, paths),
        DatasetCommand::Preview(args) => preview(args, paths),
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
        let companion_uri = format!("gs://{}/{}/{}/{}", parts[2], parts[3], component, parts[5]);
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

    let rgb_size = args.rgb_size.unwrap_or(dataset_config.rgb_size);
    let range_size = args.range_size.unwrap_or(dataset_config.range_size);
    let configured_channels = args
        .range_channels
        .as_deref()
        .unwrap_or(dataset_config.range_channels.as_slice());
    let range_channels = parse_range_channels(configured_channels)?;
    let process_options = sfx_preprocess::ProcessOptions {
        rgb_height: rgb_size[0],
        rgb_width: rgb_size[1],
        range_height: range_size[0],
        range_width: range_size[1],
        range_channels,
    };

    let mut all_entries: Vec<sfx_core::manifest::ProcessedSampleEntry> = Vec::new();
    let mut extraction_entries = Vec::new();
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
        let paired_files = pair_component_parquet_files(&camera_files, &lidar_files);

        if paired_files.len() < camera_files.len() {
            println!(
                "warn {} camera parquet file(s) have no matching lidar file",
                camera_files.len() - paired_files.len()
            );
        }
        if paired_files.len() < lidar_files.len() {
            println!(
                "warn {} lidar parquet file(s) have no matching camera file",
                lidar_files.len() - paired_files.len()
            );
        }
        if paired_files.is_empty() {
            println!("warn no matching camera/lidar parquet files for {split_dir_name}; skipping");
            continue;
        }

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

        for (cam_path, lid_path) in paired_files {
            println!(
                "extracting {} + {}",
                cam_path.file_name().unwrap_or_default().to_string_lossy(),
                lid_path.file_name().unwrap_or_default().to_string_lossy()
            );

            let camera_frames = sfx_waymo::extract::read_camera_frames_for_sensor(
                &cam_path,
                args.camera_name,
                args.max_frames,
            )?;
            let lidar_frames = sfx_waymo::extract::read_lidar_frames_for_sensor(
                &lid_path,
                args.laser_name,
                args.max_frames,
            )?;

            println!(
                "  camera frames: {}, lidar frames: {}",
                camera_frames.len(),
                lidar_frames.len()
            );

            let (pairs, align_stats) = sfx_preprocess::align_frames_with_options(
                camera_frames,
                lidar_frames,
                sfx_preprocess::AlignOptions {
                    max_timestamp_delta_micros: args.max_timestamp_delta_us.max(0),
                },
            );
            println!("  aligned pairs: {}", align_stats.matched_pairs);
            if align_stats.unmatched_camera > 0 || align_stats.unmatched_lidar > 0 {
                println!(
                    "  unmatched camera: {}, unmatched lidar: {}",
                    align_stats.unmatched_camera, align_stats.unmatched_lidar
                );
            }
            if align_stats.max_abs_timestamp_delta_micros > 0 {
                println!(
                    "  max |camera_ts - lidar_ts|: {} µs",
                    align_stats.max_abs_timestamp_delta_micros
                );
            }

            let first_sample_id = pairs
                .first()
                .map(|_| sfx_core::manifest::SampleId(format!("sample_{sample_counter:06}")));

            for pair in &pairs {
                let entry = sfx_preprocess::write_sample_with_options(
                    &out_dir,
                    pair,
                    &split,
                    sample_counter,
                    &process_options,
                )?;
                let id = entry.meta.id.clone();
                all_entries.push(entry);
                match &split {
                    Split::Train => train_ids.push(id),
                    Split::Val => val_ids.push(id),
                    Split::Test => test_ids.push(id),
                }
                sample_counter += 1;
            }

            let last_sample_id = if sample_counter == 0 || pairs.is_empty() {
                None
            } else {
                Some(sfx_core::manifest::SampleId(format!(
                    "sample_{:06}",
                    sample_counter - 1
                )))
            };

            extraction_entries.push(ExtractionSummaryEntry {
                split: split.clone(),
                camera_parquet_path: manifest_local_path(&paths.root, &cam_path),
                lidar_parquet_path: manifest_local_path(&paths.root, &lid_path),
                camera_frames: align_stats.camera_frames,
                lidar_frames: align_stats.lidar_frames,
                matched_pairs: align_stats.matched_pairs,
                unmatched_camera: align_stats.unmatched_camera,
                unmatched_lidar: align_stats.unmatched_lidar,
                max_abs_timestamp_delta_micros: align_stats.max_abs_timestamp_delta_micros,
                first_sample_id,
                last_sample_id,
            });

            println!("  ok   wrote {} samples", pairs.len());
        }
    }

    if args.inspect {
        return Ok(());
    }

    let sample_manifest = ProcessedSampleManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name.clone()),
        rgb_shape: Some(sfx_core::manifest::TensorShape {
            channels: 3,
            height: process_options.rgb_height,
            width: process_options.rgb_width,
        }),
        range_shape: Some(sfx_core::manifest::TensorShape {
            channels: process_options.range_channels.len(),
            height: process_options.range_height,
            width: process_options.range_width,
        }),
        samples: all_entries,
    };
    sample_manifest.validate()?;
    let sample_manifest_path = paths.processed_sample_manifest_path();
    write_manifest(&sample_manifest_path, &sample_manifest)?;
    println!(
        "ok   {}",
        display_from_root(&paths.root, &sample_manifest_path)
    );

    let extraction_summary = ExtractionSummaryManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some(dataset_config.name.clone()),
        files: extraction_entries,
    };
    extraction_summary.validate()?;
    let extraction_summary_path = paths.extraction_summary_manifest_path();
    write_manifest(&extraction_summary_path, &extraction_summary)?;
    println!(
        "ok   {}",
        display_from_root(&paths.root, &extraction_summary_path)
    );

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

fn pair_component_parquet_files(
    camera_files: &[PathBuf],
    lidar_files: &[PathBuf],
) -> Vec<(PathBuf, PathBuf)> {
    let mut lidar_by_name = BTreeMap::new();
    for path in lidar_files {
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            lidar_by_name.insert(name.to_string(), path.clone());
        }
    }

    let mut pairs = Vec::new();
    for camera_path in camera_files {
        let Some(name) = camera_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if let Some(lidar_path) = lidar_by_name.remove(name) {
            pairs.push((camera_path.clone(), lidar_path));
        }
    }

    pairs
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
    // URI: gs://bucket/split/component/file.parquet — include component subdir
    let parts: Vec<&str> = file.uri.splitn(6, '/').collect();
    if parts.len() == 6 {
        let component = parts[4];
        out_dir
            .join(split_label(&file.split))
            .join(component)
            .join(&file.file_name)
    } else {
        out_dir.join(split_label(&file.split)).join(&file.file_name)
    }
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

fn parse_range_channels(values: &[String]) -> Result<Vec<sfx_preprocess::RangeChannel>> {
    let mut channels = Vec::new();
    for value in values {
        let key = value.trim().to_ascii_lowercase();
        let channel = match key.as_str() {
            "range" | "distance" => sfx_preprocess::RangeChannel::Range,
            "intensity" => sfx_preprocess::RangeChannel::Intensity,
            "valid" | "validity" | "mask" | "validity-mask" => {
                sfx_preprocess::RangeChannel::ValidityMask
            }
            _ => {
                anyhow::bail!(
                    "unsupported range channel `{value}`; expected range, intensity, or validity"
                )
            }
        };
        channels.push(channel);
    }
    if channels.is_empty() {
        anyhow::bail!("at least one range channel is required");
    }
    Ok(channels)
}

fn resolve_processed_dir(dataset_arg: Option<&Path>, dataset_config: &sfx_config::DatasetConfig) -> PathBuf {
    dataset_arg
        .map(Path::to_path_buf)
        .unwrap_or_else(|| dataset_config.processed_dir.clone())
}

#[derive(Default)]
struct SuspiciousSampleReport {
    missing_files: usize,
    wrong_tensor_size: usize,
    non_finite_values: usize,
    rgb_out_of_range: usize,
    range_out_of_range: usize,
    low_valid_fraction: usize,
}

fn scan_suspicious_samples(
    manifest: &ProcessedSampleManifest,
    processed_dir: &Path,
) -> Result<SuspiciousSampleReport> {
    let mut report = SuspiciousSampleReport::default();
    let rgb_shape = manifest
        .rgb_shape
        .as_ref()
        .context("processed sample manifest is missing rgb_shape")?;
    let range_shape = manifest
        .range_shape
        .as_ref()
        .context("processed sample manifest is missing range_shape")?;

    let rgb_expected = rgb_shape.value_count();
    let range_expected = range_shape.value_count();
    let range_pixels = range_shape.height * range_shape.width;

    for sample in &manifest.samples {
        let rgb_path = processed_dir.join(&sample.meta.rgb_path);
        let range_path = processed_dir.join(&sample.meta.range_path);
        if !rgb_path.exists() || !range_path.exists() {
            report.missing_files += 1;
            continue;
        }

        let rgb = bytes_to_f32_le(&std::fs::read(&rgb_path)?);
        let range = bytes_to_f32_le(&std::fs::read(&range_path)?);
        if rgb.len() != rgb_expected || range.len() != range_expected {
            report.wrong_tensor_size += 1;
            continue;
        }

        if rgb.iter().any(|v| !v.is_finite()) || range.iter().any(|v| !v.is_finite()) {
            report.non_finite_values += 1;
        }

        if rgb.iter().any(|v| *v < 0.0 || *v > 1.0) {
            report.rgb_out_of_range += 1;
        }

        if range.iter().any(|v| *v < 0.0 || *v > 1.0) {
            report.range_out_of_range += 1;
        }

        let valid = range
            .iter()
            .take(range_pixels.min(range.len()))
            .filter(|v| **v > 0.0)
            .count();
        let valid_frac = valid as f64 / range_pixels.max(1) as f64;
        if valid_frac < 0.01 {
            report.low_valid_fraction += 1;
        }
    }

    Ok(report)
}

fn render_rgb_from_tensor(path: &Path, shape: &sfx_core::manifest::TensorShape) -> Result<image::RgbImage> {
    let values = bytes_to_f32_le(&std::fs::read(path)?);
    let expected = shape.value_count();
    if values.len() != expected {
        anyhow::bail!(
            "{} has {} values; expected {}",
            path.display(),
            values.len(),
            expected
        );
    }
    if shape.channels < 3 {
        anyhow::bail!("RGB tensor requires at least 3 channels");
    }

    let plane = shape.height * shape.width;
    let mut pixels = vec![0u8; 3 * plane];
    for i in 0..plane {
        pixels[3 * i] = (values[i].clamp(0.0, 1.0) * 255.0) as u8;
        pixels[3 * i + 1] = (values[plane + i].clamp(0.0, 1.0) * 255.0) as u8;
        pixels[3 * i + 2] = (values[2 * plane + i].clamp(0.0, 1.0) * 255.0) as u8;
    }

    image::RgbImage::from_raw(shape.width as u32, shape.height as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to construct RGB image"))
}

fn render_range_from_tensor(path: &Path, shape: &sfx_core::manifest::TensorShape) -> Result<image::GrayImage> {
    let values = bytes_to_f32_le(&std::fs::read(path)?);
    let expected = shape.value_count();
    if values.len() != expected {
        anyhow::bail!(
            "{} has {} values; expected {}",
            path.display(),
            values.len(),
            expected
        );
    }

    let plane = shape.height * shape.width;
    let pixels: Vec<u8> = values
        .iter()
        .take(plane)
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    image::GrayImage::from_raw(shape.width as u32, shape.height as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to construct range image"))
}

// ---------------------------------------------------------------------------
// dataset inspect
// ---------------------------------------------------------------------------

fn inspect(args: DatasetInspectArgs, paths: &ProjectPaths) -> Result<()> {
    let dataset_config = sfx_config::load_dataset_config(&paths.root, &args.config)
        .with_context(|| format!("loading dataset config {}", args.config.display()))?;

    let manifest_path = paths.processed_sample_manifest_path();
    if !manifest_path.exists() {
        anyhow::bail!(
            "processed sample manifest not found at {}\nhint: run `cargo xtask dataset prepare` first",
            display_from_root(&paths.root, &manifest_path)
        );
    }
    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)?;
    manifest.validate()?;

    let processed_dir = resolve_processed_dir(args.dataset.as_deref(), &dataset_config);

    println!("dataset:       {}", dataset_config.name);
    println!(
        "processed dir: {}",
        display_from_root(&paths.root, &processed_dir)
    );
    println!("schema:        v{}", manifest.schema_version);

    if let Some(shape) = &manifest.rgb_shape {
        let values = shape.value_count();
        println!(
            "rgb shape:     [{}, {}, {}]  ({} values, {} bytes/sample)",
            shape.channels,
            shape.height,
            shape.width,
            values,
            values * 4
        );
    } else {
        println!("rgb shape:     (not recorded)");
    }
    if let Some(shape) = &manifest.range_shape {
        let values = shape.value_count();
        println!(
            "range shape:   [{}, {}, {}]  ({} values, {} bytes/sample)",
            shape.channels,
            shape.height,
            shape.width,
            values,
            values * 4
        );
    } else {
        println!("range shape:   (not recorded)");
    }

    // --- split counts ---
    let (train_n, val_n, test_n) =
        manifest
            .samples
            .iter()
            .fold((0usize, 0usize, 0usize), |(tr, v, te), s| {
                match s.meta.split {
                    sfx_core::manifest::Split::Train => (tr + 1, v, te),
                    sfx_core::manifest::Split::Val => (tr, v + 1, te),
                    sfx_core::manifest::Split::Test => (tr, v, te + 1),
                }
            });
    println!();
    println!("split counts:");
    println!("  train: {train_n}");
    println!("  val:   {val_n}");
    println!("  test:  {test_n}");
    println!("  total: {}", manifest.samples.len());

    // --- tensor statistics (sampled subset) ---
    let n = args.sample_count.min(manifest.samples.len());
    if n == 0 {
        return Ok(());
    }
    let step = (manifest.samples.len() as f64 / n as f64).ceil() as usize;
    let sample_entries: Vec<_> = manifest.samples.iter().step_by(step).take(n).collect();

    let rgb_shape = manifest
        .rgb_shape
        .as_ref()
        .context("processed sample manifest is missing rgb_shape")?;
    let range_shape = manifest
        .range_shape
        .as_ref()
        .context("processed sample manifest is missing range_shape")?;

    let (rgb_stats, range_stats0, range_stats1, valid_frac) =
        compute_tensor_stats(&sample_entries, &processed_dir, rgb_shape, range_shape)?;

    println!();
    println!("rgb tensor statistics ({n} samples sampled):");
    print_stats(&rgb_stats);

    println!();
    println!("range channel 0 (distance) statistics ({n} samples sampled):");
    print_stats(&range_stats0);
    println!("  valid pixels: {:.1}%", valid_frac * 100.0);

    println!();
    println!("range channel 1 (intensity) statistics ({n} samples sampled):");
    print_stats(&range_stats1);

    let suspicious = scan_suspicious_samples(&manifest, &processed_dir)?;
    println!();
    println!("suspicious samples:");
    println!("  missing files:         {}", suspicious.missing_files);
    println!("  wrong tensor length:   {}", suspicious.wrong_tensor_size);
    println!("  non-finite values:     {}", suspicious.non_finite_values);
    println!("  rgb out of [0,1]:      {}", suspicious.rgb_out_of_range);
    println!("  range out of [0,1]:    {}", suspicious.range_out_of_range);
    println!("  near-empty range mask: {}", suspicious.low_valid_fraction);

    Ok(())
}

struct TensorStats {
    min: f32,
    max: f32,
    mean: f64,
    std: f64,
}

fn print_stats(s: &TensorStats) {
    println!(
        "  min: {:.4}  max: {:.4}  mean: {:.4}  std: {:.4}",
        s.min, s.max, s.mean, s.std
    );
}

fn compute_tensor_stats(
    entries: &[&sfx_core::manifest::ProcessedSampleEntry],
    processed_dir: &Path,
    rgb_shape: &sfx_core::manifest::TensorShape,
    range_shape: &sfx_core::manifest::TensorShape,
) -> Result<(TensorStats, TensorStats, TensorStats, f64)> {
    let rgb_values = rgb_shape.value_count();
    let range_pixels = range_shape.height * range_shape.width;

    let mut rgb_acc = StatsAccum::new();
    let mut range0_acc = StatsAccum::new();
    let mut range1_acc = StatsAccum::new();
    let mut valid_pixels: u64 = 0;
    let mut total_range_pixels: u64 = 0;

    for entry in entries {
        let rgb_path = processed_dir.join(&entry.meta.rgb_path);
        let bytes =
            std::fs::read(&rgb_path).with_context(|| format!("reading {}", rgb_path.display()))?;
        let values = bytes_to_f32_le(&bytes);
        if values.len() != rgb_values {
            anyhow::bail!(
                "{} has {} values; expected {}",
                rgb_path.display(),
                values.len(),
                rgb_values
            );
        }
        for &v in &values {
            rgb_acc.push(v);
        }

        let range_path = processed_dir.join(&entry.meta.range_path);
        let bytes = std::fs::read(&range_path)
            .with_context(|| format!("reading {}", range_path.display()))?;
        let values = bytes_to_f32_le(&bytes);
        // Channel 0: first range_pixels values
        for i in 0..range_pixels.min(values.len()) {
            let v = values[i];
            range0_acc.push(v);
            if v > 0.0 {
                valid_pixels += 1;
            }
            total_range_pixels += 1;
        }
        // Channel 1: next range_pixels values (if present)
        for i in range_pixels..(2 * range_pixels).min(values.len()) {
            range1_acc.push(values[i]);
        }
    }

    let valid_frac = if total_range_pixels > 0 {
        valid_pixels as f64 / total_range_pixels as f64
    } else {
        0.0
    };

    Ok((
        rgb_acc.finish(),
        range0_acc.finish(),
        range1_acc.finish(),
        valid_frac,
    ))
}

struct StatsAccum {
    count: u64,
    sum: f64,
    sum_sq: f64,
    min: f32,
    max: f32,
}

impl StatsAccum {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f32::MAX,
            max: f32::MIN,
        }
    }

    fn push(&mut self, v: f32) {
        if !v.is_finite() {
            return;
        }
        self.count += 1;
        self.sum += v as f64;
        self.sum_sq += (v as f64) * (v as f64);
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }

    fn finish(self) -> TensorStats {
        if self.count == 0 {
            return TensorStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
            };
        }
        let mean = self.sum / self.count as f64;
        let variance = (self.sum_sq / self.count as f64) - mean * mean;
        TensorStats {
            min: self.min,
            max: self.max,
            mean,
            std: variance.max(0.0).sqrt(),
        }
    }
}

fn bytes_to_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// dataset preview
// ---------------------------------------------------------------------------

fn preview(args: DatasetPreviewArgs, paths: &ProjectPaths) -> Result<()> {
    let dataset_config = sfx_config::load_dataset_config(&paths.root, &args.config)
        .with_context(|| format!("loading dataset config {}", args.config.display()))?;

    let manifest_path = paths.processed_sample_manifest_path();
    if !manifest_path.exists() {
        anyhow::bail!(
            "processed sample manifest not found at {}\nhint: run `cargo xtask dataset prepare` first",
            display_from_root(&paths.root, &manifest_path)
        );
    }
    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)?;

    let processed_dir = resolve_processed_dir(args.dataset.as_deref(), &dataset_config);

    let rgb_shape = manifest
        .rgb_shape
        .as_ref()
        .context("processed sample manifest is missing rgb_shape")?
        .clone();
    let range_shape = manifest
        .range_shape
        .as_ref()
        .context("processed sample manifest is missing range_shape")?
        .clone();

    // Filter by requested split string ("train", "val", "test", or "all")
    let target_split = args.split.trim().to_ascii_lowercase();
    let selected: Vec<_> = manifest
        .samples
        .iter()
        .filter(|s| {
            target_split == "all"
                || matches!(
                    (&s.meta.split, target_split.as_str()),
                    (sfx_core::manifest::Split::Train, "train")
                        | (sfx_core::manifest::Split::Val, "val")
                        | (sfx_core::manifest::Split::Test, "test")
                )
        })
        .take(args.count)
        .collect();

    if selected.is_empty() {
        anyhow::bail!("no samples found for split `{}`", args.split);
    }

    let out_dir = paths.root.join(&args.out);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let cols = 4usize;
    let rows = selected.len().div_ceil(cols);

    let rgb_grid_path = out_dir.join("rgb_grid.png");
    let range_grid_path = out_dir.join("range_grid.png");

    let rgb_tile_w = rgb_shape.width as u32;
    let rgb_tile_h = rgb_shape.height as u32;
    let range_tile_w = range_shape.width as u32;
    let range_tile_h = range_shape.height as u32;

    let mut rgb_canvas = image::RgbImage::new(cols as u32 * rgb_tile_w, rows as u32 * rgb_tile_h);
    let mut range_canvas =
        image::GrayImage::new(cols as u32 * range_tile_w, rows as u32 * range_tile_h);

    let mut loaded = 0usize;
    for (idx, entry) in selected.iter().enumerate() {
        let col = (idx % cols) as u32;
        let row = (idx / cols) as u32;

        // RGB tile
        let mut rgb_loaded = false;
        if let Some(preview_path) = &entry.preview_rgb_path {
            let src = processed_dir.join(preview_path);
            if src.exists() {
                match image::open(&src) {
                    Ok(img) => {
                        let tile = img.to_rgb8();
                        let tile = image::imageops::resize(
                            &tile,
                            rgb_tile_w,
                            rgb_tile_h,
                            image::imageops::FilterType::Nearest,
                        );
                        image::imageops::replace(
                            &mut rgb_canvas,
                            &tile,
                            (col * rgb_tile_w) as i64,
                            (row * rgb_tile_h) as i64,
                        );
                        loaded += 1;
                        rgb_loaded = true;
                    }
                    Err(e) => eprintln!("warn: could not load {}: {e}", src.display()),
                }
            }
        }
        if !rgb_loaded {
            let src = processed_dir.join(&entry.meta.rgb_path);
            match render_rgb_from_tensor(&src, &rgb_shape) {
                Ok(tile) => {
                    image::imageops::replace(
                        &mut rgb_canvas,
                        &tile,
                        (col * rgb_tile_w) as i64,
                        (row * rgb_tile_h) as i64,
                    );
                    loaded += 1;
                }
                Err(e) => eprintln!("warn: could not render RGB tensor {}: {e}", src.display()),
            }
        }

        // Range tile
        let mut range_loaded = false;
        if let Some(preview_path) = &entry.preview_range_path {
            let src = processed_dir.join(preview_path);
            if src.exists() {
                match image::open(&src) {
                    Ok(img) => {
                        let tile = img.to_luma8();
                        let tile = image::imageops::resize(
                            &tile,
                            range_tile_w,
                            range_tile_h,
                            image::imageops::FilterType::Nearest,
                        );
                        image::imageops::replace(
                            &mut range_canvas,
                            &tile,
                            (col * range_tile_w) as i64,
                            (row * range_tile_h) as i64,
                        );
                        range_loaded = true;
                    }
                    Err(e) => eprintln!("warn: could not load {}: {e}", src.display()),
                }
            }
        }
        if !range_loaded {
            let src = processed_dir.join(&entry.meta.range_path);
            match render_range_from_tensor(&src, &range_shape) {
                Ok(tile) => {
                    image::imageops::replace(
                        &mut range_canvas,
                        &tile,
                        (col * range_tile_w) as i64,
                        (row * range_tile_h) as i64,
                    );
                }
                Err(e) => {
                    eprintln!("warn: could not render range tensor {}: {e}", src.display())
                }
            }
        }
    }

    rgb_canvas
        .save(&rgb_grid_path)
        .with_context(|| format!("saving {}", rgb_grid_path.display()))?;
    range_canvas
        .save(&range_grid_path)
        .with_context(|| format!("saving {}", range_grid_path.display()))?;

    println!("loaded {loaded} tile(s) from {} samples", selected.len());
    println!("ok   {}", display_from_root(&paths.root, &rgb_grid_path));
    println!("ok   {}", display_from_root(&paths.root, &range_grid_path));
    Ok(())
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

    #[test]
    fn pair_component_parquet_files_matches_on_file_name() {
        let camera_files = vec![
            PathBuf::from("raw/training/camera_image/a.parquet"),
            PathBuf::from("raw/training/camera_image/b.parquet"),
        ];
        let lidar_files = vec![
            PathBuf::from("raw/training/lidar/b.parquet"),
            PathBuf::from("raw/training/lidar/c.parquet"),
        ];

        let pairs = pair_component_parquet_files(&camera_files, &lidar_files);

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, PathBuf::from("raw/training/camera_image/b.parquet"));
        assert_eq!(pairs[0].1, PathBuf::from("raw/training/lidar/b.parquet"));
    }

    #[test]
    fn manifest_local_path_strips_workspace_root() {
        let root = PathBuf::from("D:/repos/sensor-fusion-experiment");
        let path = root.join("data/raw/waymo/training/camera_image/a.parquet");

        let local = manifest_local_path(&root, &path);

        assert_eq!(
            local,
            PathBuf::from("data/raw/waymo/training/camera_image/a.parquet")
        );
    }

    #[test]
    fn parse_range_channels_accepts_aliases() {
        let channels = parse_range_channels(&[
            "distance".to_string(),
            "intensity".to_string(),
            "valid".to_string(),
        ])
        .unwrap();

        assert_eq!(channels.len(), 3);
        assert!(matches!(
            channels[0],
            sfx_preprocess::RangeChannel::Range
        ));
        assert!(matches!(
            channels[2],
            sfx_preprocess::RangeChannel::ValidityMask
        ));
    }

    #[test]
    fn render_rgb_from_tensor_uses_chw_layout() {
        let root = temp_root("render_rgb_from_tensor_uses_chw_layout");
        let path = root.join("rgb.f32.bin");
        let values = vec![
            1.0, 0.0, // R
            0.0, 1.0, // G
            0.0, 0.0, // B
        ];
        write_f32_file(&path, &values);

        let img = render_rgb_from_tensor(
            &path,
            &sfx_core::manifest::TensorShape {
                channels: 3,
                height: 1,
                width: 2,
            },
        )
        .unwrap();

        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0]);
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

    fn write_f32_file(path: &Path, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }
}
