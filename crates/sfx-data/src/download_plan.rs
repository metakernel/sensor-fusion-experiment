use crate::manifest::{
    DOWNLOAD_PLAN_SCHEMA_VERSION, DownloadPlanManifest, DownloadPlanRecord, DownloadPlanTargets,
    DownloadPlanWarning, RawFileRecord, RawFilesManifest, SplitName, ValidateManifest,
};
use chrono::Utc;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DownloadPlanOptions {
    pub source_manifest_path: Option<PathBuf>,
    pub targets: DownloadPlanTargets,
    pub limit: Option<usize>,
}

pub fn plan_downloads(
    raw_manifest: &RawFilesManifest,
    options: DownloadPlanOptions,
) -> Result<DownloadPlanManifest, crate::manifest::ManifestValidationError> {
    raw_manifest.validate()?;

    let mut grouped: BTreeMap<SplitName, Vec<&RawFileRecord>> = BTreeMap::new();
    let mut warnings = Vec::new();

    for file in &raw_manifest.files {
        match infer_split(file) {
            Some(split) => grouped.entry(split).or_default().push(file),
            None => warnings.push(DownloadPlanWarning {
                split: None,
                requested: None,
                available: 0,
                message: format!(
                    "skipped raw file with unrecognized split prefix: {}",
                    file.object_path
                ),
            }),
        }
    }

    for files in grouped.values_mut() {
        files.sort_by(|left, right| {
            left.object_path
                .cmp(&right.object_path)
                .then_with(|| left.object_uri.cmp(&right.object_uri))
        });
    }

    let mut selected = Vec::new();
    for split in [SplitName::Train, SplitName::Val, SplitName::Test] {
        let available = grouped.get(&split).map(Vec::len).unwrap_or_default();
        let requested = target_for_split(&options.targets, split);
        let take_count = requested.unwrap_or(available).min(available);

        if requested.is_some_and(|count| count > available) {
            warnings.push(DownloadPlanWarning {
                split: Some(split),
                requested,
                available,
                message: format!(
                    "requested {} {} file(s), but only {available} available",
                    requested.unwrap_or_default(),
                    split_label(split)
                ),
            });
        }

        if let Some(files) = grouped.get(&split) {
            selected.extend(
                files
                    .iter()
                    .take(take_count)
                    .copied()
                    .map(|file| (split, file)),
            );
        }
    }

    if let Some(limit) = options.limit {
        selected.truncate(limit);
    }

    let files = selected
        .into_iter()
        .enumerate()
        .map(|(plan_index, (split, file))| DownloadPlanRecord {
            plan_index,
            split,
            object_uri: file.object_uri.clone(),
            object_path: file.object_path.clone(),
            size_bytes: file.size_bytes,
        })
        .collect();

    let manifest = DownloadPlanManifest {
        schema_version: DOWNLOAD_PLAN_SCHEMA_VERSION,
        generated_at: Utc::now(),
        source_manifest_path: options.source_manifest_path,
        source_manifest_generated_at: raw_manifest.generated_at,
        limit: options.limit,
        targets: options.targets,
        files,
        warnings,
    };
    manifest.validate()?;
    Ok(manifest)
}

fn target_for_split(targets: &DownloadPlanTargets, split: SplitName) -> Option<usize> {
    match split {
        SplitName::Train => targets.train,
        SplitName::Val => targets.val,
        SplitName::Test => targets.test,
    }
}

fn infer_split(file: &RawFileRecord) -> Option<SplitName> {
    infer_split_from_path(&file.object_path).or_else(|| infer_split_from_path(&file.object_uri))
}

fn infer_split_from_path(path: &str) -> Option<SplitName> {
    path.split('/')
        .find_map(|segment| match segment.to_ascii_lowercase().as_str() {
            "train" | "training" => Some(SplitName::Train),
            "val" | "valid" | "validation" => Some(SplitName::Val),
            "test" | "testing" => Some(SplitName::Test),
            _ => None,
        })
}

fn split_label(split: SplitName) -> &'static str {
    match split {
        SplitName::Train => "train",
        SplitName::Val => "val",
        SplitName::Test => "test",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{RAW_FILES_SCHEMA_VERSION, SchemaVersion, Timestamp};
    use chrono::{DateTime, Utc};

    fn ts(value: &str) -> Timestamp {
        DateTime::parse_from_rfc3339(value)
            .expect("timestamp must parse")
            .with_timezone(&Utc)
    }

    fn raw(path: &str) -> RawFileRecord {
        RawFileRecord {
            object_uri: format!("gs://waymo/{path}"),
            object_path: path.to_string(),
            size_bytes: 42,
            discovered_at: ts("2025-01-01T00:00:00Z"),
            source_updated_at: None,
        }
    }

    fn manifest(paths: &[&str]) -> RawFilesManifest {
        RawFilesManifest {
            schema_version: RAW_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: paths.iter().map(|path| raw(path)).collect(),
        }
    }

    #[test]
    fn selects_requested_counts_by_split() {
        let plan = plan_downloads(
            &manifest(&[
                "training/a.tfrecord",
                "training/b.tfrecord",
                "validation/a.tfrecord",
                "testing/a.tfrecord",
                "testing/b.tfrecord",
            ]),
            DownloadPlanOptions {
                targets: DownloadPlanTargets {
                    train: Some(1),
                    val: Some(1),
                    test: Some(2),
                },
                ..Default::default()
            },
        )
        .expect("plan succeeds");

        assert_eq!(
            plan.files
                .iter()
                .map(|file| (file.split, file.object_path.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (SplitName::Train, "training/a.tfrecord"),
                (SplitName::Val, "validation/a.tfrecord"),
                (SplitName::Test, "testing/a.tfrecord"),
                (SplitName::Test, "testing/b.tfrecord"),
            ]
        );
    }

    #[test]
    fn records_warning_when_split_has_insufficient_files() {
        let plan = plan_downloads(
            &manifest(&["training/a.tfrecord"]),
            DownloadPlanOptions {
                targets: DownloadPlanTargets {
                    train: Some(2),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("plan succeeds with warning");

        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.warnings.len(), 1);
        assert_eq!(plan.warnings[0].split, Some(SplitName::Train));
        assert_eq!(plan.warnings[0].requested, Some(2));
        assert_eq!(plan.warnings[0].available, 1);
    }

    #[test]
    fn uses_stable_lexical_order_before_applying_limit() {
        let plan = plan_downloads(
            &manifest(&[
                "testing/z.tfrecord",
                "training/c.tfrecord",
                "training/a.tfrecord",
                "validation/b.tfrecord",
                "training/b.tfrecord",
            ]),
            DownloadPlanOptions {
                targets: DownloadPlanTargets {
                    train: Some(3),
                    val: Some(1),
                    test: Some(1),
                },
                limit: Some(3),
                ..Default::default()
            },
        )
        .expect("plan succeeds");

        assert_eq!(
            plan.files
                .iter()
                .map(|file| (file.plan_index, file.object_path.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (0, "training/a.tfrecord"),
                (1, "training/b.tfrecord"),
                (2, "training/c.tfrecord"),
            ]
        );
    }

    #[test]
    fn rejects_invalid_raw_manifest_before_planning() {
        let mut raw_manifest = manifest(&["training/a.tfrecord"]);
        raw_manifest.schema_version = SchemaVersion(99);

        let error = plan_downloads(&raw_manifest, DownloadPlanOptions::default())
            .expect_err("wrong raw schema should fail planning");

        assert_eq!(error.kind, crate::manifest::ManifestKind::RawFiles);
    }
}
