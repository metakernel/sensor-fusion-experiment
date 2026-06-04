use crate::manifest::{
    DOWNLOAD_PLAN_SCHEMA_VERSION, DOWNLOADED_FILES_SCHEMA_VERSION, DownloadPlanManifest,
    DownloadedFilesManifest, EXTRACTED_FRAMES_SCHEMA_VERSION, ExtractedFramesManifest,
    HasSchemaVersion, LATEST_SCHEMA_VERSION, LatestManifest, ManifestDiagnostic, ManifestKind,
    PROCESSED_SAMPLES_SCHEMA_VERSION, ProcessedSamplesManifest, RAW_FILES_SCHEMA_VERSION,
    RUN_INDEX_SCHEMA_VERSION, RawFilesManifest, RunIndexManifest, SPLITS_SCHEMA_VERSION,
    SchemaVersion, SplitsManifest, ValidateManifest,
};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::{Display, Formatter};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub type Result<T> = std::result::Result<T, ManifestIoError>;

#[derive(Debug)]
pub enum ManifestIoError {
    MissingFile {
        path: PathBuf,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    Serialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    SchemaVersionMismatch {
        path: PathBuf,
        expected: SchemaVersion,
        found: SchemaVersion,
    },
}

impl Display for ManifestIoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFile { path } => {
                write!(f, "manifest file does not exist: {}", path.display())
            }
            Self::Io { path, source } => {
                write!(f, "i/o error for {}: {}", path.display(), source)
            }
            Self::Parse { path, source } => {
                write!(f, "failed to parse manifest {}: {}", path.display(), source)
            }
            Self::Serialize { path, source } => {
                write!(
                    f,
                    "failed to serialize manifest {}: {}",
                    path.display(),
                    source
                )
            }
            Self::SchemaVersionMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "schema version mismatch for {}: expected {}, found {}",
                    path.display(),
                    expected.0,
                    found.0
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestCorruptionReport {
    pub kind: ManifestKind,
    pub path: PathBuf,
    pub diagnostics: Vec<ManifestDiagnostic>,
}

impl ManifestCorruptionReport {
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

fn detect_manifest_corruption<T>(
    path: &Path,
    read_manifest: impl FnOnce(&Path) -> Result<T>,
) -> ManifestCorruptionReport
where
    T: ValidateManifest,
{
    let diagnostics = match read_manifest(path) {
        Ok(manifest) => manifest.validation_diagnostics(),
        Err(error) => vec![io_error_diagnostic(&error)],
    };

    ManifestCorruptionReport {
        kind: T::KIND,
        path: path.to_path_buf(),
        diagnostics,
    }
}

fn io_error_diagnostic(error: &ManifestIoError) -> ManifestDiagnostic {
    match error {
        ManifestIoError::MissingFile { path } => ManifestDiagnostic::new(
            "manifest_path",
            format!("manifest file is missing: {}", path.display()),
            "Regenerate the manifest by rerunning the pipeline step that owns this manifest.",
        ),
        ManifestIoError::Io { path, source } => ManifestDiagnostic::new(
            "manifest_path",
            format!("could not read manifest {}: {}", path.display(), source),
            "Check file permissions and storage health, then retry manifest validation.",
        ),
        ManifestIoError::Parse { path, source } => ManifestDiagnostic::new(
            "json",
            format!(
                "manifest JSON is malformed in {}: {}",
                path.display(),
                source
            ),
            "Restore the manifest from a known-good copy or regenerate it with the manifest writer.",
        ),
        ManifestIoError::Serialize { path, source } => ManifestDiagnostic::new(
            "json",
            format!(
                "manifest could not be serialized for {}: {}",
                path.display(),
                source
            ),
            "Report this serialization failure; it indicates an internal manifest writer bug.",
        ),
        ManifestIoError::SchemaVersionMismatch {
            path,
            expected,
            found,
        } => ManifestDiagnostic::new(
            "schema_version",
            format!(
                "schema version mismatch in {}: expected {}, found {}",
                path.display(),
                expected.0,
                found.0
            ),
            "Regenerate the manifest with the current toolchain or migrate it before use.",
        ),
    }
}

impl std::error::Error for ManifestIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::Serialize { source, .. } => Some(source),
            Self::MissingFile { .. } | Self::SchemaVersionMismatch { .. } => None,
        }
    }
}

fn read_manifest_with_version<T>(path: &Path, expected: SchemaVersion) -> Result<T>
where
    T: DeserializeOwned + HasSchemaVersion,
{
    if !path.is_file() {
        return Err(ManifestIoError::MissingFile {
            path: path.to_path_buf(),
        });
    }

    let bytes = std::fs::read(path).map_err(|source| ManifestIoError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let manifest: T = serde_json::from_slice(&bytes).map_err(|source| ManifestIoError::Parse {
        path: path.to_path_buf(),
        source,
    })?;

    let found = manifest.schema_version();
    if found != expected {
        return Err(ManifestIoError::SchemaVersionMismatch {
            path: path.to_path_buf(),
            expected,
            found,
        });
    }

    Ok(manifest)
}

fn write_manifest_atomic<T>(path: &Path, manifest: &T, expected: SchemaVersion) -> Result<()>
where
    T: Serialize + HasSchemaVersion,
{
    let found = manifest.schema_version();
    if found != expected {
        return Err(ManifestIoError::SchemaVersionMismatch {
            path: path.to_path_buf(),
            expected,
            found,
        });
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ManifestIoError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let payload =
        serde_json::to_vec_pretty(manifest).map_err(|source| ManifestIoError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
    let temp_path = temporary_manifest_path(path);

    let mut file = std::fs::File::create(&temp_path).map_err(|source| ManifestIoError::Io {
        path: temp_path.clone(),
        source,
    })?;
    file.write_all(&payload)
        .map_err(|source| ManifestIoError::Io {
            path: temp_path.clone(),
            source,
        })?;
    file.sync_all().map_err(|source| ManifestIoError::Io {
        path: temp_path.clone(),
        source,
    })?;

    if path.exists() {
        std::fs::remove_file(path).map_err(|source| ManifestIoError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }

    std::fs::rename(&temp_path, path).map_err(|source| ManifestIoError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(())
}

fn temporary_manifest_path(path: &Path) -> PathBuf {
    let stem = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("manifest.json");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.with_file_name(format!("{stem}.tmp.{nonce}"))
}

pub fn read_raw_files_manifest(path: &Path) -> Result<RawFilesManifest> {
    read_manifest_with_version(path, RAW_FILES_SCHEMA_VERSION)
}

pub fn write_raw_files_manifest(path: &Path, manifest: &RawFilesManifest) -> Result<()> {
    write_manifest_atomic(path, manifest, RAW_FILES_SCHEMA_VERSION)
}

pub fn read_download_plan_manifest(path: &Path) -> Result<DownloadPlanManifest> {
    read_manifest_with_version(path, DOWNLOAD_PLAN_SCHEMA_VERSION)
}

pub fn write_download_plan_manifest(path: &Path, manifest: &DownloadPlanManifest) -> Result<()> {
    write_manifest_atomic(path, manifest, DOWNLOAD_PLAN_SCHEMA_VERSION)
}

pub fn read_downloaded_files_manifest(path: &Path) -> Result<DownloadedFilesManifest> {
    read_manifest_with_version(path, DOWNLOADED_FILES_SCHEMA_VERSION)
}

pub fn write_downloaded_files_manifest(
    path: &Path,
    manifest: &DownloadedFilesManifest,
) -> Result<()> {
    write_manifest_atomic(path, manifest, DOWNLOADED_FILES_SCHEMA_VERSION)
}

pub fn read_extracted_frames_manifest(path: &Path) -> Result<ExtractedFramesManifest> {
    read_manifest_with_version(path, EXTRACTED_FRAMES_SCHEMA_VERSION)
}

pub fn write_extracted_frames_manifest(
    path: &Path,
    manifest: &ExtractedFramesManifest,
) -> Result<()> {
    write_manifest_atomic(path, manifest, EXTRACTED_FRAMES_SCHEMA_VERSION)
}

pub fn read_processed_samples_manifest(path: &Path) -> Result<ProcessedSamplesManifest> {
    read_manifest_with_version(path, PROCESSED_SAMPLES_SCHEMA_VERSION)
}

pub fn write_processed_samples_manifest(
    path: &Path,
    manifest: &ProcessedSamplesManifest,
) -> Result<()> {
    write_manifest_atomic(path, manifest, PROCESSED_SAMPLES_SCHEMA_VERSION)
}

pub fn read_splits_manifest(path: &Path) -> Result<SplitsManifest> {
    read_manifest_with_version(path, SPLITS_SCHEMA_VERSION)
}

pub fn write_splits_manifest(path: &Path, manifest: &SplitsManifest) -> Result<()> {
    write_manifest_atomic(path, manifest, SPLITS_SCHEMA_VERSION)
}

pub fn read_run_index_manifest(path: &Path) -> Result<RunIndexManifest> {
    read_manifest_with_version(path, RUN_INDEX_SCHEMA_VERSION)
}

pub fn write_run_index_manifest(path: &Path, manifest: &RunIndexManifest) -> Result<()> {
    write_manifest_atomic(path, manifest, RUN_INDEX_SCHEMA_VERSION)
}

pub fn read_latest_manifest(path: &Path) -> Result<LatestManifest> {
    read_manifest_with_version(path, LATEST_SCHEMA_VERSION)
}

pub fn write_latest_manifest(path: &Path, manifest: &LatestManifest) -> Result<()> {
    write_manifest_atomic(path, manifest, LATEST_SCHEMA_VERSION)
}

pub fn detect_raw_files_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_raw_files_manifest)
}

pub fn detect_download_plan_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_download_plan_manifest)
}

pub fn detect_downloaded_files_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_downloaded_files_manifest)
}

pub fn detect_extracted_frames_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_extracted_frames_manifest)
}

pub fn detect_processed_samples_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_processed_samples_manifest)
}

pub fn detect_splits_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_splits_manifest)
}

pub fn detect_run_index_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_run_index_manifest)
}

pub fn detect_latest_manifest_corruption(path: &Path) -> ManifestCorruptionReport {
    detect_manifest_corruption(path, read_latest_manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        ArtifactReference, ExtractedFrameRecord, ExtractionProvenance, RawFileRecord,
        SourceSegmentReference, SplitName, Timestamp,
    };
    use chrono::{DateTime, Utc};
    use tempfile::TempDir;

    fn ts(value: &str) -> Timestamp {
        DateTime::parse_from_rfc3339(value)
            .expect("timestamp must parse")
            .with_timezone(&Utc)
    }

    fn sample_raw_manifest(schema_version: SchemaVersion) -> RawFilesManifest {
        RawFilesManifest {
            schema_version,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![RawFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: "training/file-0001.tfrecord".to_string(),
                size_bytes: 42,
                discovered_at: ts("2025-01-01T00:01:00Z"),
                source_updated_at: None,
            }],
        }
    }

    fn sample_extracted_manifest(schema_version: SchemaVersion) -> ExtractedFramesManifest {
        ExtractedFramesManifest {
            schema_version,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: crate::manifest::extraction_output_layout("data/intermediate/extracted"),
            frames: vec![ExtractedFrameRecord {
                sample_id: "sample-001".to_string(),
                frame_id: "segment-001:000003".to_string(),
                split: SplitName::Train,
                frame_index: 3,
                timestamp_micros: 1_735_689_600_123_456,
                source: SourceSegmentReference {
                    segment_id: "segment-001".to_string(),
                    object_uri: Some("gs://wod/training/segment-001.tfrecord".to_string()),
                    object_path: "training/segment-001.tfrecord".to_string(),
                    local_path: Some(PathBuf::from(
                        "data/raw/waymo/training/segment-001.tfrecord",
                    )),
                    file_index: Some(0),
                    sha256: Some("a".repeat(64)),
                },
                rgb: ArtifactReference {
                    path: PathBuf::from("data/intermediate/extracted/rgb/sample-001.jpg"),
                    media_type: "image/jpeg".to_string(),
                    encoding: "jpeg".to_string(),
                    width: 1920,
                    height: 1280,
                    channels: Some(3),
                    size_bytes: Some(100),
                    sha256: Some("b".repeat(64)),
                },
                range: ArtifactReference {
                    path: PathBuf::from("data/intermediate/extracted/range/sample-001.npz"),
                    media_type: "application/x-npz".to_string(),
                    encoding: "npz:f32".to_string(),
                    width: 2650,
                    height: 64,
                    channels: Some(1),
                    size_bytes: Some(200),
                    sha256: Some("c".repeat(64)),
                },
                provenance: ExtractionProvenance {
                    extractor_name: "sfx-waymo-extractor".to_string(),
                    extractor_version: "0.1.0".to_string(),
                    extracted_at: ts("2025-01-01T00:03:00Z"),
                    config_hash: Some("config-v1".to_string()),
                    command: Some("sfx waymo extract".to_string()),
                },
            }],
        }
    }

    #[test]
    fn writes_and_reads_raw_manifest_round_trip() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        let expected = sample_raw_manifest(RAW_FILES_SCHEMA_VERSION);

        write_raw_files_manifest(&manifest_path, &expected).expect("manifest should write");
        let actual = read_raw_files_manifest(&manifest_path).expect("manifest should read");

        assert_eq!(actual, expected);
    }

    #[test]
    fn writes_and_reads_extracted_frames_manifest_round_trip() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("extracted_frames.json");
        let expected = sample_extracted_manifest(EXTRACTED_FRAMES_SCHEMA_VERSION);

        write_extracted_frames_manifest(&manifest_path, &expected).expect("manifest should write");
        let actual = read_extracted_frames_manifest(&manifest_path).expect("manifest should read");

        assert_eq!(actual, expected);
    }

    #[test]
    fn write_creates_parent_directories() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir
            .path()
            .join("nested")
            .join("manifests")
            .join("raw_files.json");
        let manifest = sample_raw_manifest(RAW_FILES_SCHEMA_VERSION);

        write_raw_files_manifest(&manifest_path, &manifest).expect("manifest should write");

        assert!(manifest_path.is_file());
    }

    #[test]
    fn read_returns_missing_file_error() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("missing.json");

        let error = read_raw_files_manifest(&manifest_path).expect_err("missing path should fail");
        match error {
            ManifestIoError::MissingFile { path } => assert_eq!(path, manifest_path),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn read_returns_parse_error_for_malformed_json() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        std::fs::write(&manifest_path, "{ not valid json").expect("fixture should write");

        let error =
            read_raw_files_manifest(&manifest_path).expect_err("malformed json should fail");
        match error {
            ManifestIoError::Parse { path, .. } => assert_eq!(path, manifest_path),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn read_returns_schema_version_mismatch_for_wrong_version() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        let wrong_manifest = sample_raw_manifest(SchemaVersion(99));
        let payload = serde_json::to_vec_pretty(&wrong_manifest).expect("fixture should serialize");
        std::fs::write(&manifest_path, payload).expect("fixture should write");

        let error = read_raw_files_manifest(&manifest_path).expect_err("wrong version should fail");
        match error {
            ManifestIoError::SchemaVersionMismatch {
                path,
                expected,
                found,
            } => {
                assert_eq!(path, manifest_path);
                assert_eq!(expected, RAW_FILES_SCHEMA_VERSION);
                assert_eq!(found, SchemaVersion(99));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn corruption_detection_reports_valid_manifest_without_diagnostics() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        let manifest = sample_raw_manifest(RAW_FILES_SCHEMA_VERSION);
        write_raw_files_manifest(&manifest_path, &manifest).expect("manifest should write");

        let report = detect_raw_files_manifest_corruption(&manifest_path);

        assert_eq!(report.kind, ManifestKind::RawFiles);
        assert_eq!(report.path, manifest_path);
        assert!(report.is_valid());
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn corruption_detection_reports_parse_errors_with_hints() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        std::fs::write(&manifest_path, "{ not valid json").expect("fixture should write");

        let report = detect_raw_files_manifest_corruption(&manifest_path);

        assert!(!report.is_valid());
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.diagnostics[0].location, "json");
        assert!(report.diagnostics[0].message.contains("malformed"));
        assert!(!report.diagnostics[0].hint.is_empty());
    }

    #[test]
    fn corruption_detection_reports_semantic_validation_errors() {
        let temp_dir = TempDir::new().expect("tempdir should be created");
        let manifest_path = temp_dir.path().join("raw_files.json");
        let mut manifest = sample_raw_manifest(RAW_FILES_SCHEMA_VERSION);
        manifest.files[0].object_uri.clear();
        manifest.files[0].size_bytes = 0;
        let payload = serde_json::to_vec_pretty(&manifest).expect("fixture should serialize");
        std::fs::write(&manifest_path, payload).expect("fixture should write");

        let report = detect_raw_files_manifest_corruption(&manifest_path);

        assert!(!report.is_valid());
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("object_uri"))
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("size_bytes"))
        );
        assert!(
            report
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.hint.is_empty())
        );
    }
}
