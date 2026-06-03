use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

pub type Result<T> = std::result::Result<T, ManifestError>;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create parent directory for {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to encode {path}: {source}")]
    Encode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid {manifest}: {message}")]
    Invalid {
        manifest: &'static str,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceSplit {
    Training,
    Validation,
    Testing,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SampleId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub fn value_count(&self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn validate(&self, manifest: &'static str, label: &str) -> Result<()> {
        ensure(
            manifest,
            self.channels > 0,
            format!("{label}.channels must be greater than zero"),
        )?;
        ensure(
            manifest,
            self.height > 0,
            format!("{label}.height must be greater than zero"),
        )?;
        ensure(
            manifest,
            self.width > 0,
            format!("{label}.width must be greater than zero"),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalSampleMeta {
    pub id: SampleId,
    pub split: Split,
    pub rgb_path: PathBuf,
    pub range_path: PathBuf,
    pub timestamp_micros: i64,
    pub source_segment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawFileManifest {
    pub schema_version: u32,
    pub dataset: Option<String>,
    pub files: Vec<RawFileEntry>,
}

impl Default for RawFileManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: None,
            files: Vec::new(),
        }
    }
}

impl RawFileManifest {
    pub fn validate(&self) -> Result<()> {
        validate_schema("raw_files", self.schema_version)?;
        validate_optional_name("raw_files", "dataset", self.dataset.as_deref())?;
        for (index, file) in self.files.iter().enumerate() {
            let label = format!("files[{index}]");
            validate_text("raw_files", &format!("{label}.uri"), &file.uri)?;
            validate_text("raw_files", &format!("{label}.file_name"), &file.file_name)?;
            validate_optional_name(
                "raw_files",
                &format!("{label}.checksum"),
                file.checksum.as_deref(),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawFileEntry {
    pub uri: String,
    pub split: SourceSplit,
    pub file_name: String,
    pub size_bytes: Option<u64>,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadedFileManifest {
    pub schema_version: u32,
    pub dataset: Option<String>,
    pub files: Vec<DownloadedFileEntry>,
}

impl Default for DownloadedFileManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: None,
            files: Vec::new(),
        }
    }
}

impl DownloadedFileManifest {
    pub fn validate(&self) -> Result<()> {
        validate_schema("downloaded_files", self.schema_version)?;
        validate_optional_name("downloaded_files", "dataset", self.dataset.as_deref())?;
        for (index, file) in self.files.iter().enumerate() {
            let label = format!("files[{index}]");
            validate_text(
                "downloaded_files",
                &format!("{label}.source_uri"),
                &file.source_uri,
            )?;
            validate_path(
                "downloaded_files",
                &format!("{label}.local_path"),
                &file.local_path,
            )?;
            validate_optional_name(
                "downloaded_files",
                &format!("{label}.error"),
                file.error.as_deref(),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadedFileEntry {
    pub source_uri: String,
    pub local_path: PathBuf,
    pub split: SourceSplit,
    pub size_bytes: Option<u64>,
    pub status: DownloadStatus,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DownloadStatus {
    Planned,
    InProgress,
    Downloaded,
    Verified,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessedSampleManifest {
    pub schema_version: u32,
    pub dataset: Option<String>,
    pub rgb_shape: Option<TensorShape>,
    pub range_shape: Option<TensorShape>,
    pub samples: Vec<ProcessedSampleEntry>,
}

impl Default for ProcessedSampleManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: None,
            rgb_shape: None,
            range_shape: None,
            samples: Vec::new(),
        }
    }
}

impl ProcessedSampleManifest {
    pub fn validate(&self) -> Result<()> {
        validate_schema("processed_samples", self.schema_version)?;
        validate_optional_name("processed_samples", "dataset", self.dataset.as_deref())?;
        if let Some(shape) = &self.rgb_shape {
            shape.validate("processed_samples", "rgb_shape")?;
        }
        if let Some(shape) = &self.range_shape {
            shape.validate("processed_samples", "range_shape")?;
        }

        let mut seen = BTreeSet::new();
        for (index, sample) in self.samples.iter().enumerate() {
            let label = format!("samples[{index}]");
            validate_sample_id(
                "processed_samples",
                &format!("{label}.meta.id"),
                &sample.meta.id,
            )?;
            ensure(
                "processed_samples",
                seen.insert(sample.meta.id.clone()),
                format!("duplicate sample id {}", sample.meta.id.0),
            )?;
            validate_path(
                "processed_samples",
                &format!("{label}.meta.rgb_path"),
                &sample.meta.rgb_path,
            )?;
            validate_path(
                "processed_samples",
                &format!("{label}.meta.range_path"),
                &sample.meta.range_path,
            )?;
            validate_path(
                "processed_samples",
                &format!("{label}.meta_path"),
                &sample.meta_path,
            )?;
            validate_text(
                "processed_samples",
                &format!("{label}.meta.source_segment"),
                &sample.meta.source_segment,
            )?;
            if let Some(path) = &sample.preview_rgb_path {
                validate_path(
                    "processed_samples",
                    &format!("{label}.preview_rgb_path"),
                    path,
                )?;
            }
            if let Some(path) = &sample.preview_range_path {
                validate_path(
                    "processed_samples",
                    &format!("{label}.preview_range_path"),
                    path,
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionSummaryManifest {
    pub schema_version: u32,
    pub dataset: Option<String>,
    pub files: Vec<ExtractionSummaryEntry>,
}

impl Default for ExtractionSummaryManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: None,
            files: Vec::new(),
        }
    }
}

impl ExtractionSummaryManifest {
    pub fn validate(&self) -> Result<()> {
        validate_schema("extraction_summary", self.schema_version)?;
        validate_optional_name("extraction_summary", "dataset", self.dataset.as_deref())?;
        for (index, file) in self.files.iter().enumerate() {
            let label = format!("files[{index}]");
            validate_path(
                "extraction_summary",
                &format!("{label}.camera_parquet_path"),
                &file.camera_parquet_path,
            )?;
            validate_path(
                "extraction_summary",
                &format!("{label}.lidar_parquet_path"),
                &file.lidar_parquet_path,
            )?;
            ensure(
                "extraction_summary",
                file.matched_pairs <= file.camera_frames,
                format!(
                    "{label}.matched_pairs cannot exceed camera_frames ({} > {})",
                    file.matched_pairs, file.camera_frames
                ),
            )?;
            ensure(
                "extraction_summary",
                file.matched_pairs <= file.lidar_frames,
                format!(
                    "{label}.matched_pairs cannot exceed lidar_frames ({} > {})",
                    file.matched_pairs, file.lidar_frames
                ),
            )?;
            if let Some(id) = &file.first_sample_id {
                validate_sample_id(
                    "extraction_summary",
                    &format!("{label}.first_sample_id"),
                    id,
                )?;
            }
            if let Some(id) = &file.last_sample_id {
                validate_sample_id(
                    "extraction_summary",
                    &format!("{label}.last_sample_id"),
                    id,
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessedSampleEntry {
    pub meta: MultimodalSampleMeta,
    pub meta_path: PathBuf,
    pub preview_rgb_path: Option<PathBuf>,
    pub preview_range_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionSummaryEntry {
    pub split: Split,
    pub camera_parquet_path: PathBuf,
    pub lidar_parquet_path: PathBuf,
    pub camera_frames: usize,
    pub lidar_frames: usize,
    pub matched_pairs: usize,
    pub unmatched_camera: usize,
    pub unmatched_lidar: usize,
    pub max_abs_timestamp_delta_micros: i64,
    pub first_sample_id: Option<SampleId>,
    pub last_sample_id: Option<SampleId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitsManifest {
    pub schema_version: u32,
    pub dataset: Option<String>,
    pub train: Vec<SampleId>,
    pub val: Vec<SampleId>,
    pub test: Vec<SampleId>,
}

impl Default for SplitsManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: None,
            train: Vec::new(),
            val: Vec::new(),
            test: Vec::new(),
        }
    }
}

impl SplitsManifest {
    pub fn validate(&self) -> Result<()> {
        validate_schema("splits", self.schema_version)?;
        validate_optional_name("splits", "dataset", self.dataset.as_deref())?;
        let mut seen = BTreeSet::new();
        validate_split_ids("splits", "train", &self.train, &mut seen)?;
        validate_split_ids("splits", "val", &self.val, &mut seen)?;
        validate_split_ids("splits", "test", &self.test, &mut seen)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIndex {
    pub schema_version: u32,
    pub runs: Vec<RunIndexEntry>,
}

impl Default for RunIndex {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            runs: Vec::new(),
        }
    }
}

impl RunIndex {
    pub fn validate(&self) -> Result<()> {
        validate_schema("run_index", self.schema_version)?;
        let mut seen = BTreeSet::new();
        for (index, run) in self.runs.iter().enumerate() {
            let label = format!("runs[{index}]");
            validate_text("run_index", &format!("{label}.run_id"), &run.run_id)?;
            ensure(
                "run_index",
                seen.insert(run.run_id.clone()),
                format!("duplicate run id {}", run.run_id),
            )?;
            validate_text("run_index", &format!("{label}.model_kind"), &run.model_kind)?;
            validate_path("run_index", &format!("{label}.path"), &run.path)?;
            validate_optional_name(
                "run_index",
                &format!("{label}.started_at"),
                run.started_at.as_deref(),
            )?;
            validate_optional_name(
                "run_index",
                &format!("{label}.completed_at"),
                run.completed_at.as_deref(),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIndexEntry {
    pub run_id: String,
    pub model_kind: String,
    pub path: PathBuf,
    pub status: RunStatus,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Planned,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatestRun {
    pub schema_version: u32,
    pub run_id: Option<String>,
    pub path: Option<PathBuf>,
}

impl Default for LatestRun {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            run_id: None,
            path: None,
        }
    }
}

impl LatestRun {
    pub fn validate(&self) -> Result<()> {
        validate_schema("latest", self.schema_version)?;
        validate_optional_name("latest", "run_id", self.run_id.as_deref())?;
        if let Some(path) = &self.path {
            validate_path("latest", "path", path)?;
        }
        Ok(())
    }
}

pub fn read_manifest<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| ManifestError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

pub fn write_manifest<T: Serialize>(path: impl AsRef<Path>, manifest: &T) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ManifestError::CreateDir {
            path: path.to_path_buf(),
            source,
        })?;
    }
    let text = serde_json::to_string_pretty(manifest).map_err(|source| ManifestError::Encode {
        path: path.to_path_buf(),
        source,
    })?;
    std::fs::write(path, format!("{text}\n")).map_err(|source| ManifestError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn validate_schema(manifest: &'static str, schema_version: u32) -> Result<()> {
    ensure(
        manifest,
        schema_version == MANIFEST_SCHEMA_VERSION,
        format!("unsupported schema_version {schema_version}"),
    )
}

fn validate_split_ids(
    manifest: &'static str,
    split: &str,
    ids: &[SampleId],
    seen: &mut BTreeSet<SampleId>,
) -> Result<()> {
    for (index, id) in ids.iter().enumerate() {
        let label = format!("{split}[{index}]");
        validate_sample_id(manifest, &label, id)?;
        ensure(
            manifest,
            seen.insert(id.clone()),
            format!("duplicate sample id {}", id.0),
        )?;
    }
    Ok(())
}

fn validate_sample_id(manifest: &'static str, label: &str, id: &SampleId) -> Result<()> {
    validate_text(manifest, label, &id.0)
}

fn validate_optional_name(manifest: &'static str, label: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        validate_text(manifest, label, value)?;
    }
    Ok(())
}

fn validate_text(manifest: &'static str, label: &str, value: &str) -> Result<()> {
    ensure(
        manifest,
        !value.trim().is_empty(),
        format!("{label} cannot be empty"),
    )
}

fn validate_path(manifest: &'static str, label: &str, path: &Path) -> Result<()> {
    ensure(
        manifest,
        !path.as_os_str().is_empty(),
        format!("{label} cannot be empty"),
    )
}

fn ensure(manifest: &'static str, condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Invalid {
            manifest,
            message: message.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_manifest_round_trip() {
        let root = temp_root("empty_manifest_round_trip");
        let path = root.join("raw_files.json");
        let manifest = RawFileManifest::default();

        write_manifest(&path, &manifest).unwrap();
        let loaded: RawFileManifest = read_manifest(&path).unwrap();

        assert_eq!(loaded, manifest);
        loaded.validate().unwrap();
    }

    #[test]
    fn splits_reject_duplicate_sample_ids() {
        let manifest = SplitsManifest {
            train: vec![SampleId("sample_0001".to_string())],
            val: vec![SampleId("sample_0001".to_string())],
            ..SplitsManifest::default()
        };

        let err = manifest.validate().unwrap_err();

        assert!(err.to_string().contains("duplicate sample id"));
    }

    #[test]
    fn processed_manifest_rejects_invalid_shape() {
        let manifest = ProcessedSampleManifest {
            rgb_shape: Some(TensorShape {
                channels: 0,
                height: 128,
                width: 256,
            }),
            ..ProcessedSampleManifest::default()
        };

        let err = manifest.validate().unwrap_err();

        assert!(err.to_string().contains("rgb_shape.channels"));
    }

    #[test]
    fn extraction_summary_rejects_invalid_match_counts() {
        let manifest = ExtractionSummaryManifest {
            files: vec![ExtractionSummaryEntry {
                split: Split::Train,
                camera_parquet_path: PathBuf::from("training/camera_image/a.parquet"),
                lidar_parquet_path: PathBuf::from("training/lidar/a.parquet"),
                camera_frames: 1,
                lidar_frames: 1,
                matched_pairs: 2,
                unmatched_camera: 0,
                unmatched_lidar: 0,
                max_abs_timestamp_delta_micros: 0,
                first_sample_id: None,
                last_sample_id: None,
            }],
            ..ExtractionSummaryManifest::default()
        };

        let err = manifest.validate().unwrap_err();

        assert!(err.to_string().contains("matched_pairs cannot exceed camera_frames"));
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("sfx-core-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
