use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

pub type Timestamp = DateTime<Utc>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    pub const V1: Self = Self(1);
}

pub const RAW_FILES_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const DOWNLOAD_PLAN_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const DOWNLOADED_FILES_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const EXTRACTED_FRAMES_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const PROCESSED_SAMPLES_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const SPLITS_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const RUN_INDEX_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;
pub const LATEST_SCHEMA_VERSION: SchemaVersion = SchemaVersion::V1;

pub trait HasSchemaVersion {
    fn schema_version(&self) -> SchemaVersion;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawFilesManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub files: Vec<RawFileRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawFileRecord {
    pub object_uri: String,
    pub object_path: String,
    pub size_bytes: u64,
    pub discovered_at: Timestamp,
    pub source_updated_at: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadPlanManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub source_manifest_path: Option<PathBuf>,
    pub source_manifest_generated_at: Timestamp,
    pub limit: Option<usize>,
    pub targets: DownloadPlanTargets,
    pub files: Vec<DownloadPlanRecord>,
    pub warnings: Vec<DownloadPlanWarning>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadPlanTargets {
    pub train: Option<usize>,
    pub val: Option<usize>,
    pub test: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadPlanRecord {
    pub plan_index: usize,
    pub split: SplitName,
    pub object_uri: String,
    pub object_path: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadPlanWarning {
    pub split: Option<SplitName>,
    pub requested: Option<usize>,
    pub available: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadedFilesManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub files: Vec<DownloadedFileRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadedFileRecord {
    pub object_uri: String,
    pub object_path: String,
    pub local_path: PathBuf,
    pub split: SplitName,
    pub size_bytes: u64,
    pub status: DownloadedFileStatus,
    pub bytes_transferred: u64,
    pub sha256: Option<String>,
    pub downloaded_at: Timestamp,
    pub error: Option<String>,
}

pub const EXTRACTED_FRAMES_MANIFEST_FILE: &str = "extracted_frames.json";
pub const EXTRACTED_RGB_DIR: &str = "rgb";
pub const EXTRACTED_RANGE_DIR: &str = "range";
pub const PROCESSED_SAMPLES_MANIFEST_FILE: &str = "processed_samples.json";
pub const PROCESSED_RGB_TENSOR_DIR: &str = "tensors/rgb";
pub const PROCESSED_RANGE_TENSOR_DIR: &str = "tensors/range";
pub const PROCESSED_PREVIEW_DIR: &str = "previews";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractedFramesManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub output_layout: ExtractionOutputLayout,
    pub frames: Vec<ExtractedFrameRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionOutputLayout {
    pub root_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub rgb_dir: PathBuf,
    pub range_dir: PathBuf,
}

impl ExtractionOutputLayout {
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        let root_dir = root_dir.into();
        Self {
            manifest_path: root_dir.join(EXTRACTED_FRAMES_MANIFEST_FILE),
            rgb_dir: root_dir.join(EXTRACTED_RGB_DIR),
            range_dir: root_dir.join(EXTRACTED_RANGE_DIR),
            root_dir,
        }
    }

    pub fn rgb_artifact_path(&self, sample_id: &str, extension: &str) -> PathBuf {
        self.rgb_dir.join(format!(
            "{}.{}",
            sample_id,
            extension.trim_start_matches('.')
        ))
    }

    pub fn range_artifact_path(&self, sample_id: &str, extension: &str) -> PathBuf {
        self.range_dir.join(format!(
            "{}.{}",
            sample_id,
            extension.trim_start_matches('.')
        ))
    }
}

pub fn extraction_output_layout(root_dir: impl Into<PathBuf>) -> ExtractionOutputLayout {
    ExtractionOutputLayout::new(root_dir)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractedFrameRecord {
    pub sample_id: String,
    pub frame_id: String,
    pub split: SplitName,
    pub frame_index: u32,
    pub timestamp_micros: i64,
    pub source: SourceSegmentReference,
    pub rgb: ArtifactReference,
    pub range: ArtifactReference,
    pub provenance: ExtractionProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSegmentReference {
    pub segment_id: String,
    pub object_uri: Option<String>,
    pub object_path: String,
    pub local_path: Option<PathBuf>,
    pub file_index: Option<usize>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactReference {
    pub path: PathBuf,
    pub media_type: String,
    pub encoding: String,
    pub width: u32,
    pub height: u32,
    pub channels: Option<u8>,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionProvenance {
    pub extractor_name: String,
    pub extractor_version: String,
    pub extracted_at: Timestamp,
    pub config_hash: Option<String>,
    pub command: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadedFileStatus {
    Downloaded,
    Skipped,
    Resumed,
    Replaced,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessedSamplesManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub output_layout: ProcessedDatasetOutputLayout,
    pub source_extracted_frames: Option<ManifestReference>,
    pub samples: Vec<ProcessedSampleRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessedDatasetOutputLayout {
    pub root_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub rgb_tensor_dir: PathBuf,
    pub range_tensor_dir: PathBuf,
    pub preview_dir: PathBuf,
}

impl ProcessedDatasetOutputLayout {
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        let root_dir = root_dir.into();
        Self {
            manifest_path: root_dir.join(PROCESSED_SAMPLES_MANIFEST_FILE),
            rgb_tensor_dir: root_dir.join(PROCESSED_RGB_TENSOR_DIR),
            range_tensor_dir: root_dir.join(PROCESSED_RANGE_TENSOR_DIR),
            preview_dir: root_dir.join(PROCESSED_PREVIEW_DIR),
            root_dir,
        }
    }

    pub fn rgb_tensor_path(&self, sample_id: &str, extension: &str) -> PathBuf {
        self.rgb_tensor_dir.join(format!(
            "{}.{}",
            sample_id,
            extension.trim_start_matches('.')
        ))
    }

    pub fn range_tensor_path(&self, sample_id: &str, extension: &str) -> PathBuf {
        self.range_tensor_dir.join(format!(
            "{}.{}",
            sample_id,
            extension.trim_start_matches('.')
        ))
    }

    pub fn preview_path(
        &self,
        sample_id: &str,
        kind: ProcessedPreviewKind,
        extension: &str,
    ) -> PathBuf {
        self.preview_dir.join(kind.as_str()).join(format!(
            "{}.{}",
            sample_id,
            extension.trim_start_matches('.')
        ))
    }
}

pub fn processed_dataset_output_layout(
    root_dir: impl Into<PathBuf>,
) -> ProcessedDatasetOutputLayout {
    ProcessedDatasetOutputLayout::new(root_dir)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessedSampleRecord {
    pub sample_id: String,
    pub split: SplitName,
    pub rgb: TensorArtifactReference,
    pub range: TensorArtifactReference,
    pub metadata: ProcessedSampleMetadata,
    pub previews: Vec<ProcessedPreviewArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorArtifactReference {
    pub path: PathBuf,
    pub shape: TensorShape,
    pub dtype: TensorDType,
    pub layout: TensorLayout,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorShape {
    pub dimensions: Vec<u32>,
}

impl TensorShape {
    pub fn new(dimensions: impl Into<Vec<u32>>) -> Self {
        Self {
            dimensions: dimensions.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    F16,
    F32,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    Hwc,
    Chw,
    Hw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessedSampleMetadata {
    pub source_sample_id: String,
    pub frame_id: String,
    pub frame_index: u32,
    pub timestamp_micros: i64,
    pub source: SourceSegmentReference,
    pub extracted_rgb_path: PathBuf,
    pub extracted_range_path: PathBuf,
    pub provenance: ProcessingProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessingProvenance {
    pub preprocessor_name: String,
    pub preprocessor_version: String,
    pub processed_at: Timestamp,
    pub config_hash: Option<String>,
    pub command: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessedPreviewArtifact {
    pub kind: ProcessedPreviewKind,
    pub path: PathBuf,
    pub media_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessedPreviewKind {
    Rgb,
    Range,
    Overlay,
}

impl ProcessedPreviewKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Range => "range",
            Self::Overlay => "overlay",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitsManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub split_seed: u64,
    pub assignments: Vec<SplitAssignment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitName {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitAssignment {
    pub sample_id: String,
    pub split: SplitName,
    pub assigned_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunIndexManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub runs: Vec<RunIndexEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    Fetch,
    Preprocess,
    Split,
    Train,
    Eval,
    Inspect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunIndexEntry {
    pub run_id: String,
    pub run_kind: RunKind,
    pub status: RunStatus,
    pub started_at: Timestamp,
    pub finished_at: Option<Timestamp>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestReference {
    pub manifest_path: PathBuf,
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatestManifest {
    pub schema_version: SchemaVersion,
    pub generated_at: Timestamp,
    pub raw_files: Option<ManifestReference>,
    pub downloaded_files: Option<ManifestReference>,
    #[serde(default)]
    pub extracted_frames: Option<ManifestReference>,
    pub processed_samples: Option<ManifestReference>,
    pub splits: Option<ManifestReference>,
    pub run_index: Option<ManifestReference>,
}

macro_rules! impl_has_schema_version {
    ($manifest:ty) => {
        impl HasSchemaVersion for $manifest {
            fn schema_version(&self) -> SchemaVersion {
                self.schema_version
            }
        }
    };
}

impl_has_schema_version!(RawFilesManifest);
impl_has_schema_version!(DownloadPlanManifest);
impl_has_schema_version!(DownloadedFilesManifest);
impl_has_schema_version!(ExtractedFramesManifest);
impl_has_schema_version!(ProcessedSamplesManifest);
impl_has_schema_version!(SplitsManifest);
impl_has_schema_version!(RunIndexManifest);
impl_has_schema_version!(LatestManifest);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestKind {
    RawFiles,
    DownloadPlan,
    DownloadedFiles,
    ExtractedFrames,
    ProcessedSamples,
    Splits,
    RunIndex,
    Latest,
}

impl ManifestKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawFiles => "raw_files",
            Self::DownloadPlan => "download_plan",
            Self::DownloadedFiles => "downloaded_files",
            Self::ExtractedFrames => "extracted_frames",
            Self::ProcessedSamples => "processed_samples",
            Self::Splits => "splits",
            Self::RunIndex => "run_index",
            Self::Latest => "latest",
        }
    }
}

impl std::fmt::Display for ManifestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestDiagnostic {
    pub location: String,
    pub message: String,
    pub hint: String,
}

impl ManifestDiagnostic {
    pub fn new(
        location: impl Into<String>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            location: location.into(),
            message: message.into(),
            hint: hint.into(),
        }
    }
}

impl std::fmt::Display for ManifestDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} (hint: {})",
            self.location, self.message, self.hint
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestValidationError {
    pub kind: ManifestKind,
    pub diagnostics: Vec<ManifestDiagnostic>,
}

impl std::fmt::Display for ManifestValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} manifest validation failed", self.kind)?;
        for diagnostic in &self.diagnostics {
            write!(f, "; {diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ManifestValidationError {}

pub type ManifestValidationResult = Result<(), ManifestValidationError>;

pub trait ValidateManifest {
    const KIND: ManifestKind;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic>;

    fn validate(&self) -> ManifestValidationResult {
        let diagnostics = self.validation_diagnostics();
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(ManifestValidationError {
                kind: Self::KIND,
                diagnostics,
            })
        }
    }
}

pub fn validate_raw_files_manifest(manifest: &RawFilesManifest) -> ManifestValidationResult {
    manifest.validate()
}

pub fn validate_download_plan_manifest(
    manifest: &DownloadPlanManifest,
) -> ManifestValidationResult {
    manifest.validate()
}

pub fn validate_downloaded_files_manifest(
    manifest: &DownloadedFilesManifest,
) -> ManifestValidationResult {
    manifest.validate()
}

pub fn validate_extracted_frames_manifest(
    manifest: &ExtractedFramesManifest,
) -> ManifestValidationResult {
    manifest.validate()
}

pub fn extracted_frame_alignment_diagnostics(
    manifest: &ExtractedFramesManifest,
) -> Vec<ManifestDiagnostic> {
    let mut diagnostics = Vec::new();
    validate_extracted_frame_alignment_records(manifest, &mut diagnostics);
    diagnostics
}

pub fn validate_extracted_frame_alignment(
    manifest: &ExtractedFramesManifest,
) -> ManifestValidationResult {
    let diagnostics = extracted_frame_alignment_diagnostics(manifest);
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ManifestValidationError {
            kind: ManifestKind::ExtractedFrames,
            diagnostics,
        })
    }
}

pub fn validate_processed_samples_manifest(
    manifest: &ProcessedSamplesManifest,
) -> ManifestValidationResult {
    manifest.validate()
}

pub fn processed_sample_contract_diagnostics(
    manifest: &ProcessedSamplesManifest,
) -> Vec<ManifestDiagnostic> {
    manifest.validation_diagnostics()
}

pub fn validate_processed_sample_contract(
    manifest: &ProcessedSamplesManifest,
) -> ManifestValidationResult {
    let diagnostics = processed_sample_contract_diagnostics(manifest);
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ManifestValidationError {
            kind: ManifestKind::ProcessedSamples,
            diagnostics,
        })
    }
}

pub fn validate_splits_manifest(manifest: &SplitsManifest) -> ManifestValidationResult {
    manifest.validate()
}

pub fn validate_run_index_manifest(manifest: &RunIndexManifest) -> ManifestValidationResult {
    manifest.validate()
}

pub fn validate_latest_manifest(manifest: &LatestManifest) -> ManifestValidationResult {
    manifest.validate()
}

fn require_schema_version(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    found: SchemaVersion,
    expected: SchemaVersion,
) {
    if found != expected {
        diagnostics.push(ManifestDiagnostic::new(
            "schema_version",
            format!("expected schema version {}, found {}", expected.0, found.0),
            "Regenerate the manifest with the current sfx-data manifest writer.",
        ));
    }
}

fn require_non_empty_string(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: impl Into<String>,
    value: &str,
    field_name: &str,
) {
    if value.trim().is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            format!("{field_name} must not be empty"),
            format!("Populate {field_name} from the source dataset metadata before writing the manifest."),
        ));
    }
}

fn require_non_empty_path(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: impl Into<String>,
    path: &PathBuf,
    field_name: &str,
) {
    if path.as_os_str().is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            format!("{field_name} must not be empty"),
            format!("Populate {field_name} with the relative path produced by the pipeline step."),
        ));
    }
}

fn require_positive_size(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: impl Into<String>,
    size_bytes: u64,
) {
    if size_bytes == 0 {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            "size_bytes must be greater than zero",
            "Re-scan the source file size and exclude zero-byte placeholder files.",
        ));
    }
}

fn validate_optional_sha256(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: impl Into<String>,
    sha256: &Option<String>,
) {
    let Some(value) = sha256 else {
        return;
    };
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            "sha256 must be a 64-character hexadecimal digest when present",
            "Recompute the file checksum or omit sha256 until a valid digest is available.",
        ));
    }
}

fn duplicate_string_diagnostics<'a>(
    values: impl IntoIterator<Item = (String, &'a str)>,
    field_name: &str,
    hint: &str,
) -> Vec<ManifestDiagnostic> {
    let mut seen = std::collections::HashSet::new();
    let mut reported = std::collections::HashSet::new();
    let mut diagnostics = Vec::new();
    for (location, value) in values {
        if !value.trim().is_empty()
            && !seen.insert(value.to_string())
            && reported.insert(value.to_string())
        {
            diagnostics.push(ManifestDiagnostic::new(
                location,
                format!("duplicate {field_name}: {value}"),
                hint,
            ));
        }
    }
    diagnostics
}

impl ValidateManifest for RawFilesManifest {
    const KIND: ManifestKind = ManifestKind::RawFiles;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            RAW_FILES_SCHEMA_VERSION,
        );
        for (index, file) in self.files.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_uri"),
                &file.object_uri,
                "object_uri",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_path"),
                &file.object_path,
                "object_path",
            );
            require_positive_size(
                &mut diagnostics,
                format!("files[{index}].size_bytes"),
                file.size_bytes,
            );
        }
        diagnostics.extend(duplicate_string_diagnostics(
            self.files.iter().enumerate().map(|(index, file)| {
                (
                    format!("files[{index}].object_uri"),
                    file.object_uri.as_str(),
                )
            }),
            "object_uri",
            "Remove duplicate raw file records or keep the newest discovery for this object URI.",
        ));
        diagnostics.extend(duplicate_string_diagnostics(
            self.files.iter().enumerate().map(|(index, file)| {
                (
                    format!("files[{index}].object_path"),
                    file.object_path.as_str(),
                )
            }),
            "object_path",
            "Ensure each raw object path maps to exactly one source object.",
        ));
        diagnostics
    }
}

impl ValidateManifest for DownloadPlanManifest {
    const KIND: ManifestKind = ManifestKind::DownloadPlan;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            DOWNLOAD_PLAN_SCHEMA_VERSION,
        );
        if let Some(path) = &self.source_manifest_path {
            require_non_empty_path(
                &mut diagnostics,
                "source_manifest_path",
                path,
                "source_manifest_path",
            );
        }
        for (index, file) in self.files.iter().enumerate() {
            if file.plan_index != index {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("files[{index}].plan_index"),
                    format!("expected plan_index {index}, found {}", file.plan_index),
                    "Regenerate the download plan so records are ordered with contiguous plan indices.",
                ));
            }
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_uri"),
                &file.object_uri,
                "object_uri",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_path"),
                &file.object_path,
                "object_path",
            );
            require_positive_size(
                &mut diagnostics,
                format!("files[{index}].size_bytes"),
                file.size_bytes,
            );
        }
        diagnostics.extend(duplicate_string_diagnostics(
            self.files.iter().enumerate().map(|(index, file)| {
                (
                    format!("files[{index}].object_uri"),
                    file.object_uri.as_str(),
                )
            }),
            "object_uri",
            "Remove duplicate raw file records before planning downloads.",
        ));
        diagnostics.extend(duplicate_string_diagnostics(
            self.files.iter().enumerate().map(|(index, file)| {
                (
                    format!("files[{index}].object_path"),
                    file.object_path.as_str(),
                )
            }),
            "object_path",
            "Ensure each planned download has a unique source object path.",
        ));
        for (index, warning) in self.warnings.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("warnings[{index}].message"),
                &warning.message,
                "message",
            );
        }
        diagnostics
    }
}

impl ValidateManifest for DownloadedFilesManifest {
    const KIND: ManifestKind = ManifestKind::DownloadedFiles;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            DOWNLOADED_FILES_SCHEMA_VERSION,
        );
        for (index, file) in self.files.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_uri"),
                &file.object_uri,
                "object_uri",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("files[{index}].object_path"),
                &file.object_path,
                "object_path",
            );
            require_non_empty_path(
                &mut diagnostics,
                format!("files[{index}].local_path"),
                &file.local_path,
                "local_path",
            );
            require_positive_size(
                &mut diagnostics,
                format!("files[{index}].size_bytes"),
                file.size_bytes,
            );
            validate_optional_sha256(
                &mut diagnostics,
                format!("files[{index}].sha256"),
                &file.sha256,
            );
            if file.bytes_transferred > file.size_bytes {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("files[{index}].bytes_transferred"),
                    format!(
                        "bytes_transferred {} exceeds size_bytes {}",
                        file.bytes_transferred, file.size_bytes
                    ),
                    "Record only the bytes transferred during this download attempt.",
                ));
            }
            match (file.status, &file.error) {
                (DownloadedFileStatus::Failed, Some(error)) if !error.trim().is_empty() => {}
                (DownloadedFileStatus::Failed, _) => diagnostics.push(ManifestDiagnostic::new(
                    format!("files[{index}].error"),
                    "failed records must include a non-empty error",
                    "Capture the transfer or validation error that caused the failed outcome.",
                )),
                (_, Some(error)) if error.trim().is_empty() => {
                    diagnostics.push(ManifestDiagnostic::new(
                        format!("files[{index}].error"),
                        "error must be non-empty when present",
                        "Omit error for successful records or populate it with a meaningful failure.",
                    ));
                }
                (_, _) => {}
            }
        }
        diagnostics.extend(duplicate_string_diagnostics(
            self.files
                .iter()
                .enumerate()
                .map(|(index, file)| (format!("files[{index}].object_uri"), file.object_uri.as_str())),
            "object_uri",
            "Remove duplicate downloaded records or keep the latest completed download for this object URI.",
        ));
        diagnostics.extend(duplicate_string_diagnostics(
            self.files.iter().enumerate().map(|(index, file)| {
                (
                    format!("files[{index}].object_path"),
                    file.object_path.as_str(),
                )
            }),
            "object_path",
            "Ensure each downloaded source object path maps to exactly one local file.",
        ));
        let mut seen_paths = std::collections::HashSet::new();
        let mut reported_paths = std::collections::HashSet::new();
        for (index, file) in self.files.iter().enumerate() {
            if file.local_path.as_os_str().is_empty() {
                continue;
            }
            let value = file.local_path.to_string_lossy().into_owned();
            if !seen_paths.insert(value.clone()) && reported_paths.insert(value.clone()) {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("files[{index}].local_path"),
                    format!("duplicate local_path: {value}"),
                    "Ensure each downloaded source object has a unique local destination path.",
                ));
            }
        }
        diagnostics
    }
}

impl ValidateManifest for ExtractedFramesManifest {
    const KIND: ManifestKind = ManifestKind::ExtractedFrames;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            EXTRACTED_FRAMES_SCHEMA_VERSION,
        );
        require_non_empty_path(
            &mut diagnostics,
            "output_layout.root_dir",
            &self.output_layout.root_dir,
            "root_dir",
        );
        require_non_empty_path(
            &mut diagnostics,
            "output_layout.manifest_path",
            &self.output_layout.manifest_path,
            "manifest_path",
        );
        require_non_empty_path(
            &mut diagnostics,
            "output_layout.rgb_dir",
            &self.output_layout.rgb_dir,
            "rgb_dir",
        );
        require_non_empty_path(
            &mut diagnostics,
            "output_layout.range_dir",
            &self.output_layout.range_dir,
            "range_dir",
        );

        for (index, frame) in self.frames.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].sample_id"),
                &frame.sample_id,
                "sample_id",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].frame_id"),
                &frame.frame_id,
                "frame_id",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].source.segment_id"),
                &frame.source.segment_id,
                "segment_id",
            );
            if let Some(object_uri) = &frame.source.object_uri {
                require_non_empty_string(
                    &mut diagnostics,
                    format!("frames[{index}].source.object_uri"),
                    object_uri,
                    "object_uri",
                );
            }
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].source.object_path"),
                &frame.source.object_path,
                "object_path",
            );
            if let Some(local_path) = &frame.source.local_path {
                require_non_empty_path(
                    &mut diagnostics,
                    format!("frames[{index}].source.local_path"),
                    local_path,
                    "local_path",
                );
            }
            validate_optional_sha256(
                &mut diagnostics,
                format!("frames[{index}].source.sha256"),
                &frame.source.sha256,
            );
            validate_artifact_reference(
                &mut diagnostics,
                format!("frames[{index}].rgb"),
                &frame.rgb,
            );
            validate_artifact_reference(
                &mut diagnostics,
                format!("frames[{index}].range"),
                &frame.range,
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].provenance.extractor_name"),
                &frame.provenance.extractor_name,
                "extractor_name",
            );
            require_non_empty_string(
                &mut diagnostics,
                format!("frames[{index}].provenance.extractor_version"),
                &frame.provenance.extractor_version,
                "extractor_version",
            );
            if let Some(config_hash) = &frame.provenance.config_hash {
                require_non_empty_string(
                    &mut diagnostics,
                    format!("frames[{index}].provenance.config_hash"),
                    config_hash,
                    "config_hash",
                );
            }
            if let Some(command) = &frame.provenance.command {
                require_non_empty_string(
                    &mut diagnostics,
                    format!("frames[{index}].provenance.command"),
                    command,
                    "command",
                );
            }
        }
        validate_extracted_frame_alignment_records(self, &mut diagnostics);

        diagnostics.extend(duplicate_string_diagnostics(
            self.frames.iter().enumerate().map(|(index, frame)| {
                (
                    format!("frames[{index}].sample_id"),
                    frame.sample_id.as_str(),
                )
            }),
            "sample_id",
            "Assign one stable sample ID per extracted synchronized RGB/range frame.",
        ));
        diagnostics.extend(duplicate_string_diagnostics(
            self.frames.iter().enumerate().map(|(index, frame)| {
                (format!("frames[{index}].frame_id"), frame.frame_id.as_str())
            }),
            "frame_id",
            "Assign one stable frame ID per source segment/frame pair.",
        ));
        let mut seen_source_frames = std::collections::HashSet::new();
        let mut reported_source_frames = std::collections::HashSet::new();
        for (index, frame) in self.frames.iter().enumerate() {
            if frame.source.segment_id.trim().is_empty() {
                continue;
            }
            let key = (frame.source.segment_id.clone(), frame.frame_index);
            if !seen_source_frames.insert(key.clone()) && reported_source_frames.insert(key.clone())
            {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("frames[{index}].frame_index"),
                    format!("duplicate source segment/frame pair: {}#{}", key.0, key.1),
                    "Emit exactly one extracted sample per source segment frame.",
                ));
            }
        }

        diagnostics
    }
}

fn validate_extracted_frame_alignment_records(
    manifest: &ExtractedFramesManifest,
    diagnostics: &mut Vec<ManifestDiagnostic>,
) {
    let mut seen_artifact_paths = std::collections::HashSet::new();
    let mut reported_artifact_paths = std::collections::HashSet::new();
    let mut expected_provenance: Option<&ExtractionProvenance> = None;

    for (index, frame) in manifest.frames.iter().enumerate() {
        validate_frame_context_alignment(index, frame, diagnostics);
        validate_frame_split_alignment(index, frame, diagnostics);
        validate_artifact_pair_alignment(index, frame, &manifest.output_layout, diagnostics);
        validate_artifact_path_uniqueness(
            diagnostics,
            &mut seen_artifact_paths,
            &mut reported_artifact_paths,
            format!("frames[{index}].rgb.path"),
            &frame.rgb.path,
        );
        validate_artifact_path_uniqueness(
            diagnostics,
            &mut seen_artifact_paths,
            &mut reported_artifact_paths,
            format!("frames[{index}].range.path"),
            &frame.range.path,
        );

        if frame.provenance.extractor_name.trim().is_empty()
            || frame.provenance.extractor_version.trim().is_empty()
        {
            continue;
        }
        if let Some(expected) = expected_provenance {
            if !provenance_contract_matches(expected, &frame.provenance) {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("frames[{index}].provenance"),
                    "provenance differs from earlier extracted frame records",
                    "Write one extracted-frames manifest per extractor/config invocation so all aligned RGB/range records share provenance.",
                ));
            }
        } else {
            expected_provenance = Some(&frame.provenance);
        }
    }
}

fn validate_frame_context_alignment(
    index: usize,
    frame: &ExtractedFrameRecord,
    diagnostics: &mut Vec<ManifestDiagnostic>,
) {
    if frame.timestamp_micros < 0 {
        diagnostics.push(ManifestDiagnostic::new(
            format!("frames[{index}].timestamp_micros"),
            "timestamp_micros must be zero or greater",
            "Use the source frame timestamp in Unix microseconds for the synchronized RGB/range pair.",
        ));
    }

    if frame.frame_id.trim().is_empty() || frame.source.segment_id.trim().is_empty() {
        return;
    }

    let expected_frame_id = format!("{}:{:06}", frame.source.segment_id, frame.frame_index);
    if frame.frame_id != expected_frame_id {
        diagnostics.push(ManifestDiagnostic::new(
            format!("frames[{index}].frame_id"),
            format!(
                "frame_id must match source segment/frame_index context: expected {expected_frame_id}, found {}",
                frame.frame_id
            ),
            "Build frame_id as '<segment_id>:<zero-padded frame_index>' from the same source frame used for both modalities.",
        ));
    }
}

fn validate_frame_split_alignment(
    index: usize,
    frame: &ExtractedFrameRecord,
    diagnostics: &mut Vec<ManifestDiagnostic>,
) {
    let tokens = split_tokens(frame.split);
    validate_split_text(
        diagnostics,
        format!("frames[{index}].source.object_path"),
        &frame.source.object_path,
        tokens,
    );
    if let Some(object_uri) = &frame.source.object_uri {
        validate_split_text(
            diagnostics,
            format!("frames[{index}].source.object_uri"),
            object_uri,
            tokens,
        );
    }
    if let Some(local_path) = &frame.source.local_path {
        validate_split_text(
            diagnostics,
            format!("frames[{index}].source.local_path"),
            &local_path.to_string_lossy(),
            tokens,
        );
    }
}

fn validate_split_text(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    value: &str,
    accepted_tokens: &[&str],
) {
    if value.trim().is_empty() {
        return;
    }
    let normalized = value.to_ascii_lowercase().replace('\\', "/");
    let contains_split = normalized
        .split('/')
        .any(|component| accepted_tokens.contains(&component));
    if !contains_split {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            format!(
                "source path does not contain expected split token ({})",
                accepted_tokens.join(" or ")
            ),
            "Keep the frame split consistent with the source object URI/path used to extract it.",
        ));
    }
}

fn split_tokens(split: SplitName) -> &'static [&'static str] {
    match split {
        SplitName::Train => &["train", "training"],
        SplitName::Val => &["val", "validation"],
        SplitName::Test => &["test", "testing"],
    }
}

fn validate_artifact_pair_alignment(
    index: usize,
    frame: &ExtractedFrameRecord,
    layout: &ExtractionOutputLayout,
    diagnostics: &mut Vec<ManifestDiagnostic>,
) {
    if frame.rgb.path.as_os_str().is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            format!("frames[{index}].rgb.path"),
            "rgb artifact must be present for every extracted frame",
            "Emit an RGB artifact path and range artifact path for the same sample_id before writing the record.",
        ));
    }
    if frame.range.path.as_os_str().is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            format!("frames[{index}].range.path"),
            "range artifact must be present for every extracted frame",
            "Emit an RGB artifact path and range artifact path for the same sample_id before writing the record.",
        ));
    }
    if !frame.rgb.path.as_os_str().is_empty()
        && !frame.range.path.as_os_str().is_empty()
        && frame.rgb.path == frame.range.path
    {
        diagnostics.push(ManifestDiagnostic::new(
            format!("frames[{index}]"),
            "rgb and range artifacts must use distinct paths",
            "Write each modality to its own artifact under the configured rgb_dir and range_dir.",
        ));
    }

    validate_modality_artifact_path(
        diagnostics,
        format!("frames[{index}].rgb.path"),
        &frame.rgb.path,
        &layout.rgb_dir,
        &frame.sample_id,
        "rgb",
    );
    validate_modality_artifact_path(
        diagnostics,
        format!("frames[{index}].range.path"),
        &frame.range.path,
        &layout.range_dir,
        &frame.sample_id,
        "range",
    );
}

fn validate_modality_artifact_path(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    path: &Path,
    expected_dir: &Path,
    sample_id: &str,
    modality: &str,
) {
    if path.as_os_str().is_empty() {
        return;
    }
    if !path_is_relative_artifact(path) {
        diagnostics.push(ManifestDiagnostic::new(
            location.clone(),
            "artifact path must be relative and must not contain parent-directory components",
            "Store artifact paths relative to the repository/dataset root without drive prefixes, roots, or '..'.",
        ));
    }
    if !expected_dir.as_os_str().is_empty() && !path.starts_with(expected_dir) {
        diagnostics.push(ManifestDiagnostic::new(
            location.clone(),
            format!(
                "{modality} artifact path is outside configured {modality}_dir {}",
                expected_dir.display()
            ),
            "Use ExtractionOutputLayout::{rgb_artifact_path,range_artifact_path} when writing extracted artifacts.",
        ));
    }
    if !sample_id.trim().is_empty() {
        let stem = path.file_stem().and_then(|stem| stem.to_str());
        if stem != Some(sample_id) {
            diagnostics.push(ManifestDiagnostic::new(
                location,
                format!("{modality} artifact filename stem must match sample_id {sample_id}"),
                "Name both modality artifacts with the aligned sample_id so inspection tools can pair them by path.",
            ));
        }
    }
}

fn path_is_relative_artifact(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn validate_artifact_path_uniqueness(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    seen_artifact_paths: &mut std::collections::HashSet<String>,
    reported_artifact_paths: &mut std::collections::HashSet<String>,
    location: String,
    path: &Path,
) {
    if path.as_os_str().is_empty() {
        return;
    }
    let value = path.to_string_lossy().into_owned();
    if !seen_artifact_paths.insert(value.clone()) && reported_artifact_paths.insert(value.clone()) {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            format!("duplicate artifact path: {value}"),
            "Each artifact should be written once and referenced by only one manifest record.",
        ));
    }
}

fn provenance_contract_matches(left: &ExtractionProvenance, right: &ExtractionProvenance) -> bool {
    left.extractor_name == right.extractor_name
        && left.extractor_version == right.extractor_version
        && left.config_hash == right.config_hash
        && left.command == right.command
}

fn validate_artifact_reference(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    artifact: &ArtifactReference,
) {
    require_non_empty_path(
        diagnostics,
        format!("{location}.path"),
        &artifact.path,
        "artifact path",
    );
    require_non_empty_string(
        diagnostics,
        format!("{location}.media_type"),
        &artifact.media_type,
        "media_type",
    );
    require_non_empty_string(
        diagnostics,
        format!("{location}.encoding"),
        &artifact.encoding,
        "encoding",
    );
    if artifact.width == 0 {
        diagnostics.push(ManifestDiagnostic::new(
            format!("{location}.width"),
            "width must be greater than zero",
            "Populate artifact shape from the decoded extracted output.",
        ));
    }
    if artifact.height == 0 {
        diagnostics.push(ManifestDiagnostic::new(
            format!("{location}.height"),
            "height must be greater than zero",
            "Populate artifact shape from the decoded extracted output.",
        ));
    }
    if artifact.channels == Some(0) {
        diagnostics.push(ManifestDiagnostic::new(
            format!("{location}.channels"),
            "channels must be greater than zero when present",
            "Populate channels from the decoded artifact or omit it for scalar range encodings.",
        ));
    }
    if artifact.size_bytes == Some(0) {
        diagnostics.push(ManifestDiagnostic::new(
            format!("{location}.size_bytes"),
            "size_bytes must be greater than zero when present",
            "Record the extracted artifact size after writing the file.",
        ));
    }
    validate_optional_sha256(diagnostics, format!("{location}.sha256"), &artifact.sha256);
}

impl ValidateManifest for ProcessedSamplesManifest {
    const KIND: ManifestKind = ManifestKind::ProcessedSamples;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            PROCESSED_SAMPLES_SCHEMA_VERSION,
        );
        validate_processed_output_layout(&mut diagnostics, &self.output_layout);
        if let Some(reference) = &self.source_extracted_frames {
            require_non_empty_path(
                &mut diagnostics,
                "source_extracted_frames.manifest_path",
                &reference.manifest_path,
                "manifest_path",
            );
            if reference.schema_version != EXTRACTED_FRAMES_SCHEMA_VERSION {
                diagnostics.push(ManifestDiagnostic::new(
                    "source_extracted_frames.schema_version",
                    format!(
                        "expected extracted_frames schema version {}, found {}",
                        EXTRACTED_FRAMES_SCHEMA_VERSION.0, reference.schema_version.0
                    ),
                    "Reference the extracted_frames manifest used as preprocessing input.",
                ));
            }
        }
        for (index, sample) in self.samples.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("samples[{index}].sample_id"),
                &sample.sample_id,
                "sample_id",
            );
            validate_tensor_artifact_reference(
                &mut diagnostics,
                format!("samples[{index}].rgb"),
                &sample.rgb,
                &self.output_layout.rgb_tensor_dir,
                &sample.sample_id,
                ProcessedTensorModality::Rgb,
            );
            validate_tensor_artifact_reference(
                &mut diagnostics,
                format!("samples[{index}].range"),
                &sample.range,
                &self.output_layout.range_tensor_dir,
                &sample.sample_id,
                ProcessedTensorModality::Range,
            );
            validate_processed_sample_metadata(&mut diagnostics, index, sample);
            validate_processed_previews(&mut diagnostics, index, sample, &self.output_layout);
        }
        validate_processed_sample_contract_records(self, &mut diagnostics);
        diagnostics.extend(duplicate_string_diagnostics(
            self.samples.iter().enumerate().map(|(index, sample)| {
                (
                    format!("samples[{index}].sample_id"),
                    sample.sample_id.as_str(),
                )
            }),
            "sample_id",
            "Remove duplicate processed sample records or assign stable unique sample IDs.",
        ));
        diagnostics
    }
}

fn validate_processed_output_layout(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    layout: &ProcessedDatasetOutputLayout,
) {
    require_non_empty_path(
        diagnostics,
        "output_layout.root_dir",
        &layout.root_dir,
        "root_dir",
    );
    require_non_empty_path(
        diagnostics,
        "output_layout.manifest_path",
        &layout.manifest_path,
        "manifest_path",
    );
    require_non_empty_path(
        diagnostics,
        "output_layout.rgb_tensor_dir",
        &layout.rgb_tensor_dir,
        "rgb_tensor_dir",
    );
    require_non_empty_path(
        diagnostics,
        "output_layout.range_tensor_dir",
        &layout.range_tensor_dir,
        "range_tensor_dir",
    );
    require_non_empty_path(
        diagnostics,
        "output_layout.preview_dir",
        &layout.preview_dir,
        "preview_dir",
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessedTensorModality {
    Rgb,
    Range,
}

impl ProcessedTensorModality {
    fn as_str(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Range => "range",
        }
    }
}

fn validate_tensor_artifact_reference(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    tensor: &TensorArtifactReference,
    expected_dir: &Path,
    sample_id: &str,
    modality: ProcessedTensorModality,
) {
    require_non_empty_path(
        diagnostics,
        format!("{location}.path"),
        &tensor.path,
        "tensor path",
    );
    validate_modality_artifact_path(
        diagnostics,
        format!("{location}.path"),
        &tensor.path,
        expected_dir,
        sample_id,
        modality.as_str(),
    );
    validate_tensor_shape(diagnostics, format!("{location}.shape"), &tensor.shape);
    validate_tensor_shape_layout(
        diagnostics,
        format!("{location}.shape"),
        &tensor.shape,
        tensor.layout,
        modality,
    );
    if tensor.size_bytes == Some(0) {
        diagnostics.push(ManifestDiagnostic::new(
            format!("{location}.size_bytes"),
            "size_bytes must be greater than zero when present",
            "Record the tensor artifact size after writing the file.",
        ));
    }
    validate_optional_sha256(diagnostics, format!("{location}.sha256"), &tensor.sha256);
}

fn validate_tensor_shape(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    shape: &TensorShape,
) {
    if shape.dimensions.is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            location,
            "tensor shape must include at least one dimension",
            "Populate tensor dimensions in storage order before writing the manifest.",
        ));
        return;
    }
    for (dimension_index, dimension) in shape.dimensions.iter().enumerate() {
        if *dimension == 0 {
            diagnostics.push(ManifestDiagnostic::new(
                format!("{location}.dimensions[{dimension_index}]"),
                "tensor shape dimensions must be greater than zero",
                "Populate tensor dimensions from the processed tensor artifact.",
            ));
        }
    }
}

fn validate_tensor_shape_layout(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    location: String,
    shape: &TensorShape,
    layout: TensorLayout,
    modality: ProcessedTensorModality,
) {
    match (layout, shape.dimensions.as_slice(), modality) {
        (TensorLayout::Hwc, [_, _, 3], ProcessedTensorModality::Rgb)
        | (TensorLayout::Chw, [3, _, _], ProcessedTensorModality::Rgb) => {}
        (TensorLayout::Hwc | TensorLayout::Chw, dims, ProcessedTensorModality::Rgb) => {
            diagnostics.push(ManifestDiagnostic::new(
                location,
                format!(
                    "rgb tensor shape/layout must have 3 channels, found {dims:?} with {layout:?}"
                ),
                "Store RGB tensors as HWC or CHW with exactly three channels.",
            ));
        }
        (TensorLayout::Hw, [_, _], ProcessedTensorModality::Range)
        | (TensorLayout::Hwc, [_, _, _], ProcessedTensorModality::Range)
        | (TensorLayout::Chw, [_, _, _], ProcessedTensorModality::Range) => {}
        (_, dims, ProcessedTensorModality::Range) => diagnostics.push(ManifestDiagnostic::new(
            location,
            format!("range tensor shape/layout is inconsistent: {dims:?} with {layout:?}"),
            "Store range tensors as HW or as a 3D tensor with an explicit channel dimension.",
        )),
        (TensorLayout::Hw, dims, ProcessedTensorModality::Rgb) => {
            diagnostics.push(ManifestDiagnostic::new(
                location,
                format!("rgb tensor shape/layout must include a channel dimension, found {dims:?}"),
                "Store RGB tensors as HWC or CHW with exactly three channels.",
            ))
        }
    }
}

fn validate_processed_sample_metadata(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    index: usize,
    sample: &ProcessedSampleRecord,
) {
    let metadata = &sample.metadata;
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.source_sample_id"),
        &metadata.source_sample_id,
        "source_sample_id",
    );
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.frame_id"),
        &metadata.frame_id,
        "frame_id",
    );
    if metadata.timestamp_micros < 0 {
        diagnostics.push(ManifestDiagnostic::new(
            format!("samples[{index}].metadata.timestamp_micros"),
            "timestamp_micros must be zero or greater",
            "Preserve the source frame timestamp in Unix microseconds.",
        ));
    }
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.source.segment_id"),
        &metadata.source.segment_id,
        "segment_id",
    );
    if let Some(object_uri) = &metadata.source.object_uri {
        require_non_empty_string(
            diagnostics,
            format!("samples[{index}].metadata.source.object_uri"),
            object_uri,
            "object_uri",
        );
    }
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.source.object_path"),
        &metadata.source.object_path,
        "object_path",
    );
    if let Some(local_path) = &metadata.source.local_path {
        require_non_empty_path(
            diagnostics,
            format!("samples[{index}].metadata.source.local_path"),
            local_path,
            "local_path",
        );
    }
    validate_optional_sha256(
        diagnostics,
        format!("samples[{index}].metadata.source.sha256"),
        &metadata.source.sha256,
    );
    require_non_empty_path(
        diagnostics,
        format!("samples[{index}].metadata.extracted_rgb_path"),
        &metadata.extracted_rgb_path,
        "extracted_rgb_path",
    );
    require_non_empty_path(
        diagnostics,
        format!("samples[{index}].metadata.extracted_range_path"),
        &metadata.extracted_range_path,
        "extracted_range_path",
    );
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.provenance.preprocessor_name"),
        &metadata.provenance.preprocessor_name,
        "preprocessor_name",
    );
    require_non_empty_string(
        diagnostics,
        format!("samples[{index}].metadata.provenance.preprocessor_version"),
        &metadata.provenance.preprocessor_version,
        "preprocessor_version",
    );
    if let Some(config_hash) = &metadata.provenance.config_hash {
        require_non_empty_string(
            diagnostics,
            format!("samples[{index}].metadata.provenance.config_hash"),
            config_hash,
            "config_hash",
        );
    }
    if let Some(command) = &metadata.provenance.command {
        require_non_empty_string(
            diagnostics,
            format!("samples[{index}].metadata.provenance.command"),
            command,
            "command",
        );
    }
}

fn validate_processed_previews(
    diagnostics: &mut Vec<ManifestDiagnostic>,
    index: usize,
    sample: &ProcessedSampleRecord,
    layout: &ProcessedDatasetOutputLayout,
) {
    if sample.previews.is_empty() {
        diagnostics.push(ManifestDiagnostic::new(
            format!("samples[{index}].previews"),
            "previews must contain at least one artifact",
            "Write lightweight preview artifacts for inspection alongside processed tensors.",
        ));
    }
    let mut seen_kinds = std::collections::HashSet::new();
    for (preview_index, preview) in sample.previews.iter().enumerate() {
        let location = format!("samples[{index}].previews[{preview_index}]");
        require_non_empty_path(
            diagnostics,
            format!("{location}.path"),
            &preview.path,
            "preview path",
        );
        require_non_empty_string(
            diagnostics,
            format!("{location}.media_type"),
            &preview.media_type,
            "media_type",
        );
        validate_modality_artifact_path(
            diagnostics,
            format!("{location}.path"),
            &preview.path,
            &layout.preview_dir.join(preview.kind.as_str()),
            &sample.sample_id,
            "preview",
        );
        if !seen_kinds.insert(preview.kind) {
            diagnostics.push(ManifestDiagnostic::new(
                format!("{location}.kind"),
                format!("duplicate preview kind: {}", preview.kind.as_str()),
                "Emit at most one preview artifact for each preview kind per sample.",
            ));
        }
    }
}

fn validate_processed_sample_contract_records(
    manifest: &ProcessedSamplesManifest,
    diagnostics: &mut Vec<ManifestDiagnostic>,
) {
    let mut seen_artifact_paths = std::collections::HashSet::new();
    let mut reported_artifact_paths = std::collections::HashSet::new();
    let mut seen_source_frames = std::collections::HashSet::new();
    let mut reported_source_frames = std::collections::HashSet::new();
    for (index, sample) in manifest.samples.iter().enumerate() {
        if sample.rgb.path == sample.range.path && !sample.rgb.path.as_os_str().is_empty() {
            diagnostics.push(ManifestDiagnostic::new(
                format!("samples[{index}]"),
                "rgb and range tensors must use distinct paths",
                "Write each modality tensor under its configured modality directory.",
            ));
        }
        validate_artifact_path_uniqueness(
            diagnostics,
            &mut seen_artifact_paths,
            &mut reported_artifact_paths,
            format!("samples[{index}].rgb.path"),
            &sample.rgb.path,
        );
        validate_artifact_path_uniqueness(
            diagnostics,
            &mut seen_artifact_paths,
            &mut reported_artifact_paths,
            format!("samples[{index}].range.path"),
            &sample.range.path,
        );
        for (preview_index, preview) in sample.previews.iter().enumerate() {
            validate_artifact_path_uniqueness(
                diagnostics,
                &mut seen_artifact_paths,
                &mut reported_artifact_paths,
                format!("samples[{index}].previews[{preview_index}].path"),
                &preview.path,
            );
        }
        if sample.metadata.source.segment_id.trim().is_empty() {
            continue;
        }
        let key = (
            sample.metadata.source.segment_id.clone(),
            sample.metadata.frame_index,
        );
        if !seen_source_frames.insert(key.clone()) && reported_source_frames.insert(key.clone()) {
            diagnostics.push(ManifestDiagnostic::new(
                format!("samples[{index}].metadata.frame_index"),
                format!("duplicate source segment/frame pair: {}#{}", key.0, key.1),
                "Ensure each source frame is represented by one processed sample record.",
            ));
        }
    }
}

impl ValidateManifest for SplitsManifest {
    const KIND: ManifestKind = ManifestKind::Splits;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(&mut diagnostics, self.schema_version, SPLITS_SCHEMA_VERSION);
        for (index, assignment) in self.assignments.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("assignments[{index}].sample_id"),
                &assignment.sample_id,
                "sample_id",
            );
        }
        diagnostics.extend(duplicate_string_diagnostics(
            self.assignments
                .iter()
                .enumerate()
                .map(|(index, assignment)| {
                    (
                        format!("assignments[{index}].sample_id"),
                        assignment.sample_id.as_str(),
                    )
                }),
            "sample_id",
            "Keep exactly one split assignment per processed sample.",
        ));
        diagnostics
    }
}

impl ValidateManifest for RunIndexManifest {
    const KIND: ManifestKind = ManifestKind::RunIndex;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(
            &mut diagnostics,
            self.schema_version,
            RUN_INDEX_SCHEMA_VERSION,
        );
        for (index, run) in self.runs.iter().enumerate() {
            require_non_empty_string(
                &mut diagnostics,
                format!("runs[{index}].run_id"),
                &run.run_id,
                "run_id",
            );
            require_non_empty_path(
                &mut diagnostics,
                format!("runs[{index}].output_dir"),
                &run.output_dir,
                "output_dir",
            );
            if let Some(finished_at) = run.finished_at {
                if finished_at < run.started_at {
                    diagnostics.push(ManifestDiagnostic::new(
                        format!("runs[{index}].finished_at"),
                        "finished_at must be greater than or equal to started_at",
                        "Repair the run timestamps from the run metadata or rerun the affected command.",
                    ));
                }
            }
            match (run.status, run.finished_at) {
                (RunStatus::Pending | RunStatus::Running, Some(_)) => diagnostics.push(ManifestDiagnostic::new(
                    format!("runs[{index}].finished_at"),
                    "unfinished run statuses must not have finished_at set",
                    "Set status to a terminal state or clear finished_at for in-progress runs.",
                )),
                (RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled, None) => diagnostics.push(ManifestDiagnostic::new(
                    format!("runs[{index}].finished_at"),
                    "terminal run statuses must include finished_at",
                    "Populate finished_at from the run completion time or mark the run as running/pending.",
                )),
                _ => {}
            }
        }
        diagnostics.extend(duplicate_string_diagnostics(
            self.runs
                .iter()
                .enumerate()
                .map(|(index, run)| (format!("runs[{index}].run_id"), run.run_id.as_str())),
            "run_id",
            "Keep the newest entry for each run ID or allocate unique run IDs.",
        ));
        diagnostics
    }
}

impl ValidateManifest for LatestManifest {
    const KIND: ManifestKind = ManifestKind::Latest;

    fn validation_diagnostics(&self) -> Vec<ManifestDiagnostic> {
        let mut diagnostics = Vec::new();
        require_schema_version(&mut diagnostics, self.schema_version, LATEST_SCHEMA_VERSION);
        let refs = [
            ("raw_files", &self.raw_files, RAW_FILES_SCHEMA_VERSION),
            (
                "downloaded_files",
                &self.downloaded_files,
                DOWNLOADED_FILES_SCHEMA_VERSION,
            ),
            (
                "extracted_frames",
                &self.extracted_frames,
                EXTRACTED_FRAMES_SCHEMA_VERSION,
            ),
            (
                "processed_samples",
                &self.processed_samples,
                PROCESSED_SAMPLES_SCHEMA_VERSION,
            ),
            ("splits", &self.splits, SPLITS_SCHEMA_VERSION),
            ("run_index", &self.run_index, RUN_INDEX_SCHEMA_VERSION),
        ];
        if refs.iter().all(|(_, reference, _)| reference.is_none()) {
            diagnostics.push(ManifestDiagnostic::new(
                "latest",
                "latest manifest must reference at least one concrete manifest",
                "Write latest after producing at least one pipeline manifest.",
            ));
        }
        let mut seen_paths = std::collections::HashSet::new();
        let mut reported_paths = std::collections::HashSet::new();
        for (name, reference, expected_schema) in refs {
            let Some(reference) = reference else {
                continue;
            };
            require_non_empty_path(
                &mut diagnostics,
                format!("{name}.manifest_path"),
                &reference.manifest_path,
                "manifest_path",
            );
            if reference.schema_version != expected_schema {
                diagnostics.push(ManifestDiagnostic::new(
                    format!("{name}.schema_version"),
                    format!(
                        "expected referenced schema version {}, found {}",
                        expected_schema.0, reference.schema_version.0
                    ),
                    format!("Point {name} at a manifest written with the current schema version."),
                ));
            }
            if !reference.manifest_path.as_os_str().is_empty() {
                let value = reference.manifest_path.to_string_lossy().into_owned();
                if !seen_paths.insert(value.clone()) && reported_paths.insert(value.clone()) {
                    diagnostics.push(ManifestDiagnostic::new(
                        format!("{name}.manifest_path"),
                        format!("duplicate manifest_path: {value}"),
                        "Each latest reference should point at the manifest for its own type.",
                    ));
                }
            }
        }
        diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(value: &str) -> Timestamp {
        DateTime::parse_from_rfc3339(value)
            .expect("timestamp must parse")
            .with_timezone(&Utc)
    }

    fn sample_extracted_frame(sample_id: &str) -> ExtractedFrameRecord {
        ExtractedFrameRecord {
            sample_id: sample_id.to_string(),
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
                path: PathBuf::from(format!("data/intermediate/extracted/rgb/{sample_id}.jpg")),
                media_type: "image/jpeg".to_string(),
                encoding: "jpeg".to_string(),
                width: 1920,
                height: 1280,
                channels: Some(3),
                size_bytes: Some(100),
                sha256: Some("b".repeat(64)),
            },
            range: ArtifactReference {
                path: PathBuf::from(format!("data/intermediate/extracted/range/{sample_id}.npz")),
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
        }
    }

    fn sample_extracted_manifest(frames: Vec<ExtractedFrameRecord>) -> ExtractedFramesManifest {
        ExtractedFramesManifest {
            schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: extraction_output_layout("data/intermediate/extracted"),
            frames,
        }
    }

    fn sample_processed_record(sample_id: &str) -> ProcessedSampleRecord {
        ProcessedSampleRecord {
            sample_id: sample_id.to_string(),
            split: SplitName::Train,
            rgb: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/rgb/{sample_id}.npy")),
                shape: TensorShape::new([224, 224, 3]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hwc,
                size_bytes: Some(224 * 224 * 3 * 4),
                sha256: Some("d".repeat(64)),
            },
            range: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/range/{sample_id}.npy")),
                shape: TensorShape::new([64, 2650]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hw,
                size_bytes: Some(64 * 2650 * 4),
                sha256: Some("e".repeat(64)),
            },
            metadata: ProcessedSampleMetadata {
                source_sample_id: sample_id.to_string(),
                frame_id: "segment-001:000003".to_string(),
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
                extracted_rgb_path: PathBuf::from(format!(
                    "data/intermediate/extracted/rgb/{sample_id}.jpg"
                )),
                extracted_range_path: PathBuf::from(format!(
                    "data/intermediate/extracted/range/{sample_id}.npz"
                )),
                provenance: ProcessingProvenance {
                    preprocessor_name: "sfx-preprocess".to_string(),
                    preprocessor_version: "0.1.0".to_string(),
                    processed_at: ts("2025-01-01T00:03:00Z"),
                    config_hash: Some("config-v1".to_string()),
                    command: Some("sfx preprocess".to_string()),
                },
            },
            previews: vec![
                ProcessedPreviewArtifact {
                    kind: ProcessedPreviewKind::Rgb,
                    path: PathBuf::from(format!("data/processed/previews/rgb/{sample_id}.jpg")),
                    media_type: "image/jpeg".to_string(),
                },
                ProcessedPreviewArtifact {
                    kind: ProcessedPreviewKind::Range,
                    path: PathBuf::from(format!("data/processed/previews/range/{sample_id}.png")),
                    media_type: "image/png".to_string(),
                },
            ],
        }
    }

    fn set_frame_index(frame: &mut ExtractedFrameRecord, frame_index: u32) {
        frame.frame_index = frame_index;
        frame.frame_id = format!("{}:{frame_index:06}", frame.source.segment_id);
        frame.timestamp_micros += i64::from(frame_index);
    }

    #[test]
    fn raw_files_manifest_has_versioned_top_level_shape() {
        let manifest = RawFilesManifest {
            schema_version: RAW_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![RawFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: "train/file-0001.tfrecord".to_string(),
                size_bytes: 42,
                discovered_at: ts("2025-01-01T00:01:00Z"),
                source_updated_at: None,
            }],
        };

        let json = serde_json::to_value(manifest).expect("raw_files manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:00:00Z");
        assert!(json["files"].is_array());
    }

    #[test]
    fn downloaded_files_manifest_has_versioned_top_level_shape() {
        let manifest = DownloadedFilesManifest {
            schema_version: DOWNLOADED_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![DownloadedFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: "train/file-0001.tfrecord".to_string(),
                local_path: PathBuf::from("data/raw/waymo/train/file-0001.tfrecord"),
                split: SplitName::Train,
                size_bytes: 42,
                status: DownloadedFileStatus::Downloaded,
                bytes_transferred: 42,
                sha256: Some("abc123".to_string()),
                downloaded_at: ts("2025-01-01T00:02:00Z"),
                error: None,
            }],
        };

        let json =
            serde_json::to_value(manifest).expect("downloaded_files manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:00:00Z");
        assert!(json["files"].is_array());
    }

    #[test]
    fn processed_samples_manifest_has_versioned_top_level_shape() {
        let manifest = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: processed_dataset_output_layout("data/processed"),
            source_extracted_frames: Some(ManifestReference {
                manifest_path: PathBuf::from("data/intermediate/extracted/extracted_frames.json"),
                schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
                generated_at: ts("2025-01-01T00:00:00Z"),
            }),
            samples: vec![sample_processed_record("sample-001")],
        };

        let json =
            serde_json::to_value(&manifest).expect("processed_samples manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:00:00Z");
        assert!(
            json["output_layout"]["manifest_path"]
                .as_str()
                .expect("manifest path should be a string")
                .ends_with(PROCESSED_SAMPLES_MANIFEST_FILE)
        );
        assert_eq!(json["samples"][0]["sample_id"], "sample-001");
        assert_eq!(json["samples"][0]["split"], "train");
        assert_eq!(
            json["samples"][0]["rgb"]["shape"]["dimensions"],
            serde_json::json!([224, 224, 3])
        );
        assert_eq!(json["samples"][0]["range"]["layout"], "hw");
        assert_eq!(
            json["samples"][0]["metadata"]["source"]["segment_id"],
            "segment-001"
        );
        assert!(json["samples"].is_array());

        let round_trip: ProcessedSamplesManifest =
            serde_json::from_value(json).expect("processed samples manifest should deserialize");
        assert_eq!(round_trip, manifest);
    }

    #[test]
    fn processed_dataset_output_layout_builds_expected_artifact_paths() {
        let layout = processed_dataset_output_layout("data/processed");

        assert_eq!(
            layout.manifest_path,
            PathBuf::from("data/processed").join(PROCESSED_SAMPLES_MANIFEST_FILE)
        );
        assert_eq!(
            layout.rgb_tensor_path("sample-001", ".npy"),
            PathBuf::from("data/processed/tensors/rgb/sample-001.npy")
        );
        assert_eq!(
            layout.range_tensor_path("sample-001", "npy"),
            PathBuf::from("data/processed/tensors/range/sample-001.npy")
        );
        assert_eq!(
            layout.preview_path("sample-001", ProcessedPreviewKind::Overlay, "jpg"),
            PathBuf::from("data/processed/previews/overlay/sample-001.jpg")
        );
    }

    #[test]
    fn extracted_frames_manifest_has_versioned_contract_shape() {
        let manifest = ExtractedFramesManifest {
            schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: extraction_output_layout("data/intermediate/extracted"),
            frames: vec![sample_extracted_frame("sample-001")],
        };

        let json =
            serde_json::to_value(&manifest).expect("extracted_frames manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert!(
            json["output_layout"]["manifest_path"]
                .as_str()
                .expect("manifest path should be a string")
                .ends_with(EXTRACTED_FRAMES_MANIFEST_FILE)
        );
        assert!(
            json["output_layout"]["rgb_dir"]
                .as_str()
                .expect("rgb_dir should be a string")
                .ends_with(EXTRACTED_RGB_DIR)
        );
        assert!(
            json["output_layout"]["range_dir"]
                .as_str()
                .expect("range_dir should be a string")
                .ends_with(EXTRACTED_RANGE_DIR)
        );
        assert_eq!(json["frames"][0]["sample_id"], "sample-001");
        assert_eq!(json["frames"][0]["split"], "train");
        assert_eq!(json["frames"][0]["rgb"]["width"], 1920);
        assert_eq!(json["frames"][0]["range"]["height"], 64);

        let round_trip: ExtractedFramesManifest =
            serde_json::from_value(json).expect("extracted_frames manifest should deserialize");
        assert_eq!(round_trip, manifest);
    }

    #[test]
    fn extraction_output_layout_builds_expected_artifact_paths() {
        let layout = extraction_output_layout("data/intermediate/extracted");

        assert_eq!(
            layout.manifest_path,
            PathBuf::from("data/intermediate/extracted").join(EXTRACTED_FRAMES_MANIFEST_FILE)
        );
        assert_eq!(
            layout.rgb_artifact_path("sample-001", ".jpg"),
            PathBuf::from("data/intermediate/extracted/rgb/sample-001.jpg")
        );
        assert_eq!(
            layout.range_artifact_path("sample-001", "npz"),
            PathBuf::from("data/intermediate/extracted/range/sample-001.npz")
        );
    }

    #[test]
    fn splits_manifest_uses_named_split_enum() {
        let manifest = SplitsManifest {
            schema_version: SPLITS_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            split_seed: 42,
            assignments: vec![SplitAssignment {
                sample_id: "sample-001".to_string(),
                split: SplitName::Train,
                assigned_at: ts("2025-01-01T00:04:00Z"),
            }],
        };

        let json = serde_json::to_value(manifest).expect("splits manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:00:00Z");
        assert_eq!(json["assignments"][0]["split"], "train");
    }

    #[test]
    fn run_index_manifest_uses_versioned_run_records() {
        let manifest = RunIndexManifest {
            schema_version: RUN_INDEX_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            runs: vec![RunIndexEntry {
                run_id: "run-0001".to_string(),
                run_kind: RunKind::Preprocess,
                status: RunStatus::Succeeded,
                started_at: ts("2025-01-01T00:05:00Z"),
                finished_at: Some(ts("2025-01-01T00:06:00Z")),
                output_dir: PathBuf::from(".xtask/runs/run-0001"),
            }],
        };

        let json = serde_json::to_value(manifest).expect("run_index manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:00:00Z");
        assert_eq!(json["runs"][0]["run_kind"], "preprocess");
        assert_eq!(json["runs"][0]["status"], "succeeded");
    }

    #[test]
    fn latest_manifest_tracks_all_manifest_heads() {
        let reference = ManifestReference {
            manifest_path: PathBuf::from(".xtask/manifests/raw_files.json"),
            schema_version: RAW_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
        };

        let manifest = LatestManifest {
            schema_version: LATEST_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:10:00Z"),
            raw_files: Some(reference),
            downloaded_files: None,
            extracted_frames: None,
            processed_samples: None,
            splits: None,
            run_index: None,
        };

        let json = serde_json::to_value(manifest).expect("latest manifest should serialize");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["generated_at"], "2025-01-01T00:10:00Z");
        assert!(json["raw_files"].is_object());
        assert!(json["run_index"].is_null());
    }

    #[test]
    fn validation_accepts_representative_valid_manifests() {
        let digest = "a".repeat(64);
        validate_raw_files_manifest(&RawFilesManifest {
            schema_version: RAW_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![RawFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: "train/file-0001.tfrecord".to_string(),
                size_bytes: 42,
                discovered_at: ts("2025-01-01T00:01:00Z"),
                source_updated_at: None,
            }],
        })
        .expect("raw manifest should validate");

        validate_downloaded_files_manifest(&DownloadedFilesManifest {
            schema_version: DOWNLOADED_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![DownloadedFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: "train/file-0001.tfrecord".to_string(),
                local_path: PathBuf::from("data/raw/waymo/train/file-0001.tfrecord"),
                split: SplitName::Train,
                size_bytes: 42,
                status: DownloadedFileStatus::Downloaded,
                bytes_transferred: 42,
                sha256: Some(digest),
                downloaded_at: ts("2025-01-01T00:02:00Z"),
                error: None,
            }],
        })
        .expect("downloaded manifest should validate");

        validate_processed_samples_manifest(&ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: processed_dataset_output_layout("data/processed"),
            source_extracted_frames: Some(ManifestReference {
                manifest_path: PathBuf::from("data/intermediate/extracted/extracted_frames.json"),
                schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
                generated_at: ts("2025-01-01T00:00:00Z"),
            }),
            samples: vec![sample_processed_record("sample-001")],
        })
        .expect("processed manifest should validate");

        validate_extracted_frames_manifest(&ExtractedFramesManifest {
            schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: extraction_output_layout("data/intermediate/extracted"),
            frames: vec![sample_extracted_frame("sample-001")],
        })
        .expect("extracted frames manifest should validate");

        validate_splits_manifest(&SplitsManifest {
            schema_version: SPLITS_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            split_seed: 42,
            assignments: vec![SplitAssignment {
                sample_id: "sample-001".to_string(),
                split: SplitName::Train,
                assigned_at: ts("2025-01-01T00:04:00Z"),
            }],
        })
        .expect("splits manifest should validate");

        validate_run_index_manifest(&RunIndexManifest {
            schema_version: RUN_INDEX_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            runs: vec![RunIndexEntry {
                run_id: "run-0001".to_string(),
                run_kind: RunKind::Preprocess,
                status: RunStatus::Succeeded,
                started_at: ts("2025-01-01T00:05:00Z"),
                finished_at: Some(ts("2025-01-01T00:06:00Z")),
                output_dir: PathBuf::from(".xtask/runs/run-0001"),
            }],
        })
        .expect("run index manifest should validate");

        validate_latest_manifest(&LatestManifest {
            schema_version: LATEST_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:10:00Z"),
            raw_files: Some(ManifestReference {
                manifest_path: PathBuf::from(".xtask/manifests/raw_files.json"),
                schema_version: RAW_FILES_SCHEMA_VERSION,
                generated_at: ts("2025-01-01T00:00:00Z"),
            }),
            downloaded_files: None,
            extracted_frames: None,
            processed_samples: None,
            splits: None,
            run_index: None,
        })
        .expect("latest manifest should validate");
    }

    #[test]
    fn extracted_frame_alignment_accepts_coherent_rgb_range_pairs() {
        let mut second = sample_extracted_frame("sample-002");
        set_frame_index(&mut second, 4);
        let manifest =
            sample_extracted_manifest(vec![sample_extracted_frame("sample-001"), second]);

        validate_extracted_frame_alignment(&manifest).expect("alignment should validate");
        validate_extracted_frames_manifest(&manifest).expect("manifest should validate");
    }

    #[test]
    fn extracted_frame_alignment_reports_context_split_path_and_provenance_errors() {
        let mut first = sample_extracted_frame("sample-001");
        first.frame_id = "segment-999:000099".to_string();
        first.timestamp_micros = -1;
        first.split = SplitName::Val;

        let mut second = sample_extracted_frame("sample-002");
        set_frame_index(&mut second, 4);
        second.rgb.path = PathBuf::from("data/intermediate/extracted/range/not-sample-002.npz");
        second.range.path = second.rgb.path.clone();
        second.provenance.config_hash = Some("different-config".to_string());

        let manifest = sample_extracted_manifest(vec![first, second]);
        let error =
            validate_extracted_frame_alignment(&manifest).expect_err("alignment should fail");

        assert_eq!(error.kind, ManifestKind::ExtractedFrames);
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[0].frame_id"
                && diagnostic.message.contains("source segment/frame_index")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[0].timestamp_micros"
                && diagnostic.message.contains("zero or greater")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[0].source.object_path"
                && diagnostic.message.contains("expected split token")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[1]" && diagnostic.message.contains("distinct paths")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[1].rgb.path"
                && diagnostic.message.contains("outside configured rgb_dir")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[1].rgb.path"
                && diagnostic
                    .message
                    .contains("filename stem must match sample_id")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[1].provenance"
                && diagnostic.message.contains("differs from earlier")
        }));
        assert!(
            error
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.hint.is_empty())
        );
    }

    #[test]
    fn extracted_frame_alignment_rejects_unsafe_and_reused_artifact_paths() {
        let mut first = sample_extracted_frame("sample-001");
        first.rgb.path = PathBuf::from("data/intermediate/extracted/rgb/../sample-001.jpg");

        let mut second = sample_extracted_frame("sample-002");
        set_frame_index(&mut second, 4);
        second.rgb.path = first.range.path.clone();

        let mut third = sample_extracted_frame("sample-003");
        set_frame_index(&mut third, 5);
        third.range.path = PathBuf::new();

        let manifest = sample_extracted_manifest(vec![first, second, third]);
        let error =
            validate_extracted_frame_alignment(&manifest).expect_err("alignment should fail");

        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[0].rgb.path" && diagnostic.message.contains("relative")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[1].rgb.path"
                && diagnostic.message.contains("duplicate artifact path")
        }));
        assert!(error.diagnostics.iter().any(|diagnostic| {
            diagnostic.location == "frames[2].range.path"
                && diagnostic
                    .message
                    .contains("range artifact must be present")
        }));
    }

    #[test]
    fn validation_reports_raw_file_corruption() {
        let manifest = RawFilesManifest {
            schema_version: SchemaVersion(99),
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![
                RawFileRecord {
                    object_uri: "".to_string(),
                    object_path: "train/file-0001.tfrecord".to_string(),
                    size_bytes: 0,
                    discovered_at: ts("2025-01-01T00:01:00Z"),
                    source_updated_at: None,
                },
                RawFileRecord {
                    object_uri: "gs://wod/train/file-0002.tfrecord".to_string(),
                    object_path: "train/file-0001.tfrecord".to_string(),
                    size_bytes: 12,
                    discovered_at: ts("2025-01-01T00:02:00Z"),
                    source_updated_at: None,
                },
            ],
        };

        let error = validate_raw_files_manifest(&manifest).expect_err("manifest should be invalid");
        assert_eq!(error.kind, ManifestKind::RawFiles);
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.location == "schema_version")
        );
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("object_uri must not be empty"))
        );
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("size_bytes"))
        );
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("duplicate object_path"))
        );
        assert!(
            error
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.hint.is_empty())
        );
    }

    #[test]
    fn validation_reports_manifest_specific_corruption() {
        let downloaded = DownloadedFilesManifest {
            schema_version: DOWNLOADED_FILES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            files: vec![DownloadedFileRecord {
                object_uri: "gs://wod/train/file-0001.tfrecord".to_string(),
                object_path: String::new(),
                local_path: PathBuf::new(),
                split: SplitName::Train,
                size_bytes: 42,
                status: DownloadedFileStatus::Failed,
                bytes_transferred: 43,
                sha256: Some("not-a-sha".to_string()),
                downloaded_at: ts("2025-01-01T00:02:00Z"),
                error: None,
            }],
        };
        assert!(validate_downloaded_files_manifest(&downloaded).is_err());

        let processed = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: ProcessedDatasetOutputLayout {
                root_dir: PathBuf::new(),
                manifest_path: PathBuf::new(),
                rgb_tensor_dir: PathBuf::new(),
                range_tensor_dir: PathBuf::new(),
                preview_dir: PathBuf::new(),
            },
            source_extracted_frames: Some(ManifestReference {
                manifest_path: PathBuf::from("data/processed/not-extracted.json"),
                schema_version: SchemaVersion(99),
                generated_at: ts("2025-01-01T00:00:00Z"),
            }),
            samples: vec![{
                let mut sample = sample_processed_record("sample-001");
                sample.rgb.shape = TensorShape::new([224, 224, 1]);
                sample.range.path = sample.rgb.path.clone();
                sample.metadata.timestamp_micros = -1;
                sample.metadata.provenance.preprocessor_name.clear();
                sample.previews.clear();
                sample
            }],
        };
        assert!(validate_processed_samples_manifest(&processed).is_err());

        let mut extracted = ExtractedFramesManifest {
            schema_version: EXTRACTED_FRAMES_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            output_layout: ExtractionOutputLayout {
                root_dir: PathBuf::new(),
                manifest_path: PathBuf::new(),
                rgb_dir: PathBuf::new(),
                range_dir: PathBuf::new(),
            },
            frames: vec![
                sample_extracted_frame("sample-001"),
                sample_extracted_frame("sample-001"),
            ],
        };
        extracted.frames[0].frame_id.clear();
        extracted.frames[0].source.object_path.clear();
        extracted.frames[0].rgb.width = 0;
        extracted.frames[0].range.channels = Some(0);
        extracted.frames[0].provenance.extractor_name.clear();
        let error = validate_extracted_frames_manifest(&extracted)
            .expect_err("extracted manifest should be invalid");
        assert_eq!(error.kind, ManifestKind::ExtractedFrames);
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("duplicate sample_id"))
        );
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.location.ends_with(".rgb.width"))
        );
        assert!(
            error
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.location.ends_with(".range.channels"))
        );

        let splits = SplitsManifest {
            schema_version: SPLITS_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            split_seed: 42,
            assignments: vec![
                SplitAssignment {
                    sample_id: "sample-001".to_string(),
                    split: SplitName::Train,
                    assigned_at: ts("2025-01-01T00:04:00Z"),
                },
                SplitAssignment {
                    sample_id: "sample-001".to_string(),
                    split: SplitName::Val,
                    assigned_at: ts("2025-01-01T00:05:00Z"),
                },
            ],
        };
        assert!(validate_splits_manifest(&splits).is_err());

        let run_index = RunIndexManifest {
            schema_version: RUN_INDEX_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:00:00Z"),
            runs: vec![RunIndexEntry {
                run_id: "run-0001".to_string(),
                run_kind: RunKind::Fetch,
                status: RunStatus::Running,
                started_at: ts("2025-01-01T00:05:00Z"),
                finished_at: Some(ts("2025-01-01T00:04:00Z")),
                output_dir: PathBuf::from(".xtask/runs/run-0001"),
            }],
        };
        assert!(validate_run_index_manifest(&run_index).is_err());

        let latest = LatestManifest {
            schema_version: LATEST_SCHEMA_VERSION,
            generated_at: ts("2025-01-01T00:10:00Z"),
            raw_files: None,
            downloaded_files: None,
            extracted_frames: None,
            processed_samples: None,
            splits: None,
            run_index: None,
        };
        assert!(validate_latest_manifest(&latest).is_err());
    }
}
