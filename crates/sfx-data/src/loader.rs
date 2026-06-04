use crate::io::{ManifestIoError, read_processed_samples_manifest};
use crate::manifest::{
    ManifestValidationError, ProcessedSampleMetadata, ProcessedSampleRecord,
    ProcessedSamplesManifest, SplitName, TensorArtifactReference, TensorDType, TensorLayout,
    TensorShape, validate_processed_samples_manifest,
};
use sha2::{Digest, Sha256};
use std::fmt::{Display, Formatter};
use std::path::{Component, Path, PathBuf};

pub type Result<T> = std::result::Result<T, ProcessedLoaderError>;

#[derive(Debug)]
pub enum ProcessedLoaderError {
    ManifestRead {
        path: PathBuf,
        source: ManifestIoError,
    },
    ManifestValidation {
        path: PathBuf,
        source: ManifestValidationError,
    },
    MissingArtifact {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
    },
    ArtifactIo {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
        source: std::io::Error,
    },
    ArtifactSizeMismatch {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    ArtifactDeclaredSizeMismatch {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
        declared_bytes: u64,
        actual_bytes: usize,
    },
    ArtifactSha256Mismatch {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
        expected_sha256: String,
        actual_sha256: String,
    },
    ArtifactShapeOverflow {
        sample_id: String,
        modality: TensorModality,
        path: PathBuf,
    },
    InvalidBatchSize {
        batch_size: usize,
    },
}

impl Display for ProcessedLoaderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManifestRead { path, source } => {
                write!(
                    f,
                    "failed to read processed manifest {}: {}",
                    path.display(),
                    source
                )
            }
            Self::ManifestValidation { path, source } => {
                write!(
                    f,
                    "processed manifest {} is invalid: {}",
                    path.display(),
                    source
                )
            }
            Self::MissingArtifact {
                sample_id,
                modality,
                path,
            } => write!(
                f,
                "missing {modality} tensor artifact for sample {sample_id}: {}",
                path.display()
            ),
            Self::ArtifactIo {
                sample_id,
                modality,
                path,
                source,
            } => write!(
                f,
                "failed to read {modality} tensor artifact for sample {sample_id} at {}: {}",
                path.display(),
                source
            ),
            Self::ArtifactSizeMismatch {
                sample_id,
                modality,
                path,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "corrupt {modality} tensor artifact for sample {sample_id} at {}: expected {expected_bytes} bytes from dtype/shape, found {actual_bytes}",
                path.display()
            ),
            Self::ArtifactDeclaredSizeMismatch {
                sample_id,
                modality,
                path,
                declared_bytes,
                actual_bytes,
            } => write!(
                f,
                "corrupt {modality} tensor artifact for sample {sample_id} at {}: manifest declares {declared_bytes} bytes, found {actual_bytes}",
                path.display()
            ),
            Self::ArtifactSha256Mismatch {
                sample_id,
                modality,
                path,
                expected_sha256,
                actual_sha256,
            } => write!(
                f,
                "corrupt {modality} tensor artifact for sample {sample_id} at {}: expected sha256 {expected_sha256}, found {actual_sha256}",
                path.display()
            ),
            Self::ArtifactShapeOverflow {
                sample_id,
                modality,
                path,
            } => write!(
                f,
                "{modality} tensor shape for sample {sample_id} overflows addressable byte length: {}",
                path.display()
            ),
            Self::InvalidBatchSize { batch_size } => {
                write!(f, "invalid processed sample batch size: {batch_size}")
            }
        }
    }
}

impl std::error::Error for ProcessedLoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ManifestRead { source, .. } => Some(source),
            Self::ManifestValidation { source, .. } => Some(source),
            Self::ArtifactIo { source, .. } => Some(source),
            Self::MissingArtifact { .. }
            | Self::ArtifactSizeMismatch { .. }
            | Self::ArtifactDeclaredSizeMismatch { .. }
            | Self::ArtifactSha256Mismatch { .. }
            | Self::ArtifactShapeOverflow { .. }
            | Self::InvalidBatchSize { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorModality {
    Rgb,
    Range,
}

impl TensorModality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Range => "range",
        }
    }
}

impl Display for TensorModality {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedDataset {
    pub manifest_path: PathBuf,
    pub artifact_root: PathBuf,
    pub manifest: ProcessedSamplesManifest,
}

impl ProcessedDataset {
    pub fn read(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = manifest_path.as_ref();
        let manifest = read_processed_samples_manifest(manifest_path).map_err(|source| {
            ProcessedLoaderError::ManifestRead {
                path: manifest_path.to_path_buf(),
                source,
            }
        })?;
        validate_processed_samples_manifest(&manifest).map_err(|source| {
            ProcessedLoaderError::ManifestValidation {
                path: manifest_path.to_path_buf(),
                source,
            }
        })?;
        let artifact_root = infer_artifact_root(
            manifest_path,
            manifest.output_layout.manifest_path.as_path(),
        );
        Ok(Self {
            manifest_path: manifest_path.to_path_buf(),
            artifact_root,
            manifest,
        })
    }

    pub fn with_artifact_root(
        manifest_path: impl Into<PathBuf>,
        artifact_root: impl Into<PathBuf>,
        manifest: ProcessedSamplesManifest,
    ) -> Result<Self> {
        let manifest_path = manifest_path.into();
        validate_processed_samples_manifest(&manifest).map_err(|source| {
            ProcessedLoaderError::ManifestValidation {
                path: manifest_path.clone(),
                source,
            }
        })?;
        Ok(Self {
            manifest_path,
            artifact_root: artifact_root.into(),
            manifest,
        })
    }

    pub fn samples_for_split(&self, split: SplitName) -> Vec<&ProcessedSampleRecord> {
        samples_for_split(&self.manifest, split)
    }

    pub fn load_split(&self, split: SplitName) -> Result<Vec<LoadedProcessedSample>> {
        self.samples_for_split(split)
            .into_iter()
            .map(|sample| self.load_sample(sample))
            .collect()
    }

    pub fn load_sample(&self, sample: &ProcessedSampleRecord) -> Result<LoadedProcessedSample> {
        load_processed_sample(&self.artifact_root, sample)
    }

    pub fn batches(&self, split: SplitName, config: BatchConfig) -> Result<ProcessedBatchIter<'_>> {
        ProcessedBatchIter::new(self, split, config)
    }

    pub fn batches_for_split(
        &self,
        split: SplitName,
        batch_size: usize,
    ) -> Result<ProcessedBatchIter<'_>> {
        self.batches(split, BatchConfig::new(batch_size))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedProcessedSample {
    pub sample_id: String,
    pub source_sample_id: String,
    pub split: SplitName,
    pub source: ProcessedSampleMetadata,
    pub rgb: LoadedTensorArtifact,
    pub range: LoadedTensorArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedTensorArtifact {
    pub modality: TensorModality,
    pub path: PathBuf,
    pub shape: TensorShape,
    pub dtype: TensorDType,
    pub layout: TensorLayout,
    pub bytes: Vec<u8>,
    pub declared_size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

impl LoadedTensorArtifact {
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
}

impl BatchConfig {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            shuffle_seed: None,
        }
    }

    pub fn shuffled(batch_size: usize, seed: u64) -> Self {
        Self {
            batch_size,
            shuffle_seed: Some(seed),
        }
    }

    pub fn with_shuffle_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    pub fn without_shuffle(mut self) -> Self {
        self.shuffle_seed = None;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedSampleBatch {
    pub split: SplitName,
    pub batch_index: usize,
    pub samples: Vec<LoadedProcessedSample>,
}

impl ProcessedSampleBatch {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Debug)]
pub struct ProcessedBatchIter<'a> {
    dataset: &'a ProcessedDataset,
    split: SplitName,
    batch_size: usize,
    sample_indices: Vec<usize>,
    cursor: usize,
    batch_index: usize,
}

impl<'a> ProcessedBatchIter<'a> {
    fn new(dataset: &'a ProcessedDataset, split: SplitName, config: BatchConfig) -> Result<Self> {
        if config.batch_size == 0 {
            return Err(ProcessedLoaderError::InvalidBatchSize {
                batch_size: config.batch_size,
            });
        }

        let mut sample_indices = dataset
            .manifest
            .samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| (sample.split == split).then_some(index))
            .collect::<Vec<_>>();
        if let Some(seed) = config.shuffle_seed {
            shuffle_indices(&mut sample_indices, seed);
        }

        Ok(Self {
            dataset,
            split,
            batch_size: config.batch_size,
            sample_indices,
            cursor: 0,
            batch_index: 0,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn sample_count(&self) -> usize {
        self.sample_indices.len()
    }

    pub fn batch_count(&self) -> usize {
        self.sample_indices.len().div_ceil(self.batch_size)
    }
}

impl Iterator for ProcessedBatchIter<'_> {
    type Item = Result<ProcessedSampleBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.sample_indices.len() {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.sample_indices.len());
        let samples = self.sample_indices[self.cursor..end]
            .iter()
            .map(|index| {
                self.dataset
                    .load_sample(&self.dataset.manifest.samples[*index])
            })
            .collect::<Result<Vec<_>>>();
        let batch_index = self.batch_index;
        self.cursor = end;
        self.batch_index += 1;

        Some(samples.map(|samples| ProcessedSampleBatch {
            split: self.split,
            batch_index,
            samples,
        }))
    }
}

pub fn read_processed_dataset(manifest_path: impl AsRef<Path>) -> Result<ProcessedDataset> {
    ProcessedDataset::read(manifest_path)
}

pub fn load_processed_split(
    manifest_path: impl AsRef<Path>,
    split: SplitName,
) -> Result<Vec<LoadedProcessedSample>> {
    ProcessedDataset::read(manifest_path)?.load_split(split)
}

pub fn load_processed_batches(
    manifest_path: impl AsRef<Path>,
    split: SplitName,
    config: BatchConfig,
) -> Result<Vec<ProcessedSampleBatch>> {
    ProcessedDataset::read(manifest_path)?
        .batches(split, config)?
        .collect()
}

pub fn samples_for_split(
    manifest: &ProcessedSamplesManifest,
    split: SplitName,
) -> Vec<&ProcessedSampleRecord> {
    manifest
        .samples
        .iter()
        .filter(|sample| sample.split == split)
        .collect()
}

pub fn load_processed_sample(
    artifact_root: &Path,
    sample: &ProcessedSampleRecord,
) -> Result<LoadedProcessedSample> {
    Ok(LoadedProcessedSample {
        sample_id: sample.sample_id.clone(),
        source_sample_id: sample.metadata.source_sample_id.clone(),
        split: sample.split,
        source: sample.metadata.clone(),
        rgb: load_tensor_artifact(artifact_root, sample, TensorModality::Rgb)?,
        range: load_tensor_artifact(artifact_root, sample, TensorModality::Range)?,
    })
}

pub fn load_tensor_artifact(
    artifact_root: &Path,
    sample: &ProcessedSampleRecord,
    modality: TensorModality,
) -> Result<LoadedTensorArtifact> {
    let reference = match modality {
        TensorModality::Rgb => &sample.rgb,
        TensorModality::Range => &sample.range,
    };
    let path = resolve_artifact_path(artifact_root, &reference.path);
    let bytes = read_tensor_bytes(&sample.sample_id, modality, &path)?;
    validate_tensor_bytes(&sample.sample_id, modality, &path, reference, &bytes)?;
    Ok(LoadedTensorArtifact {
        modality,
        path,
        shape: reference.shape.clone(),
        dtype: reference.dtype,
        layout: reference.layout,
        bytes,
        declared_size_bytes: reference.size_bytes,
        sha256: reference.sha256.clone(),
    })
}

fn read_tensor_bytes(sample_id: &str, modality: TensorModality, path: &Path) -> Result<Vec<u8>> {
    if !path.is_file() {
        return Err(ProcessedLoaderError::MissingArtifact {
            sample_id: sample_id.to_string(),
            modality,
            path: path.to_path_buf(),
        });
    }
    std::fs::read(path).map_err(|source| ProcessedLoaderError::ArtifactIo {
        sample_id: sample_id.to_string(),
        modality,
        path: path.to_path_buf(),
        source,
    })
}

fn validate_tensor_bytes(
    sample_id: &str,
    modality: TensorModality,
    path: &Path,
    reference: &TensorArtifactReference,
    bytes: &[u8],
) -> Result<()> {
    let expected_bytes = expected_tensor_bytes(reference).ok_or_else(|| {
        ProcessedLoaderError::ArtifactShapeOverflow {
            sample_id: sample_id.to_string(),
            modality,
            path: path.to_path_buf(),
        }
    })?;
    if bytes.len() != expected_bytes {
        return Err(ProcessedLoaderError::ArtifactSizeMismatch {
            sample_id: sample_id.to_string(),
            modality,
            path: path.to_path_buf(),
            expected_bytes,
            actual_bytes: bytes.len(),
        });
    }
    if let Some(declared_bytes) = reference.size_bytes {
        if usize::try_from(declared_bytes).ok() != Some(bytes.len()) {
            return Err(ProcessedLoaderError::ArtifactDeclaredSizeMismatch {
                sample_id: sample_id.to_string(),
                modality,
                path: path.to_path_buf(),
                declared_bytes,
                actual_bytes: bytes.len(),
            });
        }
    }
    if let Some(expected_sha256) = &reference.sha256 {
        let actual_sha256 = sha256_hex(bytes);
        if !expected_sha256.eq_ignore_ascii_case(&actual_sha256) {
            return Err(ProcessedLoaderError::ArtifactSha256Mismatch {
                sample_id: sample_id.to_string(),
                modality,
                path: path.to_path_buf(),
                expected_sha256: expected_sha256.clone(),
                actual_sha256,
            });
        }
    }
    Ok(())
}

fn expected_tensor_bytes(reference: &TensorArtifactReference) -> Option<usize> {
    let elements = reference
        .shape
        .dimensions
        .iter()
        .try_fold(1_usize, |acc, dimension| {
            acc.checked_mul(usize::try_from(*dimension).ok()?)
        })?;
    elements.checked_mul(dtype_size_bytes(reference.dtype))
}

fn dtype_size_bytes(dtype: TensorDType) -> usize {
    match dtype {
        TensorDType::F16 => 2,
        TensorDType::F32 => 4,
        TensorDType::U8 => 1,
    }
}

fn resolve_artifact_path(artifact_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let mut rng = DeterministicRng::new(seed);
    for cursor in (1..indices.len()).rev() {
        let swap_with = rng.gen_index(cursor + 1);
        indices.swap(cursor, swap_with);
    }
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn gen_index(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        (self.next_u64() % upper_bound as u64) as usize
    }
}

fn infer_artifact_root(manifest_path: &Path, declared_manifest_path: &Path) -> PathBuf {
    if declared_manifest_path.is_absolute() {
        return declared_manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
    }
    if let Some(root) = strip_path_suffix(manifest_path, declared_manifest_path) {
        return root;
    }
    manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
}

fn strip_path_suffix(path: &Path, suffix: &Path) -> Option<PathBuf> {
    path.ancestors()
        .find(|ancestor| {
            comparable_components(&ancestor.join(suffix)) == comparable_components(path)
        })
        .map(Path::to_path_buf)
}

fn comparable_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().to_ascii_lowercase()),
            Component::Prefix(prefix) => {
                Some(prefix.as_os_str().to_string_lossy().to_ascii_lowercase())
            }
            Component::RootDir | Component::CurDir | Component::ParentDir => None,
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to string cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::write_processed_samples_manifest;
    use crate::manifest::{
        PROCESSED_SAMPLES_SCHEMA_VERSION, ProcessedDatasetOutputLayout, ProcessedPreviewArtifact,
        ProcessedPreviewKind, ProcessedSampleMetadata, ProcessingProvenance,
        SourceSegmentReference,
    };
    use chrono::{DateTime, Utc};
    use tempfile::TempDir;

    fn ts() -> crate::manifest::Timestamp {
        DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .expect("timestamp parses")
            .with_timezone(&Utc)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn sample_record(
        root: &Path,
        sample_id: &str,
        split: SplitName,
        rgb_bytes: &[u8],
        range_bytes: &[u8],
    ) -> ProcessedSampleRecord {
        let frame_index = sample_id
            .rsplit('-')
            .next()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(1);
        let rgb_path = root
            .join("data")
            .join("processed")
            .join("tensors")
            .join("rgb")
            .join(format!("{sample_id}.bin"));
        let range_path = root
            .join("data")
            .join("processed")
            .join("tensors")
            .join("range")
            .join(format!("{sample_id}.bin"));
        std::fs::create_dir_all(rgb_path.parent().expect("rgb parent")).expect("rgb dir");
        std::fs::create_dir_all(range_path.parent().expect("range parent")).expect("range dir");
        std::fs::write(&rgb_path, rgb_bytes).expect("rgb fixture");
        std::fs::write(&range_path, range_bytes).expect("range fixture");

        ProcessedSampleRecord {
            sample_id: sample_id.to_string(),
            split,
            rgb: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/rgb/{sample_id}.bin")),
                shape: TensorShape::new([1, 1, 3]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hwc,
                size_bytes: Some(rgb_bytes.len() as u64),
                sha256: Some(sha256_hex(rgb_bytes)),
            },
            range: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/range/{sample_id}.bin")),
                shape: TensorShape::new([1, 2]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hw,
                size_bytes: Some(range_bytes.len() as u64),
                sha256: Some(sha256_hex(range_bytes)),
            },
            metadata: ProcessedSampleMetadata {
                source_sample_id: format!("source-{sample_id}"),
                frame_id: format!("segment-001:{frame_index:06}"),
                frame_index,
                timestamp_micros: 1_735_689_600_123_456,
                source: SourceSegmentReference {
                    segment_id: "segment-001".to_string(),
                    object_uri: Some("gs://wod/training/segment-001.tfrecord".to_string()),
                    object_path: "training/segment-001.tfrecord".to_string(),
                    local_path: None,
                    file_index: Some(0),
                    sha256: None,
                },
                extracted_rgb_path: PathBuf::from(format!(
                    "data/intermediate/extracted/rgb/{sample_id}.ppm"
                )),
                extracted_range_path: PathBuf::from(format!(
                    "data/intermediate/extracted/range/{sample_id}.bin"
                )),
                provenance: ProcessingProvenance {
                    preprocessor_name: "sfx-test-preprocessor".to_string(),
                    preprocessor_version: "0.1.0".to_string(),
                    processed_at: ts(),
                    config_hash: None,
                    command: None,
                },
            },
            previews: vec![ProcessedPreviewArtifact {
                kind: ProcessedPreviewKind::Rgb,
                path: PathBuf::from(format!("data/processed/previews/rgb/{sample_id}.ppm")),
                media_type: "image/x-portable-pixmap".to_string(),
            }],
        }
    }

    fn write_fixture_dataset(root: &Path) -> PathBuf {
        let manifest_path = root
            .join("data")
            .join("processed")
            .join("processed_samples.json");
        let samples = vec![
            sample_record(root, "sample-001", SplitName::Train, &[1; 12], &[2; 8]),
            sample_record(root, "sample-002", SplitName::Val, &[3; 12], &[4; 8]),
        ];
        let manifest = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts(),
            output_layout: ProcessedDatasetOutputLayout {
                root_dir: PathBuf::from("data/processed"),
                manifest_path: PathBuf::from("data/processed/processed_samples.json"),
                rgb_tensor_dir: PathBuf::from("data/processed/tensors/rgb"),
                range_tensor_dir: PathBuf::from("data/processed/tensors/range"),
                preview_dir: PathBuf::from("data/processed/previews"),
            },
            source_extracted_frames: None,
            samples,
        };
        write_processed_samples_manifest(&manifest_path, &manifest).expect("manifest fixture");
        manifest_path
    }

    fn write_batch_fixture_dataset(root: &Path) -> PathBuf {
        let manifest_path = root
            .join("data")
            .join("processed")
            .join("processed_samples.json");
        let samples = vec![
            sample_record(root, "sample-001", SplitName::Train, &[1; 12], &[2; 8]),
            sample_record(root, "sample-002", SplitName::Train, &[3; 12], &[4; 8]),
            sample_record(root, "sample-003", SplitName::Val, &[5; 12], &[6; 8]),
            sample_record(root, "sample-004", SplitName::Train, &[7; 12], &[8; 8]),
            sample_record(root, "sample-005", SplitName::Train, &[9; 12], &[10; 8]),
            sample_record(root, "sample-006", SplitName::Test, &[11; 12], &[12; 8]),
            sample_record(root, "sample-007", SplitName::Train, &[13; 12], &[14; 8]),
        ];
        let manifest = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts(),
            output_layout: ProcessedDatasetOutputLayout {
                root_dir: PathBuf::from("data/processed"),
                manifest_path: PathBuf::from("data/processed/processed_samples.json"),
                rgb_tensor_dir: PathBuf::from("data/processed/tensors/rgb"),
                range_tensor_dir: PathBuf::from("data/processed/tensors/range"),
                preview_dir: PathBuf::from("data/processed/previews"),
            },
            source_extracted_frames: None,
            samples,
        };
        write_processed_samples_manifest(&manifest_path, &manifest).expect("manifest fixture");
        manifest_path
    }

    fn write_tensor_smoke_dataset(root: &Path) -> PathBuf {
        let manifest_path = root
            .join("data")
            .join("processed")
            .join("processed_samples.json");
        let first_rgb = f32_bytes(&[1.0, 2.0, 3.0]);
        let first_range = f32_bytes(&[4.0, 5.0]);
        let second_rgb = f32_bytes(&[6.0, 7.0, 8.0]);
        let second_range = f32_bytes(&[9.0, 10.0]);
        let samples = vec![
            sample_record(
                root,
                "sample-001",
                SplitName::Train,
                &first_rgb,
                &first_range,
            ),
            sample_record(
                root,
                "sample-002",
                SplitName::Train,
                &second_rgb,
                &second_range,
            ),
        ];
        let manifest = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts(),
            output_layout: ProcessedDatasetOutputLayout {
                root_dir: PathBuf::from("data/processed"),
                manifest_path: PathBuf::from("data/processed/processed_samples.json"),
                rgb_tensor_dir: PathBuf::from("data/processed/tensors/rgb"),
                range_tensor_dir: PathBuf::from("data/processed/tensors/range"),
                preview_dir: PathBuf::from("data/processed/previews"),
            },
            source_extracted_frames: None,
            samples,
        };
        write_processed_samples_manifest(&manifest_path, &manifest).expect("manifest fixture");
        manifest_path
    }

    fn batch_sample_ids(batches: &[ProcessedSampleBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|batch| batch.samples.iter().map(|sample| sample.sample_id.clone()))
            .collect()
    }

    #[test]
    fn loads_valid_processed_split_with_metadata_and_artifact_bytes() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_fixture_dataset(temp_dir.path());

        let loaded = load_processed_split(&manifest_path, SplitName::Train).expect("split loads");

        assert_eq!(loaded.len(), 1);
        let sample = &loaded[0];
        assert_eq!(sample.sample_id, "sample-001");
        assert_eq!(sample.source_sample_id, "source-sample-001");
        assert_eq!(sample.split, SplitName::Train);
        assert_eq!(sample.source.frame_index, 1);
        assert_eq!(sample.rgb.modality, TensorModality::Rgb);
        assert_eq!(sample.rgb.byte_len(), 12);
        assert_eq!(sample.rgb.bytes, vec![1; 12]);
        assert_eq!(sample.range.modality, TensorModality::Range);
        assert_eq!(sample.range.bytes, vec![2; 8]);
    }

    #[test]
    fn exposes_sample_records_by_split_without_loading_artifacts() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_fixture_dataset(temp_dir.path());
        let dataset = read_processed_dataset(&manifest_path).expect("dataset reads");

        let val = dataset.samples_for_split(SplitName::Val);

        assert_eq!(val.len(), 1);
        assert_eq!(val[0].sample_id, "sample-002");
    }

    #[test]
    fn batches_split_samples_with_expected_count_and_no_loss() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_batch_fixture_dataset(temp_dir.path());
        let dataset = read_processed_dataset(&manifest_path).expect("dataset reads");
        let iter = dataset
            .batches_for_split(SplitName::Train, 2)
            .expect("batch iterator");

        assert_eq!(iter.sample_count(), 5);
        assert_eq!(iter.batch_count(), 3);

        let batches = iter.collect::<Result<Vec<_>>>().expect("batches load");

        assert_eq!(
            batches
                .iter()
                .map(ProcessedSampleBatch::len)
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(batch_sample_ids(&batches).len(), 5);
        assert_eq!(
            batch_sample_ids(&batches),
            vec![
                "sample-001",
                "sample-002",
                "sample-004",
                "sample-005",
                "sample-007"
            ]
        );
        assert!(batches.iter().all(|batch| batch.split == SplitName::Train));
    }

    #[test]
    fn shuffled_batches_are_deterministic_for_same_seed_without_duplication() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_batch_fixture_dataset(temp_dir.path());
        let dataset = read_processed_dataset(&manifest_path).expect("dataset reads");
        let config = BatchConfig::shuffled(2, 0x5eed);

        let first = dataset
            .batches(SplitName::Train, config)
            .expect("first iterator")
            .collect::<Result<Vec<_>>>()
            .expect("first batches load");
        let second = dataset
            .batches(SplitName::Train, config)
            .expect("second iterator")
            .collect::<Result<Vec<_>>>()
            .expect("second batches load");
        let first_ids = batch_sample_ids(&first);
        let mut sorted_first_ids = first_ids.clone();
        sorted_first_ids.sort();
        sorted_first_ids.dedup();

        assert_eq!(first_ids, batch_sample_ids(&second));
        assert_ne!(
            first_ids,
            vec![
                "sample-001",
                "sample-002",
                "sample-004",
                "sample-005",
                "sample-007"
            ]
        );
        assert_eq!(
            sorted_first_ids,
            vec![
                "sample-001",
                "sample-002",
                "sample-004",
                "sample-005",
                "sample-007"
            ]
        );
    }

    #[test]
    fn non_shuffled_batches_keep_manifest_order_stable() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_batch_fixture_dataset(temp_dir.path());

        let batches = load_processed_batches(
            &manifest_path,
            SplitName::Train,
            BatchConfig::new(3).without_shuffle(),
        )
        .expect("batches load");

        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.batch_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            batch_sample_ids(&batches),
            vec![
                "sample-001",
                "sample-002",
                "sample-004",
                "sample-005",
                "sample-007"
            ]
        );
    }

    #[test]
    fn smoke_loads_batches_and_converts_tensors_for_model_use() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_tensor_smoke_dataset(temp_dir.path());
        let dataset = read_processed_dataset(&manifest_path).expect("dataset reads");
        let mut batches = dataset
            .batches_for_split(SplitName::Train, 2)
            .expect("batch iterator");

        let batch = batches
            .next()
            .expect("first batch")
            .expect("first batch loads");
        assert!(batches.next().is_none());
        let model_batch =
            crate::tensor::convert_batch_tensors(&batch.samples).expect("batch converts");

        assert_eq!(batch.batch_index, 0);
        assert_eq!(model_batch.sample_ids, vec!["sample-001", "sample-002"]);
        assert_eq!(model_batch.rgb.batch_shape, vec![2, 1, 1, 3]);
        assert_eq!(model_batch.rgb.values, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
        assert_eq!(model_batch.range.batch_shape, vec![2, 1, 2]);
        assert_eq!(model_batch.range.values, vec![4.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn rejects_zero_batch_size() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_batch_fixture_dataset(temp_dir.path());
        let dataset = read_processed_dataset(&manifest_path).expect("dataset reads");

        let error = dataset
            .batches_for_split(SplitName::Train, 0)
            .expect_err("zero batch size fails");

        match error {
            ProcessedLoaderError::InvalidBatchSize { batch_size } => assert_eq!(batch_size, 0),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn reports_missing_artifacts_explicitly() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_fixture_dataset(temp_dir.path());
        let missing_path = temp_dir
            .path()
            .join("data")
            .join("processed")
            .join("tensors")
            .join("rgb")
            .join("sample-001.bin");
        std::fs::remove_file(&missing_path).expect("remove fixture");

        let error =
            load_processed_split(&manifest_path, SplitName::Train).expect_err("missing fails");

        match error {
            ProcessedLoaderError::MissingArtifact {
                sample_id,
                modality,
                path,
            } => {
                assert_eq!(sample_id, "sample-001");
                assert_eq!(modality, TensorModality::Rgb);
                assert_eq!(path, missing_path);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn reports_corrupt_artifact_size_explicitly() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_fixture_dataset(temp_dir.path());
        let corrupt_path = temp_dir
            .path()
            .join("data")
            .join("processed")
            .join("tensors")
            .join("range")
            .join("sample-001.bin");
        std::fs::write(&corrupt_path, [9_u8; 4]).expect("corrupt fixture");

        let error =
            load_processed_split(&manifest_path, SplitName::Train).expect_err("corrupt fails");

        match error {
            ProcessedLoaderError::ArtifactSizeMismatch {
                sample_id,
                modality,
                path,
                expected_bytes,
                actual_bytes,
            } => {
                assert_eq!(sample_id, "sample-001");
                assert_eq!(modality, TensorModality::Range);
                assert_eq!(path, corrupt_path);
                assert_eq!(expected_bytes, 8);
                assert_eq!(actual_bytes, 4);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn reports_corrupt_artifact_digest_explicitly() {
        let temp_dir = TempDir::new().expect("tempdir");
        let manifest_path = write_fixture_dataset(temp_dir.path());
        let corrupt_path = temp_dir
            .path()
            .join("data")
            .join("processed")
            .join("tensors")
            .join("rgb")
            .join("sample-001.bin");
        std::fs::write(&corrupt_path, [8_u8; 12]).expect("corrupt fixture");

        let error =
            load_processed_split(&manifest_path, SplitName::Train).expect_err("corrupt fails");

        match error {
            ProcessedLoaderError::ArtifactSha256Mismatch {
                sample_id,
                modality,
                path,
                ..
            } => {
                assert_eq!(sample_id, "sample-001");
                assert_eq!(modality, TensorModality::Rgb);
                assert_eq!(path, corrupt_path);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
