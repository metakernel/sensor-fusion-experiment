use crate::manifest::{
    PROCESSED_SAMPLES_SCHEMA_VERSION, ProcessedSamplesManifest, SPLITS_SCHEMA_VERSION,
    SplitAssignment, SplitName, SplitsManifest,
};
use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitRatios {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.8,
            val: 0.1,
            test: 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitCounts {
    pub train: usize,
    pub val: usize,
    pub test: usize,
}

impl SplitCounts {
    pub fn total(self) -> usize {
        self.train + self.val + self.test
    }

    pub fn get(self, split: SplitName) -> usize {
        match split {
            SplitName::Train => self.train,
            SplitName::Val => self.val,
            SplitName::Test => self.test,
        }
    }

    fn increment(&mut self, split: SplitName) {
        match split {
            SplitName::Train => self.train += 1,
            SplitName::Val => self.val += 1,
            SplitName::Test => self.test += 1,
        }
    }

    fn decrement(&mut self, split: SplitName) {
        match split {
            SplitName::Train => self.train -= 1,
            SplitName::Val => self.val -= 1,
            SplitName::Test => self.test -= 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitAssignmentError {
    InvalidRatio { split: SplitName, value: String },
    EmptyRatios,
    EmptySampleId { index: usize },
    DuplicateSampleId { sample_id: String },
    SchemaVersionMismatch,
}

impl Display for SplitAssignmentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRatio { split, value } => {
                write!(f, "invalid {split:?} split ratio: {value}")
            }
            Self::EmptyRatios => write!(f, "at least one split ratio must be greater than zero"),
            Self::EmptySampleId { index } => write!(f, "sample id at index {index} is empty"),
            Self::DuplicateSampleId { sample_id } => write!(f, "duplicate sample id: {sample_id}"),
            Self::SchemaVersionMismatch => write!(f, "processed samples manifest schema mismatch"),
        }
    }
}

impl std::error::Error for SplitAssignmentError {}

pub fn split_counts(
    total_samples: usize,
    ratios: SplitRatios,
) -> Result<SplitCounts, SplitAssignmentError> {
    let normalized = normalized_ratios(ratios)?;
    if total_samples == 0 {
        return Ok(SplitCounts {
            train: 0,
            val: 0,
            test: 0,
        });
    }

    let active = active_splits(normalized);
    if total_samples < active.len() {
        let mut ranked = active;
        ranked.sort_by(|left, right| {
            ratio_for(normalized, *right)
                .partial_cmp(&ratio_for(normalized, *left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| split_order(*left).cmp(&split_order(*right)))
        });

        let mut counts = SplitCounts {
            train: 0,
            val: 0,
            test: 0,
        };
        for split in ranked.into_iter().take(total_samples) {
            counts.increment(split);
        }
        return Ok(counts);
    }

    let raw = [
        (SplitName::Train, normalized.train * total_samples as f64),
        (SplitName::Val, normalized.val * total_samples as f64),
        (SplitName::Test, normalized.test * total_samples as f64),
    ];
    let mut counts = SplitCounts {
        train: raw[0].1.floor() as usize,
        val: raw[1].1.floor() as usize,
        test: raw[2].1.floor() as usize,
    };

    let mut remainders = raw
        .iter()
        .map(|(split, value)| (*split, value - value.floor()))
        .collect::<Vec<_>>();
    remainders.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| split_order(left.0).cmp(&split_order(right.0)))
    });

    for (split, _) in remainders
        .iter()
        .copied()
        .cycle()
        .take(total_samples - counts.total())
    {
        counts.increment(split);
    }

    for split in active_splits(normalized) {
        if counts.get(split) == 0 {
            let donor = donor_split(counts, split)
                .expect("a donor split must exist when total samples covers active splits");
            counts.decrement(donor);
            counts.increment(split);
        }
    }

    debug_assert_eq!(counts.total(), total_samples);
    Ok(counts)
}

pub fn assign_sample_splits(
    sample_ids: impl IntoIterator<Item = impl Into<String>>,
    split_seed: u64,
    ratios: SplitRatios,
    assigned_at: DateTime<Utc>,
) -> Result<SplitsManifest, SplitAssignmentError> {
    let mut sample_ids = validate_sample_ids(sample_ids)?;
    sample_ids.sort_by(|left, right| {
        split_sort_key(split_seed, left)
            .cmp(&split_sort_key(split_seed, right))
            .then_with(|| left.cmp(right))
    });

    let counts = split_counts(sample_ids.len(), ratios)?;
    let mut assignments = Vec::with_capacity(sample_ids.len());
    let mut next_index = 0;
    for split in [SplitName::Train, SplitName::Val, SplitName::Test] {
        for sample_id in sample_ids[next_index..next_index + counts.get(split)]
            .iter()
            .cloned()
        {
            assignments.push(SplitAssignment {
                sample_id,
                split,
                assigned_at,
            });
        }
        next_index += counts.get(split);
    }
    assignments.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

    Ok(SplitsManifest {
        schema_version: SPLITS_SCHEMA_VERSION,
        generated_at: assigned_at,
        split_seed,
        assignments,
    })
}

pub fn enforce_processed_sample_splits(
    manifest: &mut ProcessedSamplesManifest,
    split_seed: u64,
    ratios: SplitRatios,
    assigned_at: DateTime<Utc>,
) -> Result<SplitsManifest, SplitAssignmentError> {
    if manifest.schema_version != PROCESSED_SAMPLES_SCHEMA_VERSION {
        return Err(SplitAssignmentError::SchemaVersionMismatch);
    }

    let splits = assign_sample_splits(
        manifest
            .samples
            .iter()
            .map(|sample| sample.sample_id.clone()),
        split_seed,
        ratios,
        assigned_at,
    )?;
    let split_by_sample = splits
        .assignments
        .iter()
        .map(|assignment| (assignment.sample_id.as_str(), assignment.split))
        .collect::<BTreeMap<_, _>>();

    for sample in &mut manifest.samples {
        sample.split = *split_by_sample
            .get(sample.sample_id.as_str())
            .expect("assignment was generated from processed sample ids");
    }
    manifest.generated_at = assigned_at;

    Ok(splits)
}

pub fn split_assignment_counts(manifest: &SplitsManifest) -> SplitCounts {
    let mut counts = SplitCounts {
        train: 0,
        val: 0,
        test: 0,
    };
    for assignment in &manifest.assignments {
        counts.increment(assignment.split);
    }
    counts
}

fn validate_sample_ids(
    sample_ids: impl IntoIterator<Item = impl Into<String>>,
) -> Result<Vec<String>, SplitAssignmentError> {
    let mut seen = BTreeSet::new();
    let mut validated = Vec::new();
    for (index, sample_id) in sample_ids.into_iter().enumerate() {
        let sample_id = sample_id.into();
        if sample_id.trim().is_empty() {
            return Err(SplitAssignmentError::EmptySampleId { index });
        }
        if !seen.insert(sample_id.clone()) {
            return Err(SplitAssignmentError::DuplicateSampleId { sample_id });
        }
        validated.push(sample_id);
    }
    Ok(validated)
}

fn normalized_ratios(ratios: SplitRatios) -> Result<SplitRatios, SplitAssignmentError> {
    for (split, value) in [
        (SplitName::Train, ratios.train),
        (SplitName::Val, ratios.val),
        (SplitName::Test, ratios.test),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(SplitAssignmentError::InvalidRatio {
                split,
                value: value.to_string(),
            });
        }
    }

    let sum = ratios.train + ratios.val + ratios.test;
    if sum <= 0.0 {
        return Err(SplitAssignmentError::EmptyRatios);
    }

    Ok(SplitRatios {
        train: ratios.train / sum,
        val: ratios.val / sum,
        test: ratios.test / sum,
    })
}

fn active_splits(ratios: SplitRatios) -> Vec<SplitName> {
    [SplitName::Train, SplitName::Val, SplitName::Test]
        .into_iter()
        .filter(|split| ratio_for(ratios, *split) > 0.0)
        .collect()
}

fn donor_split(counts: SplitCounts, needed: SplitName) -> Option<SplitName> {
    [SplitName::Train, SplitName::Val, SplitName::Test]
        .into_iter()
        .filter(|split| *split != needed && counts.get(*split) > 1)
        .max_by_key(|split| (counts.get(*split), std::cmp::Reverse(split_order(*split))))
}

fn ratio_for(ratios: SplitRatios, split: SplitName) -> f64 {
    match split {
        SplitName::Train => ratios.train,
        SplitName::Val => ratios.val,
        SplitName::Test => ratios.test,
    }
}

fn split_order(split: SplitName) -> u8 {
    match split {
        SplitName::Train => 0,
        SplitName::Val => 1,
        SplitName::Test => 2,
    }
}

fn split_sort_key(seed: u64, sample_id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed.rotate_left(17);
    for byte in seed.to_le_bytes().into_iter().chain(sample_id.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        ManifestReference, ProcessedDatasetOutputLayout, ProcessedSampleMetadata,
        ProcessedSampleRecord, ProcessingProvenance, SourceSegmentReference,
        TensorArtifactReference, TensorDType, TensorLayout, TensorShape,
    };
    use std::path::PathBuf;

    fn ts() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .expect("timestamp parses")
            .with_timezone(&Utc)
    }

    #[test]
    fn split_counts_follow_ratios_and_total() {
        let counts = split_counts(10, SplitRatios::default()).expect("counts should build");
        assert_eq!(
            counts,
            SplitCounts {
                train: 8,
                val: 1,
                test: 1
            }
        );
        assert_eq!(counts.total(), 10);
    }

    #[test]
    fn split_counts_handle_small_datasets_deterministically() {
        assert_eq!(
            split_counts(1, SplitRatios::default()).expect("one sample counts"),
            SplitCounts {
                train: 1,
                val: 0,
                test: 0
            }
        );
        assert_eq!(
            split_counts(2, SplitRatios::default()).expect("two sample counts"),
            SplitCounts {
                train: 1,
                val: 1,
                test: 0
            }
        );
        assert_eq!(
            split_counts(4, SplitRatios::default()).expect("four sample counts"),
            SplitCounts {
                train: 2,
                val: 1,
                test: 1
            }
        );
    }

    #[test]
    fn assignments_are_deterministic_for_same_seed() {
        let sample_ids = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let first = assign_sample_splits(sample_ids, 42, SplitRatios::default(), ts())
            .expect("first assignment");
        let second = assign_sample_splits(sample_ids, 42, SplitRatios::default(), ts())
            .expect("second assignment");

        assert_eq!(first, second);
    }

    #[test]
    fn every_sample_gets_exactly_one_assignment() {
        let manifest = assign_sample_splits(
            ["sample-001", "sample-002", "sample-003", "sample-004"],
            7,
            SplitRatios::default(),
            ts(),
        )
        .expect("assignments should build");

        let ids = manifest
            .assignments
            .iter()
            .map(|assignment| assignment.sample_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec!["sample-001", "sample-002", "sample-003", "sample-004"]
        );
        assert_eq!(split_assignment_counts(&manifest).total(), 4);
    }

    #[test]
    fn enforcement_updates_processed_manifest_and_emits_split_manifest() {
        let mut processed = ProcessedSamplesManifest {
            schema_version: PROCESSED_SAMPLES_SCHEMA_VERSION,
            generated_at: ts(),
            output_layout: ProcessedDatasetOutputLayout::new("data/processed"),
            source_extracted_frames: Some(ManifestReference {
                manifest_path: PathBuf::from("data/intermediate/extracted/extracted_frames.json"),
                schema_version: crate::manifest::EXTRACTED_FRAMES_SCHEMA_VERSION,
                generated_at: ts(),
            }),
            samples: ["sample-001", "sample-002", "sample-003", "sample-004"]
                .into_iter()
                .map(sample_record)
                .collect(),
        };

        let splits =
            enforce_processed_sample_splits(&mut processed, 99, SplitRatios::default(), ts())
                .expect("split enforcement should succeed");

        assert_eq!(splits.schema_version, SPLITS_SCHEMA_VERSION);
        assert_eq!(splits.split_seed, 99);
        assert_eq!(splits.assignments.len(), processed.samples.len());
        assert_eq!(
            split_assignment_counts(&splits),
            SplitCounts {
                train: 2,
                val: 1,
                test: 1
            }
        );

        for sample in &processed.samples {
            let assignment = splits
                .assignments
                .iter()
                .find(|assignment| assignment.sample_id == sample.sample_id)
                .expect("every processed sample has one split assignment");
            assert_eq!(sample.split, assignment.split);
        }
    }

    fn sample_record(sample_id: &str) -> ProcessedSampleRecord {
        ProcessedSampleRecord {
            sample_id: sample_id.to_string(),
            split: SplitName::Train,
            rgb: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/rgb/{sample_id}.npy")),
                shape: TensorShape::new([2, 2, 3]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hwc,
                size_bytes: Some(48),
                sha256: None,
            },
            range: TensorArtifactReference {
                path: PathBuf::from(format!("data/processed/tensors/range/{sample_id}.npy")),
                shape: TensorShape::new([2, 2]),
                dtype: TensorDType::F32,
                layout: TensorLayout::Hw,
                size_bytes: Some(16),
                sha256: None,
            },
            metadata: ProcessedSampleMetadata {
                source_sample_id: sample_id.to_string(),
                frame_id: format!("{sample_id}:000000"),
                frame_index: 0,
                timestamp_micros: 1,
                source: SourceSegmentReference {
                    segment_id: sample_id.to_string(),
                    object_uri: None,
                    object_path: format!("{sample_id}.tfrecord"),
                    local_path: None,
                    file_index: None,
                    sha256: None,
                },
                extracted_rgb_path: PathBuf::from(format!("data/intermediate/rgb/{sample_id}.ppm")),
                extracted_range_path: PathBuf::from(format!(
                    "data/intermediate/range/{sample_id}.json"
                )),
                provenance: ProcessingProvenance {
                    preprocessor_name: "test".to_string(),
                    preprocessor_version: "0.0.0".to_string(),
                    processed_at: ts(),
                    config_hash: None,
                    command: None,
                },
            },
            previews: Vec::new(),
        }
    }
}
