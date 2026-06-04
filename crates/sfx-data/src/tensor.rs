use crate::loader::{LoadedProcessedSample, LoadedTensorArtifact, TensorModality};
use crate::manifest::{TensorDType, TensorLayout, TensorShape};
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, TensorConversionError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorConversionError {
    BufferLengthMismatch {
        sample_id: String,
        modality: TensorModality,
        shape: TensorShape,
        dtype: TensorDType,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    ShapeMismatch {
        sample_id: String,
        modality: TensorModality,
        expected_shape: TensorShape,
        actual_shape: TensorShape,
    },
    BatchShapeMismatch {
        sample_id: String,
        modality: TensorModality,
        expected_shape: TensorShape,
        actual_shape: TensorShape,
    },
    ShapeElementOverflow {
        sample_id: String,
        modality: TensorModality,
        shape: TensorShape,
    },
    EmptyBatch,
}

impl Display for TensorConversionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferLengthMismatch {
                sample_id,
                modality,
                shape,
                dtype,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "{modality} tensor buffer for sample {sample_id} has {actual_bytes} bytes, expected {expected_bytes} bytes for dtype {dtype:?} and shape {:?}",
                shape.dimensions
            ),
            Self::ShapeMismatch {
                sample_id,
                modality,
                expected_shape,
                actual_shape,
            } => write!(
                f,
                "{modality} tensor for sample {sample_id} has shape {:?}, expected {:?}",
                actual_shape.dimensions, expected_shape.dimensions
            ),
            Self::BatchShapeMismatch {
                sample_id,
                modality,
                expected_shape,
                actual_shape,
            } => write!(
                f,
                "{modality} tensor for sample {sample_id} cannot be batched: shape {:?} does not match batch shape {:?}",
                actual_shape.dimensions, expected_shape.dimensions
            ),
            Self::ShapeElementOverflow {
                sample_id,
                modality,
                shape,
            } => write!(
                f,
                "{modality} tensor shape for sample {sample_id} overflows addressable element count: {:?}",
                shape.dimensions
            ),
            Self::EmptyBatch => f.write_str("cannot convert an empty processed-sample batch"),
        }
    }
}

impl std::error::Error for TensorConversionError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelReadyTensor {
    pub modality: TensorModality,
    pub shape: TensorShape,
    pub layout: TensorLayout,
    pub values: Vec<f32>,
}

impl ModelReadyTensor {
    pub fn element_count(&self) -> usize {
        self.values.len()
    }

    pub fn shape_dimensions(&self) -> &[u32] {
        &self.shape.dimensions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelReadySample {
    pub sample_id: String,
    pub rgb: ModelReadyTensor,
    pub range: ModelReadyTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchedModelTensor {
    pub modality: TensorModality,
    pub sample_shape: TensorShape,
    pub batch_shape: Vec<u32>,
    pub layout: TensorLayout,
    pub values: Vec<f32>,
}

impl BatchedModelTensor {
    pub fn batch_size(&self) -> u32 {
        self.batch_shape.first().copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelReadyBatch {
    pub sample_ids: Vec<String>,
    pub rgb: BatchedModelTensor,
    pub range: BatchedModelTensor,
}

pub fn convert_tensor_artifact(
    sample_id: impl AsRef<str>,
    artifact: &LoadedTensorArtifact,
) -> Result<ModelReadyTensor> {
    let sample_id = sample_id.as_ref();
    validate_buffer_len(sample_id, artifact)?;
    Ok(ModelReadyTensor {
        modality: artifact.modality,
        shape: artifact.shape.clone(),
        layout: artifact.layout,
        values: decode_to_f32(&artifact.bytes, artifact.dtype),
    })
}

pub fn convert_tensor_artifact_with_shape(
    sample_id: impl AsRef<str>,
    artifact: &LoadedTensorArtifact,
    expected_shape: &TensorShape,
) -> Result<ModelReadyTensor> {
    let sample_id = sample_id.as_ref();
    validate_declared_shape(sample_id, artifact, expected_shape)?;
    convert_tensor_artifact(sample_id, artifact)
}

pub fn convert_sample_tensors(sample: &LoadedProcessedSample) -> Result<ModelReadySample> {
    Ok(ModelReadySample {
        sample_id: sample.sample_id.clone(),
        rgb: convert_tensor_artifact(&sample.sample_id, &sample.rgb)?,
        range: convert_tensor_artifact(&sample.sample_id, &sample.range)?,
    })
}

pub fn convert_sample_tensors_with_shapes(
    sample: &LoadedProcessedSample,
    expected_rgb_shape: &TensorShape,
    expected_range_shape: &TensorShape,
) -> Result<ModelReadySample> {
    Ok(ModelReadySample {
        sample_id: sample.sample_id.clone(),
        rgb: convert_tensor_artifact_with_shape(
            &sample.sample_id,
            &sample.rgb,
            expected_rgb_shape,
        )?,
        range: convert_tensor_artifact_with_shape(
            &sample.sample_id,
            &sample.range,
            expected_range_shape,
        )?,
    })
}

pub fn convert_batch_tensors(samples: &[LoadedProcessedSample]) -> Result<ModelReadyBatch> {
    let first = samples.first().ok_or(TensorConversionError::EmptyBatch)?;
    convert_batch_tensors_with_shapes(samples, &first.rgb.shape, &first.range.shape)
}

pub fn convert_batch_tensors_with_shapes(
    samples: &[LoadedProcessedSample],
    expected_rgb_shape: &TensorShape,
    expected_range_shape: &TensorShape,
) -> Result<ModelReadyBatch> {
    if samples.is_empty() {
        return Err(TensorConversionError::EmptyBatch);
    }

    let mut sample_ids = Vec::with_capacity(samples.len());
    let mut rgb_values = Vec::new();
    let mut range_values = Vec::new();
    let mut rgb_layout = samples[0].rgb.layout;
    let mut range_layout = samples[0].range.layout;

    for sample in samples {
        validate_batch_shape(&sample.sample_id, &sample.rgb, expected_rgb_shape)?;
        validate_batch_shape(&sample.sample_id, &sample.range, expected_range_shape)?;

        let converted =
            convert_sample_tensors_with_shapes(sample, expected_rgb_shape, expected_range_shape)?;
        rgb_layout = converted.rgb.layout;
        range_layout = converted.range.layout;
        sample_ids.push(converted.sample_id);
        rgb_values.extend(converted.rgb.values);
        range_values.extend(converted.range.values);
    }

    let batch_size = u32::try_from(samples.len()).unwrap_or(u32::MAX);
    Ok(ModelReadyBatch {
        sample_ids,
        rgb: BatchedModelTensor {
            modality: TensorModality::Rgb,
            sample_shape: expected_rgb_shape.clone(),
            batch_shape: batch_shape(batch_size, expected_rgb_shape),
            layout: rgb_layout,
            values: rgb_values,
        },
        range: BatchedModelTensor {
            modality: TensorModality::Range,
            sample_shape: expected_range_shape.clone(),
            batch_shape: batch_shape(batch_size, expected_range_shape),
            layout: range_layout,
            values: range_values,
        },
    })
}

fn validate_declared_shape(
    sample_id: &str,
    artifact: &LoadedTensorArtifact,
    expected_shape: &TensorShape,
) -> Result<()> {
    if &artifact.shape != expected_shape {
        return Err(TensorConversionError::ShapeMismatch {
            sample_id: sample_id.to_string(),
            modality: artifact.modality,
            expected_shape: expected_shape.clone(),
            actual_shape: artifact.shape.clone(),
        });
    }
    Ok(())
}

fn validate_batch_shape(
    sample_id: &str,
    artifact: &LoadedTensorArtifact,
    expected_shape: &TensorShape,
) -> Result<()> {
    if &artifact.shape != expected_shape {
        return Err(TensorConversionError::BatchShapeMismatch {
            sample_id: sample_id.to_string(),
            modality: artifact.modality,
            expected_shape: expected_shape.clone(),
            actual_shape: artifact.shape.clone(),
        });
    }
    Ok(())
}

fn validate_buffer_len(sample_id: &str, artifact: &LoadedTensorArtifact) -> Result<()> {
    let expected_bytes =
        expected_tensor_bytes(&artifact.shape, artifact.dtype).ok_or_else(|| {
            TensorConversionError::ShapeElementOverflow {
                sample_id: sample_id.to_string(),
                modality: artifact.modality,
                shape: artifact.shape.clone(),
            }
        })?;
    let actual_bytes = artifact.bytes.len();
    if actual_bytes != expected_bytes {
        return Err(TensorConversionError::BufferLengthMismatch {
            sample_id: sample_id.to_string(),
            modality: artifact.modality,
            shape: artifact.shape.clone(),
            dtype: artifact.dtype,
            expected_bytes,
            actual_bytes,
        });
    }
    Ok(())
}

fn batch_shape(batch_size: u32, sample_shape: &TensorShape) -> Vec<u32> {
    let mut dimensions = Vec::with_capacity(sample_shape.dimensions.len() + 1);
    dimensions.push(batch_size);
    dimensions.extend(sample_shape.dimensions.iter().copied());
    dimensions
}

fn expected_tensor_bytes(shape: &TensorShape, dtype: TensorDType) -> Option<usize> {
    element_count(shape)?.checked_mul(dtype_size_bytes(dtype))
}

fn element_count(shape: &TensorShape) -> Option<usize> {
    shape.dimensions.iter().try_fold(1_usize, |acc, dimension| {
        acc.checked_mul(usize::try_from(*dimension).ok()?)
    })
}

fn dtype_size_bytes(dtype: TensorDType) -> usize {
    match dtype {
        TensorDType::F16 => 2,
        TensorDType::F32 => 4,
        TensorDType::U8 => 1,
    }
}

fn decode_to_f32(bytes: &[u8], dtype: TensorDType) -> Vec<f32> {
    match dtype {
        TensorDType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        TensorDType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        TensorDType::U8 => bytes.iter().map(|value| f32::from(*value)).collect(),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;

    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * 2_f32.powi(-14) * (fraction as f32 / 1024.0),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * 2_f32.powi(exponent - 15) * (1.0 + fraction as f32 / 1024.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::LoadedTensorArtifact;
    use crate::manifest::SplitName;
    use crate::manifest::{ProcessedSampleMetadata, ProcessingProvenance, SourceSegmentReference};
    use chrono::{DateTime, Utc};
    use std::path::PathBuf;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn tensor(
        modality: TensorModality,
        shape: TensorShape,
        layout: TensorLayout,
        values: &[f32],
    ) -> LoadedTensorArtifact {
        LoadedTensorArtifact {
            modality,
            path: PathBuf::from(format!("{}.bin", modality.as_str())),
            shape,
            dtype: TensorDType::F32,
            layout,
            bytes: f32_bytes(values),
            declared_size_bytes: Some((values.len() * 4) as u64),
            sha256: None,
        }
    }

    fn sample(
        sample_id: &str,
        rgb: LoadedTensorArtifact,
        range: LoadedTensorArtifact,
    ) -> LoadedProcessedSample {
        LoadedProcessedSample {
            sample_id: sample_id.to_string(),
            source_sample_id: format!("source-{sample_id}"),
            split: SplitName::Train,
            source: ProcessedSampleMetadata {
                source_sample_id: format!("source-{sample_id}"),
                frame_id: "segment:000001".to_string(),
                frame_index: 1,
                timestamp_micros: 1,
                source: SourceSegmentReference {
                    segment_id: "segment".to_string(),
                    object_uri: None,
                    object_path: "segment.tfrecord".to_string(),
                    local_path: None,
                    file_index: None,
                    sha256: None,
                },
                extracted_rgb_path: PathBuf::from("rgb.ppm"),
                extracted_range_path: PathBuf::from("range.bin"),
                provenance: ProcessingProvenance {
                    preprocessor_name: "test".to_string(),
                    preprocessor_version: "0.1.0".to_string(),
                    processed_at: DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                        .expect("timestamp")
                        .with_timezone(&Utc),
                    config_hash: None,
                    command: None,
                },
            },
            rgb,
            range,
        }
    }

    #[test]
    fn converts_loaded_sample_buffers_into_model_ready_tensors_and_batches() {
        let rgb_shape = TensorShape::new([1, 1, 3]);
        let range_shape = TensorShape::new([1, 2]);
        let first = sample(
            "sample-001",
            tensor(
                TensorModality::Rgb,
                rgb_shape.clone(),
                TensorLayout::Hwc,
                &[1.0, 2.0, 3.0],
            ),
            tensor(
                TensorModality::Range,
                range_shape.clone(),
                TensorLayout::Hw,
                &[4.0, 5.0],
            ),
        );
        let second = sample(
            "sample-002",
            tensor(
                TensorModality::Rgb,
                rgb_shape.clone(),
                TensorLayout::Hwc,
                &[6.0, 7.0, 8.0],
            ),
            tensor(
                TensorModality::Range,
                range_shape.clone(),
                TensorLayout::Hw,
                &[9.0, 10.0],
            ),
        );

        let converted = convert_sample_tensors_with_shapes(&first, &rgb_shape, &range_shape)
            .expect("sample converts");
        assert_eq!(converted.rgb.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(converted.rgb.shape_dimensions(), &[1, 1, 3]);
        assert_eq!(converted.range.values, vec![4.0, 5.0]);

        let batch = convert_batch_tensors(&[first, second]).expect("batch converts");
        assert_eq!(batch.sample_ids, vec!["sample-001", "sample-002"]);
        assert_eq!(batch.rgb.batch_shape, vec![2, 1, 1, 3]);
        assert_eq!(batch.rgb.values, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
        assert_eq!(batch.range.batch_shape, vec![2, 1, 2]);
        assert_eq!(batch.range.values, vec![4.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn reports_buffer_length_mismatches_before_decoding() {
        let mut artifact = tensor(
            TensorModality::Rgb,
            TensorShape::new([1, 1, 3]),
            TensorLayout::Hwc,
            &[1.0, 2.0],
        );
        artifact.declared_size_bytes = Some(artifact.bytes.len() as u64);

        let error = convert_tensor_artifact("sample-001", &artifact).expect_err("length mismatch");

        assert_eq!(
            error,
            TensorConversionError::BufferLengthMismatch {
                sample_id: "sample-001".to_string(),
                modality: TensorModality::Rgb,
                shape: TensorShape::new([1, 1, 3]),
                dtype: TensorDType::F32,
                expected_bytes: 12,
                actual_bytes: 8,
            }
        );
    }

    #[test]
    fn reports_declared_shape_mismatches() {
        let artifact = tensor(
            TensorModality::Range,
            TensorShape::new([1, 2]),
            TensorLayout::Hw,
            &[1.0, 2.0],
        );

        let error =
            convert_tensor_artifact_with_shape("sample-001", &artifact, &TensorShape::new([2, 2]))
                .expect_err("shape mismatch");

        assert_eq!(
            error,
            TensorConversionError::ShapeMismatch {
                sample_id: "sample-001".to_string(),
                modality: TensorModality::Range,
                expected_shape: TensorShape::new([2, 2]),
                actual_shape: TensorShape::new([1, 2]),
            }
        );
    }

    #[test]
    fn reports_batch_shape_mismatches() {
        let range_shape = TensorShape::new([1, 2]);
        let first = sample(
            "sample-001",
            tensor(
                TensorModality::Rgb,
                TensorShape::new([1, 1, 3]),
                TensorLayout::Hwc,
                &[1.0, 2.0, 3.0],
            ),
            tensor(
                TensorModality::Range,
                range_shape.clone(),
                TensorLayout::Hw,
                &[4.0, 5.0],
            ),
        );
        let second = sample(
            "sample-002",
            tensor(
                TensorModality::Rgb,
                TensorShape::new([1, 3, 1]),
                TensorLayout::Hwc,
                &[6.0, 7.0, 8.0],
            ),
            tensor(
                TensorModality::Range,
                range_shape,
                TensorLayout::Hw,
                &[9.0, 10.0],
            ),
        );

        let error = convert_batch_tensors(&[first, second]).expect_err("batch shape mismatch");

        assert_eq!(
            error,
            TensorConversionError::BatchShapeMismatch {
                sample_id: "sample-002".to_string(),
                modality: TensorModality::Rgb,
                expected_shape: TensorShape::new([1, 1, 3]),
                actual_shape: TensorShape::new([1, 3, 1]),
            }
        );
    }
}
