use flate2::Compression;
use flate2::write::DeflateEncoder;
use std::io::Write;

const I8_MIN: f32 = i8::MIN as f32;
const I8_MAX: f32 = i8::MAX as f32;
const I8_FULL_RANGE: f32 = I8_MAX - I8_MIN;
const SERIALIZED_HEADER_BYTES: usize = 4 + 1 + 4 + 4 + 1 + 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMinMaxPolicy {
    Symmetric,
    Asymmetric,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int8QuantizationMetadata {
    pub scale: f32,
    pub zero_point: i8,
    pub min_value: f32,
    pub max_value: f32,
    pub policy: QuantizationMinMaxPolicy,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LatentAccountingError {
    #[error("latent vector must not be empty")]
    EmptyInput,
    #[error("latent value at index {index} is not finite")]
    NonFiniteInput { index: usize },
    #[error("invalid quantization metadata: {message}")]
    InvalidMetadata { message: &'static str },
}

pub fn raw_f32_latent_byte_size(latent: &[f32]) -> usize {
    latent.len() * std::mem::size_of::<f32>()
}

pub fn compute_int8_quantization_metadata(
    latent: &[f32],
    policy: QuantizationMinMaxPolicy,
) -> Result<Int8QuantizationMetadata, LatentAccountingError> {
    if latent.is_empty() {
        return Err(LatentAccountingError::EmptyInput);
    }
    validate_latent_values(latent)?;

    let (min_value, max_value) = latent.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(value), max_v.max(value)),
    );

    if min_value == max_value {
        return Ok(Int8QuantizationMetadata {
            scale: 1.0,
            zero_point: 0,
            min_value,
            max_value,
            policy,
        });
    }

    let metadata = match policy {
        QuantizationMinMaxPolicy::Symmetric => {
            let max_abs = min_value.abs().max(max_value.abs());
            Int8QuantizationMetadata {
                scale: max_abs / I8_MAX,
                zero_point: 0,
                min_value: -max_abs,
                max_value: max_abs,
                policy,
            }
        }
        QuantizationMinMaxPolicy::Asymmetric => {
            let scale = (max_value - min_value) / I8_FULL_RANGE;
            let zero_point = (I8_MIN - min_value / scale).round().clamp(I8_MIN, I8_MAX) as i8;
            Int8QuantizationMetadata {
                scale,
                zero_point,
                min_value,
                max_value,
                policy,
            }
        }
    };

    validate_metadata(&metadata)?;
    Ok(metadata)
}

pub fn quantize_latent_to_int8(
    latent: &[f32],
    metadata: &Int8QuantizationMetadata,
) -> Result<Vec<i8>, LatentAccountingError> {
    validate_metadata(metadata)?;
    validate_latent_values(latent)?;

    if latent.is_empty() {
        return Ok(Vec::new());
    }
    if metadata.min_value == metadata.max_value {
        return Ok(vec![0; latent.len()]);
    }

    let mut quantized = Vec::with_capacity(latent.len());
    for &value in latent {
        let clipped = value.clamp(metadata.min_value, metadata.max_value);
        let scaled = (clipped / metadata.scale + metadata.zero_point as f32).round() as i32;
        let clamped = scaled.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        quantized.push(clamped);
    }
    Ok(quantized)
}

pub fn dequantize_latent_from_int8(
    quantized_latent: &[i8],
    metadata: &Int8QuantizationMetadata,
) -> Result<Vec<f32>, LatentAccountingError> {
    validate_metadata(metadata)?;

    if quantized_latent.is_empty() {
        return Ok(Vec::new());
    }
    if metadata.min_value == metadata.max_value {
        return Ok(vec![metadata.min_value; quantized_latent.len()]);
    }

    let dequantized = quantized_latent
        .iter()
        .map(|&value| {
            let centered = value as i32 - metadata.zero_point as i32;
            (centered as f32 * metadata.scale).clamp(metadata.min_value, metadata.max_value)
        })
        .collect();
    Ok(dequantized)
}

pub fn serialize_quantized_latent_payload(
    quantized_latent: &[i8],
    metadata: &Int8QuantizationMetadata,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(SERIALIZED_HEADER_BYTES + quantized_latent.len());
    payload.extend_from_slice(&metadata.scale.to_le_bytes());
    payload.extend_from_slice(&metadata.zero_point.to_le_bytes());
    payload.extend_from_slice(&metadata.min_value.to_le_bytes());
    payload.extend_from_slice(&metadata.max_value.to_le_bytes());
    payload.push(policy_tag(metadata.policy));
    payload.extend_from_slice(&(quantized_latent.len() as u64).to_le_bytes());
    payload.extend(quantized_latent.iter().map(|value| *value as u8));
    payload
}

pub fn deflate_compressed_size_for_serialized_quantized_payload(
    quantized_latent: &[i8],
    metadata: &Int8QuantizationMetadata,
    compression: Compression,
) -> std::io::Result<usize> {
    let payload = serialize_quantized_latent_payload(quantized_latent, metadata);
    deflate_compressed_size(&payload, compression)
}

pub fn deflate_compressed_size(payload: &[u8], compression: Compression) -> std::io::Result<usize> {
    let mut encoder = DeflateEncoder::new(Vec::new(), compression);
    encoder.write_all(payload)?;
    let compressed = encoder.finish()?;
    Ok(compressed.len())
}

fn validate_latent_values(latent: &[f32]) -> Result<(), LatentAccountingError> {
    for (index, value) in latent.iter().enumerate() {
        if !value.is_finite() {
            return Err(LatentAccountingError::NonFiniteInput { index });
        }
    }
    Ok(())
}

fn validate_metadata(metadata: &Int8QuantizationMetadata) -> Result<(), LatentAccountingError> {
    if !metadata.min_value.is_finite() || !metadata.max_value.is_finite() {
        return Err(LatentAccountingError::InvalidMetadata {
            message: "min and max must be finite",
        });
    }
    if metadata.min_value > metadata.max_value {
        return Err(LatentAccountingError::InvalidMetadata {
            message: "min must be <= max",
        });
    }
    if metadata.min_value != metadata.max_value
        && (!metadata.scale.is_finite() || metadata.scale <= 0.0)
    {
        return Err(LatentAccountingError::InvalidMetadata {
            message: "scale must be finite and > 0 for non-constant ranges",
        });
    }
    Ok(())
}

fn policy_tag(policy: QuantizationMinMaxPolicy) -> u8 {
    match policy {
        QuantizationMinMaxPolicy::Symmetric => 0,
        QuantizationMinMaxPolicy::Asymmetric => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_latent_size_matches_f32_storage() {
        assert_eq!(raw_f32_latent_byte_size(&[]), 0);
        assert_eq!(raw_f32_latent_byte_size(&[1.0, -2.0, 3.5]), 12);
    }

    #[test]
    fn metadata_rejects_empty_input() {
        let result = compute_int8_quantization_metadata(&[], QuantizationMinMaxPolicy::Asymmetric);
        assert_eq!(result, Err(LatentAccountingError::EmptyInput));
    }

    #[test]
    fn metadata_rejects_non_finite_values() {
        let result = compute_int8_quantization_metadata(
            &[0.0, f32::NAN, 1.0],
            QuantizationMinMaxPolicy::Asymmetric,
        );
        assert_eq!(
            result,
            Err(LatentAccountingError::NonFiniteInput { index: 1 })
        );
    }

    #[test]
    fn constant_vectors_quantize_and_dequantize_exactly() {
        let latent = vec![3.25; 64];
        let metadata =
            compute_int8_quantization_metadata(&latent, QuantizationMinMaxPolicy::Asymmetric)
                .expect("metadata should be computed");
        assert_eq!(metadata.min_value, 3.25);
        assert_eq!(metadata.max_value, 3.25);

        let quantized = quantize_latent_to_int8(&latent, &metadata).expect("quantization succeeds");
        assert!(quantized.iter().all(|value| *value == 0));

        let restored =
            dequantize_latent_from_int8(&quantized, &metadata).expect("dequantization succeeds");
        assert_eq!(restored, latent);
    }

    #[test]
    fn asymmetric_policy_handles_negative_and_positive_ranges() {
        let latent = vec![-3.2, -1.0, -0.1, 0.0, 1.7, 4.2];
        let metadata =
            compute_int8_quantization_metadata(&latent, QuantizationMinMaxPolicy::Asymmetric)
                .expect("metadata should be computed");
        assert_eq!(metadata.policy, QuantizationMinMaxPolicy::Asymmetric);
        assert_eq!(metadata.min_value, -3.2);
        assert_eq!(metadata.max_value, 4.2);

        let quantized = quantize_latent_to_int8(&latent, &metadata).expect("quantization succeeds");
        let restored =
            dequantize_latent_from_int8(&quantized, &metadata).expect("dequantization succeeds");
        for (original, decoded) in latent.iter().zip(restored.iter()) {
            assert!(
                (original - decoded).abs() <= metadata.scale + 1e-6,
                "expected {original} ~= {decoded} within {}",
                metadata.scale + 1e-6
            );
        }
    }

    #[test]
    fn symmetric_policy_uses_zero_point_zero_and_symmetric_bounds() {
        let latent = vec![-1.0, 0.25, 2.5];
        let metadata =
            compute_int8_quantization_metadata(&latent, QuantizationMinMaxPolicy::Symmetric)
                .expect("metadata should be computed");
        assert_eq!(metadata.policy, QuantizationMinMaxPolicy::Symmetric);
        assert_eq!(metadata.zero_point, 0);
        assert!((metadata.min_value + metadata.max_value).abs() <= 1e-6);
        assert!((metadata.max_value - 2.5).abs() <= 1e-6);
    }

    #[test]
    fn quantize_and_dequantize_empty_input() {
        let metadata = Int8QuantizationMetadata {
            scale: 0.1,
            zero_point: 0,
            min_value: -1.0,
            max_value: 1.0,
            policy: QuantizationMinMaxPolicy::Asymmetric,
        };
        let quantized = quantize_latent_to_int8(&[], &metadata).expect("quantization succeeds");
        assert!(quantized.is_empty());
        let dequantized =
            dequantize_latent_from_int8(&quantized, &metadata).expect("dequantization succeeds");
        assert!(dequantized.is_empty());
    }

    #[test]
    fn quantization_clamps_values_outside_metadata_range() {
        let metadata = Int8QuantizationMetadata {
            scale: 1.0 / I8_MAX,
            zero_point: 0,
            min_value: -1.0,
            max_value: 1.0,
            policy: QuantizationMinMaxPolicy::Symmetric,
        };

        let quantized =
            quantize_latent_to_int8(&[-4.0, -1.0, 0.0, 1.0, 4.0], &metadata).expect("quantize");
        assert_eq!(quantized, vec![-127, -127, 0, 127, 127]);

        let dequantized = dequantize_latent_from_int8(&quantized, &metadata).expect("dequantize");
        let expected = [-1.0, -1.0, 0.0, 1.0, 1.0];
        for (actual, expected) in dequantized.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn quantization_rejects_invalid_non_constant_metadata() {
        let invalid_metadata = Int8QuantizationMetadata {
            scale: 0.0,
            zero_point: 0,
            min_value: -1.0,
            max_value: 1.0,
            policy: QuantizationMinMaxPolicy::Asymmetric,
        };

        let quantize_err =
            quantize_latent_to_int8(&[0.25], &invalid_metadata).expect_err("invalid metadata");
        assert_eq!(
            quantize_err,
            LatentAccountingError::InvalidMetadata {
                message: "scale must be finite and > 0 for non-constant ranges",
            }
        );

        let dequantize_err =
            dequantize_latent_from_int8(&[0], &invalid_metadata).expect_err("invalid metadata");
        assert_eq!(dequantize_err, quantize_err);
    }

    #[test]
    fn dequantization_clamps_extreme_codes_to_metadata_bounds() {
        let metadata = Int8QuantizationMetadata {
            scale: 0.01,
            zero_point: 20,
            min_value: -0.5,
            max_value: 0.5,
            policy: QuantizationMinMaxPolicy::Asymmetric,
        };

        let dequantized =
            dequantize_latent_from_int8(&[i8::MIN, 20, i8::MAX], &metadata).expect("dequantize");
        let expected = [-0.5, 0.0, 0.5];
        for (actual, expected) in dequantized.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn deflate_size_can_be_measured_for_quantized_payloads() {
        let latent = vec![0.125; 4096];
        let metadata =
            compute_int8_quantization_metadata(&latent, QuantizationMinMaxPolicy::Asymmetric)
                .expect("metadata should be computed");
        let quantized = quantize_latent_to_int8(&latent, &metadata).expect("quantization succeeds");
        let payload = serialize_quantized_latent_payload(&quantized, &metadata);
        assert_eq!(payload.len(), SERIALIZED_HEADER_BYTES + quantized.len());

        let compressed_size = deflate_compressed_size_for_serialized_quantized_payload(
            &quantized,
            &metadata,
            Compression::default(),
        )
        .expect("deflate should succeed");
        assert!(compressed_size > 0);
        assert!(compressed_size < payload.len());
    }

    #[test]
    fn deflate_compresses_repetitive_payload_better_than_high_entropy_payload() {
        let repetitive = vec![0x5a_u8; 4096];
        let high_entropy = pseudo_random_bytes(4096);

        let repetitive_size =
            deflate_compressed_size(&repetitive, Compression::default()).expect("deflate");
        let high_entropy_size =
            deflate_compressed_size(&high_entropy, Compression::default()).expect("deflate");

        assert!(repetitive_size < repetitive.len());
        assert!(repetitive_size < high_entropy_size);
    }

    fn pseudo_random_bytes(len: usize) -> Vec<u8> {
        let mut state = 0x1234_5678_u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xff) as u8
            })
            .collect()
    }
}
