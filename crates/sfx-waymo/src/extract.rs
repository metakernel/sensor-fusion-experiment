use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, BinaryArray, Int64Array, Int8Array, LargeBinaryArray, StringArray};
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const CAMERA_FRONT: i8 = 1;
pub const LIDAR_TOP: i8 = 1;

const COL_SEGMENT: &str = "key.segment_context_name";
const COL_TIMESTAMP: &str = "key.frame_timestamp_micros";
const COL_CAMERA_NAME: &str = "key.camera_name";
const COL_LIDAR_NAME: &str = "key.lidar_name";

#[derive(Debug, Clone)]
pub struct SchemaField {
    pub name: String,
    pub data_type: String,
}

#[derive(Debug, Clone)]
pub struct CameraFrame {
    pub segment: String,
    pub timestamp_micros: i64,
    pub jpeg_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct LidarFrame {
    pub segment: String,
    pub timestamp_micros: i64,
    pub range_values: Vec<f32>,
    pub shape: [usize; 3],
}

pub fn read_schema(path: &Path) -> Result<Vec<SchemaField>> {
    let file = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet header of {}", path.display()))?;
    let schema = builder.schema();
    Ok(schema
        .fields()
        .iter()
        .map(|f| SchemaField {
            name: f.name().clone(),
            data_type: format!("{:?}", f.data_type()),
        })
        .collect())
}

pub fn read_camera_frames(path: &Path, max: Option<usize>) -> Result<Vec<CameraFrame>> {
    let file = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata of {}", path.display()))?;

    let schema = builder.schema().clone();
    let image_col = find_binary_column(&schema, "image")
        .ok_or_else(|| anyhow!("no binary image column found; schema:\n{}", schema_summary(&schema)))?;

    let reader = builder.build()?;
    let mut frames = Vec::new();

    'outer: for batch in reader {
        let batch = batch.context("reading record batch")?;

        let segments = col_as::<StringArray>(&batch, COL_SEGMENT, "StringArray")?;
        let timestamps = col_as::<Int64Array>(&batch, COL_TIMESTAMP, "Int64Array")?;
        let camera_names = col_as::<Int8Array>(&batch, COL_CAMERA_NAME, "Int8Array")?;

        let img_arr = batch.column_by_name(&image_col)
            .ok_or_else(|| anyhow!("column {image_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if camera_names.value(i) != CAMERA_FRONT {
                continue;
            }
            let jpeg_bytes = extract_binary(img_arr, i)
                .with_context(|| format!("reading image bytes at row {i}"))?;

            frames.push(CameraFrame {
                segment: segments.value(i).to_string(),
                timestamp_micros: timestamps.value(i),
                jpeg_bytes,
            });

            if max.is_some_and(|m| frames.len() >= m) {
                break 'outer;
            }
        }
    }

    Ok(frames)
}

pub fn read_lidar_frames(path: &Path, max: Option<usize>) -> Result<Vec<LidarFrame>> {
    let file = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata of {}", path.display()))?;

    let schema = builder.schema().clone();
    let range_col = find_column_containing(&schema, "range_image_return1")
        .ok_or_else(|| anyhow!("no range_image_return1 column found; schema:\n{}", schema_summary(&schema)))?;

    let reader = builder.build()?;
    let mut frames = Vec::new();

    'outer: for batch in reader {
        let batch = batch.context("reading record batch")?;

        let segments = col_as::<StringArray>(&batch, COL_SEGMENT, "StringArray")?;
        let timestamps = col_as::<Int64Array>(&batch, COL_TIMESTAMP, "Int64Array")?;
        let lidar_names = col_as::<Int8Array>(&batch, COL_LIDAR_NAME, "Int8Array")?;

        let range_arr = batch.column_by_name(&range_col)
            .ok_or_else(|| anyhow!("column {range_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if lidar_names.value(i) != LIDAR_TOP {
                continue;
            }
            let raw = extract_binary(range_arr, i)
                .with_context(|| format!("reading range image bytes at row {i}"))?;

            let (range_values, shape) = decode_range_image(&raw)
                .with_context(|| format!("decoding range image at row {i}"))?;

            frames.push(LidarFrame {
                segment: segments.value(i).to_string(),
                timestamp_micros: timestamps.value(i),
                range_values,
                shape,
            });

            if max.is_some_and(|m| frames.len() >= m) {
                break 'outer;
            }
        }
    }

    Ok(frames)
}

fn find_binary_column(schema: &SchemaRef, name_fragment: &str) -> Option<String> {
    schema.fields().iter().find(|f| {
        f.name().contains(name_fragment)
            && matches!(f.data_type(), DataType::Binary | DataType::LargeBinary)
    }).map(|f| f.name().clone())
}

fn find_column_containing(schema: &SchemaRef, fragment: &str) -> Option<String> {
    schema.fields().iter().find(|f| f.name().contains(fragment))
        .map(|f| f.name().clone())
}

fn col_as<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    type_name: &str,
) -> Result<&'a T> {
    let col = batch.column_by_name(name)
        .ok_or_else(|| anyhow!("missing column '{name}'"))?;
    col.as_any().downcast_ref::<T>()
        .ok_or_else(|| anyhow!("column '{name}' is not {type_name}; actual: {:?}", col.data_type()))
}

fn extract_binary(col: &dyn Array, row: usize) -> Result<Vec<u8>> {
    if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        return Ok(arr.value(row).to_vec());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(arr.value(row).to_vec());
    }
    Err(anyhow!("column is not a binary array; type: {:?}", col.data_type()))
}

fn decode_range_image(bytes: &[u8]) -> Result<(Vec<f32>, [usize; 3])> {
    // Try parsing as MatrixFloat protobuf (Waymo v2 format)
    if let Ok((values, dims)) = decode_matrix_float_proto(bytes) {
        if dims.len() >= 2 {
            let h = dims[0] as usize;
            let w = dims[1] as usize;
            let c = if dims.len() >= 3 { dims[2] as usize } else { values.len() / (h * w).max(1) };
            return Ok((values, [h, w, c]));
        }
    }

    // Fallback: try zlib-decompressed raw float32
    let decompressed = if is_zlib(bytes) {
        decompress_zlib(bytes)?
    } else {
        bytes.to_vec()
    };

    if decompressed.len() % 4 != 0 {
        anyhow::bail!(
            "decoded {} bytes, not divisible by 4; cannot interpret as f32 array",
            decompressed.len()
        );
    }

    let values: Vec<f32> = decompressed
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let shape = infer_top_lidar_shape(values.len());
    Ok((values, shape))
}

/// Minimal protobuf decoder for Waymo MatrixFloat:
///   message MatrixFloat { repeated float data = 1; MatrixShape shape = 2; }
///   message MatrixShape  { repeated int32 dims = 1; }
fn decode_matrix_float_proto(bytes: &[u8]) -> Result<(Vec<f32>, Vec<i32>)> {
    let mut pos = 0;
    let mut data = Vec::new();
    let mut dims = Vec::new();

    while pos < bytes.len() {
        let (tag, wire, advance) = read_proto_tag(bytes, pos)?;
        pos += advance;
        match (tag, wire) {
            (1, 2) => {
                let (len, a) = read_proto_varint(bytes, pos)?;
                pos += a;
                let end = pos + len as usize;
                if end > bytes.len() {
                    anyhow::bail!("data field overflows buffer");
                }
                let slice = &bytes[pos..end];
                for chunk in slice.chunks_exact(4) {
                    data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                pos = end;
            }
            (2, 2) => {
                let (len, a) = read_proto_varint(bytes, pos)?;
                pos += a;
                let end = pos + len as usize;
                if end > bytes.len() {
                    anyhow::bail!("shape field overflows buffer");
                }
                dims = decode_matrix_shape_proto(&bytes[pos..end])?;
                pos = end;
            }
            (_, 0) => {
                let (_, a) = read_proto_varint(bytes, pos)?;
                pos += a;
            }
            (_, 2) => {
                let (len, a) = read_proto_varint(bytes, pos)?;
                pos += a + len as usize;
            }
            (_, 5) => {
                pos += 4;
            }
            (_, 1) => {
                pos += 8;
            }
            (_, wt) => anyhow::bail!("unknown wire type {wt}"),
        }
    }

    Ok((data, dims))
}

fn decode_matrix_shape_proto(bytes: &[u8]) -> Result<Vec<i32>> {
    let mut pos = 0;
    let mut dims = Vec::new();
    while pos < bytes.len() {
        let (tag, wire, advance) = read_proto_tag(bytes, pos)?;
        pos += advance;
        match (tag, wire) {
            (1, 0) => {
                let (v, a) = read_proto_varint(bytes, pos)?;
                dims.push(v as i32);
                pos += a;
            }
            (1, 2) => {
                // packed int32 dims
                let (len, a) = read_proto_varint(bytes, pos)?;
                pos += a;
                let end = pos + len as usize;
                while pos < end {
                    let (v, a) = read_proto_varint(bytes, pos)?;
                    dims.push(v as i32);
                    pos += a;
                }
            }
            (_, 0) => {
                let (_, a) = read_proto_varint(bytes, pos)?;
                pos += a;
            }
            (_, 2) => {
                let (len, a) = read_proto_varint(bytes, pos)?;
                pos += a + len as usize;
            }
            (_, wt) => anyhow::bail!("unknown wire type {wt} in shape"),
        }
    }
    Ok(dims)
}

fn read_proto_tag(bytes: &[u8], pos: usize) -> Result<(u64, u8, usize)> {
    let (v, advance) = read_proto_varint(bytes, pos)?;
    let field = v >> 3;
    let wire = (v & 0x7) as u8;
    Ok((field, wire, advance))
}

fn read_proto_varint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let start = pos;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        if pos >= bytes.len() {
            anyhow::bail!("varint overflows buffer at {pos}");
        }
        let b = bytes[pos] as u64;
        pos += 1;
        value |= (b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            anyhow::bail!("varint too long");
        }
    }
    Ok((value, pos - start))
}

fn is_zlib(bytes: &[u8]) -> bool {
    bytes.len() >= 2
        && bytes[0] == 0x78
        && matches!(bytes[1], 0x01 | 0x5e | 0x9c | 0xda)
}

fn decompress_zlib(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::ZlibDecoder::new(bytes);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .context("decompressing zlib range image")?;
    Ok(out)
}

fn infer_top_lidar_shape(n: usize) -> [usize; 3] {
    const TOP_H: usize = 64;
    const TOP_W: usize = 2650;
    const CHANNELS: usize = 4;
    if n == TOP_H * TOP_W * CHANNELS {
        [TOP_H, TOP_W, CHANNELS]
    } else {
        [n, 1, 1]
    }
}

fn schema_summary(schema: &SchemaRef) -> String {
    schema
        .fields()
        .iter()
        .map(|f| format!("  {}: {:?}", f.name(), f.data_type()))
        .collect::<Vec<_>>()
        .join("\n")
}
