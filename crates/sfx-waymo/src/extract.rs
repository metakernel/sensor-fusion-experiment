use anyhow::{Context, Result, anyhow};
use arrow::array::{
    Array, BinaryArray, FixedSizeListArray, Int8Array, Int32Array, Int64Array, LargeBinaryArray,
    ListArray, StringArray,
};
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

pub const CAMERA_FRONT: i8 = 1;
pub const LASER_TOP: i8 = 1;

const COL_SEGMENT: &str = "key.segment_context_name";
const COL_TIMESTAMP: &str = "key.frame_timestamp_micros";
const COL_CAMERA_NAME: &str = "key.camera_name";
const COL_LASER_NAME: &str = "key.laser_name";

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
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
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
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata of {}", path.display()))?;

    let schema = builder.schema().clone();
    let image_col = find_binary_column(&schema, "image").ok_or_else(|| {
        anyhow!(
            "no binary image column found; schema:\n{}",
            schema_summary(&schema)
        )
    })?;

    let reader = builder.build()?;
    let mut frames = Vec::new();

    'outer: for batch in reader {
        let batch = batch.context("reading record batch")?;

        let segments = col_as::<StringArray>(&batch, COL_SEGMENT, "StringArray")?;
        let timestamps = col_as::<Int64Array>(&batch, COL_TIMESTAMP, "Int64Array")?;
        let camera_names = col_as::<Int8Array>(&batch, COL_CAMERA_NAME, "Int8Array")?;

        let img_arr = batch
            .column_by_name(&image_col)
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
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata of {}", path.display()))?;

    let schema = builder.schema().clone();
    let values_col =
        find_column_containing(&schema, "range_image_return1.values").ok_or_else(|| {
            anyhow!(
                "no range_image_return1.values column; schema:\n{}",
                schema_summary(&schema)
            )
        })?;
    let shape_col =
        find_column_containing(&schema, "range_image_return1.shape").ok_or_else(|| {
            anyhow!(
                "no range_image_return1.shape column; schema:\n{}",
                schema_summary(&schema)
            )
        })?;

    let reader = builder.build()?;
    let mut frames = Vec::new();

    'outer: for batch in reader {
        let batch = batch.context("reading record batch")?;

        let segments = col_as::<StringArray>(&batch, COL_SEGMENT, "StringArray")?;
        let timestamps = col_as::<Int64Array>(&batch, COL_TIMESTAMP, "Int64Array")?;
        let laser_names = col_as::<Int8Array>(&batch, COL_LASER_NAME, "Int8Array")?;

        let values_arr = batch
            .column_by_name(&values_col)
            .ok_or_else(|| anyhow!("column {values_col} missing in batch"))?;
        let shape_arr = batch
            .column_by_name(&shape_col)
            .ok_or_else(|| anyhow!("column {shape_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if laser_names.value(i) != LASER_TOP {
                continue;
            }

            let range_values = extract_float_list(values_arr, i)
                .with_context(|| format!("reading range values at row {i}"))?;
            let shape = extract_int32_fixed_list::<3>(shape_arr, i)
                .with_context(|| format!("reading range shape at row {i}"))?;

            frames.push(LidarFrame {
                segment: segments.value(i).to_string(),
                timestamp_micros: timestamps.value(i),
                range_values,
                shape: [shape[0] as usize, shape[1] as usize, shape[2] as usize],
            });

            if max.is_some_and(|m| frames.len() >= m) {
                break 'outer;
            }
        }
    }

    Ok(frames)
}

fn find_binary_column(schema: &SchemaRef, name_fragment: &str) -> Option<String> {
    schema
        .fields()
        .iter()
        .find(|f| {
            f.name().contains(name_fragment)
                && matches!(f.data_type(), DataType::Binary | DataType::LargeBinary)
        })
        .map(|f| f.name().clone())
}

fn find_column_containing(schema: &SchemaRef, fragment: &str) -> Option<String> {
    schema
        .fields()
        .iter()
        .find(|f| f.name().contains(fragment))
        .map(|f| f.name().clone())
}

fn col_as<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    type_name: &str,
) -> Result<&'a T> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column '{name}'"))?;
    col.as_any().downcast_ref::<T>().ok_or_else(|| {
        anyhow!(
            "column '{name}' is not {type_name}; actual: {:?}",
            col.data_type()
        )
    })
}

fn extract_binary(col: &dyn Array, row: usize) -> Result<Vec<u8>> {
    if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        return Ok(arr.value(row).to_vec());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(arr.value(row).to_vec());
    }
    Err(anyhow!(
        "column is not a binary array; type: {:?}",
        col.data_type()
    ))
}

/// Extract a row from a List<Float32> column.
fn extract_float_list(col: &dyn Array, row: usize) -> Result<Vec<f32>> {
    let list = col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("expected ListArray, got {:?}", col.data_type()))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let floats = values
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .ok_or_else(|| anyhow!("List element type is not Float32"))?;
    Ok((0..floats.len()).map(|i| floats.value(i)).collect())
}

/// Extract a row from a FixedSizeList<Int32, N> column.
fn extract_int32_fixed_list<const N: usize>(col: &dyn Array, row: usize) -> Result<[i32; N]> {
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("expected FixedSizeListArray, got {:?}", col.data_type()))?;
    if list.is_null(row) {
        return Ok([0i32; N]);
    }
    let values = list.value(row);
    let ints = values
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| anyhow!("FixedSizeList element type is not Int32"))?;
    if ints.len() < N {
        anyhow::bail!("FixedSizeList has {} elements, expected {}", ints.len(), N);
    }
    let mut out = [0i32; N];
    for (i, v) in out.iter_mut().enumerate() {
        *v = ints.value(i);
    }
    Ok(out)
}

fn schema_summary(schema: &SchemaRef) -> String {
    schema
        .fields()
        .iter()
        .map(|f| format!("  {}: {:?}", f.name(), f.data_type()))
        .collect::<Vec<_>>()
        .join("\n")
}
