use anyhow::{Context, Result, anyhow};
use arrow::array::{
    Array, BinaryArray, FixedSizeListArray, Int8Array, Int32Array, Int64Array, LargeBinaryArray,
    ListArray, StringArray,
};
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

pub const CAMERA_FRONT: i8 = 1;
pub const LASER_TOP: i8 = 1;

const COL_SEGMENT: &str = "key.segment_context_name";
const COL_TIMESTAMP: &str = "key.frame_timestamp_micros";
const COL_CAMERA_NAME: &str = "key.camera_name";
const COL_LASER_NAME: &str = "key.laser_name";
const COL_BOX_TYPE: &str = "[CameraBoxComponent].type";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CameraBoxClassCounts {
    pub vehicle: u32,
    pub pedestrian: u32,
    pub cyclist: u32,
    pub sign: u32,
}

impl CameraBoxClassCounts {
    fn increment_type(&mut self, class_type: i8) {
        match class_type {
            1 => self.vehicle += 1,
            2 => self.pedestrian += 1,
            3 => self.sign += 1,
            4 => self.cyclist += 1,
            _ => {}
        }
    }
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
    read_camera_frames_for_sensor(path, CAMERA_FRONT, max)
}

pub fn read_camera_frames_for_sensor(
    path: &Path,
    sensor_name: i8,
    max: Option<usize>,
) -> Result<Vec<CameraFrame>> {
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
        let camera_names = batch
            .column_by_name(COL_CAMERA_NAME)
            .ok_or_else(|| anyhow!("missing column '{COL_CAMERA_NAME}'"))?;

        let img_arr = batch
            .column_by_name(&image_col)
            .ok_or_else(|| anyhow!("column {image_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if sensor_name_at(camera_names.as_ref(), i)? != sensor_name {
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
    read_lidar_frames_for_sensor(path, LASER_TOP, max)
}

pub fn read_lidar_frames_for_sensor(
    path: &Path,
    sensor_name: i8,
    max: Option<usize>,
) -> Result<Vec<LidarFrame>> {
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
        let laser_names = batch
            .column_by_name(COL_LASER_NAME)
            .ok_or_else(|| anyhow!("missing column '{COL_LASER_NAME}'"))?;

        let values_arr = batch
            .column_by_name(&values_col)
            .ok_or_else(|| anyhow!("column {values_col} missing in batch"))?;
        let shape_arr = batch
            .column_by_name(&shape_col)
            .ok_or_else(|| anyhow!("column {shape_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if sensor_name_at(laser_names.as_ref(), i)? != sensor_name {
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

pub fn read_camera_box_counts_for_sensor(
    path: &Path,
    sensor_name: i8,
) -> Result<BTreeMap<(String, i64), CameraBoxClassCounts>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata of {}", path.display()))?;

    let schema = builder.schema().clone();
    let type_col = find_column_containing(&schema, COL_BOX_TYPE).ok_or_else(|| {
        anyhow!(
            "no camera box type column found; schema:\n{}",
            schema_summary(&schema)
        )
    })?;

    let reader = builder.build()?;
    let mut by_frame = BTreeMap::new();

    for batch in reader {
        let batch = batch.context("reading record batch")?;

        let segments = col_as::<StringArray>(&batch, COL_SEGMENT, "StringArray")?;
        let timestamps = col_as::<Int64Array>(&batch, COL_TIMESTAMP, "Int64Array")?;
        let camera_names = batch
            .column_by_name(COL_CAMERA_NAME)
            .ok_or_else(|| anyhow!("missing column '{COL_CAMERA_NAME}'"))?;
        let class_types = batch
            .column_by_name(&type_col)
            .ok_or_else(|| anyhow!("column {type_col} missing in batch"))?;

        for i in 0..batch.num_rows() {
            if sensor_name_at(camera_names.as_ref(), i)? != sensor_name {
                continue;
            }

            let class_type = sensor_name_at(class_types.as_ref(), i)?;
            if !matches!(class_type, 1..=4) {
                continue;
            }
            let key = (segments.value(i).to_string(), timestamps.value(i));
            by_frame
                .entry(key)
                .or_insert_with(CameraBoxClassCounts::default)
                .increment_type(class_type);
        }
    }

    Ok(by_frame)
}

fn sensor_name_at(col: &dyn Array, row: usize) -> Result<i8> {
    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        return Ok(arr.value(row));
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return i8::try_from(arr.value(row))
            .map_err(|_| anyhow!("sensor id {} out of i8 range", arr.value(row)));
    }

    Err(anyhow!(
        "expected Int8 or Int32 sensor column, got {:?}",
        col.data_type()
    ))
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
    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        if list.is_null(row) {
            return Ok(Vec::new());
        }
        let values = list.value(row);
        let floats = values
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .ok_or_else(|| anyhow!("List element type is not Float32"))?;
        return Ok((0..floats.len()).map(|i| floats.value(i)).collect());
    }

    if let Some(list) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        if list.is_null(row) {
            return Ok(Vec::new());
        }
        let values = list.value(row);
        let floats = values
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .ok_or_else(|| anyhow!("FixedSizeList element type is not Float32"))?;
        return Ok((0..floats.len()).map(|i| floats.value(i)).collect());
    }

    Err(anyhow!(
        "expected ListArray or FixedSizeListArray, got {:?}",
        col.data_type()
    ))
}

/// Extract a row from a FixedSizeList<Int32, N> column.
fn extract_int32_fixed_list<const N: usize>(col: &dyn Array, row: usize) -> Result<[i32; N]> {
    if let Some(list) = col.as_any().downcast_ref::<FixedSizeListArray>() {
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
        return Ok(out);
    }

    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        if list.is_null(row) {
            return Ok([0i32; N]);
        }
        let values = list.value(row);
        let ints = values
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| anyhow!("List element type is not Int32"))?;
        if ints.len() < N {
            anyhow::bail!("List has {} elements, expected {}", ints.len(), N);
        }
        let mut out = [0i32; N];
        for (i, v) in out.iter_mut().enumerate() {
            *v = ints.value(i);
        }
        return Ok(out);
    }

    Err(anyhow!(
        "expected FixedSizeListArray or ListArray, got {:?}",
        col.data_type()
    ))
}

fn schema_summary(schema: &SchemaRef) -> String {
    schema
        .fields()
        .iter()
        .map(|f| format!("  {}: {:?}", f.name(), f.data_type()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    #[test]
    fn reads_front_camera_box_counts_per_frame() {
        let root = temp_root("reads_front_camera_box_counts_per_frame");
        let path = root.join("camera_box.parquet");
        write_camera_box_fixture(&path);

        let counts = read_camera_box_counts_for_sensor(&path, CAMERA_FRONT).unwrap();

        let frame_100 = counts
            .get(&(String::from("segment-a"), 100))
            .expect("frame 100 should be present");
        assert_eq!(frame_100.vehicle, 1);
        assert_eq!(frame_100.pedestrian, 1);
        assert_eq!(frame_100.cyclist, 0);
        assert_eq!(frame_100.sign, 0);

        let frame_200 = counts
            .get(&(String::from("segment-a"), 200))
            .expect("frame 200 should be present");
        assert_eq!(frame_200.sign, 1);

        assert!(
            !counts.contains_key(&(String::from("segment-b"), 300)),
            "unknown class types should not create a frame entry"
        );
    }

    fn write_camera_box_fixture(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_SEGMENT, DataType::Utf8, false),
            Field::new(COL_TIMESTAMP, DataType::Int64, false),
            Field::new(COL_CAMERA_NAME, DataType::Int8, false),
            Field::new(COL_BOX_TYPE, DataType::Int8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "segment-a",
                    "segment-a",
                    "segment-a",
                    "segment-a",
                    "segment-b",
                ])),
                Arc::new(Int64Array::from(vec![100, 100, 100, 200, 300])),
                Arc::new(Int8Array::from(vec![
                    CAMERA_FRONT,
                    CAMERA_FRONT,
                    2,
                    CAMERA_FRONT,
                    CAMERA_FRONT,
                ])),
                Arc::new(Int8Array::from(vec![1, 2, 4, 3, 9])),
            ],
        )
        .expect("batch");

        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close parquet");
    }

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("sfx-waymo-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
