use anyhow::{Context, Result, bail};
use sfx_core::manifest::{
    MultimodalSampleMeta, ProcessedSampleEntry, ProcessedSampleManifest, SampleId, Split,
    TensorShape, read_manifest,
};
use std::path::{Path, PathBuf};

pub const CRATE_NAME: &str = "sfx-data";

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

/// Runtime view of a processed multimodal dataset.
///
/// The dataset keeps manifest metadata plus a filtered list of sample indices.
/// Tensor values are loaded lazily from `rgb.f32.bin` and `range.f32.bin` files.
#[derive(Debug, Clone)]
pub struct FusionDataset {
    processed_dir: PathBuf,
    manifest: ProcessedSampleManifest,
    rgb_shape: TensorShape,
    range_shape: TensorShape,
    indices: Vec<usize>,
    split: Option<Split>,
}

impl FusionDataset {
    /// Open a processed dataset from its root directory and manifest file.
    pub fn open(processed_dir: impl AsRef<Path>, manifest_path: impl AsRef<Path>) -> Result<Self> {
        Self::open_split(processed_dir, manifest_path, None)
    }

    /// Open a processed dataset and filter it to a single split.
    pub fn open_split(
        processed_dir: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
        split: Option<Split>,
    ) -> Result<Self> {
        let manifest_path = manifest_path.as_ref();
        let manifest: ProcessedSampleManifest = read_manifest(manifest_path)
            .with_context(|| format!("loading manifest {}", manifest_path.display()))?;
        Self::from_manifest(processed_dir, manifest, split)
    }

    /// Construct a dataset from an already-loaded manifest.
    pub fn from_manifest(
        processed_dir: impl AsRef<Path>,
        manifest: ProcessedSampleManifest,
        split: Option<Split>,
    ) -> Result<Self> {
        manifest.validate()?;
        let rgb_shape = manifest
            .rgb_shape
            .clone()
            .context("processed sample manifest is missing rgb_shape")?;
        let range_shape = manifest
            .range_shape
            .clone()
            .context("processed sample manifest is missing range_shape")?;

        let indices = manifest
            .samples
            .iter()
            .enumerate()
            .filter(|(_, sample)| split.as_ref().is_none_or(|s| &sample.meta.split == s))
            .map(|(index, _)| index)
            .collect();

        Ok(Self {
            processed_dir: processed_dir.as_ref().to_path_buf(),
            manifest,
            rgb_shape,
            range_shape,
            indices,
            split,
        })
    }

    pub fn processed_dir(&self) -> &Path {
        &self.processed_dir
    }

    pub fn manifest(&self) -> &ProcessedSampleManifest {
        &self.manifest
    }

    pub fn split(&self) -> Option<&Split> {
        self.split.as_ref()
    }

    pub fn rgb_shape(&self) -> &TensorShape {
        &self.rgb_shape
    }

    pub fn range_shape(&self) -> &TensorShape {
        &self.range_shape
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn entries(&self) -> impl Iterator<Item = &ProcessedSampleEntry> {
        self.indices
            .iter()
            .map(|&manifest_index| &self.manifest.samples[manifest_index])
    }

    pub fn entry(&self, index: usize) -> Result<&ProcessedSampleEntry> {
        let manifest_index = *self.indices.get(index).with_context(|| {
            format!("sample index {index} out of bounds for len {}", self.len())
        })?;
        Ok(&self.manifest.samples[manifest_index])
    }

    pub fn load_sample(&self, index: usize) -> Result<FusionSample> {
        let entry = self.entry(index)?.clone();
        self.load_entry(entry)
    }

    pub fn load_sample_by_id(&self, sample_id: &SampleId) -> Result<FusionSample> {
        let entry = self
            .entries()
            .find(|entry| &entry.meta.id == sample_id)
            .with_context(|| format!("sample id `{}` not found", sample_id.0))?
            .clone();
        self.load_entry(entry)
    }

    pub fn validate_tensor_files(&self) -> Result<DatasetValidationReport> {
        for entry in self.entries() {
            let rgb_path = self.tensor_path(&entry.meta.rgb_path);
            ensure_tensor_file_len(&rgb_path, self.rgb_shape.value_count())?;

            let range_path = self.tensor_path(&entry.meta.range_path);
            ensure_tensor_file_len(&range_path, self.range_shape.value_count())?;
        }

        Ok(DatasetValidationReport {
            sample_count: self.len(),
            rgb_bytes_per_sample: self.rgb_shape.value_count() * std::mem::size_of::<f32>(),
            range_bytes_per_sample: self.range_shape.value_count() * std::mem::size_of::<f32>(),
        })
    }

    pub fn batches(&self, options: BatchOptions) -> Result<BatchIter<'_>> {
        options.validate()?;
        let mut order: Vec<usize> = (0..self.len()).collect();
        if options.shuffle {
            shuffle_order(&mut order, options.seed);
        }

        Ok(BatchIter {
            dataset: self,
            options,
            order,
            cursor: 0,
        })
    }

    fn load_entry(&self, entry: ProcessedSampleEntry) -> Result<FusionSample> {
        let rgb_path = self.tensor_path(&entry.meta.rgb_path);
        let rgb = read_f32_tensor(&rgb_path, self.rgb_shape.value_count())?;

        let range_path = self.tensor_path(&entry.meta.range_path);
        let range = read_f32_tensor(&range_path, self.range_shape.value_count())?;

        Ok(FusionSample { entry, rgb, range })
    }

    fn tensor_path(&self, path: &Path) -> PathBuf {
        self.processed_dir.join(path)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionSample {
    pub entry: ProcessedSampleEntry,
    /// CHW RGB tensor values: `[3, H, W]`.
    pub rgb: Vec<f32>,
    /// CHW range tensor values: `[C, H, W]`.
    pub range: Vec<f32>,
}

impl FusionSample {
    pub fn id(&self) -> &SampleId {
        &self.entry.meta.id
    }

    pub fn meta(&self) -> &MultimodalSampleMeta {
        &self.entry.meta
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionBatch {
    /// Contiguous NCHW RGB tensor values: `[B, 3, H, W]`.
    pub rgb: Vec<f32>,
    /// Contiguous NCHW range tensor values: `[B, C, H, W]`.
    pub range: Vec<f32>,
    pub entries: Vec<ProcessedSampleEntry>,
    pub sample_ids: Vec<SampleId>,
    pub rgb_shape: TensorShape,
    pub range_shape: TensorShape,
}

impl FusionBatch {
    pub fn batch_size(&self) -> usize {
        self.sample_ids.len()
    }

    #[cfg(feature = "burn")]
    pub fn into_burn<B>(self, device: &B::Device) -> BurnFusionBatch<B>
    where
        B: burn::prelude::Backend,
    {
        use burn::prelude::{Tensor, TensorData};

        let rgb_shape = self.rgb_nchw_shape();
        let range_shape = self.range_nchw_shape();
        let rgb = Tensor::<B, 4>::from_data(TensorData::new(self.rgb, rgb_shape), device);
        let range = Tensor::<B, 4>::from_data(TensorData::new(self.range, range_shape), device);

        BurnFusionBatch {
            rgb,
            range,
            entries: self.entries,
            sample_ids: self.sample_ids,
            rgb_shape: self.rgb_shape,
            range_shape: self.range_shape,
        }
    }

    pub fn rgb_nchw_shape(&self) -> [usize; 4] {
        [
            self.batch_size(),
            self.rgb_shape.channels,
            self.rgb_shape.height,
            self.rgb_shape.width,
        ]
    }

    pub fn range_nchw_shape(&self) -> [usize; 4] {
        [
            self.batch_size(),
            self.range_shape.channels,
            self.range_shape.height,
            self.range_shape.width,
        ]
    }

    fn from_samples(
        samples: Vec<FusionSample>,
        rgb_shape: TensorShape,
        range_shape: TensorShape,
    ) -> Result<Self> {
        let rgb_values = rgb_shape.value_count();
        let range_values = range_shape.value_count();
        let mut rgb = Vec::with_capacity(samples.len() * rgb_values);
        let mut range = Vec::with_capacity(samples.len() * range_values);
        let mut entries = Vec::with_capacity(samples.len());
        let mut sample_ids = Vec::with_capacity(samples.len());

        for sample in samples {
            if sample.rgb.len() != rgb_values {
                bail!(
                    "sample {} has {} RGB values; expected {rgb_values}",
                    sample.id().0,
                    sample.rgb.len()
                );
            }
            if sample.range.len() != range_values {
                bail!(
                    "sample {} has {} range values; expected {range_values}",
                    sample.id().0,
                    sample.range.len()
                );
            }

            sample_ids.push(sample.id().clone());
            rgb.extend_from_slice(&sample.rgb);
            range.extend_from_slice(&sample.range);
            entries.push(sample.entry);
        }

        Ok(Self {
            rgb,
            range,
            entries,
            sample_ids,
            rgb_shape,
            range_shape,
        })
    }
}

#[cfg(feature = "burn")]
#[derive(Debug, Clone)]
pub struct BurnFusionBatch<B>
where
    B: burn::prelude::Backend,
{
    pub rgb: burn::prelude::Tensor<B, 4>,
    pub range: burn::prelude::Tensor<B, 4>,
    pub entries: Vec<ProcessedSampleEntry>,
    pub sample_ids: Vec<SampleId>,
    pub rgb_shape: TensorShape,
    pub range_shape: TensorShape,
}

#[cfg(feature = "burn")]
impl<B> BurnFusionBatch<B>
where
    B: burn::prelude::Backend,
{
    pub fn batch_size(&self) -> usize {
        self.sample_ids.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchOptions {
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub drop_last: bool,
}

impl BatchOptions {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            ..Self::default()
        }
    }

    pub fn shuffled(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.seed = seed;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            bail!("batch_size must be greater than zero");
        }
        Ok(())
    }
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            seed: 0,
            drop_last: false,
        }
    }
}

pub struct BatchIter<'a> {
    dataset: &'a FusionDataset,
    options: BatchOptions,
    order: Vec<usize>,
    cursor: usize,
}

impl Iterator for BatchIter<'_> {
    type Item = Result<FusionBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.order.len() {
            return None;
        }

        let end = (self.cursor + self.options.batch_size).min(self.order.len());
        if self.options.drop_last && end - self.cursor < self.options.batch_size {
            self.cursor = self.order.len();
            return None;
        }

        let samples = self.order[self.cursor..end]
            .iter()
            .map(|&dataset_index| self.dataset.load_sample(dataset_index))
            .collect::<Result<Vec<_>>>();
        self.cursor = end;

        Some(samples.and_then(|samples| {
            FusionBatch::from_samples(
                samples,
                self.dataset.rgb_shape.clone(),
                self.dataset.range_shape.clone(),
            )
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetValidationReport {
    pub sample_count: usize,
    pub rgb_bytes_per_sample: usize,
    pub range_bytes_per_sample: usize,
}

fn read_f32_tensor(path: &Path, expected_values: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let expected_bytes = expected_values * std::mem::size_of::<f32>();
    if bytes.len() != expected_bytes {
        bail!(
            "{} has {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        );
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn ensure_tensor_file_len(path: &Path, expected_values: usize) -> Result<()> {
    let expected_bytes = expected_values * std::mem::size_of::<f32>();
    let actual = std::fs::metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len() as usize;
    if actual != expected_bytes {
        bail!(
            "{} has {actual} bytes; expected {expected_bytes}",
            path.display()
        );
    }
    Ok(())
}

fn shuffle_order(order: &mut [usize], seed: u64) {
    let mut state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };

    for i in (1..order.len()).rev() {
        let j = (next_u64(&mut state) as usize) % (i + 1);
        order.swap(i, j);
    }
}

fn next_u64(state: &mut u64) -> u64 {
    // xorshift64*: tiny deterministic PRNG, sufficient for reproducible batching.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfx_core::manifest::{MANIFEST_SCHEMA_VERSION, write_manifest};

    #[test]
    fn loads_split_and_sample_tensors() {
        let root = temp_root("loads_split_and_sample_tensors");
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = fixture_manifest(&processed, 3).unwrap();
        write_manifest(&manifest_path, &manifest).unwrap();

        let dataset =
            FusionDataset::open_split(&processed, &manifest_path, Some(Split::Train)).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.rgb_shape().value_count(), 12);
        assert_eq!(dataset.range_shape().value_count(), 4);

        let sample = dataset.load_sample(0).unwrap();
        assert_eq!(sample.id().0, "sample_000000");
        assert_eq!(sample.rgb[0], 0.0);
        assert_eq!(sample.rgb[11], 11.0);
        assert_eq!(sample.range, vec![100.0, 101.0, 102.0, 103.0]);
    }

    #[test]
    fn batches_have_nchw_shapes_and_deterministic_shuffle() {
        let root = temp_root("batches_have_nchw_shapes_and_deterministic_shuffle");
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = fixture_manifest(&processed, 6).unwrap();
        write_manifest(&manifest_path, &manifest).unwrap();
        let dataset =
            FusionDataset::open_split(&processed, &manifest_path, Some(Split::Train)).unwrap();

        let options = BatchOptions::new(2).shuffled(42);
        let ids_a = collect_batch_ids(dataset.batches(options).unwrap());
        let ids_b = collect_batch_ids(dataset.batches(options).unwrap());

        assert_eq!(ids_a, ids_b);
        assert_eq!(ids_a.len(), 4);

        let mut iter = dataset.batches(BatchOptions::new(2)).unwrap();
        let batch = iter.next().unwrap().unwrap();
        assert_eq!(batch.rgb_nchw_shape(), [2, 3, 2, 2]);
        assert_eq!(batch.range_nchw_shape(), [2, 2, 1, 2]);
        assert_eq!(batch.rgb.len(), 24);
        assert_eq!(batch.range.len(), 8);
    }

    #[test]
    fn validates_tensor_file_sizes() {
        let root = temp_root("validates_tensor_file_sizes");
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = fixture_manifest(&processed, 1).unwrap();
        write_manifest(&manifest_path, &manifest).unwrap();
        std::fs::write(
            processed.join("train/sample_000000/rgb.f32.bin"),
            [1u8, 2, 3],
        )
        .unwrap();
        let dataset = FusionDataset::open(&processed, &manifest_path).unwrap();

        let err = dataset.validate_tensor_files().unwrap_err();

        assert!(err.to_string().contains("expected 48"));
    }

    #[test]
    fn loads_sample_by_id() {
        let root = temp_root("loads_sample_by_id");
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = fixture_manifest(&processed, 3).unwrap();
        write_manifest(&manifest_path, &manifest).unwrap();
        let dataset = FusionDataset::open(&processed, &manifest_path).unwrap();

        let sample = dataset
            .load_sample_by_id(&SampleId("sample_000001".to_string()))
            .unwrap();

        assert_eq!(sample.id().0, "sample_000001");
        assert_eq!(sample.meta().timestamp_micros, 1);
    }

    #[test]
    fn batches_drop_last_when_requested() {
        let root = temp_root("batches_drop_last_when_requested");
        let processed = root.join("processed");
        let manifest_path = root.join("processed_samples.json");
        let manifest = fixture_manifest(&processed, 5).unwrap();
        write_manifest(&manifest_path, &manifest).unwrap();
        let dataset = FusionDataset::open(&processed, &manifest_path).unwrap();

        let count_keep_partial = dataset
            .batches(BatchOptions::new(2))
            .unwrap()
            .map(|b| b.unwrap().batch_size())
            .collect::<Vec<_>>();
        let count_drop_partial = dataset
            .batches(BatchOptions::new(2).drop_last(true))
            .unwrap()
            .map(|b| b.unwrap().batch_size())
            .collect::<Vec<_>>();

        assert_eq!(count_keep_partial, vec![2, 2, 1]);
        assert_eq!(count_drop_partial, vec![2, 2]);
    }

    fn collect_batch_ids(iter: BatchIter<'_>) -> Vec<String> {
        iter.flat_map(|batch| batch.unwrap().sample_ids)
            .map(|id| id.0)
            .collect()
    }

    fn fixture_manifest(processed: &Path, count: usize) -> Result<ProcessedSampleManifest> {
        let rgb_shape = TensorShape {
            channels: 3,
            height: 2,
            width: 2,
        };
        let range_shape = TensorShape {
            channels: 2,
            height: 1,
            width: 2,
        };
        let mut samples = Vec::new();

        for i in 0..count {
            let split = if i % 3 == 2 { Split::Val } else { Split::Train };
            let split_dir = match split {
                Split::Train => "train",
                Split::Val => "val",
                Split::Test => "test",
            };
            let id = SampleId(format!("sample_{i:06}"));
            let sample_rel = PathBuf::from(split_dir).join(&id.0);
            let sample_dir = processed.join(&sample_rel);
            std::fs::create_dir_all(&sample_dir).unwrap();

            let base = i as f32 * 1000.0;
            write_f32_file(
                &sample_dir.join("rgb.f32.bin"),
                &(0..rgb_shape.value_count())
                    .map(|v| base + v as f32)
                    .collect::<Vec<_>>(),
            )?;
            write_f32_file(
                &sample_dir.join("range.f32.bin"),
                &(0..range_shape.value_count())
                    .map(|v| base + 100.0 + v as f32)
                    .collect::<Vec<_>>(),
            )?;

            samples.push(ProcessedSampleEntry {
                meta: MultimodalSampleMeta {
                    id,
                    split,
                    rgb_path: sample_rel.join("rgb.f32.bin"),
                    range_path: sample_rel.join("range.f32.bin"),
                    timestamp_micros: i as i64,
                    source_segment: "segment".to_string(),
                },
                meta_path: sample_rel.join("meta.json"),
                preview_rgb_path: Some(sample_rel.join("preview_rgb.png")),
                preview_range_path: Some(sample_rel.join("preview_range.png")),
            });
        }

        Ok(ProcessedSampleManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            dataset: Some("fixture".to_string()),
            rgb_shape: Some(rgb_shape),
            range_shape: Some(range_shape),
            samples,
        })
    }

    fn write_f32_file(path: &Path, values: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("sfx-data-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
