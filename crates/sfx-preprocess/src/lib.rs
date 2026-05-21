use anyhow::{Context, Result};
use sfx_core::manifest::{
    MultimodalSampleMeta, ProcessedSampleEntry, SampleId, Split,
};
use sfx_waymo::extract::{CameraFrame, LidarFrame};
use std::path::{Path, PathBuf};

pub struct ExtractedPair {
    pub segment: String,
    pub timestamp_micros: i64,
    pub jpeg_bytes: Vec<u8>,
    pub range_values: Vec<f32>,
    pub range_shape: [usize; 3],
}

pub fn align_frames(
    camera: Vec<CameraFrame>,
    lidar: Vec<LidarFrame>,
) -> Vec<ExtractedPair> {
    let mut pairs = Vec::new();

    for cam in &camera {
        if let Some(lid) = lidar
            .iter()
            .find(|l| l.segment == cam.segment && l.timestamp_micros == cam.timestamp_micros)
        {
            pairs.push(ExtractedPair {
                segment: cam.segment.clone(),
                timestamp_micros: cam.timestamp_micros,
                jpeg_bytes: cam.jpeg_bytes.clone(),
                range_values: lid.range_values.clone(),
                range_shape: lid.shape,
            });
        }
    }

    pairs
}

pub fn write_sample(
    output_dir: &Path,
    pair: &ExtractedPair,
    split: &Split,
    sample_idx: usize,
) -> Result<ProcessedSampleEntry> {
    let sample_id = format!("sample_{sample_idx:06}");
    let sample_dir = output_dir
        .join(split_dir(split))
        .join(&sample_id);
    std::fs::create_dir_all(&sample_dir)
        .with_context(|| format!("creating {}", sample_dir.display()))?;

    let rgb_path = sample_dir.join("rgb.jpg");
    let range_path = sample_dir.join("range.f32.bin");
    let meta_path = sample_dir.join("meta.json");
    let preview_rgb_path = sample_dir.join("preview_rgb.jpg");
    let preview_range_path = sample_dir.join("preview_range.png");

    std::fs::write(&rgb_path, &pair.jpeg_bytes)
        .with_context(|| format!("writing {}", rgb_path.display()))?;

    std::fs::copy(&rgb_path, &preview_rgb_path)
        .with_context(|| format!("writing {}", preview_rgb_path.display()))?;

    let range_bytes: Vec<u8> = pair
        .range_values
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    std::fs::write(&range_path, &range_bytes)
        .with_context(|| format!("writing {}", range_path.display()))?;

    write_range_preview(&pair.range_values, &pair.range_shape, &preview_range_path)
        .with_context(|| format!("writing {}", preview_range_path.display()))?;

    let meta = MultimodalSampleMeta {
        id: SampleId(sample_id),
        split: split.clone(),
        rgb_path: relative_path(output_dir, &rgb_path),
        range_path: relative_path(output_dir, &range_path),
        timestamp_micros: pair.timestamp_micros,
        source_segment: pair.segment.clone(),
    };
    let meta_json = serde_json::to_string_pretty(&meta).context("encoding sample meta")?;
    std::fs::write(&meta_path, format!("{meta_json}\n"))
        .with_context(|| format!("writing {}", meta_path.display()))?;

    Ok(ProcessedSampleEntry {
        meta,
        meta_path: relative_path(output_dir, &meta_path),
        preview_rgb_path: Some(relative_path(output_dir, &preview_rgb_path)),
        preview_range_path: Some(relative_path(output_dir, &preview_range_path)),
    })
}

fn write_range_preview(values: &[f32], shape: &[usize; 3], path: &Path) -> Result<()> {
    let [h, w, c] = *shape;
    if h == 0 || w == 0 || c == 0 {
        anyhow::bail!("invalid range image shape {h}×{w}×{c}");
    }

    let channel: Vec<f32> = (0..h * w)
        .map(|i| {
            let idx = i * c;
            values.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let valid: Vec<f32> = channel
        .iter()
        .filter(|&&v| v > 0.0 && v.is_finite())
        .copied()
        .collect();
    let max_val = valid.iter().cloned().fold(0.0f32, f32::max).max(1.0);

    let pixels: Vec<u8> = channel
        .iter()
        .map(|&v| {
            if v > 0.0 && v.is_finite() {
                ((v / max_val).clamp(0.0, 1.0) * 255.0) as u8
            } else {
                0
            }
        })
        .collect();

    let img = image::GrayImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create range preview image"))?;
    img.save(path)
        .with_context(|| format!("saving range preview to {}", path.display()))?;

    Ok(())
}

fn split_dir(split: &Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

fn relative_path(base: &Path, full: &Path) -> PathBuf {
    full.strip_prefix(base).unwrap_or(full).to_path_buf()
}

