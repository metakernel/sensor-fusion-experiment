use anyhow::{Context, Result};
use image::imageops::FilterType;
use sfx_core::manifest::{MultimodalSampleMeta, ProcessedSampleEntry, SampleId, Split};
use sfx_waymo::extract::{CameraFrame, LidarFrame};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const RGB_H: usize = 128;
pub const RGB_W: usize = 256;
pub const RANGE_H: usize = 64;
pub const RANGE_W: usize = 256;
pub const RANGE_CHANNELS: usize = 2;
/// Physical max range in metres used for normalisation.
const MAX_RANGE_METERS: f32 = 75.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeChannel {
    Range,
    Intensity,
    ValidityMask,
}

#[derive(Debug, Clone)]
pub struct ProcessOptions {
    pub rgb_height: usize,
    pub rgb_width: usize,
    pub range_height: usize,
    pub range_width: usize,
    pub range_channels: Vec<RangeChannel>,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            rgb_height: RGB_H,
            rgb_width: RGB_W,
            range_height: RANGE_H,
            range_width: RANGE_W,
            range_channels: vec![RangeChannel::Range, RangeChannel::Intensity],
        }
    }
}

impl ProcessOptions {
    fn validate(&self) -> Result<()> {
        if self.rgb_height == 0 || self.rgb_width == 0 {
            anyhow::bail!("rgb size must be greater than zero");
        }
        if self.range_height == 0 || self.range_width == 0 {
            anyhow::bail!("range size must be greater than zero");
        }
        if self.range_channels.is_empty() {
            anyhow::bail!("range_channels must contain at least one channel");
        }
        Ok(())
    }
}

pub struct ExtractedPair {
    pub segment: String,
    pub timestamp_micros: i64,
    pub jpeg_bytes: Vec<u8>,
    pub range_values: Vec<f32>,
    pub range_shape: [usize; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct AlignOptions {
    pub max_timestamp_delta_micros: i64,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            max_timestamp_delta_micros: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlignStats {
    pub camera_frames: usize,
    pub lidar_frames: usize,
    pub matched_pairs: usize,
    pub unmatched_camera: usize,
    pub unmatched_lidar: usize,
    pub max_abs_timestamp_delta_micros: i64,
}

pub fn align_frames(camera: Vec<CameraFrame>, lidar: Vec<LidarFrame>) -> Vec<ExtractedPair> {
    align_frames_with_options(camera, lidar, AlignOptions::default()).0
}

pub fn align_frames_with_options(
    camera: Vec<CameraFrame>,
    lidar: Vec<LidarFrame>,
    options: AlignOptions,
) -> (Vec<ExtractedPair>, AlignStats) {
    let max_delta = options.max_timestamp_delta_micros.max(0);
    let mut stats = AlignStats {
        camera_frames: camera.len(),
        lidar_frames: lidar.len(),
        ..AlignStats::default()
    };

    let mut lidar_timeline = HashMap::<String, BTreeMap<i64, Vec<LidarFrame>>>::new();
    for frame in lidar {
        lidar_timeline
            .entry(frame.segment.clone())
            .or_default()
            .entry(frame.timestamp_micros)
            .or_default()
            .push(frame);
    }

    let mut pairs = Vec::new();

    for cam in camera {
        let Some(segment_timeline) = lidar_timeline.get_mut(&cam.segment) else {
            stats.unmatched_camera += 1;
            continue;
        };

        if let Some((lid, abs_delta)) = pop_best_lidar(segment_timeline, cam.timestamp_micros, max_delta) {
            pairs.push(ExtractedPair {
                segment: cam.segment.clone(),
                timestamp_micros: cam.timestamp_micros,
                jpeg_bytes: cam.jpeg_bytes.clone(),
                range_values: lid.range_values.clone(),
                range_shape: lid.shape,
            });
            stats.max_abs_timestamp_delta_micros =
                stats.max_abs_timestamp_delta_micros.max(abs_delta);
        } else {
            stats.unmatched_camera += 1;
        }
    }

    stats.matched_pairs = pairs.len();
    stats.unmatched_lidar = remaining_lidar_frames(&lidar_timeline);

    (pairs, stats)
}

fn pop_best_lidar(
    timeline: &mut BTreeMap<i64, Vec<LidarFrame>>,
    target_ts: i64,
    max_delta: i64,
) -> Option<(LidarFrame, i64)> {
    let lower = timeline
        .range(..=target_ts)
        .next_back()
        .map(|(&ts, _)| ts);
    let upper = timeline.range(target_ts..).next().map(|(&ts, _)| ts);

    let choose = match (lower, upper) {
        (Some(a), Some(b)) => {
            let da = (target_ts - a).abs();
            let db = (target_ts - b).abs();
            if da <= db {
                (a, da)
            } else {
                (b, db)
            }
        }
        (Some(a), None) => (a, (target_ts - a).abs()),
        (None, Some(b)) => (b, (target_ts - b).abs()),
        (None, None) => return None,
    };

    if choose.1 > max_delta {
        return None;
    }

    let (frame, remove_key) = {
        let frames = timeline.get_mut(&choose.0)?;
        let frame = frames.pop()?;
        (frame, frames.is_empty())
    };

    if remove_key {
        timeline.remove(&choose.0);
    }

    Some((frame, choose.1))
}

fn remaining_lidar_frames(timeline: &HashMap<String, BTreeMap<i64, Vec<LidarFrame>>>) -> usize {
    timeline
        .values()
        .flat_map(|by_ts| by_ts.values())
        .map(Vec::len)
        .sum()
}

pub fn write_sample(
    output_dir: &Path,
    pair: &ExtractedPair,
    split: &Split,
    sample_idx: usize,
) -> Result<ProcessedSampleEntry> {
    write_sample_with_options(output_dir, pair, split, sample_idx, &ProcessOptions::default())
}

pub fn write_sample_with_options(
    output_dir: &Path,
    pair: &ExtractedPair,
    split: &Split,
    sample_idx: usize,
    options: &ProcessOptions,
) -> Result<ProcessedSampleEntry> {
    options.validate()?;

    let sample_id = format!("sample_{sample_idx:06}");
    let sample_dir = output_dir.join(split_dir(split)).join(&sample_id);
    std::fs::create_dir_all(&sample_dir)
        .with_context(|| format!("creating {}", sample_dir.display()))?;

    let rgb_path = sample_dir.join("rgb.f32.bin");
    let range_path = sample_dir.join("range.f32.bin");
    let meta_path = sample_dir.join("meta.json");
    let preview_rgb_path = sample_dir.join("preview_rgb.png");
    let preview_range_path = sample_dir.join("preview_range.png");

    // --- RGB: decode JPEG → resize [H, W] → normalise → CHW f32 binary ---
    let rgb_chw =
        process_rgb(&pair.jpeg_bytes, options.rgb_height, options.rgb_width).context("processing RGB frame")?;
    let rgb_bytes: Vec<u8> = rgb_chw.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&rgb_path, &rgb_bytes)
        .with_context(|| format!("writing {}", rgb_path.display()))?;

    write_rgb_preview(&rgb_chw, options.rgb_height, options.rgb_width, &preview_rgb_path)
        .with_context(|| format!("writing {}", preview_rgb_path.display()))?;

    // --- Range: extract channels [0,1] → resize width → normalise → CHW f32 binary ---
    let range_chw = process_range(
        &pair.range_values,
        &pair.range_shape,
        options.range_height,
        options.range_width,
        &options.range_channels,
    )
    .context("processing range frame")?;
    let range_bytes: Vec<u8> = range_chw.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&range_path, &range_bytes)
        .with_context(|| format!("writing {}", range_path.display()))?;

    write_range_preview(&range_chw, options.range_height, options.range_width, &preview_range_path)
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

/// Decode JPEG → resize to `[H, W]` → normalise to `[0, 1]` → CHW layout `[3, H, W]`.
fn process_rgb(jpeg: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
    let img = image::load_from_memory(jpeg).context("decoding JPEG")?;
    let resized = img.resize_exact(w as u32, h as u32, FilterType::Triangle);
    let rgb8 = resized.to_rgb8();

    let pixels = rgb8.as_raw();
    let num = h * w;
    let mut chw = vec![0.0f32; 3 * num];

    for i in 0..num {
        chw[i] = pixels[3 * i] as f32 / 255.0;
        chw[num + i] = pixels[3 * i + 1] as f32 / 255.0;
        chw[2 * num + i] = pixels[3 * i + 2] as f32 / 255.0;
    }

    Ok(chw)
}

/// Extract channels `[0, 1]` from raw LiDAR range image `[H_src, W_src, C_src]`,
/// resize width to `W_target`, normalise, return CHW `[2, H, W]`.
fn process_range(
    values: &[f32],
    shape: &[usize; 3],
    h: usize,
    w: usize,
    channels: &[RangeChannel],
) -> Result<Vec<f32>> {
    let [h_src, w_src, c_src] = *shape;
    if h_src == 0 || w_src == 0 || c_src == 0 {
        anyhow::bail!("invalid range shape {h_src}×{w_src}×{c_src}");
    }
    if c_src < 2 {
        anyhow::bail!("range image has only {c_src} channel(s); need at least 2");
    }
    if channels.is_empty() {
        anyhow::bail!("at least one range channel is required");
    }

    let scale_x = w_src as f32 / w as f32;
    let scale_y = h_src as f32 / h as f32;
    let mut range = vec![0.0f32; h * w];
    let mut intensity = vec![0.0f32; h * w];

    for row in 0..h {
        let src_y = (row as f32 + 0.5) * scale_y - 0.5;
        let y0 = (src_y.floor() as isize).clamp(0, h_src as isize - 1) as usize;
        let y1 = (y0 + 1).min(h_src - 1);
        let ty = (src_y - y0 as f32).clamp(0.0, 1.0);

        for col in 0..w {
            let src_x = (col as f32 + 0.5) * scale_x - 0.5;
            let x0 = (src_x.floor() as isize).clamp(0, w_src as isize - 1) as usize;
            let x1 = (x0 + 1).min(w_src - 1);
            let tx = (src_x - x0 as f32).clamp(0.0, 1.0);

            let i00 = (y0 * w_src + x0) * c_src;
            let i01 = (y0 * w_src + x1) * c_src;
            let i10 = (y1 * w_src + x0) * c_src;
            let i11 = (y1 * w_src + x1) * c_src;

            let r00 = values.get(i00).copied().unwrap_or(0.0);
            let r01 = values.get(i01).copied().unwrap_or(0.0);
            let r10 = values.get(i10).copied().unwrap_or(0.0);
            let r11 = values.get(i11).copied().unwrap_or(0.0);

            let t00 = values.get(i00 + 1).copied().unwrap_or(0.0);
            let t01 = values.get(i01 + 1).copied().unwrap_or(0.0);
            let t10 = values.get(i10 + 1).copied().unwrap_or(0.0);
            let t11 = values.get(i11 + 1).copied().unwrap_or(0.0);

            let r0 = r00 + tx * (r01 - r00);
            let r1 = r10 + tx * (r11 - r10);
            let v_range = r0 + ty * (r1 - r0);

            let t0 = t00 + tx * (t01 - t00);
            let t1 = t10 + tx * (t11 - t10);
            let v_intensity = t0 + ty * (t1 - t0);

            range[row * w + col] = v_range;
            intensity[row * w + col] = v_intensity;
        }
    }

    // Normalise channel 0: divide by physical max range, clamp to [0, 1]
    for v in &mut range {
        *v = (*v / MAX_RANGE_METERS).clamp(0.0, 1.0);
    }

    // Normalise channel 1 (intensity): to [0, 1] by observed max
    let int_max = intensity.iter().cloned().fold(0.0f32, f32::max);
    if int_max > 0.0 {
        for v in &mut intensity {
            *v = (*v / int_max).clamp(0.0, 1.0);
        }
    }

    let validity: Vec<f32> = range
        .iter()
        .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let mut chw = Vec::with_capacity(channels.len() * h * w);
    for channel in channels {
        match channel {
            RangeChannel::Range => chw.extend_from_slice(&range),
            RangeChannel::Intensity => chw.extend_from_slice(&intensity),
            RangeChannel::ValidityMask => chw.extend_from_slice(&validity),
        }
    }

    Ok(chw)
}

/// Save the processed CHW RGB tensor (already f32 [0,1]) as a PNG preview.
fn write_rgb_preview(chw: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    let num = h * w;
    let mut pixels = vec![0u8; 3 * num];
    for i in 0..num {
        pixels[3 * i] = (chw[i].clamp(0.0, 1.0) * 255.0) as u8;
        pixels[3 * i + 1] = (chw[num + i].clamp(0.0, 1.0) * 255.0) as u8;
        pixels[3 * i + 2] = (chw[2 * num + i].clamp(0.0, 1.0) * 255.0) as u8;
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create RGB preview image"))?;
    img.save(path)
        .with_context(|| format!("saving RGB preview to {}", path.display()))?;
    Ok(())
}

/// Save channel 0 of the processed CHW range tensor as a grayscale PNG preview.
fn write_range_preview(chw: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    let ch0 = &chw[..h * w]; // channel 0 already normalised to [0, 1]
    let pixels: Vec<u8> = ch0
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_frames_matches_exact_timestamps() {
        let camera = vec![camera("seg", 100), camera("seg", 200)];
        let lidar = vec![lidar("seg", 100), lidar("seg", 300)];

        let (pairs, stats) = align_frames_with_options(camera, lidar, AlignOptions::default());

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].timestamp_micros, 100);
        assert_eq!(stats.camera_frames, 2);
        assert_eq!(stats.lidar_frames, 2);
        assert_eq!(stats.matched_pairs, 1);
        assert_eq!(stats.unmatched_camera, 1);
        assert_eq!(stats.unmatched_lidar, 1);
    }

    #[test]
    fn align_frames_supports_small_timestamp_tolerance() {
        let camera = vec![camera("seg", 1_000_000)];
        let lidar = vec![lidar("seg", 1_000_012)];

        let (pairs, stats) = align_frames_with_options(
            camera,
            lidar,
            AlignOptions {
                max_timestamp_delta_micros: 20,
            },
        );

        assert_eq!(pairs.len(), 1);
        assert_eq!(stats.max_abs_timestamp_delta_micros, 12);
    }

    #[test]
    fn align_frames_does_not_reuse_same_lidar_frame() {
        let camera = vec![camera("seg", 100), camera("seg", 100)];
        let lidar = vec![lidar("seg", 100)];

        let (pairs, stats) = align_frames_with_options(camera, lidar, AlignOptions::default());

        assert_eq!(pairs.len(), 1);
        assert_eq!(stats.unmatched_camera, 1);
        assert_eq!(stats.unmatched_lidar, 0);
    }

    #[test]
    fn process_range_supports_selected_channels() {
        let values = vec![
            10.0, 1.0, 20.0, 2.0, // row 0
            0.0, 0.0, 40.0, 4.0, // row 1
        ];
        let shape = [2, 2, 2];

        let chw = process_range(
            &values,
            &shape,
            2,
            2,
            &[RangeChannel::Range, RangeChannel::ValidityMask],
        )
        .unwrap();

        assert_eq!(chw.len(), 8);
        assert!(chw[0] > 0.0);
        assert_eq!(chw[4], 1.0);
        assert_eq!(chw[6], 0.0);
    }

    fn camera(segment: &str, timestamp_micros: i64) -> CameraFrame {
        CameraFrame {
            segment: segment.to_string(),
            timestamp_micros,
            jpeg_bytes: Vec::new(),
        }
    }

    fn lidar(segment: &str, timestamp_micros: i64) -> LidarFrame {
        LidarFrame {
            segment: segment.to_string(),
            timestamp_micros,
            range_values: vec![0.0, 0.0],
            shape: [1, 1, 2],
        }
    }
}
