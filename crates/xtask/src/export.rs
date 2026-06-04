use crate::{ExportArgs, ProjectPaths, display_from_root, resolve_run_dir};
use anyhow::{Context, Result, anyhow, ensure};
use image::{GrayImage, Luma, Rgb, RgbImage};
use sfx_core::manifest::{ProcessedSampleManifest, Split, TensorShape, read_manifest};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn run(args: ExportArgs, paths: &ProjectPaths) -> Result<()> {
    let manifest_path = paths.processed_sample_manifest_path();
    if !manifest_path.exists() {
        println!(
            "warn processed sample manifest not found at {}",
            display_from_root(&paths.root, &manifest_path)
        );
        return Ok(());
    }

    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let split = parse_split(&args.split)?;
    let processed_dir = resolve_processed_dir(paths, &manifest);
    let out_dir = sfx_config::resolve_from_root(&paths.root, &args.out);
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let selected: Vec<_> = manifest
        .samples
        .iter()
        .filter(|sample| sample.meta.split == split)
        .take(args.n)
        .collect();

    let mut written_files = 0usize;
    for sample in selected {
        if let Some(shape) = manifest.rgb_shape.as_ref() {
            let truth = load_tensor(
                &processed_dir.join(&sample.meta.rgb_path),
                shape.value_count(),
            )?;
            // TODO: replace zero reconstruction with Burn checkpoint loading from run_dir.
            let reconstructed = vec![0.0; truth.len()];
            let error: Vec<f32> = truth
                .iter()
                .zip(reconstructed.iter())
                .map(|(original, predicted)| (original - predicted).abs() * 4.0)
                .collect();
            let triptych = rgb_triptych(&truth, &reconstructed, &error, shape)?;
            let rgb_path = out_dir.join(format!("{}_rgb.ppm", sample.meta.id.0));
            triptych
                .save(&rgb_path)
                .with_context(|| format!("writing {}", rgb_path.display()))?;
            written_files += 1;
        }

        if let Some(shape) = manifest.range_shape.as_ref() {
            let truth = load_tensor(
                &processed_dir.join(&sample.meta.range_path),
                shape.value_count(),
            )?;
            let reconstructed = vec![0.0; truth.len()];
            let error: Vec<f32> = truth
                .iter()
                .zip(reconstructed.iter())
                .map(|(original, predicted)| (original - predicted).abs() * 4.0)
                .collect();
            let triptych = range_triptych(&truth, &reconstructed, &error, shape)?;
            let range_path = out_dir.join(format!("{}_range.pgm", sample.meta.id.0));
            triptych
                .save(&range_path)
                .with_context(|| format!("writing {}", range_path.display()))?;
            written_files += 1;
        }
    }

    println!(
        "ok   wrote {} files for run {} to {}",
        written_files,
        display_from_root(&paths.root, &run_dir),
        display_from_root(&paths.root, &out_dir)
    );
    Ok(())
}

fn parse_split(value: &str) -> Result<Split> {
    match value.trim().to_ascii_lowercase().as_str() {
        "train" => Ok(Split::Train),
        "val" | "validation" => Ok(Split::Val),
        "test" => Ok(Split::Test),
        other => Err(anyhow!(
            "unsupported split {other}; expected train, val, or test"
        )),
    }
}

fn resolve_processed_dir(paths: &ProjectPaths, manifest: &ProcessedSampleManifest) -> PathBuf {
    if let Some(sample) = manifest.samples.first() {
        let root_relative = paths.root.join(&sample.meta.rgb_path);
        if root_relative.exists() {
            return paths.root.clone();
        }
    }

    if let Some(dataset) = manifest.dataset.as_deref() {
        let candidate = paths.root.join("data").join("processed").join(dataset);
        if manifest
            .samples
            .first()
            .map(|sample| candidate.join(&sample.meta.rgb_path).exists())
            .unwrap_or(true)
        {
            return candidate;
        }
    }

    paths.root.join("data").join("processed")
}

fn load_tensor(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "{} has {} trailing bytes; expected f32-aligned data",
        path.display(),
        bytes.len() % 4
    );
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    ensure!(
        floats.len() == expected_len,
        "{} has {} values; expected {}",
        path.display(),
        floats.len(),
        expected_len
    );
    Ok(floats)
}

fn rgb_triptych(
    truth: &[f32],
    reconstructed: &[f32],
    error: &[f32],
    shape: &TensorShape,
) -> Result<RgbImage> {
    let original = rgb_image_from_chw(truth, shape)?;
    let reconstructed = rgb_image_from_chw(reconstructed, shape)?;
    let error = rgb_image_from_chw(error, shape)?;
    let mut canvas = RgbImage::new(shape.width as u32 * 3, shape.height as u32);
    image::imageops::replace(&mut canvas, &original, 0, 0);
    image::imageops::replace(&mut canvas, &reconstructed, shape.width as i64, 0);
    image::imageops::replace(&mut canvas, &error, (shape.width * 2) as i64, 0);
    Ok(canvas)
}

fn range_triptych(
    truth: &[f32],
    reconstructed: &[f32],
    error: &[f32],
    shape: &TensorShape,
) -> Result<GrayImage> {
    let original = gray_image_from_first_channel(truth, shape)?;
    let reconstructed = gray_image_from_first_channel(reconstructed, shape)?;
    let error = gray_image_from_first_channel(error, shape)?;
    let mut canvas = GrayImage::new(shape.width as u32 * 3, shape.height as u32);
    image::imageops::replace(&mut canvas, &original, 0, 0);
    image::imageops::replace(&mut canvas, &reconstructed, shape.width as i64, 0);
    image::imageops::replace(&mut canvas, &error, (shape.width * 2) as i64, 0);
    Ok(canvas)
}

fn rgb_image_from_chw(values: &[f32], shape: &TensorShape) -> Result<RgbImage> {
    ensure!(
        shape.channels >= 3,
        "RGB export requires at least 3 channels"
    );
    ensure!(
        values.len() == shape.value_count(),
        "RGB tensor has unexpected size"
    );
    let plane = shape.height * shape.width;
    let mut image = RgbImage::new(shape.width as u32, shape.height as u32);
    for y in 0..shape.height {
        for x in 0..shape.width {
            let index = y * shape.width + x;
            image.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    scale_to_u8(values[index]),
                    scale_to_u8(values[plane + index]),
                    scale_to_u8(values[2 * plane + index]),
                ]),
            );
        }
    }
    Ok(image)
}

fn gray_image_from_first_channel(values: &[f32], shape: &TensorShape) -> Result<GrayImage> {
    ensure!(
        shape.channels >= 1,
        "range export requires at least 1 channel"
    );
    ensure!(
        values.len() == shape.value_count(),
        "range tensor has unexpected size"
    );
    let mut image = GrayImage::new(shape.width as u32, shape.height as u32);
    for y in 0..shape.height {
        for x in 0..shape.width {
            let index = y * shape.width + x;
            image.put_pixel(x as u32, y as u32, Luma([scale_to_u8(values[index])]));
        }
    }
    Ok(image)
}

fn scale_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
