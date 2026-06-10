use crate::{CompressBenchArgs, ProjectPaths, display_from_root, resolve_run_dir};
use anyhow::{bail, ensure, Context, Result};
use sfx_bench::{
    compute_int8_quantization_metadata, deflate_compressed_size_for_serialized_quantized_payload,
    dequantize_latent_from_int8, interpolate_rate_at_psnr, load_compression_bench_config,
    quantize_latent_to_int8, raw_f32_latent_byte_size, render_summary_tables_markdown,
    AeOperatingPoint, AeOperatingPoints, CodecMatchedQualityPoint, CodecRdCurve, CodecRdPoint,
    CombinedSummary, CompressionBenchConfig, CompressionBenchmarkResult, CompressionMetadata,
    CompressionMode, DecodeRequest, EncodeRequest, EncodingMode, FfmpegExecutor,
    MatchedQualityPoint, MatchedQualitySummary, ModalitySummary, QuantizationConfig,
    QuantizationMinMaxPolicy, QuantizationStrategy, RdPoint, SystemFfmpegRunner, VideoCodec,
};
use sfx_core::manifest::{read_manifest, ProcessedSampleManifest, Split, TensorShape};
use sfx_data::{BatchOptions, FusionDataset};
use sfx_eval::{
    aggregate_metrics, aggregate_range_depth_metrics, compute_modality_metrics,
    compute_range_depth_metrics, compute_rgb_ssim, AggregatedRangeDepthMetrics, ModalityMetrics,
    RangeDepthMetrics,
};
use sfx_train::{Embedding, RunInference, TrainingSummary};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BENCH_BATCH: usize = 8;
pub(crate) fn run(args: CompressBenchArgs, paths: &ProjectPaths) -> Result<()> {
    let Some(run_dir) = resolve_run_dir(paths, args.run.as_deref())? else {
        println!("warn no run found; pass --run <run_dir> or create .xtask/runs/latest.json");
        return Ok(());
    };

    let config_path = sfx_config::resolve_from_root(&paths.root, &args.config);
    let mut config = load_compression_bench_config(&paths.root, &args.config)
        .with_context(|| format!("loading {}", config_path.display()))?;
    apply_overrides(&mut config, &args, paths)?;

    let (manifest_path, split_protocol, dataset_config_path) =
        resolve_manifest_path(paths, &run_dir)?;
    if !manifest_path.exists() {
        println!(
            "warn processed sample manifest not found at {}",
            display_from_root(&paths.root, &manifest_path)
        );
        return Ok(());
    }

    let manifest: ProcessedSampleManifest = read_manifest(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let processed_dir = resolve_processed_dir(paths, &manifest);
    let range_eval = resolve_range_eval_config(
        paths,
        &run_dir,
        &processed_dir,
        dataset_config_path.as_deref(),
    )?;
    let inferer = RunInference::load(&run_dir).context("loading run checkpoint")?;
    let selected_splits = parse_splits(&config.split)?;

    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("creating {}", config.output_dir.display()))?;

    let runner = SystemFfmpegRunner::default();
    for split in selected_splits {
        let split_name = split_name(&split).to_string();
        let split_result = run_split_benchmark(
            &runner,
            &inferer,
            &run_dir,
            &manifest_path,
            &processed_dir,
            split,
            &config,
            range_eval,
            &split_protocol,
        )?;

        let json_path = config.output_dir.join(format!("{split_name}.json"));
        let markdown_path = config.output_dir.join(format!("{split_name}.md"));
        let json =
            serde_json::to_string_pretty(&split_result).context("serializing benchmark JSON")?;
        fs::write(&json_path, json).with_context(|| format!("writing {}", json_path.display()))?;
        fs::write(
            &markdown_path,
            render_summary_tables_markdown(&split_result),
        )
        .with_context(|| format!("writing {}", markdown_path.display()))?;

        print_summary_table(&split_result);
        println!("ok   {}", display_from_root(&paths.root, &json_path));
        println!("ok   {}", display_from_root(&paths.root, &markdown_path));
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct PreparedSample {
    id: String,
    rgb_u8: Vec<u8>,
    rgb_chw_u8: Vec<f32>,
    range_u8: Vec<u8>,
    range_chw_u8: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
struct MetricAccumulator {
    rgb: Vec<ModalityMetrics>,
    range: Vec<ModalityMetrics>,
    range_depth: Vec<RangeDepthMetrics>,
}

impl MetricAccumulator {
    fn push_rgb(&mut self, metrics: ModalityMetrics) {
        self.rgb.push(metrics);
    }

    fn push_range(&mut self, metrics: ModalityMetrics) {
        self.range.push(metrics);
    }

    fn push_range_depth(&mut self, metrics: RangeDepthMetrics) {
        self.range_depth.push(metrics);
    }

    fn modality_summary(&self) -> Vec<ModalitySummary> {
        let mut modalities = Vec::new();
        if !self.rgb.is_empty() {
            let agg = aggregate_metrics(&self.rgb);
            modalities.push(ModalitySummary {
                modality: "rgb".to_string(),
                mse: agg.mean_mse,
                psnr_db: agg.mean_psnr_db,
                ssim: agg.mean_ssim,
            });
        }
        if !self.range.is_empty() {
            let agg = aggregate_metrics(&self.range);
            modalities.push(ModalitySummary {
                modality: "range".to_string(),
                mse: agg.mean_mse,
                psnr_db: agg.mean_psnr_db,
                ssim: agg.mean_ssim,
            });
        }
        modalities
    }

    fn combined_summary(&self) -> CombinedSummary {
        let modalities = self.modality_summary();
        if modalities.is_empty() {
            return CombinedSummary {
                mse: 0.0,
                psnr_db: 0.0,
                ssim: None,
            };
        }

        let n = modalities.len() as f32;
        let mean_mse = modalities.iter().map(|entry| entry.mse).sum::<f32>() / n;
        let mean_psnr = modalities.iter().map(|entry| entry.psnr_db).sum::<f32>() / n;
        let ssim_values: Vec<f32> = modalities.iter().filter_map(|entry| entry.ssim).collect();
        let mean_ssim = (!ssim_values.is_empty())
            .then(|| ssim_values.iter().sum::<f32>() / ssim_values.len() as f32);

        CombinedSummary {
            mse: mean_mse,
            psnr_db: mean_psnr,
            ssim: mean_ssim,
        }
    }

    fn aggregated_range_depth(&self) -> Option<AggregatedRangeDepthMetrics> {
        (!self.range_depth.is_empty()).then(|| aggregate_range_depth_metrics(&self.range_depth))
    }
}

#[derive(Debug, Clone, Copy)]
struct RangeEvalConfig {
    max_range_meters: f32,
    validity_channel_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct CodecSelection {
    label: String,
    video_codec: VideoCodec,
    encoder: Option<String>,
}

#[derive(Debug, Clone)]
struct CodecPointComputation {
    rate_bpp: f32,
    metrics: MetricAccumulator,
}

fn run_split_benchmark(
    ffmpeg: &impl FfmpegExecutor,
    inferer: &RunInference,
    run_dir: &Path,
    manifest_path: &Path,
    processed_dir: &Path,
    split: Split,
    config: &CompressionBenchConfig,
    range_eval: RangeEvalConfig,
    split_protocol: &str,
) -> Result<CompressionBenchmarkResult> {
    let dataset = FusionDataset::open_split(processed_dir, manifest_path, Some(split.clone()))
        .with_context(|| format!("opening {} dataset", split_name(&split)))?;
    if dataset.is_empty() {
        bail!("dataset split `{}` has no samples", split_name(&split));
    }

    let rgb_shape = dataset.rgb_shape().clone();
    let range_shape = dataset.range_shape().clone();
    let rgb_values = rgb_shape.value_count();
    let range_values = range_shape.value_count();
    let range_pixels = range_shape.height * range_shape.width;
    ensure!(range_pixels > 0, "range tensor has zero pixels");
    let range_channels_for_metrics = range_shape.channels.min(2).max(1);
    let codec_validity_channel = range_eval
        .validity_channel_index
        .filter(|channel| *channel < range_channels_for_metrics);

    let sample_cap = config.sample_cap.unwrap_or(usize::MAX);
    let mut prepared_samples = Vec::new();
    let mut raw_u8_metrics = MetricAccumulator::default();
    let mut raw_f32_metrics = MetricAccumulator::default();
    let mut quant_u8_metrics = MetricAccumulator::default();
    let mut quant_f32_metrics = MetricAccumulator::default();
    let mut raw_latent_bytes_total = 0usize;
    let mut quantized_latent_bytes_total = 0usize;
    let mut quantization_notes = quantization_config_notes(&config.quantization);

    for batch in dataset.batches(BatchOptions::new(BENCH_BATCH))? {
        let batch = batch?;
        let remaining = sample_cap.saturating_sub(prepared_samples.len());
        if remaining == 0 {
            break;
        }
        let take = remaining.min(batch.batch_size());
        if take == 0 {
            break;
        }

        let raw_recon = inferer.reconstruct(&batch)?;
        let embedding = inferer.embed(&batch)?;
        let mut quantized_z = Vec::with_capacity(take * embedding.dim);
        let mut quantized_ids = Vec::with_capacity(take);

        for index in 0..take {
            let latent_start = index * embedding.dim;
            let latent_end = latent_start + embedding.dim;
            let latent = &embedding.z[latent_start..latent_end];
            raw_latent_bytes_total += raw_f32_latent_byte_size(latent);

            let dequantized = quantize_and_restore_latent(
                latent,
                &config.quantization,
                &mut quantized_latent_bytes_total,
                &mut quantization_notes,
            )?;
            quantized_ids.push(embedding.sample_ids[index].clone());
            quantized_z.extend_from_slice(&dequantized);
        }

        let quantized_embedding = Embedding {
            sample_ids: quantized_ids,
            z: quantized_z,
            dim: embedding.dim,
        };
        let quantized_recon = inferer.reconstruct_from_latent(&quantized_embedding)?;

        for index in 0..take {
            let rgb_start = index * rgb_values;
            let rgb_end = rgb_start + rgb_values;
            let range_start = index * range_values;
            let range_end = range_start + range_values;

            let truth_rgb = &batch.rgb[rgb_start..rgb_end];
            let truth_range = &batch.range[range_start..range_end];
            let truth_range_slice = &truth_range[..range_channels_for_metrics * range_pixels];

            let rgb_u8 = rgb_chw_to_rgb24_u8(truth_rgb, &rgb_shape)?;
            let rgb_chw_u8 = rgb24_to_rgb_chw_f32(&rgb_u8, rgb_shape.height, rgb_shape.width)?;
            let range_u8 = range_chw_to_packed_rgb24_u8(truth_range, &range_shape)?;
            let range_chw_u8 = range_rgb24_to_two_channel_chw_f32(
                &range_u8,
                range_shape.height,
                range_shape.width,
            )?;

            prepared_samples.push(PreparedSample {
                id: batch.sample_ids[index].0.clone(),
                rgb_u8,
                rgb_chw_u8: rgb_chw_u8.clone(),
                range_u8,
                range_chw_u8: range_chw_u8.clone(),
            });

            if let Some(rgb_hat) = raw_recon.rgb_hat.as_ref() {
                let rgb_pred = &rgb_hat[rgb_start..rgb_end];
                raw_f32_metrics.push_rgb(rgb_metrics(rgb_pred, truth_rgb, &rgb_shape)?);
                let rgb_pred_u8 = rgb_chw_to_rgb24_u8(rgb_pred, &rgb_shape)?;
                let rgb_pred_u8_chw =
                    rgb24_to_rgb_chw_f32(&rgb_pred_u8, rgb_shape.height, rgb_shape.width)?;
                raw_u8_metrics.push_rgb(rgb_metrics(&rgb_pred_u8_chw, &rgb_chw_u8, &rgb_shape)?);
            }

            if let Some(range_hat) = raw_recon.range_hat.as_ref() {
                let range_pred = &range_hat[range_start..range_end];
                let range_pred_slice = &range_pred[..range_channels_for_metrics * range_pixels];
                raw_f32_metrics.push_range(compute_modality_metrics(
                    range_pred_slice,
                    truth_range_slice,
                    1.0,
                ));
                raw_f32_metrics.push_range_depth(compute_range_depth_metrics(
                    range_pred_slice,
                    truth_range_slice,
                    range_channels_for_metrics,
                    range_shape.height,
                    range_shape.width,
                    codec_validity_channel,
                    range_eval.max_range_meters,
                ));

                let range_pred_u8 = range_chw_to_packed_rgb24_u8(range_pred, &range_shape)?;
                let range_pred_u8_chw = range_rgb24_to_two_channel_chw_f32(
                    &range_pred_u8,
                    range_shape.height,
                    range_shape.width,
                )?;
                raw_u8_metrics.push_range(compute_modality_metrics(
                    &range_pred_u8_chw,
                    &range_chw_u8,
                    1.0,
                ));
                raw_u8_metrics.push_range_depth(compute_range_depth_metrics(
                    &range_pred_u8_chw,
                    &range_chw_u8,
                    range_channels_for_metrics,
                    range_shape.height,
                    range_shape.width,
                    codec_validity_channel,
                    range_eval.max_range_meters,
                ));
            }

            if let Some(rgb_hat) = quantized_recon.rgb_hat.as_ref() {
                let rgb_pred = &rgb_hat[rgb_start..rgb_end];
                quant_f32_metrics.push_rgb(rgb_metrics(rgb_pred, truth_rgb, &rgb_shape)?);
                let rgb_pred_u8 = rgb_chw_to_rgb24_u8(rgb_pred, &rgb_shape)?;
                let rgb_pred_u8_chw =
                    rgb24_to_rgb_chw_f32(&rgb_pred_u8, rgb_shape.height, rgb_shape.width)?;
                quant_u8_metrics.push_rgb(rgb_metrics(&rgb_pred_u8_chw, &rgb_chw_u8, &rgb_shape)?);
            }

            if let Some(range_hat) = quantized_recon.range_hat.as_ref() {
                let range_pred = &range_hat[range_start..range_end];
                let range_pred_slice = &range_pred[..range_channels_for_metrics * range_pixels];
                quant_f32_metrics.push_range(compute_modality_metrics(
                    range_pred_slice,
                    truth_range_slice,
                    1.0,
                ));
                quant_f32_metrics.push_range_depth(compute_range_depth_metrics(
                    range_pred_slice,
                    truth_range_slice,
                    range_channels_for_metrics,
                    range_shape.height,
                    range_shape.width,
                    codec_validity_channel,
                    range_eval.max_range_meters,
                ));

                let range_pred_u8 = range_chw_to_packed_rgb24_u8(range_pred, &range_shape)?;
                let range_pred_u8_chw = range_rgb24_to_two_channel_chw_f32(
                    &range_pred_u8,
                    range_shape.height,
                    range_shape.width,
                )?;
                quant_u8_metrics.push_range(compute_modality_metrics(
                    &range_pred_u8_chw,
                    &range_chw_u8,
                    1.0,
                ));
                quant_u8_metrics.push_range_depth(compute_range_depth_metrics(
                    &range_pred_u8_chw,
                    &range_chw_u8,
                    range_channels_for_metrics,
                    range_shape.height,
                    range_shape.width,
                    codec_validity_channel,
                    range_eval.max_range_meters,
                ));
            }
        }

        if prepared_samples.len() >= sample_cap {
            break;
        }
    }

    if prepared_samples.is_empty() {
        bail!(
            "no samples were processed for split `{}`",
            split_name(&split)
        );
    }

    let bits_per_pixel_denominator = bits_per_pixel_denominator(
        prepared_samples.len(),
        rgb_shape.height * rgb_shape.width,
        range_shape.height * range_shape.width,
    )?;
    let encoding_mode = encoding_mode(config.mode);

    let mut raw_notes = vec![
        format!("latent_bytes_total={raw_latent_bytes_total}"),
        format!(
            "sample_count={}, split_protocol={split_protocol}",
            prepared_samples.len()
        ),
    ];
    raw_notes.extend(format_f32_notes("f32", &raw_f32_metrics));
    raw_notes.extend(format_depth_notes(
        "f32-depth",
        raw_f32_metrics.aggregated_range_depth(),
    ));

    let mut quantized_notes = vec![
        format!("latent_bytes_total={quantized_latent_bytes_total}"),
        format!(
            "sample_count={}, split_protocol={split_protocol}",
            prepared_samples.len()
        ),
    ];
    quantized_notes.extend(format_f32_notes("f32", &quant_f32_metrics));
    quantized_notes.extend(format_depth_notes(
        "f32-depth",
        quant_f32_metrics.aggregated_range_depth(),
    ));
    quantized_notes.extend(quantization_notes);

    let ae_operating_points = AeOperatingPoints {
        raw_f32: AeOperatingPoint {
            encoding_mode,
            rate_bpp: bytes_to_bpp(raw_latent_bytes_total, bits_per_pixel_denominator),
            modality_summary: raw_u8_metrics.modality_summary(),
            combined_summary: raw_u8_metrics.combined_summary(),
            notes: raw_notes,
        },
        quantized: AeOperatingPoint {
            encoding_mode,
            rate_bpp: bytes_to_bpp(quantized_latent_bytes_total, bits_per_pixel_denominator),
            modality_summary: quant_u8_metrics.modality_summary(),
            combined_summary: quant_u8_metrics.combined_summary(),
            notes: quantized_notes,
        },
    };

    let split_name = split_name(&split);
    let scratch_dir =
        config
            .output_dir
            .join(format!(".tmp-{}-{}", split_name, unix_timestamp_millis()));
    fs::create_dir_all(&scratch_dir)
        .with_context(|| format!("creating {}", scratch_dir.display()))?;
    let _cleanup = CleanupDir::new(scratch_dir.clone());

    let mut codec_rd_curves = Vec::new();
    for codec_name in &config.codecs {
        let codec = resolve_codec(codec_name)?;
        let mut rd_points = Vec::new();

        for &crf in crf_values_for_codec(config, codec_name)? {
            let point = run_codec_point(
                ffmpeg,
                &codec,
                crf,
                config.mode,
                config.preset.as_str(),
                &prepared_samples,
                &rgb_shape,
                &range_shape,
                range_eval,
                &scratch_dir,
                split_name,
            )?;

            rd_points.push(CodecRdPoint {
                quality_control: format!("crf={crf}"),
                rate_bpp: point.rate_bpp,
                modality_summary: point.metrics.modality_summary(),
                combined_summary: point.metrics.combined_summary(),
            });
        }

        rd_points.sort_by(|left, right| left.rate_bpp.total_cmp(&right.rate_bpp));

        codec_rd_curves.push(CodecRdCurve {
            codec: codec.label.clone(),
            encoding_mode,
            rd_points,
            lossless_anchor: None,
            caveats: Vec::new(),
        });
    }

    let matched_quality = build_matched_quality(&ae_operating_points, &codec_rd_curves);
    let run_id = run_id_from_run_dir(run_dir);
    let metadata = CompressionMetadata {
        run_id: run_id.clone(),
        generated_by: "xtask compress-bench".to_string(),
        caveats: Vec::new(),
        notes: BTreeMap::from([
            ("manifest".to_string(), manifest_path.display().to_string()),
            (
                "processed_dir".to_string(),
                processed_dir.display().to_string(),
            ),
            (
                "sample_count".to_string(),
                prepared_samples.len().to_string(),
            ),
            ("split_protocol".to_string(), split_protocol.to_string()),
        ]),
    };

    Ok(CompressionBenchmarkResult {
        bench_name: format!("compression-{run_id}"),
        split: split_name.to_string(),
        ae_operating_points,
        codec_rd_curves,
        matched_quality,
        metadata,
    })
}

fn run_codec_point(
    ffmpeg: &impl FfmpegExecutor,
    codec: &CodecSelection,
    crf: u8,
    mode: CompressionMode,
    preset: &str,
    samples: &[PreparedSample],
    rgb_shape: &TensorShape,
    range_shape: &TensorShape,
    range_eval: RangeEvalConfig,
    scratch_root: &Path,
    split_name: &str,
) -> Result<CodecPointComputation> {
    let point_dir = scratch_root.join(format!(
        "{}_{}_crf{}",
        split_name,
        sanitize_label(&codec.label),
        crf
    ));
    fs::create_dir_all(&point_dir).with_context(|| format!("creating {}", point_dir.display()))?;

    let (rgb_encoded_bytes, rgb_decoded_frames) = run_codec_for_frames(
        ffmpeg,
        codec,
        crf,
        mode,
        preset,
        samples,
        rgb_shape.width,
        rgb_shape.height,
        &point_dir,
        "rgb",
        |sample| sample.rgb_u8.as_slice(),
    )?;
    let (range_encoded_bytes, range_decoded_frames) = run_codec_for_frames(
        ffmpeg,
        codec,
        crf,
        mode,
        preset,
        samples,
        range_shape.width,
        range_shape.height,
        &point_dir,
        "range",
        |sample| sample.range_u8.as_slice(),
    )?;

    let range_channels = range_shape.channels.min(2).max(1);
    let validity_channel = range_eval
        .validity_channel_index
        .filter(|channel| *channel < range_channels);
    let mut metrics = MetricAccumulator::default();

    for (index, sample) in samples.iter().enumerate() {
        let rgb_decoded = rgb24_to_rgb_chw_f32(
            &rgb_decoded_frames[index],
            rgb_shape.height,
            rgb_shape.width,
        )?;
        metrics.push_rgb(rgb_metrics(&rgb_decoded, &sample.rgb_chw_u8, rgb_shape)?);

        let range_decoded = range_rgb24_to_two_channel_chw_f32(
            &range_decoded_frames[index],
            range_shape.height,
            range_shape.width,
        )?;
        metrics.push_range(compute_modality_metrics(
            &range_decoded,
            &sample.range_chw_u8,
            1.0,
        ));
        metrics.push_range_depth(compute_range_depth_metrics(
            &range_decoded,
            &sample.range_chw_u8,
            range_channels,
            range_shape.height,
            range_shape.width,
            validity_channel,
            range_eval.max_range_meters,
        ));
    }

    let bits_per_pixel_denominator = bits_per_pixel_denominator(
        samples.len(),
        rgb_shape.height * rgb_shape.width,
        range_shape.height * range_shape.width,
    )?;
    let encoded_total = rgb_encoded_bytes + range_encoded_bytes;

    Ok(CodecPointComputation {
        rate_bpp: bytes_to_bpp(encoded_total, bits_per_pixel_denominator),
        metrics,
    })
}

fn run_codec_for_frames<F>(
    ffmpeg: &impl FfmpegExecutor,
    codec: &CodecSelection,
    crf: u8,
    mode: CompressionMode,
    preset: &str,
    samples: &[PreparedSample],
    width: usize,
    height: usize,
    point_dir: &Path,
    label: &str,
    frame_selector: F,
) -> Result<(usize, Vec<Vec<u8>>)>
where
    F: Fn(&PreparedSample) -> &[u8],
{
    let frame_size = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("frame dimensions overflow")?;
    let frame_count = samples.len();
    ensure!(frame_count > 0, "codec sweep requires at least one sample");

    match mode {
        CompressionMode::Independent => {
            let mut encoded_bytes_total = 0usize;
            let mut decoded_frames = Vec::with_capacity(samples.len());
            for (index, sample) in samples.iter().enumerate() {
                let input_path = point_dir.join(format!("{label}_{index:06}.rgb"));
                let output_path = point_dir.join(format!("{label}_{index:06}.mkv"));
                let decoded_path = point_dir.join(format!("{label}_{index:06}.decoded.rgb"));

                let frame = frame_selector(sample);
                ensure!(
                    frame.len() == frame_size,
                    "frame {index} for {label} has {} bytes; expected {frame_size}",
                    frame.len()
                );
                fs::write(&input_path, frame)
                    .with_context(|| format!("writing {}", input_path.display()))?;

                let mut encode_request =
                    EncodeRequest::new(&input_path, &output_path, codec.video_codec);
                encode_request.encoder = codec.encoder.clone();
                encode_request.pre_input_args = rawvideo_input_args(width, height);
                encode_request.output_args = codec_output_args(codec.video_codec, crf, preset, 1);
                ffmpeg
                    .encode(&encode_request)
                    .with_context(|| format!("encoding {} frame {}", codec.label, index))?;

                let encoded_len = fs::metadata(&output_path)
                    .with_context(|| format!("reading {}", output_path.display()))?
                    .len() as usize;
                encoded_bytes_total += encoded_len;

                let mut decode_request =
                    DecodeRequest::new(&output_path, &decoded_path, sfx_bench::PixFmtFamily::Rgb);
                decode_request.codec_hint = Some(codec.video_codec);
                decode_request.pix_fmt = Some("rgb24".to_string());
                decode_request.output_args = rawvideo_output_args(1);
                ffmpeg
                    .decode(&decode_request)
                    .with_context(|| format!("decoding {} frame {}", codec.label, index))?;

                let decoded = fs::read(&decoded_path)
                    .with_context(|| format!("reading {}", decoded_path.display()))?;
                ensure!(
                    decoded.len() == frame_size,
                    "decoded {label} frame {index} has {} bytes; expected {frame_size}",
                    decoded.len()
                );
                decoded_frames.push(decoded);
            }

            Ok((encoded_bytes_total, decoded_frames))
        }
        CompressionMode::IntraSequence => {
            let stream_path = point_dir.join(format!("{label}.stream.rgb"));
            let output_path = point_dir.join(format!("{label}.mkv"));
            let decoded_path = point_dir.join(format!("{label}.decoded.rgb"));

            let mut stream = Vec::with_capacity(frame_count * frame_size);
            for (index, sample) in samples.iter().enumerate() {
                let frame = frame_selector(sample);
                ensure!(
                    frame.len() == frame_size,
                    "frame {index} for {label} has {} bytes; expected {frame_size}",
                    frame.len()
                );
                stream.extend_from_slice(frame);
            }
            fs::write(&stream_path, &stream)
                .with_context(|| format!("writing {}", stream_path.display()))?;

            let mut encode_request =
                EncodeRequest::new(&stream_path, &output_path, codec.video_codec);
            encode_request.encoder = codec.encoder.clone();
            encode_request.pre_input_args = rawvideo_input_args(width, height);
            encode_request.output_args =
                codec_output_args(codec.video_codec, crf, preset, frame_count);
            ffmpeg
                .encode(&encode_request)
                .with_context(|| format!("encoding {} {} sequence", codec.label, label))?;

            let encoded_bytes = fs::metadata(&output_path)
                .with_context(|| format!("reading {}", output_path.display()))?
                .len() as usize;

            let mut decode_request =
                DecodeRequest::new(&output_path, &decoded_path, sfx_bench::PixFmtFamily::Rgb);
            decode_request.codec_hint = Some(codec.video_codec);
            decode_request.pix_fmt = Some("rgb24".to_string());
            decode_request.output_args = rawvideo_output_args(frame_count);
            ffmpeg
                .decode(&decode_request)
                .with_context(|| format!("decoding {} {} sequence", codec.label, label))?;

            let decoded_bytes = fs::read(&decoded_path)
                .with_context(|| format!("reading {}", decoded_path.display()))?;
            let expected_total = frame_count * frame_size;
            ensure!(
                decoded_bytes.len() == expected_total,
                "decoded {label} stream has {} bytes; expected {expected_total}",
                decoded_bytes.len()
            );

            let mut decoded_frames = Vec::with_capacity(frame_count);
            for chunk in decoded_bytes.chunks_exact(frame_size) {
                decoded_frames.push(chunk.to_vec());
            }

            Ok((encoded_bytes, decoded_frames))
        }
    }
}

fn rawvideo_input_args(width: usize, height: usize) -> Vec<String> {
    vec![
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s:v".to_string(),
        format!("{width}x{height}"),
        "-framerate".to_string(),
        "1".to_string(),
    ]
}

fn rawvideo_output_args(frame_count: usize) -> Vec<String> {
    vec![
        "-f".to_string(),
        "rawvideo".to_string(),
        "-frames:v".to_string(),
        frame_count.to_string(),
    ]
}

fn codec_output_args(codec: VideoCodec, crf: u8, preset: &str, frame_count: usize) -> Vec<String> {
    let mut args = vec!["-frames:v".to_string(), frame_count.to_string()];
    match codec {
        VideoCodec::H264 | VideoCodec::H265 => {
            args.push("-preset".to_string());
            args.push(preset.to_string());
            args.push("-crf".to_string());
            args.push(crf.to_string());
        }
        VideoCodec::Av1 => {
            args.push("-preset".to_string());
            args.push(preset.to_string());
            args.push("-crf".to_string());
            args.push(crf.to_string());
            args.push("-b:v".to_string());
            args.push("0".to_string());
        }
        VideoCodec::Png | VideoCodec::Ffv1 => {}
    }
    args
}

fn quantize_and_restore_latent(
    latent: &[f32],
    quantization: &QuantizationConfig,
    quantized_byte_total: &mut usize,
    notes: &mut Vec<String>,
) -> Result<Vec<f32>> {
    if !quantization.enabled {
        *quantized_byte_total += raw_f32_latent_byte_size(latent);
        return Ok(latent.to_vec());
    }
    if quantization.bits != 8 {
        push_note_once(
            notes,
            format!(
                "quantization.bits={} requested; using int8 helpers for now",
                quantization.bits
            ),
        );
    }
    if quantization.per_channel {
        push_note_once(
            notes,
            "quantization.per_channel is not yet implemented".to_string(),
        );
    }
    if matches!(quantization.strategy, QuantizationStrategy::Logarithmic) {
        push_note_once(
            notes,
            "quantization.strategy=logarithmic maps to asymmetric int8 currently".to_string(),
        );
    }

    let policy = quantization_policy(quantization.strategy);
    let metadata = compute_int8_quantization_metadata(latent, policy)
        .context("computing int8 quantization metadata")?;
    let quantized =
        quantize_latent_to_int8(latent, &metadata).context("quantizing latent vector")?;
    *quantized_byte_total += deflate_compressed_size_for_serialized_quantized_payload(
        &quantized,
        &metadata,
        Default::default(),
    )
    .context("estimating compressed quantized latent payload size")?;
    dequantize_latent_from_int8(&quantized, &metadata).context("dequantizing latent vector")
}

fn quantization_policy(strategy: QuantizationStrategy) -> QuantizationMinMaxPolicy {
    match strategy {
        QuantizationStrategy::Uniform => QuantizationMinMaxPolicy::Asymmetric,
        QuantizationStrategy::Logarithmic => QuantizationMinMaxPolicy::Asymmetric,
    }
}

fn quantization_config_notes(config: &QuantizationConfig) -> Vec<String> {
    let mut notes = Vec::new();
    if !config.enabled {
        notes
            .push("quantization is disabled; quantized point mirrors raw latent sizes".to_string());
    }
    notes
}

fn rgb_chw_to_rgb24_u8(values: &[f32], shape: &TensorShape) -> Result<Vec<u8>> {
    ensure!(
        shape.channels >= 3,
        "RGB tensors must provide at least 3 channels"
    );
    ensure!(
        values.len() == shape.value_count(),
        "RGB tensor has {} values; expected {}",
        values.len(),
        shape.value_count()
    );
    let pixels = shape.height * shape.width;
    let mut bytes = vec![0u8; pixels * 3];
    for index in 0..pixels {
        bytes[index * 3] = scale_to_u8(values[index]);
        bytes[index * 3 + 1] = scale_to_u8(values[pixels + index]);
        bytes[index * 3 + 2] = scale_to_u8(values[2 * pixels + index]);
    }
    Ok(bytes)
}

fn range_chw_to_packed_rgb24_u8(values: &[f32], shape: &TensorShape) -> Result<Vec<u8>> {
    ensure!(
        shape.channels >= 1,
        "range tensors must provide at least 1 channel"
    );
    ensure!(
        values.len() == shape.value_count(),
        "range tensor has {} values; expected {}",
        values.len(),
        shape.value_count()
    );
    let pixels = shape.height * shape.width;
    let depth = &values[..pixels];
    let intensity = if shape.channels > 1 {
        Some(&values[pixels..(2 * pixels)])
    } else {
        None
    };
    let mut bytes = vec![0u8; pixels * 3];
    for index in 0..pixels {
        bytes[index * 3] = scale_to_u8(depth[index]);
        bytes[index * 3 + 1] = intensity
            .map(|channel| scale_to_u8(channel[index]))
            .unwrap_or(0);
        bytes[index * 3 + 2] = 0;
    }
    Ok(bytes)
}

fn rgb24_to_rgb_chw_f32(bytes: &[u8], height: usize, width: usize) -> Result<Vec<f32>> {
    let pixels = height
        .checked_mul(width)
        .context("RGB dimensions overflow when decoding bytes")?;
    ensure!(
        bytes.len() == pixels * 3,
        "RGB frame has {} bytes; expected {}",
        bytes.len(),
        pixels * 3
    );
    let mut chw = vec![0.0f32; pixels * 3];
    for index in 0..pixels {
        chw[index] = bytes[index * 3] as f32 / 255.0;
        chw[pixels + index] = bytes[index * 3 + 1] as f32 / 255.0;
        chw[2 * pixels + index] = bytes[index * 3 + 2] as f32 / 255.0;
    }
    Ok(chw)
}

fn range_rgb24_to_two_channel_chw_f32(
    bytes: &[u8],
    height: usize,
    width: usize,
) -> Result<Vec<f32>> {
    let pixels = height
        .checked_mul(width)
        .context("range dimensions overflow when decoding bytes")?;
    ensure!(
        bytes.len() == pixels * 3,
        "range frame has {} bytes; expected {}",
        bytes.len(),
        pixels * 3
    );
    let mut chw = vec![0.0f32; pixels * 2];
    for index in 0..pixels {
        chw[index] = bytes[index * 3] as f32 / 255.0;
        chw[pixels + index] = bytes[index * 3 + 1] as f32 / 255.0;
    }
    Ok(chw)
}

fn rgb_metrics(predicted: &[f32], truth: &[f32], shape: &TensorShape) -> Result<ModalityMetrics> {
    ensure!(
        predicted.len() == truth.len(),
        "RGB prediction and truth lengths must match"
    );
    let mut metrics = compute_modality_metrics(predicted, truth, 1.0);
    metrics.ssim = Some(compute_rgb_ssim(
        predicted,
        truth,
        shape.height,
        shape.width,
    ));
    Ok(metrics)
}

fn build_matched_quality(
    ae: &AeOperatingPoints,
    curves: &[CodecRdCurve],
) -> Vec<MatchedQualitySummary> {
    let mut targets = vec![ae.raw_f32.combined_summary.psnr_db];
    if (ae.quantized.combined_summary.psnr_db - ae.raw_f32.combined_summary.psnr_db).abs() > 1e-4 {
        targets.push(ae.quantized.combined_summary.psnr_db);
    }

    targets
        .into_iter()
        .map(|target_psnr| {
            let mut notes = Vec::new();
            let mut codec_points = Vec::new();

            for curve in curves {
                let rd_points: Vec<RdPoint> = curve
                    .rd_points
                    .iter()
                    .map(|point| {
                        RdPoint::new(point.rate_bpp as f64, point.combined_summary.psnr_db as f64)
                    })
                    .collect();
                if rd_points.len() < 2 {
                    notes.push(format!(
                        "{}: insufficient RD points for interpolation",
                        curve.codec
                    ));
                    continue;
                }

                match interpolate_rate_at_psnr(&rd_points, target_psnr as f64) {
                    Ok(interpolated_rate) => {
                        if let Some(nearest) = nearest_rd_point(curve, target_psnr) {
                            codec_points.push(CodecMatchedQualityPoint {
                                codec: curve.codec.clone(),
                                encoding_mode: curve.encoding_mode,
                                quality_control: format!("interp@psnr={target_psnr:.2}"),
                                point: MatchedQualityPoint {
                                    encoding_mode: curve.encoding_mode,
                                    rate_bpp: interpolated_rate as f32,
                                    modality_summary: nearest.modality_summary.clone(),
                                    combined_summary: nearest.combined_summary.clone(),
                                },
                            });
                        }
                    }
                    Err(err) => notes.push(format!("{}: {err}", curve.codec)),
                }
            }

            MatchedQualitySummary {
                quality_metric: "psnr-db".to_string(),
                target_value: target_psnr,
                ae_raw_f32: matched_point_from_ae(&ae.raw_f32),
                ae_quantized: matched_point_from_ae(&ae.quantized),
                codec_points,
                notes,
            }
        })
        .collect()
}

fn matched_point_from_ae(point: &AeOperatingPoint) -> MatchedQualityPoint {
    MatchedQualityPoint {
        encoding_mode: point.encoding_mode,
        rate_bpp: point.rate_bpp,
        modality_summary: point.modality_summary.clone(),
        combined_summary: point.combined_summary.clone(),
    }
}

fn nearest_rd_point<'a>(curve: &'a CodecRdCurve, target_psnr: f32) -> Option<&'a CodecRdPoint> {
    curve.rd_points.iter().min_by(|left, right| {
        let left_delta = (left.combined_summary.psnr_db - target_psnr).abs();
        let right_delta = (right.combined_summary.psnr_db - target_psnr).abs();
        left_delta.total_cmp(&right_delta)
    })
}

fn bytes_to_bpp(bytes: usize, denominator_pixels: f32) -> f32 {
    if denominator_pixels <= 0.0 {
        return 0.0;
    }
    (bytes as f32 * 8.0) / denominator_pixels
}

fn bits_per_pixel_denominator(
    sample_count: usize,
    rgb_pixels: usize,
    range_pixels: usize,
) -> Result<f32> {
    let per_sample = rgb_pixels
        .checked_add(range_pixels)
        .context("pixel count overflow while computing bpp denominator")?;
    let total = per_sample
        .checked_mul(sample_count)
        .context("sample count overflow while computing bpp denominator")?;
    ensure!(total > 0, "bpp denominator cannot be zero");
    Ok(total as f32)
}

fn format_f32_notes(prefix: &str, metrics: &MetricAccumulator) -> Vec<String> {
    let mut notes = Vec::new();
    if !metrics.rgb.is_empty() {
        let agg = aggregate_metrics(&metrics.rgb);
        notes.push(format!(
            "{prefix}.rgb: mse={:.6}, psnr_db={:.4}, ssim={}",
            agg.mean_mse,
            agg.mean_psnr_db,
            agg.mean_ssim
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string())
        ));
    }
    if !metrics.range.is_empty() {
        let agg = aggregate_metrics(&metrics.range);
        notes.push(format!(
            "{prefix}.range: mse={:.6}, psnr_db={:.4}, ssim={}",
            agg.mean_mse,
            agg.mean_psnr_db,
            agg.mean_ssim
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string())
        ));
    }
    notes
}

fn format_depth_notes(prefix: &str, metrics: Option<AggregatedRangeDepthMetrics>) -> Vec<String> {
    let Some(metrics) = metrics else {
        return Vec::new();
    };
    vec![format!(
        "{prefix}: mae_m={}, rmse_m={}, delta<1.25={}, valid_frac={:.6}",
        metrics
            .mean_depth_mae_m
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string()),
        metrics
            .mean_depth_rmse_m
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string()),
        metrics
            .mean_delta_lt_1_25
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string()),
        metrics.mean_valid_pixel_fraction
    )]
}

fn resolve_codec(codec: &str) -> Result<CodecSelection> {
    let normalized = codec.trim().to_ascii_lowercase();
    ensure!(!normalized.is_empty(), "codec names cannot be empty");

    let video_codec = if is_codec_name_or_encoder(&normalized, VideoCodec::H264) {
        VideoCodec::H264
    } else if is_codec_name_or_encoder(&normalized, VideoCodec::H265) {
        VideoCodec::H265
    } else if is_codec_name_or_encoder(&normalized, VideoCodec::Av1) {
        VideoCodec::Av1
    } else if is_codec_name_or_encoder(&normalized, VideoCodec::Png) {
        VideoCodec::Png
    } else if is_codec_name_or_encoder(&normalized, VideoCodec::Ffv1) {
        VideoCodec::Ffv1
    } else {
        bail!("unsupported codec `{codec}`")
    };

    let encoder = if normalized == sfx_bench::codec_name(video_codec) {
        None
    } else {
        Some(normalized.clone())
    };

    Ok(CodecSelection {
        label: codec.to_string(),
        video_codec,
        encoder,
    })
}

fn is_codec_name_or_encoder(name: &str, codec: VideoCodec) -> bool {
    if name == sfx_bench::codec_name(codec) {
        return true;
    }
    sfx_bench::codec_encoder_candidates(codec)
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(name))
}

fn crf_values_for_codec<'a>(config: &'a CompressionBenchConfig, codec: &str) -> Result<&'a [u8]> {
    config
        .crf_sweep
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(codec))
        .map(|(_, values)| values.as_slice())
        .with_context(|| format!("missing crf_sweep entry for codec `{codec}`"))
}

fn apply_overrides(
    config: &mut CompressionBenchConfig,
    args: &CompressBenchArgs,
    paths: &ProjectPaths,
) -> Result<()> {
    if let Some(split) = args.split.as_ref() {
        config.split = split.clone();
    }
    if let Some(out) = args.out.as_ref() {
        config.output_dir = sfx_config::resolve_from_root(&paths.root, out);
    }
    if let Some(sample_cap) = args.sample_cap {
        config.sample_cap = Some(sample_cap);
    }
    if let Some(mode) = args.mode.as_deref() {
        config.mode = parse_compression_mode(mode)?;
    }
    if !args.codecs.is_empty() {
        config.codecs = args.codecs.clone();
    }
    if !args.crf.is_empty() {
        for codec in &config.codecs {
            config.crf_sweep.insert(codec.clone(), args.crf.clone());
        }
    } else {
        for codec in &config.codecs {
            if config
                .crf_sweep
                .keys()
                .any(|entry| entry.eq_ignore_ascii_case(codec))
            {
                continue;
            }
            config.crf_sweep.insert(codec.clone(), vec![18, 24, 30, 36]);
        }
    }
    Ok(())
}

fn parse_compression_mode(value: &str) -> Result<CompressionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "independent" => Ok(CompressionMode::Independent),
        "intra-sequence" | "intrasequence" | "intra" => Ok(CompressionMode::IntraSequence),
        other => bail!("unsupported encoding mode `{other}`"),
    }
}

fn parse_splits(value: &str) -> Result<Vec<Split>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" => Ok(vec![Split::Train, Split::Val, Split::Test]),
        "train" => Ok(vec![Split::Train]),
        "val" | "validation" => Ok(vec![Split::Val]),
        "test" => Ok(vec![Split::Test]),
        other => bail!("unsupported split {other}; expected all, train, val, or test"),
    }
}

fn split_name(split: &Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

fn encoding_mode(mode: CompressionMode) -> EncodingMode {
    match mode {
        CompressionMode::Independent => EncodingMode::Independent,
        CompressionMode::IntraSequence => EncodingMode::IntraSequence,
    }
}

fn resolve_manifest_path(
    paths: &ProjectPaths,
    run_dir: &Path,
) -> Result<(PathBuf, String, Option<PathBuf>)> {
    let default_manifest = paths.processed_sample_manifest_path();
    let summary_path = run_dir.join("summary.json");
    if !summary_path.exists() {
        return Ok((default_manifest, "mixed".to_string(), None));
    }

    let summary_text = fs::read_to_string(&summary_path)
        .with_context(|| format!("reading {}", summary_path.display()))?;
    let summary: TrainingSummary = serde_json::from_str(&summary_text)
        .with_context(|| format!("parsing {}", summary_path.display()))?;
    let split_protocol = if summary.split_protocol.trim().is_empty() {
        "mixed".to_string()
    } else {
        summary.split_protocol
    };
    let manifest_path = if summary.dataset_manifest_path.as_os_str().is_empty() {
        default_manifest
    } else {
        sfx_config::resolve_from_root(&paths.root, summary.dataset_manifest_path)
    };
    let dataset_config_path = (!summary.dataset_config_path.as_os_str().is_empty())
        .then(|| sfx_config::resolve_from_root(&paths.root, summary.dataset_config_path));

    Ok((manifest_path, split_protocol, dataset_config_path))
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

#[derive(Debug, serde::Deserialize)]
struct NormalizationManifest {
    range_normalization: RangeNormalizationDetails,
}

#[derive(Debug, serde::Deserialize)]
struct RangeNormalizationDetails {
    max_range_meters: f32,
    #[serde(default)]
    configured_output_channels: Vec<String>,
}

fn resolve_range_eval_config(
    paths: &ProjectPaths,
    run_dir: &Path,
    processed_dir: &Path,
    dataset_config_path: Option<&Path>,
) -> Result<RangeEvalConfig> {
    let mut max_range_meters = sfx_preprocess::MAX_RANGE_METERS;
    let mut validity_channel_index = None;
    let normalization_path = processed_dir.join("normalization.json");

    if normalization_path.exists() {
        let raw = fs::read_to_string(&normalization_path)
            .with_context(|| format!("reading {}", normalization_path.display()))?;
        let normalization: NormalizationManifest = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", normalization_path.display()))?;
        if normalization.range_normalization.max_range_meters > 0.0 {
            max_range_meters = normalization.range_normalization.max_range_meters;
        }
        validity_channel_index = validity_channel_index.or_else(|| {
            validity_channel_index_from_names(
                &normalization.range_normalization.configured_output_channels,
            )
        });
    }

    let config_candidates = [
        dataset_config_path.map(Path::to_path_buf),
        Some(run_dir.join("dataset.toml")),
    ];
    for candidate in config_candidates.into_iter().flatten() {
        if !candidate.exists() {
            continue;
        }
        let dataset_config = sfx_config::load_dataset_config(&paths.root, &candidate)
            .with_context(|| format!("loading dataset config {}", candidate.display()))?;
        validity_channel_index = validity_channel_index
            .or_else(|| validity_channel_index_from_names(&dataset_config.range_channels));
        break;
    }

    Ok(RangeEvalConfig {
        max_range_meters,
        validity_channel_index,
    })
}

fn validity_channel_index_from_names(channels: &[String]) -> Option<usize> {
    channels.iter().position(|channel| {
        matches!(
            channel.trim().to_ascii_lowercase().as_str(),
            "valid" | "validity" | "validity-mask" | "mask"
        )
    })
}

fn scale_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn print_summary_table(result: &CompressionBenchmarkResult) {
    println!("compression split `{}`", result.split);
    println!("| method | mode | quality | rate (bpp) | combined psnr (dB) | combined mse |");
    println!("| --- | --- | --- | ---: | ---: | ---: |");
    println!(
        "| ae/raw-f32 | {} | - | {:.6} | {:.6} | {:.6} |",
        result.ae_operating_points.raw_f32.encoding_mode.as_str(),
        result.ae_operating_points.raw_f32.rate_bpp,
        result.ae_operating_points.raw_f32.combined_summary.psnr_db,
        result.ae_operating_points.raw_f32.combined_summary.mse
    );
    println!(
        "| ae/quantized | {} | - | {:.6} | {:.6} | {:.6} |",
        result.ae_operating_points.quantized.encoding_mode.as_str(),
        result.ae_operating_points.quantized.rate_bpp,
        result
            .ae_operating_points
            .quantized
            .combined_summary
            .psnr_db,
        result.ae_operating_points.quantized.combined_summary.mse
    );
    for curve in &result.codec_rd_curves {
        for point in &curve.rd_points {
            println!(
                "| codec/{} | {} | {} | {:.6} | {:.6} | {:.6} |",
                curve.codec,
                curve.encoding_mode.as_str(),
                point.quality_control,
                point.rate_bpp,
                point.combined_summary.psnr_db,
                point.combined_summary.mse
            );
        }
    }
}

fn run_id_from_run_dir(run_dir: &Path) -> String {
    run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| run_dir.display().to_string())
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn sanitize_label(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn push_note_once(notes: &mut Vec<String>, note: String) {
    if notes.iter().any(|existing| existing == &note) {
        return;
    }
    notes.push(note);
}

struct CleanupDir {
    path: PathBuf,
}

impl CleanupDir {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for CleanupDir {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
