mod baselines;
mod compression;
mod compression_result;
mod config;
mod ffmpeg;
mod latent_accounting;
mod probe;
mod rd_interpolation;
mod result;

pub use baselines::{
    CLASS_PRIOR_BASELINE_NAME, COUNT_MEAN_BASELINE_NAME, CountBaselineKind,
    LOG1P_MEAN_BASELINE_NAME, PCA_ON_RAW_BASELINE_NAME, RANDOM_PROJECTION_BASELINE_NAME,
    RAW_LINEAR_VAL_REGULARIZED_BASELINE_NAME, RawLinearValRegularizedResult,
    RegularizationCandidateScore, run_class_prior_baseline, run_count_mean_baseline,
    run_pca_on_raw_baseline, run_random_projection_baseline,
    run_raw_linear_val_regularized_baseline,
};
pub use compression::{
    CompressionBenchConfig, CompressionConfigError, CompressionMode, QuantizationConfig,
    QuantizationStrategy, load_compression_bench_config,
};
pub use compression_result::{
    AeOperatingPoint, AeOperatingPoints, CodecMatchedQualityPoint, CodecRdCurve, CodecRdPoint,
    CombinedSummary, CompressionBenchmarkResult, CompressionMetadata, EncodingMode, LosslessAnchor,
    MatchedQualityPoint, MatchedQualitySummary, ModalitySummary, render_summary_tables_markdown,
};
pub use config::{
    BenchConfig, BenchMetadata, ConfigError, ProbeConfig, ProtocolConfig, ReconConfig,
    RegressionLossKind, load_bench_config, resolve_from_root,
};
pub use ffmpeg::{
    CodecProbeOutcome, CodecProbeReport, CodecProbeRequest, CodecSkipReason, DecodeExecution,
    DecodeRequest, EncodeExecution, EncodeRequest, EncoderInventory, FfmpegError, FfmpegExecutor,
    PixFmtFamily, PixFmtHooks, Result as FfmpegResult, SystemFfmpegRunner, VideoCodec,
    build_decode_args, build_encode_args, build_probe_encoders_args, codec_encoder_candidates,
    codec_name, default_pix_fmt_family_for_codec, default_pix_fmt_for_family, probe_codec_support,
};
pub use latent_accounting::{
    Int8QuantizationMetadata, LatentAccountingError, QuantizationMinMaxPolicy,
    compute_int8_quantization_metadata, deflate_compressed_size,
    deflate_compressed_size_for_serialized_quantized_payload, dequantize_latent_from_int8,
    quantize_latent_to_int8, raw_f32_latent_byte_size, serialize_quantized_latent_payload,
};
pub use probe::{
    BinaryClassMetrics, ClassificationMetrics, ClassificationProbeResult, FeatureStandardizer,
    MetricValue, ProbeDataset, ProbeError, ProbeRunResult, RegressionMetrics,
    RegressionProbeResult, RegressionTargetMetrics, classification_metrics_from_scores,
    regression_metrics_from_log_predictions, run_linear_probe,
};
pub use rd_interpolation::{
    RdInterpolationError, RdPoint, interpolate_log_rate_at_psnr, interpolate_rate_at_psnr,
    monotonic_rd_points,
};
pub use result::{
    BaselineResult, BenchResult, LeakageReport, MetricCI, ProtocolResult, ReconResult, TaskResult,
};
