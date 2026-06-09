mod baselines;
mod config;
mod probe;
mod result;

pub use baselines::{
    CLASS_PRIOR_BASELINE_NAME, COUNT_MEAN_BASELINE_NAME, CountBaselineKind,
    LOG1P_MEAN_BASELINE_NAME, PCA_ON_RAW_BASELINE_NAME, RANDOM_PROJECTION_BASELINE_NAME,
    RAW_LINEAR_VAL_REGULARIZED_BASELINE_NAME, RawLinearValRegularizedResult,
    RegularizationCandidateScore, run_class_prior_baseline, run_count_mean_baseline,
    run_pca_on_raw_baseline, run_random_projection_baseline,
    run_raw_linear_val_regularized_baseline,
};
pub use config::{
    BenchConfig, BenchMetadata, ConfigError, ProbeConfig, ProtocolConfig, ReconConfig,
    RegressionLossKind, load_bench_config, resolve_from_root,
};
pub use probe::{
    BinaryClassMetrics, ClassificationMetrics, ClassificationProbeResult, FeatureStandardizer,
    MetricValue, ProbeDataset, ProbeError, ProbeRunResult, RegressionMetrics,
    RegressionProbeResult, RegressionTargetMetrics, classification_metrics_from_scores,
    regression_metrics_from_log_predictions, run_linear_probe,
};
pub use result::{
    BaselineResult, BenchResult, LeakageReport, MetricCI, ProtocolResult, ReconResult, TaskResult,
};
