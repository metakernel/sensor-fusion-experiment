use std::path::{Path, PathBuf};

use sfx_config::{
    ConfigError, DatasetConfig, ModelConfig, ProjectConfig, TrainingConfig, ValidationIssue,
    load_project_config,
};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn sample_repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("repo-root")
}

fn validation_issues(error: ConfigError) -> Vec<ValidationIssue> {
    match error {
        ConfigError::Validation(errors) => errors.into_issues(),
        other => panic!("expected validation error, got {other:?}"),
    }
}

fn assert_issue(issues: &[ValidationIssue], field: &str, message: &str) {
    assert!(
        issues
            .iter()
            .any(|issue| issue.field == field && issue.message == message),
        "missing validation issue `{field}: {message}`; got {issues:#?}"
    );
}

#[test]
fn load_project_config_parses_representative_toml_file() {
    let config_path = fixture_path("project-valid.toml");
    let repo_root = sample_repo_root();

    let config = load_project_config(&config_path, &repo_root)
        .expect("expected representative project config to load");

    assert_eq!(config.dataset.name, "waymo-mini");
    assert_eq!(config.model.name, "fusion-net");
    assert_eq!(config.model.model_type, "late-fusion");
    assert_eq!(config.training.trainer.batch_size, 4);
    assert_eq!(config.training.run.backend, "cpu");
    assert_eq!(config.evaluation.metrics.iou_thresholds, vec![0.5, 0.7]);
    assert_eq!(config.google_cloud.project_id, "demo-project");
}

#[test]
fn load_project_config_resolves_relative_paths_against_repo_root() {
    let config_path = fixture_path("project-valid.toml");
    let repo_root = sample_repo_root();

    let config =
        load_project_config(&config_path, &repo_root).expect("expected project config to load");

    assert_eq!(config.dataset.root, repo_root.join("data/raw/waymo-mini"));
    assert_eq!(config.waymo.records_dir, repo_root.join("data/raw/waymo"));
    assert_eq!(
        config.training.outputs.dir,
        repo_root.join("artifacts/checkpoints/run-01")
    );
    assert_eq!(
        config.evaluation.inputs.dataset_config,
        repo_root.join("configs/dataset.toml")
    );
    assert_eq!(
        config.evaluation.inputs.checkpoint_path,
        repo_root.join("artifacts/checkpoints/run-01/last.ckpt")
    );
    assert_eq!(
        config.google_cloud.credentials_path,
        repo_root.join(".xtask/gcloud/credentials.json")
    );
}

#[test]
fn project_config_resolve_paths_preserves_absolute_paths() {
    let mut config = ProjectConfig::from_toml_str(include_str!("fixtures/project-valid.toml"))
        .expect("expected fixture TOML to parse");
    let repo_root = sample_repo_root();
    let alternate_repo_root = repo_root.join("alternate-root");

    let absolute_dataset_root = repo_root.join("absolute-inputs/dataset");
    let absolute_checkpoint = repo_root.join("absolute-inputs/checkpoint.ckpt");
    let absolute_history = repo_root.join("absolute-inputs/history.log");

    config.dataset.root = absolute_dataset_root.clone();
    config.evaluation.inputs.checkpoint_path = absolute_checkpoint.clone();
    config.tui.history_path = absolute_history.clone();

    config.resolve_paths(&alternate_repo_root);

    assert_eq!(config.dataset.root, absolute_dataset_root);
    assert_eq!(
        config.evaluation.inputs.checkpoint_path,
        absolute_checkpoint
    );
    assert_eq!(config.tui.history_path, absolute_history);
    assert_eq!(
        config.training.outputs.dir,
        alternate_repo_root.join("artifacts/checkpoints/run-01")
    );
}

#[test]
fn dataset_validation_reports_specific_ratio_and_size_errors() {
    let raw = r#"
name = "   "
root = ""
train_split = "train"
val_split = "train"
test_split = "test"
split_seed = 42

[split_ratios]
train = 1.2
val = -0.1
test = 0.0

[modalities]
range_image = false
rgb = false
point_cloud = false

[sampling]
max_scenes = 0
max_frames_per_scene = 0
shuffle = true

[preprocessing]
resize_width = 0
resize_height = 0
normalize_rgb = true
range_clip_min_m = -1.0
range_clip_max_m = -2.0
"#;

    let issues = validation_issues(
        DatasetConfig::from_toml_str(raw).expect_err("expected invalid dataset config"),
    );

    assert_issue(&issues, "name", "must not be empty");
    assert_issue(&issues, "root", "must not be empty");
    assert_issue(
        &issues,
        "train_split/val_split/test_split",
        "must use distinct split names",
    );
    assert_issue(&issues, "split_ratios.train", "must be > 0.0 and < 1.0");
    assert_issue(&issues, "split_ratios", "must sum to 1.0 (found 1.100000)");
    assert_issue(
        &issues,
        "modalities",
        "at least one modality must be enabled",
    );
    assert_issue(&issues, "sampling.max_scenes", "must be > 0");
    assert_issue(&issues, "preprocessing.resize_width", "must be > 0");
}

#[test]
fn model_validation_reports_specific_required_and_shape_errors() {
    let raw = r#"
name = ""
type = ""

[input]
use_range_image = true
use_rgb = true
range_channels = 0
image_channels = 0

[backbone]
channels = [64, 0]
blocks = [2]
dropout = 1.5

[head]
num_classes = 0
hidden_dim = 0
loss = ""
"#;

    let issues =
        validation_issues(ModelConfig::from_toml_str(raw).expect_err("expected invalid model"));

    assert_issue(&issues, "name", "must not be empty");
    assert_issue(&issues, "type", "must not be empty");
    assert_issue(&issues, "input.range_channels", "must be > 0");
    assert_issue(
        &issues,
        "backbone.channels/blocks",
        "must have matching lengths",
    );
    assert_issue(&issues, "backbone.channels[1]", "must be > 0");
    assert_issue(&issues, "backbone.dropout", "must be between 0.0 and 1.0");
    assert_issue(
        &issues,
        "fusion",
        "must be set when both range and rgb inputs are enabled",
    );
    assert_issue(&issues, "head.num_classes", "must be > 0");
}

#[test]
fn training_validation_reports_specific_numeric_and_required_errors() {
    let raw = r#"
[run]
name = ""
seed = 1
device = ""
backend = "vulkan"
precision = ""

[trainer]
max_epochs = 0
batch_size = 0
num_workers = 0
gradient_accumulation_steps = 0
gradient_clip_norm = -1.0
log_every_n_steps = 0
limit_train_batches = 0.0
limit_val_batches = -1.0
checkpoint_every_n_epochs = 0

[optimizer]
name = ""
lr = 0.0
weight_decay = -0.1
betas = [1.2, -0.1]

[outputs]
dir = ""
save_last = true
save_best = true
best_monitor = "   "
"#;

    let issues = validation_issues(
        TrainingConfig::from_toml_str(raw).expect_err("expected invalid training config"),
    );

    assert_issue(&issues, "run.name", "must not be empty");
    assert_issue(&issues, "run.device", "must not be empty");
    assert_issue(
        &issues,
        "run.backend",
        "unsupported backend `vulkan`; supported values: cpu, cuda, wgpu, gpu",
    );
    assert_issue(&issues, "trainer.max_epochs", "must be > 0");
    assert_issue(
        &issues,
        "trainer.gradient_accumulation_steps",
        "must be > 0",
    );
    assert_issue(
        &issues,
        "trainer.limit_train_batches",
        "must be a finite number > 0.0",
    );
    assert_issue(&issues, "optimizer.lr", "must be a finite number > 0.0");
    assert_issue(&issues, "optimizer.betas[0]", "must be between 0.0 and 1.0");
    assert_issue(
        &issues,
        "outputs.best_monitor",
        "must not be empty when save_best is true",
    );
}
