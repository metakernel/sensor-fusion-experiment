use std::path::{Path, PathBuf};

use sfx_config::{
    ConfigError, ModelKind, discover_workspace_root, load_dataset_config, load_evaluation_config,
    load_training_config,
};

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("sfx-config-it-{name}-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    root
}

fn write_configs(root: &Path, train_ratio: &str, val_ratio: &str, test_ratio: &str) {
    let dir = root.join("configs");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("dataset.toml"),
        format!(
            "[dataset]
name = \"waymo-mini\"
raw_dir = \"data/raw/waymo\"
processed_dir = \"data/processed/waymo-mini\"
rgb_size = [128, 256]
range_size = [64, 256]
range_channels = [\"range\", \"intensity\"]
train_ratio = {train_ratio}
val_ratio = {val_ratio}
test_ratio = {test_ratio}
"
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("model.toml"),
        "[model]
kind = \"fusion\"
latent_dim = 8
z_modality = 16
",
    )
    .unwrap();
    std::fs::write(
        dir.join("train.toml"),
        "[train]
run_name = \"integration\"
batch_size = 2
max_batches_per_epoch = 1
learning_rate = 0.001
epochs = 1
seed = 0

[dataset]
config = \"configs/dataset.toml\"

[model]
config = \"configs/model.toml\"
",
    )
    .unwrap();
}

#[test]
fn training_config_loads_and_resolves_paths() {
    let root = temp_root("training_config_loads_and_resolves_paths");
    write_configs(&root, "0.34", "0.33", "0.33");

    let cfg = load_training_config(&root, "configs/train.toml").expect("training config should load");

    assert_eq!(cfg.run_name, "integration");
    assert_eq!(cfg.batch_size, 2);
    assert_eq!(cfg.model.kind, ModelKind::Fusion);
    assert_eq!(cfg.dataset.raw_dir, root.join("data/raw/waymo"));
    assert_eq!(
        cfg.dataset.processed_dir,
        root.join("data/processed/waymo-mini")
    );
    assert_eq!(cfg.dataset_config_path, root.join("configs/dataset.toml"));
    assert_eq!(cfg.model_config_path, root.join("configs/model.toml"));
}

#[test]
fn dataset_validation_reports_invalid_split_sum() {
    let root = temp_root("dataset_validation_reports_invalid_split_sum");
    write_configs(&root, "0.80", "0.20", "0.20");

    let err = load_dataset_config(&root, "configs/dataset.toml").expect_err("dataset must be invalid");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("split ratios"), "unexpected message: {message}");
        }
        other => panic!("expected invalid config error, got {other:?}"),
    }
}

#[test]
fn evaluation_validation_rejects_unknown_split() {
    let root = temp_root("evaluation_validation_rejects_unknown_split");
    let dir = root.join("configs");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("eval.toml"),
        "[eval]
split = \"dev\"
batch_size = 4
preview_count = 8
",
    )
    .unwrap();

    let err = load_evaluation_config(&root, "configs/eval.toml").expect_err("split must be rejected");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(
                message.contains("split must be train, val, or test"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected invalid config error, got {other:?}"),
    }
}

#[test]
fn discover_workspace_root_finds_workspace_from_nested_path() {
    let root = temp_root("discover_workspace_root_finds_workspace_from_nested_path");
    let nested = root.join("a").join("b").join("c");
    std::fs::create_dir_all(&nested).unwrap();

    let discovered = discover_workspace_root(&nested).expect("workspace root should be discovered");
    assert_eq!(discovered, root);
}
