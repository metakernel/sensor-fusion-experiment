use sfx_core::manifest::{
    MANIFEST_SCHEMA_VERSION, MultimodalSampleMeta, ProcessedSampleEntry, ProcessedSampleManifest,
    SampleId, Split, TensorShape, write_manifest,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

const RGB_VALUES: usize = 3 * 128 * 256;
const RANGE_VALUES: usize = 2 * 64 * 256;

#[test]
fn xtask_train_eval_export_compare_report_end_to_end() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let exe = env!("CARGO_BIN_EXE_xtask");
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    stage_workspace(root)?;

    let train_args = ["train", "--config", "configs\\train.test.toml"];
    let train = run(exe, root, &train_args);
    assert_success(&train_args, &train);

    let run_dir = root
        .join("artifacts")
        .join("checkpoints")
        .join("fusion")
        .join("e2e");
    assert!(run_dir.join("model.bin").is_file());
    assert!(run_dir.join("summary.json").is_file());
    assert!(run_dir.join("model.toml").is_file());
    assert!(
        root.join(".xtask")
            .join("runs")
            .join("latest.json")
            .is_file()
    );

    let eval_args = ["eval", "--split", "val"];
    let eval = run(exe, root, &eval_args);
    assert_success(&eval_args, &eval);
    let eval_path = run_dir.join("eval").join("val.json");
    let eval_text = fs::read_to_string(&eval_path)?;
    assert!(
        eval_text.contains("\"rgb\""),
        "{} missing rgb metrics",
        eval_path.display()
    );
    assert!(
        eval_text.contains("\"range\""),
        "{} missing range metrics",
        eval_path.display()
    );

    let export_args = [
        "export",
        "--split",
        "val",
        "--n",
        "1",
        "--out",
        "artifacts\\exports\\e2e",
    ];
    let export = run(exe, root, &export_args);
    assert_success(&export_args, &export);
    let export_dir = root.join("artifacts").join("exports").join("e2e");
    assert!(contains_export_image(&export_dir)?);

    let compare_args = ["compare", "--runs", "artifacts\\checkpoints\\fusion\\e2e"];
    let compare = run(exe, root, &compare_args);
    assert_success(&compare_args, &compare);
    let compare_stdout = String::from_utf8_lossy(&compare.stdout);
    assert!(
        compare_stdout.contains("e2e"),
        "compare stdout: {compare_stdout}"
    );

    let report_args = ["report"];
    let report = run(exe, root, &report_args);
    assert_success(&report_args, &report);
    let report_path = root.join("artifacts").join("reports").join("report.md");
    let report_text = fs::read_to_string(&report_path)?;
    assert!(
        report_text.contains("## Evaluation"),
        "{} missing evaluation section",
        report_path.display()
    );

    eprintln!(
        "xtask e2e smoke test completed in {:.2?}",
        started.elapsed()
    );
    Ok(())
}

fn stage_workspace(root: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(root.join("Cargo.toml"), "[workspace]\n")?;

    let configs = root.join("configs");
    fs::create_dir_all(&configs)?;
    fs::write(
        configs.join("dataset.test.toml"),
        "[dataset]\nname = \"test\"\nraw_dir = 'data\\raw\\test'\nprocessed_dir = 'data\\processed\\test'\nrgb_size = [128, 256]\nrange_size = [64, 256]\nrange_channels = [\"range\", \"intensity\"]\ntrain_ratio = 0.8\nval_ratio = 0.2\ntest_ratio = 0.0\n",
    )?;
    fs::write(
        configs.join("model.fusion.test.toml"),
        "[model]\nkind = \"fusion\"\nlatent_dim = 8\nz_modality = 16\n",
    )?;
    fs::write(
        configs.join("train.test.toml"),
        "[train]\nrun_name = \"e2e\"\nbatch_size = 2\nmax_batches_per_epoch = 1\nlearning_rate = 0.001\nepochs = 1\nseed = 0\n\n[dataset]\nconfig = 'configs\\dataset.test.toml'\n\n[model]\nconfig = 'configs\\model.fusion.test.toml'\n",
    )?;

    fs::create_dir_all(root.join("data").join("raw").join("test"))?;
    let processed = root.join("data").join("processed").join("test");
    let manifest = fixture_manifest(&processed)?;
    write_manifest(
        root.join(".xtask")
            .join("manifests")
            .join("processed_samples.json"),
        &manifest,
    )?;

    Ok(())
}

fn fixture_manifest(processed: &Path) -> Result<ProcessedSampleManifest, Box<dyn Error>> {
    let rgb_shape = TensorShape {
        channels: 3,
        height: 128,
        width: 256,
    };
    let range_shape = TensorShape {
        channels: 2,
        height: 64,
        width: 256,
    };
    let mut samples = Vec::new();

    for index in 0..3 {
        let split = if index < 2 { Split::Train } else { Split::Val };
        let split_dir = match &split {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        };
        let id = SampleId(format!("sample_{index:06}"));
        let sample_rel = PathBuf::from(split_dir).join(&id.0);
        let sample_dir = processed.join(&sample_rel);
        fs::create_dir_all(&sample_dir)?;
        write_f32_zeros(&sample_dir.join("rgb.f32.bin"), RGB_VALUES)?;
        write_f32_zeros(&sample_dir.join("range.f32.bin"), RANGE_VALUES)?;
        fs::write(sample_dir.join("meta.json"), "{}\n")?;

        samples.push(ProcessedSampleEntry {
            meta: MultimodalSampleMeta {
                id,
                split,
                rgb_path: sample_rel.join("rgb.f32.bin"),
                range_path: sample_rel.join("range.f32.bin"),
                timestamp_micros: index as i64,
                source_segment: "e2e".to_string(),
            },
            meta_path: sample_rel.join("meta.json"),
            preview_rgb_path: None,
            preview_range_path: None,
        });
    }

    Ok(ProcessedSampleManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        dataset: Some("test".to_string()),
        rgb_shape: Some(rgb_shape),
        range_shape: Some(range_shape),
        samples,
    })
}

fn write_f32_zeros(path: &Path, values: usize) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(values * std::mem::size_of::<f32>());
    for _ in 0..values {
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn run(exe: &str, root: &Path, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|err| panic!("failed to run xtask {args:?}: {err}"))
}

fn assert_success(args: &[&str], output: &Output) {
    assert!(
        output.status.success(),
        "xtask {args:?} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn contains_export_image(dir: &Path) -> Result<bool, Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.ends_with("_rgb.ppm") || name.ends_with("_range.pgm") {
            return Ok(true);
        }
    }
    Ok(false)
}
