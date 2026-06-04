use crate::{CompareArgs, ProjectPaths, display_from_root};
use anyhow::{Context, Result};
use sfx_train::TrainingSummary;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn run(args: CompareArgs, paths: &ProjectPaths) -> Result<()> {
    if args.runs.is_empty() {
        println!("warn no runs provided; pass --runs <run_dir_a,run_dir_b>");
        return Ok(());
    }

    let summaries: Vec<_> = args
        .runs
        .iter()
        .map(|run| load_training_summary(paths, run))
        .collect::<Result<_>>()?;

    let mut table = String::new();
    table.push_str("| run_name | model | epochs | final_train_loss | final_val_loss | backend |\n");
    table.push_str("| --- | --- | ---: | ---: | ---: | --- |\n");
    for summary in &summaries {
        table.push_str(&format!(
            "| {} | {} | {} | {:.6} | {} | {} |\n",
            escape_md(&summary.run_name),
            escape_md(&summary.model_kind),
            summary.epochs,
            summary.final_train_loss,
            summary
                .final_val_loss
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string()),
            escape_md(&summary.backend)
        ));
    }

    print!("{table}");

    if let Some(out) = args.out {
        let out_path = sfx_config::resolve_from_root(&paths.root, out);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&out_path, &table).with_context(|| format!("writing {}", out_path.display()))?;
        println!("ok   {}", display_from_root(&paths.root, &out_path));
    }

    Ok(())
}

fn load_training_summary(paths: &ProjectPaths, run: &Path) -> Result<TrainingSummary> {
    let run_path = sfx_config::resolve_from_root(&paths.root, run);
    let summary_path = summary_path_for_run(&run_path);
    let file = fs::File::open(&summary_path)
        .with_context(|| format!("opening {}", summary_path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parsing {}", summary_path.display()))
}

fn summary_path_for_run(run_path: &Path) -> PathBuf {
    if run_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("summary.json"))
    {
        run_path.to_path_buf()
    } else {
        run_path.join("summary.json")
    }
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|")
}
