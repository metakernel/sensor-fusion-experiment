use crate::{ProjectPaths, TrainArgs, TrainSubcommand, display_from_root};
use anyhow::Result;
use sfx_config::ModelKind;

pub(crate) fn run(args: TrainArgs, paths: &ProjectPaths) -> Result<()> {
    if let Some(command) = args.command {
        return match command {
            TrainSubcommand::Resume(args) => resume(&args.run, paths),
        };
    }

    let config = sfx_config::load_training_config(&paths.root, &args.config)?;
    let summary = match config.model.kind {
        ModelKind::RangeOnly => sfx_train::train_range_autoencoder(&paths.root, &args.config)?,
        ModelKind::RgbOnly => sfx_train::train_rgb_autoencoder(&paths.root, &args.config)?,
        ModelKind::Fusion => sfx_train::train_fusion_autoencoder(&paths.root, &args.config)?,
    };

    println!("ok   run {}", summary.run_name);
    println!(
        "     dir:        {}",
        display_from_root(&paths.root, &summary.run_dir)
    );
    println!("     samples:    {}", summary.train_samples);
    println!("     epochs:     {}", summary.epochs);
    println!("     batch size: {}", summary.batch_size);
    println!("     latent dim: {}", summary.latent_dim);
    println!("     final loss: {:.6}", summary.final_train_loss);
    println!(
        "ok   {}",
        display_from_root(&paths.root, &summary.metrics_path)
    );
    println!(
        "ok   {}",
        display_from_root(&paths.root, &summary.summary_path)
    );
    println!(
        "ok   {}",
        display_from_root(&paths.root, &summary.checkpoint_path)
    );

    Ok(())
}

fn resume(run: &std::path::Path, paths: &ProjectPaths) -> Result<()> {
    let report = sfx_train::inspect_resume_run(&paths.root, run)?;
    println!("ok   run {}", report.summary.run_name);
    println!(
        "     dir:        {}",
        display_from_root(&paths.root, &report.run_dir)
    );
    println!("     model:      {}", report.summary.model_kind);
    println!("     status:     {}", report.summary.status);
    println!(
        "     checkpoint: {}",
        exists_label(report.checkpoint_exists)
    );
    println!("     metrics:    {}", exists_label(report.metrics_exists));
    println!(
        "     optimizer:  {}",
        exists_label(report.optimizer_metadata_exists)
    );
    println!(
        "note exact optimizer-state resume is not available yet; restart cleanly with: cargo xtask train --config {}",
        display_from_root(&paths.root, &report.restart_config_path)
    );
    if !report.can_resume_optimizer_state {
        println!(
            "note model checkpoint exists for evaluation/export, but training restart uses the saved config snapshot"
        );
    }
    Ok(())
}

fn exists_label(exists: bool) -> &'static str {
    if exists { "found" } else { "missing" }
}
