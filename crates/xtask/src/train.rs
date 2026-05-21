use crate::{ProjectPaths, TrainArgs, display_from_root};
use anyhow::Result;
use sfx_config::ModelKind;

pub(crate) fn run(args: TrainArgs, paths: &ProjectPaths) -> Result<()> {
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
