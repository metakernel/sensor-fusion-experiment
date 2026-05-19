pub mod fetch;
use anyhow::Result;

use std::path::Path;
use crate::cmds::fetch::{fetch_wod, FetchConfig};

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand,Debug)]
enum Commands {
    // Fetches the dataset
    #[command(name = "fetch-waymo")]
    Fetch {
        /// The path to save the dataset
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
}

pub fn handle_cli() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch { output } => {
            println!("Fetching the Waymo Open Dataset to {:?}", output);
            let wod_config = FetchConfig::from_env();
            //wod_fetch::fetch_wod(&wod_config, &output)?;
        }
    }

    Ok(())
}