mod wod_fetch;

use std::path::Path;

use clap::{Parser, Subcommand};
use anyhow::Result;

use crate::wod_fetch::WodFetchConfig;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand,Debug)]
enum Commands {
    /// Fetches the dataset
    FetchWD {
        /// The path to save the dataset
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
}

fn handle_cli() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::FetchWD { output } => {
            println!("Fetching the Waymo Open Dataset to {:?}", output);
            let wod_config = WodFetchConfig::from_env();
            //wod_fetch::fetch_wod(&wod_config, &output)?;
        }
    }

    Ok(())
}

fn main() -> Result<(),anyhow::Error> {

    // Loading env
    dotenvy::dotenv()?;

    // Parsing CLI arguments
    handle_cli()?;
    
    return Ok(());
}
