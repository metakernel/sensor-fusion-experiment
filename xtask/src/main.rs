mod cmds;
mod auth;
use anyhow::Result;
use cmds::handle_cli;

#[tokio::main]
async fn main() -> Result<(),anyhow::Error> {

    // Loading env
    dotenvy::dotenv()?;

    // Parsing CLI arguments
    handle_cli()?;
    
    return Ok(());
}
