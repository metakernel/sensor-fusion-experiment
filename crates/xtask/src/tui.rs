use crate::{ProjectPaths, TuiArgs};
use anyhow::Result;

pub(crate) fn run(args: TuiArgs, _paths: &ProjectPaths) -> Result<()> {
    sfx_tui::run_tui(args.dataset)
}
