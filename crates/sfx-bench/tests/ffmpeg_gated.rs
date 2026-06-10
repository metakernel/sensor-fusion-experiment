use std::process::Command;

use sfx_bench::{FfmpegExecutor, SystemFfmpegRunner};

#[test]
fn probes_system_ffmpeg_encoders_when_available() {
    if !ffmpeg_available() {
        eprintln!("skipping ffmpeg integration test: `ffmpeg` not found");
        return;
    }

    let inventory = SystemFfmpegRunner::default()
        .probe_encoders()
        .expect("probing ffmpeg encoders should succeed when ffmpeg is available");
    assert!(
        inventory.iter().next().is_some(),
        "ffmpeg encoder inventory should not be empty"
    );
}

fn ffmpeg_available() -> bool {
    match Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .output()
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
