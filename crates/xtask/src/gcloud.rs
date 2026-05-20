use crate::{GcloudCommand, ProjectPaths, display_from_root};
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output};

const STORAGE_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_only";

#[derive(Serialize)]
struct AuthState {
    schema_version: u32,
    credentials_path: String,
    credential_source: String,
    storage_scope: String,
}

pub(crate) fn run(command: GcloudCommand, paths: &ProjectPaths) -> Result<()> {
    match command {
        GcloudCommand::Auth => auth(paths),
        GcloudCommand::Check => check(paths),
        GcloudCommand::Whoami => whoami(paths),
        GcloudCommand::Logout => logout(paths),
    }
}

fn auth(paths: &ProjectPaths) -> Result<()> {
    ensure_gcloud_dir(paths)?;
    ensure_gcloud_available()?;

    let scope_arg = format!("--scopes={STORAGE_SCOPE}");
    let mut command = local_gcloud_command(paths);
    command
        .args(["auth", "application-default", "login"])
        .arg(scope_arg);

    run_interactive(&mut command, "Google Cloud application-default login")?;

    let source = copy_credentials(paths)?;
    write_auth_state(paths, &source)?;

    println!(
        "ok   credentials {}",
        display_from_root(&paths.root, &paths.gcloud_credentials_path())
    );
    println!("ok   scope {STORAGE_SCOPE}");
    Ok(())
}

fn check(paths: &ProjectPaths) -> Result<()> {
    ensure_gcloud_available()?;
    let credentials = find_credentials(paths)
        .with_context(|| "missing Google Cloud credentials; run `cargo xtask gcloud auth`")?;
    ensure_local_adc_copy(paths, &credentials)?;

    let mut command = gcloud_command_for_credentials(paths, &credentials);
    command.args(["auth", "application-default", "print-access-token"]);
    let token = run_capture(&mut command, "printing application-default access token")?;

    if token.trim().is_empty() {
        anyhow::bail!("Google Cloud returned an empty access token");
    }

    println!("ok   credentials {}", credential_label(paths, &credentials));
    println!("ok   access token available");

    if storage_command_available(paths) {
        println!("ok   gcloud storage command available");
    } else {
        println!("warn gcloud storage command was not found in this Cloud SDK install");
    }

    Ok(())
}

fn whoami(paths: &ProjectPaths) -> Result<()> {
    ensure_gcloud_available()?;

    if let Some(account) = active_account(paths, true)? {
        println!("account: {account}");
    } else if let Some(account) = active_account(paths, false)? {
        println!("account: {account}");
    } else {
        println!("account: unavailable");
        println!(
            "hint: application-default credentials can still be valid; run `cargo xtask gcloud check`"
        );
    }

    if let Some(credentials) = find_credentials(paths) {
        println!("credentials: {}", credential_label(paths, &credentials));
    } else {
        println!("credentials: missing");
    }

    Ok(())
}

fn logout(paths: &ProjectPaths) -> Result<()> {
    ensure_gcloud_dir(paths)?;

    if gcloud_available() {
        let mut command = local_gcloud_command(paths);
        command.args(["auth", "application-default", "revoke", "--quiet"]);
        if let Err(err) = run_interactive(&mut command, "revoking application-default credentials")
        {
            println!("warn {err}");
        }
    } else {
        println!("warn gcloud command not found; removing local files only");
    }

    for path in paths.gcloud_credential_files() {
        remove_file_if_exists(&path)?;
    }

    println!("ok   local Google Cloud credentials removed");
    Ok(())
}

fn ensure_gcloud_dir(paths: &ProjectPaths) -> Result<()> {
    std::fs::create_dir_all(paths.gcloud_dir())
        .with_context(|| format!("creating {}", paths.gcloud_dir().display()))
}

fn ensure_gcloud_available() -> Result<()> {
    let output = ProcessCommand::new("gcloud")
        .arg("--version")
        .output()
        .with_context(|| "running `gcloud --version`; install the Google Cloud CLI and make sure it is on PATH")?;

    if !output.status.success() {
        anyhow::bail!("gcloud is not usable: {}", command_error(&output));
    }

    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("gcloud")
        .to_string();
    println!("ok   {version}");
    Ok(())
}

fn gcloud_available() -> bool {
    ProcessCommand::new("gcloud")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn local_gcloud_command(paths: &ProjectPaths) -> ProcessCommand {
    let mut command = ProcessCommand::new("gcloud");
    command.env("CLOUDSDK_CONFIG", paths.gcloud_dir());
    command
}

fn gcloud_command_for_credentials(paths: &ProjectPaths, credentials: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("gcloud");
    if credentials.starts_with(paths.gcloud_dir()) {
        command.env("CLOUDSDK_CONFIG", paths.gcloud_dir());
    }
    command.env("GOOGLE_APPLICATION_CREDENTIALS", credentials);
    command
}

fn run_interactive(command: &mut ProcessCommand, label: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("running {label}"))?;
    if !status.success() {
        anyhow::bail!("{label} failed with {status}");
    }
    Ok(())
}

fn run_capture(command: &mut ProcessCommand, label: &str) -> Result<String> {
    let output = command
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        anyhow::bail!("{label} failed: {}", command_error(&output));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn copy_credentials(paths: &ProjectPaths) -> Result<PathBuf> {
    let target = paths.gcloud_credentials_path();
    for source in login_credential_candidates(paths) {
        if source.exists() {
            if source != target {
                std::fs::copy(&source, &target).with_context(|| {
                    format!(
                        "copying credentials from {} to {}",
                        source.display(),
                        target.display()
                    )
                })?;
            }
            return Ok(source);
        }
    }

    if target.exists() {
        return Ok(target);
    }

    anyhow::bail!(
        "authentication completed, but no application-default credentials file was found under {}",
        paths.gcloud_dir().display()
    )
}

fn ensure_local_adc_copy(paths: &ProjectPaths, credentials: &Path) -> Result<()> {
    let local_adc = paths.gcloud_adc_path();
    if credentials == paths.gcloud_credentials_path() && credentials != local_adc {
        std::fs::copy(credentials, &local_adc).with_context(|| {
            format!(
                "copying credentials from {} to {}",
                credentials.display(),
                local_adc.display()
            )
        })?;
    }
    Ok(())
}

fn write_auth_state(paths: &ProjectPaths, source: &Path) -> Result<()> {
    let state = AuthState {
        schema_version: 1,
        credentials_path: display_from_root(&paths.root, &paths.gcloud_credentials_path()),
        credential_source: credential_label(paths, source),
        storage_scope: STORAGE_SCOPE.to_string(),
    };
    let text = serde_json::to_string_pretty(&state).context("encoding Google Cloud auth state")?;
    std::fs::write(paths.gcloud_auth_state_path(), format!("{text}\n"))
        .with_context(|| format!("writing {}", paths.gcloud_auth_state_path().display()))
}

fn find_credentials(paths: &ProjectPaths) -> Option<PathBuf> {
    credential_candidates(paths)
        .into_iter()
        .find(|path| path.exists())
}

fn credential_candidates(paths: &ProjectPaths) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, paths.gcloud_credentials_path());
    for path in login_credential_candidates(paths) {
        push_unique(&mut candidates, path);
    }
    candidates
}

fn login_credential_candidates(paths: &ProjectPaths) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, paths.gcloud_adc_path());

    if let Some(path) = std::env::var_os("GOOGLE_APPLICATION_CREDENTIALS") {
        push_unique(&mut candidates, PathBuf::from(path));
    }

    if cfg!(windows) {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            push_unique(
                &mut candidates,
                PathBuf::from(appdata)
                    .join("gcloud")
                    .join("application_default_credentials.json"),
            );
        }
    } else if let Some(home) = std::env::var_os("HOME") {
        push_unique(
            &mut candidates,
            PathBuf::from(home)
                .join(".config")
                .join("gcloud")
                .join("application_default_credentials.json"),
        );
    }

    candidates
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn active_account(paths: &ProjectPaths, local: bool) -> Result<Option<String>> {
    let mut command = if local {
        local_gcloud_command(paths)
    } else {
        ProcessCommand::new("gcloud")
    };
    command.args([
        "auth",
        "list",
        "--filter=status:ACTIVE",
        "--format=value(account)",
    ]);

    let output = command
        .output()
        .with_context(|| "running `gcloud auth list`")?;
    if !output.status.success() {
        return Ok(None);
    }

    let account = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok((!account.is_empty()).then_some(account))
}

fn storage_command_available(paths: &ProjectPaths) -> bool {
    let output = local_gcloud_command(paths)
        .args(["storage", "--help"])
        .output();
    output
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("removing {}", path.display())),
    }
}

fn credential_label(paths: &ProjectPaths, path: &Path) -> String {
    display_from_root(&paths.root, path)
}

fn command_error(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        output.status.to_string()
    } else {
        stdout
    }
}
