use std::path::{Path, PathBuf};

/// Resolves a path against the repository root while preserving absolute paths.
pub fn resolve_repo_relative_path(repo_root: impl AsRef<Path>, value: impl AsRef<Path>) -> PathBuf {
    let value = value.as_ref();
    if value.is_absolute() {
        value.to_path_buf()
    } else {
        repo_root.as_ref().join(value)
    }
}

pub(crate) fn resolve_if_present(repo_root: &Path, value: &Path) -> PathBuf {
    if value.as_os_str().is_empty() {
        PathBuf::new()
    } else {
        resolve_repo_relative_path(repo_root, value)
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_repo_relative_path;
    use std::path::{Path, PathBuf};

    fn sample_repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("repo-root")
    }

    #[test]
    fn resolve_repo_relative_path_joins_relative_paths() {
        let repo_root = sample_repo_root();

        let resolved = resolve_repo_relative_path(&repo_root, Path::new("configs/project.toml"));

        assert_eq!(resolved, repo_root.join("configs/project.toml"));
    }

    #[test]
    fn resolve_repo_relative_path_preserves_absolute_paths() {
        let repo_root = sample_repo_root();
        let absolute = repo_root.join("artifacts/checkpoints/absolute.ckpt");

        let resolved = resolve_repo_relative_path(&repo_root, &absolute);

        assert_eq!(resolved, absolute);
    }
}
