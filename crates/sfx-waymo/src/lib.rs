use sfx_core::manifest::{RawFileEntry, SourceSplit};

pub const CRATE_NAME: &str = "sfx-waymo";
pub const DEFAULT_BUCKET: &str = "waymo_open_dataset_v_2_0_1";

const DEFAULT_TRAINING_PREFIX: &str = "training";
const DEFAULT_VALIDATION_PREFIX: &str = "validation";
const DEFAULT_TESTING_PREFIX: &str = "testing";

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryConfig {
    pub bucket: String,
    pub prefix: Option<String>,
    pub training_prefix: String,
    pub validation_prefix: String,
    pub testing_prefix: String,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            bucket: DEFAULT_BUCKET.to_string(),
            prefix: None,
            training_prefix: DEFAULT_TRAINING_PREFIX.to_string(),
            validation_prefix: DEFAULT_VALIDATION_PREFIX.to_string(),
            testing_prefix: DEFAULT_TESTING_PREFIX.to_string(),
        }
    }
}

impl DiscoveryConfig {
    pub fn split_prefix(&self, split: &SourceSplit) -> &str {
        match split {
            SourceSplit::Training => &self.training_prefix,
            SourceSplit::Validation => &self.validation_prefix,
            SourceSplit::Testing => &self.testing_prefix,
        }
    }

    pub fn split_uri(&self, split: &SourceSplit) -> String {
        let object_prefix = join_prefix(self.prefix.as_deref(), self.split_prefix(split));
        if object_prefix.is_empty() {
            format!("gs://{}", self.bucket)
        } else {
            format!("gs://{}/{}", self.bucket, object_prefix)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedObject {
    pub uri: String,
    pub file_name: String,
    pub size_bytes: Option<u64>,
}

pub fn parse_gcloud_storage_listing(
    output: &str,
    split: SourceSplit,
    limit: Option<usize>,
) -> Vec<RawFileEntry> {
    let mut files = Vec::new();

    for line in output.lines() {
        if let Some(object) = parse_listing_line(line) {
            files.push(RawFileEntry {
                uri: object.uri,
                split: split.clone(),
                file_name: object.file_name,
                size_bytes: object.size_bytes,
                checksum: None,
            });
        }

        if limit.is_some_and(|limit| files.len() >= limit) {
            break;
        }
    }

    files
}

pub fn split_label(split: &SourceSplit) -> &'static str {
    match split {
        SourceSplit::Training => "training",
        SourceSplit::Validation => "validation",
        SourceSplit::Testing => "testing",
    }
}

fn parse_listing_line(line: &str) -> Option<ListedObject> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("TOTAL:") {
        return None;
    }

    let uri_start = trimmed.find("gs://")?;
    let pre_uri = trimmed[..uri_start].trim();
    let uri = trimmed[uri_start..].split_whitespace().next()?;

    if uri.ends_with('/') || uri.ends_with(':') {
        return None;
    }

    let file_name = uri.rsplit('/').next()?.to_string();
    if file_name.is_empty() {
        return None;
    }

    Some(ListedObject {
        uri: uri.to_string(),
        file_name,
        size_bytes: pre_uri
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok()),
    })
}

fn join_prefix(base: Option<&str>, split: &str) -> String {
    [base.unwrap_or(""), split]
        .into_iter()
        .map(|part| part.trim_matches('/'))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_split_uri_with_prefix() {
        let config = DiscoveryConfig {
            prefix: Some("domain_adaptation".to_string()),
            ..DiscoveryConfig::default()
        };

        assert_eq!(
            config.split_uri(&SourceSplit::Training),
            "gs://waymo_open_dataset_v_2_0_1/domain_adaptation/training"
        );
    }

    #[test]
    fn parses_long_listing_output() {
        let output = "\
       42  2024-01-01T00:00:00Z  gs://bucket/training/a.tfrecord\n\
gs://bucket/training/folder/\n\
      100  2024-01-01T00:00:01Z  gs://bucket/training/b.tfrecord\n\
TOTAL: 2 objects, 142 bytes\n";

        let files = parse_gcloud_storage_listing(output, SourceSplit::Training, None);

        assert_eq!(files.len(), 2);
        assert_eq!(files[0].file_name, "a.tfrecord");
        assert_eq!(files[0].size_bytes, Some(42));
        assert_eq!(files[1].uri, "gs://bucket/training/b.tfrecord");
    }

    #[test]
    fn applies_listing_limit() {
        let output = "\
gs://bucket/training/a.tfrecord\n\
gs://bucket/training/b.tfrecord\n";

        let files = parse_gcloud_storage_listing(output, SourceSplit::Training, Some(1));

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name, "a.tfrecord");
    }
}
