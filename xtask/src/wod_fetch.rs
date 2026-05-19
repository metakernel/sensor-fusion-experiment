use futures_util::TryStreamExt;
use gcloud_auth::credentials::CredentialsFile;
/// The Waymo Open Dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.

use gcloud_storage::{self, client::ClientConfig, http::objects::download::Range};


// This module contains code to fetch the Waymo Open Dataset from Google Cloud Storage.
pub struct WodFetchConfig {
    pub bucket: String,
    pub prefix: Option<String>,
    pub credentials: Option<CredentialsFile>,
}

impl WodFetchConfig {
    pub fn new(bucket: String, prefix: Option<String>, credentials: Option<CredentialsFile>) -> Self {
        Self { bucket, prefix, credentials }
    }
    pub fn from_env() -> Self {
        Self {
            bucket: dotenvy::var("WOD_BUCKET").expect("WOD_BUCKET must be set"),
            prefix: dotenvy::var("WOD_PREFIX").ok(),
            credentials: None, // For now, we will use the default credentials from the environment
        }
    }
}

pub async fn fetch_wod(config: &WodFetchConfig, output_path: &std::path::Path) -> anyhow::Result<()> {
    
    let cred = CredentialsFile::new().await?;
    let clientConfig = ClientConfig::default().with_credentials(cred);
    let client = gcloud_storage::client::Client::new(clientConfig.await?);

    // List objects in the bucket with the given prefix
    let list_request = gcloud_storage::http::objects::list::ListObjectsRequest {
        bucket: config.bucket.clone(),
        prefix: config.prefix.clone(),
        ..Default::default()
    };
    let objects = client.list_objects(&list_request).await?;

    // Download each object to the output path
    for object in objects.items.unwrap_or_default() {
        println!("Downloading object: {}", object.name);
        download_object(&client, &config.bucket, &object.name, output_path).await?;
    }
    Ok(())
}

pub async fn download_object(client: &gcloud_storage::client::Client, bucket: &str, object_name: &str, output_path: &std::path::Path) -> anyhow::Result<()> {
    let object_request = gcloud_storage::http::objects::get::GetObjectRequest {
        bucket: bucket.to_string(),
        object: object_name.to_string(),
        ..Default::default()
    };
    let stream = client.download_streamed_object(&object_request,&Range::default()).await?;
    let data: Vec<u8> = stream
    .try_fold(Vec::new(), |mut acc, chunk| async move {
        acc.extend_from_slice(&chunk);
        Ok(acc)
    }).await?;
    let output_file_path = output_path.join(object_name);
    if let Some(parent) = output_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tokio::fs::write(output_file_path, data).await?;
    Ok(())
}


