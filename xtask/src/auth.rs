use anyhow::{Context, Result};
use gcloud_auth::credentials::CredentialsFile;
use oauth2::{
    AuthorizationCode, AuthUrl, ClientId, ClientSecret, CsrfToken,
    PkceCodeChallenge, RedirectUrl, Scope, TokenResponse, TokenUrl,
    basic::BasicClient,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};


const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const REDIRECT_PORT: u16 = 8080;
const GCS_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_only";

#[derive(Serialize, Deserialize)]
pub struct SavedCredentials {
    pub r#type: String,
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    pub token_uri: String,
}

impl SavedCredentials {
    pub fn new(client_id: String, client_secret: String, refresh_token: String) -> Self {
        Self {
            r#type: "authorized_user".to_string(),
            client_id,
            client_secret,
            refresh_token,
            token_uri: GOOGLE_TOKEN_URL.to_string(),
        }
    }
    pub fn to_credentials_file(&self) -> CredentialsFile {
        CredentialsFile {
            tp: todo!(),
            client_email: self.client_id.into(),
            private_key_id: todo!(),
            private_key: todo!(),
            auth_uri: todo!(),
            token_uri: self.token_uri.into(),
            project_id: todo!(),
            client_secret: todo!(),
            client_id: todo!(),
            refresh_token: todo!(),
            audience: todo!(),
            subject_token_type: todo!(),
            token_url_external: todo!(),
            token_info_url: todo!(),
            service_account_impersonation_url: todo!(),
            service_account_impersonation: todo!(),
            delegates: todo!(),
            credential_source: todo!(),
            quota_project_id: todo!(),
            workforce_pool_user_project: todo!(),
        }
    }
}

// TODO: Correctly check if credentials are valid (e.g., by trying to refresh the token or checking expiration)
pub async fn is_authenticated(creds_path: &Path) -> Result<()> {
    // check if file exists
    let creds = get_credentials(creds_path)?;

    // try to login with creds - can reuse pars of code in authenticate

    Ok(())
}

// Load credentials from file
pub fn get_credentials(creds_path: &Path) -> Result<SavedCredentials> {
    let creds_data = std::fs::read_to_string(creds_path)
        .with_context(|| format!("Failed to read credentials from {:?}", creds_path))?;
    let creds: SavedCredentials = serde_json::from_str(&creds_data)
        .with_context(|| "Failed to parse credentials JSON")?;
    Ok(creds)
}

pub async fn authenticate(client_id: &str, client_secret: &str, creds_path: &Path) -> Result<(), anyhow::Error> {
    // create OAuth2 client
    let client = BasicClient::new(ClientId::new(client_id.into()))
        .set_client_secret(ClientSecret::new(client_secret.into()))
        .set_auth_uri(AuthUrl::new(GOOGLE_AUTH_URL.into())?)
        .set_token_uri(TokenUrl::new(GOOGLE_TOKEN_URL.into())?)
        .set_redirect_uri(RedirectUrl::new(format!("http://localhost:{REDIRECT_PORT}"))?);

    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

    // generate authorization URL and open browser
    let (auth_url, csrf_token) = client
        .authorize_url(CsrfToken::new_random)
        .add_scope(Scope::new(GCS_SCOPE.into()))
        .add_extra_param("access_type", "offline")
        .add_extra_param("prompt", "consent")
        .set_pkce_challenge(pkce_challenge)
        .url();

    println!("Opening browser for authentication");
    println!("If it didn't open, visit:\n {auth_url}");
    open::that(auth_url.as_str())?;
    
    // loopback server to receive authorization code
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{REDIRECT_PORT}")).await?;
    let (mut socket, _) = listener.accept().await?;
    
    let mut reader = BufReader::new(&mut socket);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await?;

    let path = request_line.split_whitespace().nth(1).context("bad HTTP request")?;
    let url = url::Url::parse(&format!("http://localhost{path}"))?;

    let code  = url.query_pairs().find(|(k,_)| k=="code" ).map(|(_,v)| v.into_owned()).context("missing code")?;
    let state = url.query_pairs().find(|(k,_)| k=="state").map(|(_,v)| v.into_owned()).context("missing state")?;

    anyhow::ensure!(state == *csrf_token.secret(), "CSRF token mismatch");

    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Done! You can close this tab. </h1>").await?;
    drop(socket);

    // exchange code for access token
    let http_client = oauth2::reqwest::ClientBuilder::new()
        .redirect(oauth2::reqwest::redirect::Policy::none())
        .build()?;

    let token = client
        .exchange_code(AuthorizationCode::new(code))
        .set_pkce_verifier(pkce_verifier)
        .request_async(&http_client)
        .await?;
    
    let refresh_token = token.refresh_token()
        .context("No refresh token. Did you set access_type=offline?")?;

    // save credentials to file
    let creds = SavedCredentials {
        r#type:         "authorized_user".into(),
        client_id:      client_id.into(),
        client_secret:  client_secret.into(),
        refresh_token:  refresh_token.secret().clone(),
        token_uri:      GOOGLE_TOKEN_URL.into(),
    };

    if let Some(p) = creds_path.parent() { std::fs::create_dir_all(p)?; }
    std::fs::write(creds_path, serde_json::to_string_pretty(&creds)?)?;
    println!("Credentials savet to {}", creds_path.display());

    Ok(())
}