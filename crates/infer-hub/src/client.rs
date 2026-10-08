use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::download::{download_file, ProgressEvent};
use crate::error::{HubError, Result};

pub const DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GgufFile {
    pub filename: String,
    pub size: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepoSummary {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
}

pub struct HubClient {
    endpoint: String,
    http: reqwest::Client,
    token: Option<String>,
}

impl HubClient {
    pub fn new() -> Self {
        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        Self::with_endpoint(endpoint)
    }

    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(format!("nexllama/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build http client");
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            http,
            token: std::env::var("HF_TOKEN").ok(),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<RepoSummary>> {
        let url = format!("{}/api/models", self.endpoint);
        let limit_s = limit.to_string();
        let resp = self
            .auth(self.http.get(&url).query(&[
                ("search", query),
                ("limit", limit_s.as_str()),
                ("sort", "downloads"),
                ("direction", "-1"),
                ("filter", "gguf"),
            ]))
            .send()
            .await?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(HubError::AuthRequired);
        }
        if !status.is_success() {
            return Err(HubError::Status {
                status: status.as_u16(),
                url,
            });
        }
        let items: Vec<ApiModel> = resp.json().await?;
        Ok(items
            .into_iter()
            .map(|m| RepoSummary {
                id: m.id,
                downloads: m.downloads.unwrap_or(0),
                likes: m.likes.unwrap_or(0),
            })
            .collect())
    }

    pub async fn list_gguf(&self, repo: &str) -> Result<Vec<GgufFile>> {
        let url = format!(
            "{}/api/models/{}/tree/main?recursive=true",
            self.endpoint, repo
        );
        let resp = self.auth(self.http.get(&url)).send().await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(HubError::RepoNotFound(repo.to_string()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(HubError::AuthRequired);
        }
        if !status.is_success() {
            return Err(HubError::Status {
                status: status.as_u16(),
                url,
            });
        }
        let entries: Vec<ApiTreeEntry> = resp.json().await?;
        let mut files: Vec<GgufFile> = entries
            .into_iter()
            .filter(|e| e.entry_type.as_deref() == Some("file"))
            .filter(|e| e.path.to_ascii_lowercase().ends_with(".gguf"))
            .map(|e| GgufFile {
                filename: e.path,
                size: e.size.unwrap_or(0),
            })
            .collect();
        files.sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(files)
    }

    pub async fn download(
        &self,
        repo: &str,
        filename: &str,
        dest_dir: impl AsRef<Path>,
        on_progress: impl FnMut(ProgressEvent),
    ) -> Result<PathBuf> {
        validate_filename(filename)?;
        std::fs::create_dir_all(dest_dir.as_ref())?;
        let dest = dest_dir.as_ref().join(filename);
        let url = format!("{}/{}/resolve/main/{}", self.endpoint, repo, filename);
        download_file(&self.http, &self.token, &url, &dest, on_progress).await?;
        Ok(dest)
    }
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ApiTreeEntry {
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(rename = "type", default)]
    entry_type: Option<String>,
}

pub(crate) fn validate_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains("..")
        || filename.starts_with('.')
    {
        return Err(HubError::InvalidFilename(filename.to_string()));
    }
    Ok(())
}
