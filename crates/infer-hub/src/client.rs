use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::download::{download_file, ProgressEvent};
use crate::error::{HubError, Result};

pub const DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

/// The authors whose GGUFs are worth reaching for first.
///
/// Quantising well is a craft — a bad quant costs accuracy that no amount of
/// context or sampling gets back — and three names cover nearly everything
/// worth running locally:
///
/// * **bartowski** — the broadest catalogue, one model at many sizes;
/// * **Unsloth** — dynamic quants (`UD` series) and the releases that carry
///   the extra heads MTP drafting needs;
/// * **z-lab** — the DFlash draft models, which is how a dense model with no
///   MTP head gets speculative decoding at all.
///
/// Nothing is restricted to these: they are what the download page offers
/// first, and searching still covers the whole Hub.
pub const TRUSTED_AUTHORS: [&str; 3] = ["bartowski", "Unsloth", "z-lab"];

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

    /// Search within one author's repositories.
    ///
    /// Searching the Hub for "qwen" returns hundreds of repos of wildly uneven
    /// quality, most of them someone's first attempt at a conversion. Naming the
    /// author is the difference between that and a list a person can pick from.
    pub async fn search_by_author(
        &self,
        author: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RepoSummary>> {
        let url = format!("{}/api/models", self.endpoint);
        let limit_s = limit.to_string();
        let resp = self
            .auth(self.http.get(&url).query(&[
                ("search", query),
                ("author", author),
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
        // The file may sit in a subfolder of the destination, which does not
        // exist yet — `create_dir_all` above only made the destination itself.
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // A `#` or `?` in a name would otherwise be read as a fragment or query,
        // and the download would come back truncated — or 404, with no hint why.
        // Encoded one segment at a time so the separators stay separators.
        let encode_path = |s: &str| {
            s.split('/').map(url_encode).collect::<Vec<_>>().join("/")
        };
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.endpoint,
            encode_path(repo),
            encode_path(filename)
        );
        download_file(&self.http, &self.token, &url, &dest, on_progress).await?;
        Ok(dest)
    }
}

/// Percent-encode everything that is not an unreserved URL character.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
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

/// A repo-relative path may carry directories, because repos do: the tree
/// listing is recursive and plenty of GGUF lives in a subfolder. What must
/// never get through is anything that could escape the destination — absolute
/// paths, drive letters, and `..` at any depth — so every segment is held to
/// the same rule a bare filename always was.
pub(crate) fn validate_filename(filename: &str) -> Result<()> {
    if filename.is_empty() || filename.starts_with('/') || filename.contains('\\') {
        return Err(HubError::InvalidFilename(filename.to_string()));
    }
    let safe = filename.split('/').all(|seg| {
        !seg.is_empty()
            && seg != "."
            && seg != ".."
            && !seg.contains("..")
            && !seg.starts_with('.')
            && !seg.contains(':')
    });
    if !safe {
        return Err(HubError::InvalidFilename(filename.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod filename_tests {
    use super::*;

    #[test]
    fn subfolders_are_allowed_because_repos_have_them() {
        assert!(validate_filename("subdir/model-q4.gguf").is_ok());
        assert!(validate_filename("a/b/c/model.gguf").is_ok());
    }

    #[test]
    fn nothing_may_climb_out_of_the_destination() {
        assert!(validate_filename("../model.gguf").is_err());
        assert!(validate_filename("sub/../../model.gguf").is_err());
        assert!(validate_filename("sub/..model.gguf").is_err());
        // Absolute and drive-letter forms, including the Windows spellings.
        assert!(validate_filename("/model.gguf").is_err());
        assert!(validate_filename("C:/model.gguf").is_err());
        assert!(validate_filename("sub\\model.gguf").is_err());
    }

    #[test]
    fn the_old_bare_filename_rules_still_hold() {
        assert!(validate_filename("model.gguf").is_ok());
        assert!(validate_filename("").is_err());
        assert!(validate_filename(".hidden.gguf").is_err());
        assert!(validate_filename("sub/.hidden.gguf").is_err());
    }
}
