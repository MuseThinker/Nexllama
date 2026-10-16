use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use serde::Deserialize;

use crate::download::{download_file, ProgressEvent};
use crate::error::{HubError, Result};

/// The upstream Hub. Preferred wherever it can be reached: it is the source of
/// truth, and everything else is a courtesy for networks that cannot reach it.
pub const UPSTREAM_ENDPOINT: &str = "https://huggingface.co";

/// A mirror of the Hub, used when the upstream is unreachable from this machine.
pub const MIRROR_ENDPOINT: &str = "https://hf-mirror.com";

/// The endpoint to use when nothing else decides it.
///
/// Kept as a name rather than a value because the right answer depends on the
/// machine — see [`resolve_endpoint`].
pub const DEFAULT_ENDPOINT: &str = UPSTREAM_ENDPOINT;

/// The authors whose GGUFs are worth reaching for first.
///
/// Quantising well is a craft — a bad quant costs accuracy that no amount of
/// context or sampling gets back — and four names cover nearly everything worth
/// running locally:
///
/// * **bartowski** — the broadest catalogue, one model at many sizes;
/// * **Unsloth** — dynamic quants (`UD` series) and the releases that carry
///   the extra heads MTP drafting needs;
/// * **z-lab** — the DFlash draft models, which is how a dense model with no
///   MTP head gets speculative decoding at all;
/// * **ggml-org** — the project's own reference quants for models it ships
///   support for (gpt-oss and friends).
///
/// Nothing is restricted to these: they are what the download page offers
/// first, and searching still covers the whole Hub.
pub const TRUSTED_AUTHORS: [&str; 4] = ["bartowski", "Unsloth", "z-lab", "ggml-org"];

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

/// Which Hub this machine should talk to, decided once per process.
///
/// `HF_ENDPOINT` wins outright, so a user with a mirror of their own needs no
/// code change. Otherwise the only question worth asking is whether the
/// upstream can be reached from here. Shipping an IP-to-country table instead
/// would be stale the moment it was written, and wrong behind any VPN or proxy;
/// a connection attempt answers the case that actually occurs — a network where
/// the upstream does not answer and the mirror does — without asking a third
/// party where the user is.
fn resolve_endpoint() -> String {
    if let Ok(value) = std::env::var("HF_ENDPOINT") {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return value;
        }
    }
    static RESOLVED: OnceLock<String> = OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            let endpoint = if upstream_reachable() {
                UPSTREAM_ENDPOINT
            } else {
                MIRROR_ENDPOINT
            };
            tracing::info!(endpoint, "huggingface endpoint selected");
            endpoint.to_string()
        })
        .clone()
}

/// Whether a connection to the upstream completes inside the timeout.
///
/// A TCP connect is the cheapest honest signal, and it needs no TLS handshake
/// and no third-party lookup. It is also the right failure mode: where the
/// upstream is blocked, the name usually does not resolve or the connection is
/// dropped, and both cases answer "use the mirror".
fn upstream_reachable() -> bool {
    const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
    match (UPSTREAM_HOST, 443u16).to_socket_addrs() {
        Ok(addresses) => addresses
            .into_iter()
            .any(|address| TcpStream::connect_timeout(&address, PROBE_TIMEOUT).is_ok()),
        Err(e) => {
            tracing::debug!(error = %e, "huggingface: cannot resolve the upstream host");
            false
        }
    }
}

/// Host of [`UPSTREAM_ENDPOINT`], kept beside it so the two cannot drift apart.
const UPSTREAM_HOST: &str = "huggingface.co";

impl HubClient {
    pub fn new() -> Self {
        Self::with_endpoint(resolve_endpoint())
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
        let encode_path = |s: &str| s.split('/').map(url_encode).collect::<Vec<_>>().join("/");
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
