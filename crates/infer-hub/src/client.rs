use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use serde::Deserialize;

use crate::download::{download_file, DownloadControl, DownloadOutcome, ProgressEvent};
use crate::error::{HubError, Result};

/// The upstream Hub. Preferred wherever it can be reached: it is the source of
/// truth, and everything else is a courtesy for networks that cannot reach it.
pub const UPSTREAM_ENDPOINT: &str = "https://huggingface.co";

/// A mirror of the Hub, used when the upstream is unreachable from this machine.
pub const MIRROR_ENDPOINT: &str = "https://hf-mirror.com";

/// The endpoint to use when nothing else decides it.
///
/// A name rather than a promise: the endpoint list is ordered per machine by a
/// reachability probe, and calls fall through it when a host stops answering.
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

/// Folder name for a repository's downloads.
///
/// A model is several files — weights, sometimes a projector and a draft — so
/// they go in one folder. The library treats a folder as a model, which is what
/// makes a freshly downloaded repository appear as one entry.
pub fn repo_folder_name(repo: &str) -> String {
    let cleaned: String = repo
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect();
    let cleaned = cleaned.trim_matches('.').to_string();
    if cleaned.is_empty() {
        "model".to_string()
    } else {
        cleaned
    }
}

/// Name a download by repository and filename. The separator cannot appear in
/// either, so no two pairs collide.
pub fn download_key(repo: &str, filename: &str) -> String {
    format!("{repo}\u{1}{filename}")
}

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
    /// Candidate endpoints, preferred first.
    endpoints: Vec<String>,
    /// Index in [`all_endpoints`] of `endpoints[0]`, so a successful fallback
    /// can be written back to [`PREFERRED_ENDPOINT`].
    origin: usize,
    http: reqwest::Client,
    token: Option<String>,
}

/// The endpoints to try, and the order to try them in on a cold start.
///
/// `HF_ENDPOINT` overrides everything outright, so anyone with a mirror of their
/// own needs no code change. Otherwise the upstream is preferred and the mirror
/// is the fallback. This order is only the starting point: a probe can be right
/// about the connection and wrong about the API, so each call falls through the
/// list (see [`HubClient::get`]) and the host that last answered is remembered.
fn all_endpoints() -> &'static [String] {
    static ENDPOINTS: OnceLock<Vec<String>> = OnceLock::new();
    ENDPOINTS.get_or_init(|| {
        if let Ok(value) = std::env::var("HF_ENDPOINT") {
            let value = value.trim();
            if !value.is_empty() {
                return vec![value.trim_end_matches('/').to_string()];
            }
        }
        let order: [&str; 2] = if upstream_reachable() {
            [UPSTREAM_ENDPOINT, MIRROR_ENDPOINT]
        } else {
            [MIRROR_ENDPOINT, UPSTREAM_ENDPOINT]
        };
        let endpoints: Vec<String> = order
            .iter()
            .map(|e| e.trim_end_matches('/').to_string())
            .collect();
        tracing::info!(?endpoints, "huggingface endpoints selected");
        endpoints
    })
}

/// Index into [`all_endpoints`] of the one that answered most recently.
///
/// A host that completes a TCP handshake and then hangs, or answers every
/// request with a 5xx, would otherwise cost its timeout on every call; recording
/// the fallback that did answer means the next call starts there.
static PREFERRED_ENDPOINT: AtomicUsize = AtomicUsize::new(0);

/// The candidate endpoints, the last known-good one first, plus its index in
/// [`all_endpoints`] so a successful fallback can be recorded.
fn ordered_endpoints() -> (Vec<String>, usize) {
    let all = all_endpoints();
    let len = all.len().max(1);
    let start = PREFERRED_ENDPOINT.load(Ordering::Relaxed) % len;
    let endpoints = (0..all.len())
        .map(|i| all[(start + i) % all.len()].clone())
        .collect();
    (endpoints, start)
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
        let (endpoints, origin) = ordered_endpoints();
        Self::build(endpoints, origin)
    }

    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self::build(vec![endpoint.into()], 0)
    }

    fn build(mut endpoints: Vec<String>, origin: usize) -> Self {
        for e in &mut endpoints {
            *e = e.trim_end_matches('/').to_string();
        }
        if endpoints.is_empty() {
            // Never happens with the compiled-in list, but an empty one would
            // leave every request with nowhere to go and no error to show.
            endpoints.push(UPSTREAM_ENDPOINT.to_string());
        }
        let http = reqwest::Client::builder()
            .user_agent(format!("nexllama/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build http client");
        Self {
            endpoints,
            origin,
            http,
            token: std::env::var("HF_TOKEN").ok(),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoints[0]
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

    /// Send a GET to each candidate endpoint until one answers.
    ///
    /// The first response that is neither a transport failure nor a server error
    /// is returned; a 5xx or 429 moves to the next endpoint, while a 4xx comes
    /// back as-is — the request itself is wrong (a missing repo, a bad token)
    /// and another host would say the same. When a fallback answers, it is
    /// promoted so later calls start there.
    async fn get<F>(&self, build: F) -> Result<reqwest::Response>
    where
        F: Fn(&str) -> reqwest::RequestBuilder,
    {
        let mut last: Option<HubError> = None;
        for (offset, endpoint) in self.endpoints.iter().enumerate() {
            match build(endpoint).send().await {
                Ok(resp)
                    if resp.status().is_server_error()
                        || resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS =>
                {
                    tracing::warn!(
                        endpoint = %endpoint,
                        status = %resp.status(),
                        "huggingface endpoint returned a retryable status; trying another"
                    );
                    last = Some(HubError::Status {
                        status: resp.status().as_u16(),
                        url: resp.url().to_string(),
                    });
                }
                Ok(resp) => {
                    if offset > 0 {
                        PREFERRED_ENDPOINT.store(
                            (self.origin + offset) % self.endpoints.len(),
                            Ordering::Relaxed,
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    tracing::warn!(
                        endpoint = %endpoint,
                        error = %e,
                        "huggingface endpoint unreachable; trying another"
                    );
                    last = Some(HubError::Http(e));
                }
            }
        }
        Err(last.unwrap_or_else(|| HubError::Status {
            status: 0,
            url: self.endpoint().to_string(),
        }))
    }

    async fn parse_models(&self, resp: reqwest::Response) -> Result<Vec<RepoSummary>> {
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(HubError::AuthRequired);
        }
        if !status.is_success() {
            return Err(HubError::Status {
                status: status.as_u16(),
                url: resp.url().to_string(),
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

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<RepoSummary>> {
        let limit_s = limit.to_string();
        let resp = self
            .get(|base| {
                self.auth(self.http.get(format!("{base}/api/models")).query(&[
                    ("search", query),
                    ("limit", limit_s.as_str()),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("filter", "gguf"),
                ]))
            })
            .await?;
        self.parse_models(resp).await
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
        let limit_s = limit.to_string();
        let resp = self
            .get(|base| {
                self.auth(self.http.get(format!("{base}/api/models")).query(&[
                    ("search", query),
                    ("author", author),
                    ("limit", limit_s.as_str()),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("filter", "gguf"),
                ]))
            })
            .await?;
        self.parse_models(resp).await
    }

    pub async fn list_gguf(&self, repo: &str) -> Result<Vec<GgufFile>> {
        let resp = self
            .get(|base| {
                self.auth(
                    self.http
                        .get(format!("{base}/api/models/{repo}/tree/main?recursive=true")),
                )
            })
            .await?;
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
                url: resp.url().to_string(),
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
        control: DownloadControl,
        on_progress: impl FnMut(ProgressEvent) + Send,
    ) -> Result<DownloadOutcome> {
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
        // The endpoint that answered the last API call, so listing and fetching
        // agree on the host. A transfer is one long request, so it is not retried
        // mid-stream against another endpoint — the resume logic in `download_file`
        // keys the partial file to its URL, and switching would only restart it.
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.endpoint(),
            encode_path(repo),
            encode_path(filename)
        );
        download_file(&self.http, &self.token, &url, &dest, &control, on_progress).await
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

#[cfg(test)]
mod endpoint_tests {
    use super::*;

    #[test]
    fn an_explicit_endpoint_is_kept_and_normalised() {
        // The trailing slash is trimmed so request paths never double up, and
        // the single endpoint is the only one tried — an explicit choice is not
        // second-guessed with a fallback.
        let client = HubClient::with_endpoint("https://example.test/".to_string());
        assert_eq!(client.endpoint(), "https://example.test");
        assert_eq!(client.endpoints, vec!["https://example.test".to_string()]);
    }

    #[test]
    fn the_candidate_list_is_never_empty() {
        // `get` walks this list; an empty one would leave every request with
        // nowhere to go.
        let (endpoints, _) = ordered_endpoints();
        assert!(!endpoints.is_empty());
        assert!(endpoints.iter().all(|e| !e.ends_with('/')));
    }
}
