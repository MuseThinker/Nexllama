use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    #[error("invalid filename: {0}")]
    InvalidFilename(String),
    #[error("repository not found: {0}")]
    RepoNotFound(String),
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("authentication required (set HF_TOKEN)")]
    AuthRequired,
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("download incomplete: got {got} of {expected} bytes")]
    Incomplete { got: u64, expected: u64 },
    #[error("unexpected status {status} for {url}")]
    Status { status: u16, url: String },
}

pub type Result<T> = std::result::Result<T, HubError>;
