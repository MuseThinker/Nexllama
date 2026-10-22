use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::error::{HubError, Result};

const RUNNING: u8 = 0;
const PAUSE_REQUESTED: u8 = 1;
const CANCEL_REQUESTED: u8 = 2;

/// A handle a caller keeps to stop a download that is already underway.
///
/// [`HubClient::download`](crate::HubClient::download) is a long await — a model
/// is gigabytes and a slow link is minutes — and without this there was no way
/// to interrupt one short of closing the app: the transfer ran to the end
/// whether or not anyone still wanted it.
#[derive(Clone, Default)]
pub struct DownloadControl {
    state: Arc<AtomicU8>,
}

impl DownloadControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop after the current chunk, keeping the partial file for a resume.
    pub fn pause(&self) {
        // A cancel outranks a pause: the two race on one transfer and
        // cancelling is the stronger intent, so never downgrade it.
        let _ = self.state.compare_exchange(
            RUNNING,
            PAUSE_REQUESTED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    /// Stop after the current chunk and discard the partial file.
    pub fn cancel(&self) {
        self.state.store(CANCEL_REQUESTED, Ordering::SeqCst);
    }

    fn pause_requested(&self) -> bool {
        self.state.load(Ordering::SeqCst) == PAUSE_REQUESTED
    }

    fn cancel_requested(&self) -> bool {
        self.state.load(Ordering::SeqCst) == CANCEL_REQUESTED
    }
}

/// How a download ended.
#[derive(Debug, Clone)]
pub enum DownloadOutcome {
    /// The file is whole and in place.
    Completed(PathBuf),
    /// Paused at the caller's request; the partial file is kept so the next
    /// download resumes where this one stopped.
    Paused,
    /// Cancelled at the caller's request; the partial file is gone.
    ///
    /// Not an error: the caller asked for this, and a UI that reported its own
    /// stop button as a failure would be lying about what happened.
    Cancelled,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressEvent {
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl ProgressEvent {
    pub fn fraction(&self) -> Option<f32> {
        self.total
            .filter(|t| *t > 0)
            .map(|t| (self.downloaded as f32 / t as f32).min(1.0))
    }

    pub fn percent(&self) -> Option<f32> {
        self.fraction().map(|f| f * 100.0)
    }
}

pub(crate) async fn download_file(
    http: &reqwest::Client,
    token: &Option<String>,
    url: &str,
    dest: &Path,
    control: &DownloadControl,
    mut on_progress: impl FnMut(ProgressEvent) + Send,
) -> Result<DownloadOutcome> {
    let part = part_path(dest);

    // A cancel that arrived before the first byte still has to clean up: there
    // may be a partial file from an earlier attempt waiting to be resumed.
    if control.cancel_requested() {
        let _ = tokio::fs::remove_file(&part).await;
        let _ = tokio::fs::remove_file(meta_path(dest)).await;
        return Ok(DownloadOutcome::Cancelled);
    }

    // Resume only from bytes that came from this URL. The partial file used to
    // be keyed by its final name alone, so `model.gguf` interrupted in one repo
    // could be resumed as `model.gguf` from a different one — appending one
    // file onto the other and producing a model that looks complete and is not.
    let stored = read_part_meta(dest).await;
    let existing = match stored.as_ref() {
        Some(meta) if meta.url == url => tokio::fs::metadata(&part)
            .await
            .map(|m| m.len())
            .unwrap_or(0),
        _ => {
            if tokio::fs::metadata(&part).await.is_ok() {
                let _ = tokio::fs::remove_file(&part).await;
            }
            0
        }
    };

    let mut req = http.get(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    if existing > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }

    let resp = req.send().await?;
    let status = resp.status();

    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(HubError::AuthRequired);
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(HubError::FileNotFound(url.to_string()));
    }

    let etag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    if status == reqwest::StatusCode::PARTIAL_CONTENT && existing > 0 {
        // The server is letting us resume. If it also names the entity and that
        // name has changed, the file moved underneath the partial download;
        // finishing it would produce a model made of two different files.
        if let (Some(old), Some(new)) = (stored.as_ref().and_then(|m| m.etag.clone()), &etag) {
            if old != *new {
                let _ = tokio::fs::remove_file(&part).await;
                let _ = tokio::fs::remove_file(meta_path(dest)).await;
                return Err(HubError::Status {
                    status: 409,
                    url: url.to_string(),
                });
            }
        }
    }

    let (mut downloaded, total, mut file) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let remaining = resp.content_length().unwrap_or(0);
        let total = if remaining > 0 {
            Some(existing + remaining)
        } else {
            None
        };
        let file = OpenOptions::new().append(true).open(&part).await?;
        (existing, total, file)
    } else if status.is_success() {
        let total = resp.content_length();
        let file = File::create(&part).await?;
        (0u64, total, file)
    } else {
        return Err(HubError::Status {
            status: status.as_u16(),
            url: url.to_string(),
        });
    };

    // Record what these bytes are the moment we commit to writing them, not
    // when the file is whole: an interrupted transfer is the common case, and
    // metadata written only on success left the partial file anonymous, so a
    // later download could not tell what it was resuming and began again from
    // zero — the same bytes re-fetched because the file that named them was
    // deleted along with the connection.
    write_part_meta(
        dest,
        &PartMeta {
            url: url.to_string(),
            etag: etag.clone(),
        },
    )
    .await;

    on_progress(ProgressEvent { downloaded, total });

    let mut stream = resp.bytes_stream();
    let mut last_emit = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        // Checked before the write so a stop lands within one chunk, not after
        // the whole multi-gigabyte body has finished arriving.
        if control.cancel_requested() {
            drop(file);
            let _ = tokio::fs::remove_file(&part).await;
            let _ = tokio::fs::remove_file(meta_path(dest)).await;
            return Ok(DownloadOutcome::Cancelled);
        }
        if control.pause_requested() {
            file.flush().await?;
            drop(file);
            on_progress(ProgressEvent { downloaded, total });
            return Ok(DownloadOutcome::Paused);
        }

        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if last_emit.elapsed().as_millis() >= 200 {
            on_progress(ProgressEvent { downloaded, total });
            last_emit = Instant::now();
        }
    }
    file.flush().await?;
    drop(file);

    if let Some(total) = total {
        if downloaded < total {
            return Err(HubError::Incomplete {
                got: downloaded,
                expected: total,
            });
        }
    }

    on_progress(ProgressEvent { downloaded, total });

    if tokio::fs::try_exists(dest).await.unwrap_or(false) {
        tokio::fs::remove_file(dest).await?;
    }
    tokio::fs::rename(&part, dest).await?;
    let _ = tokio::fs::remove_file(meta_path(dest)).await;
    Ok(DownloadOutcome::Completed(dest.to_path_buf()))
}

/// What a partial download belongs to.
#[derive(serde::Serialize, serde::Deserialize)]
struct PartMeta {
    url: String,
    etag: Option<String>,
}

fn part_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf");
    dest.with_file_name(format!("{name}.part"))
}

fn meta_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf");
    dest.with_file_name(format!("{name}.part.json"))
}

async fn read_part_meta(dest: &Path) -> Option<PartMeta> {
    let raw = tokio::fs::read_to_string(meta_path(dest)).await.ok()?;
    serde_json::from_str(&raw).ok()
}

async fn write_part_meta(dest: &Path, meta: &PartMeta) {
    if let Ok(text) = serde_json::to_string(meta) {
        let _ = tokio::fs::write(meta_path(dest), text).await;
    }
}
