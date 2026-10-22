use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use infer_hub::{
    download_key, repo_folder_name, DownloadControl, DownloadOutcome, GgufFile, HubClient,
    RepoSummary, TRUSTED_AUTHORS,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state::AppState;

/// Transfers this daemon is running, plus the last state of each.
///
/// The controls are kept apart from the reports: a finished transfer is removed
/// from `live` but left in `downloads` so the browser that started it can read
/// the path (or the error) after the fact. `live` is the authority on whether a
/// second request for the same file may start.
#[derive(Default)]
pub struct HubState {
    live: Mutex<HashMap<String, DownloadControl>>,
    downloads: Mutex<HashMap<String, DownloadStatus>>,
}

impl HubState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Serialize)]
pub struct DownloadStatus {
    pub repo: String,
    pub filename: String,
    pub status: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct RepoQuery {
    pub repo: String,
}

#[derive(Deserialize)]
pub struct DownloadRequest {
    pub repo: String,
    pub filename: String,
}

#[derive(Deserialize)]
pub struct DownloadQuery {
    pub repo: String,
    pub filename: String,
}

fn bad_request(message: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, message.into())
}

fn unavailable(message: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_GATEWAY, message.into())
}

fn lookup_key(
    repo: &str,
    filename: &str,
) -> Result<(String, String, String), (StatusCode, String)> {
    let repo = repo.trim();
    let filename = filename.trim();
    if repo.is_empty() {
        return Err(bad_request("repo must not be empty"));
    }
    if !filename.to_ascii_lowercase().ends_with(".gguf") {
        return Err(bad_request("only .gguf files can be downloaded"));
    }
    Ok((
        repo.to_string(),
        filename.to_string(),
        download_key(repo, filename),
    ))
}

pub async fn authors() -> Json<Vec<&'static str>> {
    Json(TRUSTED_AUTHORS.to_vec())
}

pub async fn search(
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<RepoSummary>>, (StatusCode, String)> {
    let text = query.q.trim();
    if text.is_empty() {
        return Err(bad_request("q must not be empty"));
    }
    let limit = query.limit.unwrap_or(20).clamp(1, 50);
    let client = HubClient::new();
    let found = match query
        .author
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())
    {
        Some(author) => client.search_by_author(author, text, limit).await,
        None => client.search(text, limit).await,
    };
    found
        .map(Json)
        .map_err(|e| unavailable(format!("HuggingFace search failed: {e}")))
}

pub async fn files(
    Query(query): Query<RepoQuery>,
) -> Result<Json<Vec<GgufFile>>, (StatusCode, String)> {
    let repo = query.repo.trim();
    if repo.is_empty() {
        return Err(bad_request("repo must not be empty"));
    }
    HubClient::new()
        .list_gguf(repo)
        .await
        .map(Json)
        .map_err(|e| unavailable(format!("HuggingFace listing failed: {e}")))
}

pub async fn download(
    State(state): State<AppState>,
    Json(request): Json<DownloadRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let (repo, filename, key) = lookup_key(&request.repo, &request.filename)?;
    let Some(base) = state.models_dir.as_deref().cloned() else {
        return Err(bad_request(
            "the daemon has no models folder; start it with --models-dir",
        ));
    };

    let dest: PathBuf = base.join(repo_folder_name(&repo));
    std::fs::create_dir_all(&dest).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create {}: {e}", dest.display()),
        )
    })?;

    let control = DownloadControl::new();
    {
        let mut live = state.hub.live.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "download state poisoned".into(),
            )
        })?;
        if live.contains_key(&key) {
            return Err((
                StatusCode::CONFLICT,
                "this file is already downloading".into(),
            ));
        }
        live.insert(key.clone(), control.clone());
    }
    {
        let mut reports = state.hub.downloads.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "download state poisoned".into(),
            )
        })?;
        reports.insert(
            key.clone(),
            DownloadStatus {
                repo: repo.clone(),
                filename: filename.clone(),
                status: "downloading".into(),
                downloaded: 0,
                total: None,
                path: None,
                error: None,
            },
        );
    }

    let task_state = state.clone();
    let task_repo = repo.clone();
    let task_filename = filename.clone();
    let task_key = key.clone();
    tokio::spawn(async move {
        let progress_state = task_state.clone();
        let progress_key = task_key.clone();
        let result = HubClient::new()
            .download(&task_repo, &task_filename, dest, control, move |event| {
                if let Ok(mut reports) = progress_state.hub.downloads.lock() {
                    if let Some(report) = reports.get_mut(&progress_key) {
                        report.downloaded = event.downloaded;
                        report.total = event.total;
                    }
                }
            })
            .await;

        if let Ok(mut live) = task_state.hub.live.lock() {
            live.remove(&task_key);
        }
        if let Ok(mut reports) = task_state.hub.downloads.lock() {
            if let Some(report) = reports.get_mut(&task_key) {
                match result {
                    Ok(DownloadOutcome::Completed(path)) => {
                        report.status = "completed".into();
                        report.path = Some(path.to_string_lossy().into_owned());
                    }
                    Ok(DownloadOutcome::Paused) => report.status = "paused".into(),
                    Ok(DownloadOutcome::Cancelled) => report.status = "cancelled".into(),
                    Err(e) => {
                        report.status = "error".into();
                        report.error = Some(e.to_string());
                    }
                }
            }
        }
    });

    Ok(Json(json!({
        "status": "started",
        "repo": repo,
        "filename": filename,
    })))
}

pub async fn status(
    State(state): State<AppState>,
    Query(query): Query<DownloadQuery>,
) -> Result<Json<DownloadStatus>, (StatusCode, String)> {
    let (_, _, key) = lookup_key(&query.repo, &query.filename)?;
    let reports = state.hub.downloads.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download state poisoned".into(),
        )
    })?;
    reports
        .get(&key)
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "no such download".into()))
}

pub async fn pause(
    State(state): State<AppState>,
    Json(request): Json<DownloadRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let (repo, filename, key) = lookup_key(&request.repo, &request.filename)?;
    let live = state.hub.live.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download state poisoned".into(),
        )
    })?;
    let Some(control) = live.get(&key) else {
        return Err((StatusCode::NOT_FOUND, "no such download".into()));
    };
    control.pause();
    Ok(Json(json!({
        "status": "pausing",
        "repo": repo,
        "filename": filename,
    })))
}

pub async fn cancel(
    State(state): State<AppState>,
    Json(request): Json<DownloadRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let (repo, filename, key) = lookup_key(&request.repo, &request.filename)?;
    let live = state.hub.live.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download state poisoned".into(),
        )
    })?;
    let Some(control) = live.get(&key) else {
        return Err((StatusCode::NOT_FOUND, "no such download".into()));
    };
    control.cancel();
    Ok(Json(json!({
        "status": "cancelling",
        "repo": repo,
        "filename": filename,
    })))
}

pub async fn downloads(
    State(state): State<AppState>,
) -> Result<Json<Vec<DownloadStatus>>, (StatusCode, String)> {
    let reports = state.hub.downloads.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download state poisoned".into(),
        )
    })?;
    let mut all: Vec<DownloadStatus> = reports.values().cloned().collect();
    all.sort_by(|a, b| {
        a.repo
            .cmp(&b.repo)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    Ok(Json(all))
}
