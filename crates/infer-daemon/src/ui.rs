//! The built web UI, served from disk on the API's own port.
//!
//! A container that ships this app wants one published port, not a web server in
//! front of an API: the UI is a folder of static files, and the daemon is
//! already the thing that has to be reachable. So the files are served here,
//! and the API is merged on top of them by [`crate::http::router`].

use std::path::PathBuf;

use axum::Router;
use tower_http::services::{ServeDir, ServeFile};

use crate::state::AppState;

/// Serves `dir`, with `/` answering the app itself.
///
/// The app is a single page with no URL routing of its own — the tabs are state,
/// not paths — so there is nothing to rewrite to `index.html` and the only
/// decision here is what an unknown path should do. Answering the app keeps a
/// bookmarked `http://host:8090/chat` working after a restart, which costs a 404
/// on a mistyped API path and buys a page that always loads.
pub fn router(dir: PathBuf) -> Router<AppState> {
    let index = dir.join("index.html");
    Router::new()
        .route_service("/", ServeFile::new(index.clone()))
        // `ServeDir` refuses to walk out of the directory it was given, so a
        // `../../etc/passwd` in the path is a 404 rather than a file read.
        // Its fallback is the app itself, answered as a normal 200 — not
        // `not_found_service`, which would serve this same file with a 404
        // status and give the browser an error page dressed as the UI.
        .fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
}