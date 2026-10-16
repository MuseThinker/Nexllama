//! Cloud providers, proxied through the same port as the local engine.
//!
//! The desktop app owns the providers (its settings hold the non-secret parts,
//! the OS credential store holds the keys) and pushes them here over loopback
//! with `POST /admin/providers`. This module keeps them in memory and forwards
//! a request whose model id names a provider on to that provider's
//! OpenAI-compatible endpoint.
//!
//! It is deliberately independent of the local engine: the registry has its own
//! lock, proxying is async I/O on the runtime, and a request that routes to a
//! provider never touches a model context or the worker thread. A slow or broken
//! provider can only stall its own request.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, HeaderValue, Response, StatusCode};
use serde::Deserialize;
use serde_json::json;

use crate::http::ApiError;

/// One configured provider, key included. Held only in memory — never written
/// to disk here, never returned by an endpoint.
#[derive(Clone)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub models: Vec<String>,
}

/// The shared registry. `AppState` holds one.
pub type Registry = Arc<std::sync::RwLock<Vec<Provider>>>;

/// A provider as the app sends it to `POST /admin/providers`.
#[derive(Deserialize)]
pub struct ProviderIn {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub models: Vec<String>,
}

/// Split a request's model id into the provider it names and the native model.
///
/// Cloud models are namespaced `<providerId>/<model>`, so routing is the part
/// before the first slash — which leaves a native id that itself contains a
/// slash (many are `owner/name`) intact.
pub fn route(providers: &[Provider], model: Option<&str>) -> Option<(Provider, String)> {
    let model = model?.trim();
    let (prefix, native) = model.split_once('/')?;
    if prefix.is_empty() || native.is_empty() {
        return None;
    }
    providers
        .iter()
        .find(|p| p.id == prefix)
        .map(|p| (p.clone(), native.to_string()))
}

/// The OpenAI-compatible root, without a trailing slash.
fn base(p: &Provider) -> String {
    p.base_url.trim_end_matches('/').to_string()
}

/// Adjust a request body before it goes upstream: name the native model, and
/// drop a token cap the provider cannot accept.
///
/// Locally `max_tokens: 0` means "until the model stops", but no cloud provider
/// takes 0 (DeepSeek's range is `[1, 393216]`), so a zero or null cap is
/// removed and the provider's own default applies.
fn prepare_body(mut body: serde_json::Value, native_model: String) -> serde_json::Value {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".into(), json!(native_model));
        for key in ["max_tokens", "max_completion_tokens"] {
            let drop = match obj.get(key) {
                None => false,
                Some(serde_json::Value::Null) => true,
                Some(v) => v.as_i64() == Some(0),
            };
            if drop {
                obj.remove(key);
            }
        }
    }
    body
}

/// Forward one chat completion to `provider` and stream the reply straight back.
///
/// The request and response are both OpenAI-shaped, so nothing is translated:
/// the body is passed through with only `model` rewritten to the native id, and
/// a streaming reply is piped byte-for-byte. The client hanging up drops the
/// upstream body, which cancels the provider request.
pub async fn proxy_chat(
    client: &reqwest::Client,
    provider: &Provider,
    native_model: String,
    body: serde_json::Value,
) -> Result<Response<Body>, ApiError> {
    let streaming = body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let body = prepare_body(body, native_model);

    let url = format!("{}/chat/completions", base(provider));
    let resp = client
        .post(&url)
        .header(
            header::AUTHORIZATION,
            format!("Bearer {}", provider.api_key),
        )
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ApiError::Upstream(StatusCode::BAD_GATEWAY, format!("{}: {e}", provider.name))
        })?;

    let status = resp.status();
    if !status.is_success() {
        let detail = resp.text().await.unwrap_or_default();
        return Err(ApiError::Upstream(
            status,
            format!("{}: {}", provider.name, truncate(&detail, 2000)),
        ));
    }

    if streaming {
        // Straight through, so the client's SSE reader sees the provider's own
        // framing (including `[DONE]`) with nothing re-chunked in between.
        let mut response = Response::new(Body::from_stream(resp.bytes_stream()));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        Ok(response)
    } else {
        let bytes = resp.bytes().await.map_err(|e| {
            ApiError::Upstream(StatusCode::BAD_GATEWAY, format!("{}: {e}", provider.name))
        })?;
        let mut response = Response::new(Body::from(bytes));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(response)
    }
}

/// Ask a provider what it serves, for the "fetch models" button.
///
/// Uses the OpenAI `GET /models` shape, which every provider here implements.
pub async fn fetch_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<String>, ApiError> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| ApiError::Upstream(StatusCode::BAD_GATEWAY, e.to_string()))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(ApiError::Upstream(
            status,
            format!("models request failed: {}", truncate(&text, 500)),
        ));
    }
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        ApiError::Upstream(StatusCode::BAD_GATEWAY, format!("bad models reply: {e}"))
    })?;
    let mut ids: Vec<String> = value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    Ok(ids)
}

/// Cap an upstream error body so a hostile or chatty provider cannot fill a
/// response (or a log) with megabytes.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(id: &str) -> Provider {
        Provider {
            id: id.to_string(),
            name: id.to_string(),
            base_url: "https://example.test/v1".to_string(),
            api_key: "secret".to_string(),
            models: vec!["m".to_string()],
        }
    }

    #[test]
    fn routes_by_provider_prefix() {
        let providers = vec![provider("deepseek"), provider("minimax-intl")];

        let (hit, native) = route(&providers, Some("deepseek/deepseek-chat")).unwrap();
        assert_eq!(hit.id, "deepseek");
        assert_eq!(native, "deepseek-chat");

        // A native id containing its own slash is kept whole.
        let (hit, native) = route(&providers, Some("minimax-intl/MiniMax/M3")).unwrap();
        assert_eq!(hit.id, "minimax-intl");
        assert_eq!(native, "MiniMax/M3");
    }

    #[test]
    fn local_and_unknown_models_do_not_route() {
        let providers = vec![provider("deepseek")];
        assert!(route(&providers, Some("Qwen3.8-27B")).is_none(), "local id");
        assert!(
            route(&providers, Some("openai/gpt-4o")).is_none(),
            "unknown provider"
        );
        assert!(route(&providers, None).is_none());
        assert!(
            route(&providers, Some("deepseek/")).is_none(),
            "empty native"
        );
    }

    #[test]
    fn body_names_the_model_and_drops_a_zero_cap() {
        let out = prepare_body(
            json!({ "model": "deepseek/deepseek-chat", "max_tokens": 0, "messages": [] }),
            "deepseek-chat".to_string(),
        );
        assert_eq!(out["model"], "deepseek-chat");
        assert!(
            out.get("max_tokens").is_none(),
            "a zero cap must be dropped"
        );

        let out = prepare_body(json!({ "max_tokens": 1000 }), "m".to_string());
        assert_eq!(out["max_tokens"], 1000, "a real cap is kept");

        let out = prepare_body(json!({ "max_completion_tokens": null }), "m".to_string());
        assert!(out.get("max_completion_tokens").is_none());
    }

    #[test]
    fn error_bodies_are_bounded() {
        let long = "a".repeat(100);
        assert_eq!(truncate(&long, 200), long);
        assert!(truncate(&long, 10).chars().count() <= 11);
    }
}
