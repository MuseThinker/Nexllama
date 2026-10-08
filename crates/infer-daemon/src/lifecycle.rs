use std::time::Duration;

use crate::state::{AppState, Loaded};

/// Periodically unloads any model idle longer than keep_alive (0 = keep forever).
pub fn spawn_reaper(state: AppState, tick: Duration) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tick);
        loop {
            interval.tick().await;

            let removed: Vec<Loaded> = {
                let mut guard = state.inner.lock().await;
                if guard.keep_alive.is_zero() || guard.models.is_empty() {
                    continue;
                }
                let keep = guard.keep_alive;
                // Idle *and* unheld. A model serving a reply has not been idle
                // for `keep` seconds no matter what the clock says: `last_used`
                // is stamped when the request starts, so a long generation
                // looked stale the whole way through and was unloaded mid-reply.
                let stale: Vec<String> = guard
                    .models
                    .iter()
                    .filter(|(_, l)| l.last_used.elapsed() > keep && !l.is_busy())
                    .map(|(id, _)| id.clone())
                    .collect();

                let mut removed = Vec::new();
                for id in stale {
                    tracing::info!(%id, "idle timeout, unloading model");
                    if let Some(l) = guard.models.remove(&id) {
                        if guard.default_model.as_deref() == Some(id.as_str()) {
                            guard.default_model = None;
                        }
                        removed.push(l);
                    }
                }
                removed
            };

            if !removed.is_empty() {
                // Freeing a model waits for its worker thread to finish
                // whatever it is decoding, which can take seconds. Awaiting
                // that here used to stall this loop for the duration — and
                // while it stalled, no other model's idle timer ran at all.
                tokio::task::spawn_blocking(move || drop(removed));
            }
        }
    });
}
