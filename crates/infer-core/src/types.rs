use serde::{Deserialize, Serialize};

/// A single chat turn passed to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    /// Encoded images attached to this turn, in order (PNG, JPEG, WebP, …).
    ///
    /// Kept as raw bytes rather than paths or data URLs: the transport layer has
    /// already resolved whatever the client sent, and libmtmd sniffs the format
    /// itself. Skipped when serialising so a transcript never carries megabytes
    /// of base64 back out.
    #[serde(default, skip_serializing)]
    pub images: Vec<Vec<u8>>,
}

impl ChatMsg {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn with_images(mut self, images: Vec<Vec<u8>>) -> Self {
        self.images = images;
        self
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }
}

/// Timing and token counts for one generation request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_ms: f64,
    pub gen_ms: f64,
    pub first_token_ms: f64,
    /// Draft tokens the n-gram drafter proposed.
    pub draft_tokens: usize,
    /// Draft tokens the model confirmed.
    pub accepted_draft_tokens: usize,
    /// Decodes that carried a draft.
    pub draft_rounds: usize,
    /// The reply stopped because it ran out of token budget, not because the
    /// model chose to stop. The caller reports this as `finish_reason:
    /// "length"` — a truncated reply and a finished one look identical from
    /// the text alone.
    pub hit_token_cap: bool,
}

impl GenStats {
    /// Fraction of drafted tokens the model confirmed, or `None` if it never
    /// drafted. A low rate means the drafter is guessing wrong and speculative
    /// decoding only costs an extra forward pass.
    pub fn draft_acceptance(&self) -> Option<f64> {
        if self.draft_tokens == 0 {
            None
        } else {
            Some(self.accepted_draft_tokens as f64 / self.draft_tokens as f64)
        }
    }
}

impl GenStats {
    pub fn tokens_per_second(&self) -> f64 {
        if self.gen_ms > 0.0 {
            self.generated_tokens as f64 * 1000.0 / self.gen_ms
        } else {
            0.0
        }
    }

    pub fn prompt_tokens_per_second(&self) -> f64 {
        if self.prompt_ms > 0.0 {
            self.prompt_tokens as f64 * 1000.0 / self.prompt_ms
        } else {
            0.0
        }
    }
}
