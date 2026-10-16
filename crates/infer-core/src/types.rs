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
    /// The tool calls an assistant turn requested, as the OpenAI `tool_calls`
    /// array. Passed back to the chat template so the model sees what it asked
    /// for before the results arrive.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    /// For a `tool` result: the id of the call it answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl ChatMsg {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn with_images(mut self, images: Vec<Vec<u8>>) -> Self {
        self.images = images;
        self
    }

    pub fn with_tool_calls(mut self, calls: serde_json::Value) -> Self {
        self.tool_calls = Some(calls);
        self
    }

    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
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
    /// Tool calls the model asked for, as an OpenAI `tool_calls` array.
    ///
    /// Set only when the request declared tools and the reply parsed into
    /// calls. `None` means the model answered in plain text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// The reply's answer text with any tool-call markup and reasoning removed.
    ///
    /// Filled only for requests that declared tools; those hold their reply back
    /// for parsing, so this is also what was emitted to the caller. Not part of
    /// the serialised stats — it is the caller's own content, not a measurement.
    #[serde(skip)]
    pub parsed_content: Option<String>,
    /// The reasoning the model produced, split out when tools were declared.
    #[serde(skip)]
    pub reasoning_content: Option<String>,
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
