use serde::{Deserialize, Serialize};

use crate::error::{InferError, Result};

/// Largest number of tokens one request may generate.
///
/// The cap exists because `max_tokens` is the only bound on how long a request
/// occupies the model's single worker thread: an unbounded value keeps every
/// other caller queued behind it for as long as the context allows.
pub const MAX_TOKENS: usize = 1 << 16;

/// Longest draft the speculative path will build.
///
/// A draft is decoded as one batch, so its length is a batch size: `ngram_m`
/// of 65535 would allocate a 64k-token batch up front, for a 4k context that
/// could never hold it.
pub const MAX_DRAFT: u16 = 256;

/// Longest n-gram the drafter will look for.
pub const MAX_NGRAM: u16 = 64;

/// Sampling knobs mirrored from the common llama.cpp options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub repeat_freq: f32,
    pub repeat_present: f32,
    pub seed: u32,
    pub max_tokens: usize,
    /// Verify self-drafted n-gram continuations in one batched decode.
    ///
    /// This is a decoding strategy rather than a sampling knob, but it has to
    /// travel with the request that it accelerates, so it lives here.
    pub speculative: bool,
    /// Length of the n-gram whose most recent repeat seeds a draft.
    pub ngram_n: u16,
    /// How many tokens a successful match contributes to a draft.
    pub ngram_m: u16,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            repeat_freq: 0.0,
            repeat_present: 0.0,
            seed: 0xFFFF_FFFF,
            max_tokens: 512,
            speculative: false,
            ngram_n: 4,
            ngram_m: 8,
        }
    }
}

impl SamplerParams {
    /// Reject values the sampler cannot interpret.
    ///
    /// Every field here reaches the engine straight from a request body, so any
    /// of them can arrive as `NaN`, as a negative penalty, or as a token cap of
    /// `usize::MAX`. None of those fail loudly downstream: the penalty divides
    /// by the value, the cap sizes a batch, and a `NaN` temperature compares
    /// false against every bound and slips through. Catching them here is what
    /// turns "the reply is word salad" or "the daemon died" into a message that
    /// names the field.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("min_p", self.min_p),
            ("repeat_penalty", self.repeat_penalty),
            ("repeat_freq", self.repeat_freq),
            ("repeat_present", self.repeat_present),
        ] {
            if !v.is_finite() {
                return Err(InferError::InvalidParameter(format!(
                    "{name} must be a finite number, got {v}"
                )));
            }
        }

        if self.temperature < 0.0 {
            return Err(InferError::InvalidParameter(format!(
                "temperature must be >= 0, got {}",
                self.temperature
            )));
        }
        for (name, v) in [("top_p", self.top_p), ("min_p", self.min_p)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(InferError::InvalidParameter(format!(
                    "{name} must be between 0 and 1, got {v}"
                )));
            }
        }
        // The penalty is applied as `logit /= repeat_penalty` for positive
        // logits, so zero makes every repeated token infinitely likely and a
        // negative value flips the sign of the score.
        if self.repeat_penalty <= 0.0 {
            return Err(InferError::InvalidParameter(format!(
                "repeat_penalty must be > 0, got {}",
                self.repeat_penalty
            )));
        }
        // -1 means "the whole context", which llama.cpp understands; anything
        // below that is a caller mistake rather than a meaningful window.
        if self.repeat_last_n < -1 {
            return Err(InferError::InvalidParameter(format!(
                "repeat_last_n must be >= -1, got {}",
                self.repeat_last_n
            )));
        }
        if self.max_tokens > MAX_TOKENS {
            return Err(InferError::InvalidParameter(format!(
                "max_tokens must be <= {MAX_TOKENS}, got {}",
                self.max_tokens
            )));
        }
        if self.ngram_n > MAX_NGRAM {
            return Err(InferError::InvalidParameter(format!(
                "ngram_n must be <= {MAX_NGRAM}, got {}",
                self.ngram_n
            )));
        }
        if self.ngram_m > MAX_DRAFT {
            return Err(InferError::InvalidParameter(format!(
                "ngram_m must be <= {MAX_DRAFT}, got {}",
                self.ngram_m
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bad(edit: impl FnOnce(&mut SamplerParams)) -> String {
        let mut p = SamplerParams::default();
        edit(&mut p);
        p.validate().expect_err("should be rejected").to_string()
    }

    #[test]
    fn defaults_are_valid() {
        SamplerParams::default().validate().unwrap();
    }

    #[test]
    fn non_finite_values_are_rejected() {
        assert!(bad(|p| p.temperature = f32::NAN).contains("temperature"));
        assert!(bad(|p| p.top_p = f32::INFINITY).contains("top_p"));
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        assert!(bad(|p| p.temperature = -1.0).contains("temperature"));
        assert!(bad(|p| p.min_p = 1.5).contains("min_p"));
        // A zero penalty divides by zero inside the sampler.
        assert!(bad(|p| p.repeat_penalty = 0.0).contains("repeat_penalty"));
        assert!(bad(|p| p.repeat_penalty = -1.0).contains("repeat_penalty"));
        assert!(bad(|p| p.repeat_last_n = -2).contains("repeat_last_n"));
        assert!(bad(|p| p.max_tokens = usize::MAX).contains("max_tokens"));
        assert!(bad(|p| p.ngram_m = u16::MAX).contains("ngram_m"));
    }

    #[test]
    fn sentinels_still_pass() {
        // -1 is "the whole context", 0 means "until it stops".
        let p = SamplerParams {
            repeat_last_n: -1,
            max_tokens: 0,
            ..Default::default()
        };
        p.validate().unwrap();
    }
}
