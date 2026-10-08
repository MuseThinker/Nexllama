pub mod engine;
pub mod error;
pub mod sampler;
pub mod types;

pub use engine::{
    EmbedResult, Engine, EngineHandle, EngineInfo, FlashAttn, LoadConfig, LoadEvent, LoadMode,
    LoraAdapter, RerankResult,
};
pub use error::{InferError, Result};
pub use sampler::{SamplerParams, MAX_DRAFT, MAX_NGRAM, MAX_TOKENS};
pub use types::{ChatMsg, GenStats};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_is_clamped_by_context_and_batch() {
        // Plenty of room: the drafter's own limit wins.
        assert_eq!(engine::draft_len(8, 100, 8, 2048), 8);
        // The context only has room for the anchor plus two drafts.
        assert_eq!(engine::draft_len(8, 2, 8, 2048), 2);
        // A batch of one cannot hold the anchor and a draft at all.
        assert_eq!(engine::draft_len(8, 100, 8, 1), 0);
        // Never more than the drafter produced.
        assert_eq!(engine::draft_len(3, 100, 8, 2048), 3);
    }

    #[test]
    fn draft_acceptance_is_none_without_drafts() {
        let stats = GenStats {
            draft_tokens: 0,
            accepted_draft_tokens: 0,
            ..Default::default()
        };
        assert_eq!(stats.draft_acceptance(), None);

        let stats = GenStats {
            draft_tokens: 8,
            accepted_draft_tokens: 6,
            ..Default::default()
        };
        assert_eq!(stats.draft_acceptance(), Some(0.75));
    }

    #[test]
    fn speculative_params_default_off() {
        let p = SamplerParams::default();
        assert!(!p.speculative, "speculative decoding must be opt-in");
        assert_eq!((p.ngram_n, p.ngram_m), (4, 8));
    }

    #[test]
    fn ngram_model_drafts_what_it_learned() {
        use engine::NgramModel;
        use llama_cpp_4::token::LlamaToken;

        let toks = |v: &[i32]| -> Vec<LlamaToken> { v.iter().copied().map(LlamaToken).collect() };

        let mut model = NgramModel::default();
        // 1 2 3 4 1 2 3 4, three times over, so the counts are unambiguous.
        model.learn(&toks(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), 2);

        assert_eq!(
            model.draft(&toks(&[9, 1, 2]), 2),
            toks(&[3, 4]),
            "should continue the learned pattern"
        );
        assert_eq!(
            model.draft(&toks(&[9, 1, 2]), 4),
            toks(&[3, 4, 1, 2]),
            "the draft keeps extending while the pattern holds"
        );
        assert!(
            model.draft(&toks(&[7, 7]), 4).is_empty(),
            "an unknown context must not be guessed at"
        );
        assert!(
            model.draft(&toks(&[1]), 4).is_empty(),
            "history shorter than the order cannot be looked up"
        );
    }

    #[test]
    fn ngram_model_picks_the_usual_continuation() {
        use engine::NgramModel;
        use llama_cpp_4::token::LlamaToken;

        let toks = |v: &[i32]| -> Vec<LlamaToken> { v.iter().copied().map(LlamaToken).collect() };

        let mut model = NgramModel::default();
        // "1 -> 2" happens twice, "1 -> 9" once.
        model.learn(&toks(&[1, 2, 5, 1, 2, 6, 1, 9]), 1);
        assert_eq!(model.draft(&toks(&[1]), 1), toks(&[2]));
    }

    #[test]
    fn changing_the_order_invalidates_the_table() {
        use engine::NgramModel;
        use llama_cpp_4::token::LlamaToken;

        let toks = |v: &[i32]| -> Vec<LlamaToken> { v.iter().copied().map(LlamaToken).collect() };

        let mut model = NgramModel::default();
        model.learn(&toks(&[1, 2, 3, 4]), 2);
        assert!(model.contexts() > 0);
        model.learn(&toks(&[1, 2, 3, 4]), 3);
        assert_eq!(model.order, 3);
        assert!(model.draft(&toks(&[1, 2]), 2).is_empty());
    }

    #[test]
    fn draft_cache_survives_a_reload() {
        use engine::DraftCache;
        use llama_cpp_4::token::LlamaToken;

        let toks = |v: &[i32]| -> Vec<LlamaToken> { v.iter().copied().map(LlamaToken).collect() };
        let dir = std::env::temp_dir().join("nexllama-cache-test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("ngram.json");

        let mut first = DraftCache::load(Some(path.clone()));
        assert!(
            first.draft(&toks(&[1, 2]), 2, 4).is_empty(),
            "nothing has been learned yet"
        );
        first.learn(&toks(&[1, 2, 3, 4, 1, 2, 3, 4]), 2);
        assert!(first.contexts() > 0, "learning should record n-grams");
        assert!(path.exists(), "the table should be written to disk");
        let learned = first.contexts();
        drop(first);

        let second = DraftCache::load(Some(path.clone()));
        assert_eq!(
            second.contexts(),
            learned,
            "a reload should recover the table"
        );
        assert_eq!(
            second.draft(&toks(&[8, 1, 2]), 2, 2),
            toks(&[3, 4]),
            "a reloaded table should still draft"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_cache_file_is_ignored() {
        use engine::DraftCache;

        let dir = std::env::temp_dir().join("nexllama-cache-corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ngram.json");
        std::fs::write(&path, b"{not json").unwrap();

        let cache = DraftCache::load(Some(path));
        assert_eq!(
            cache.contexts(),
            0,
            "a bad file should start empty, not fail"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
