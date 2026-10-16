//! Remembers prompts already encoded, so the next turn can skip re-encoding them.
//!
//! A conversation grows by a turn at a time, which means every prompt after the
//! first contains the one before it as a prefix. Encoding that prefix again on
//! each turn is the whole of the wait before the first token appears — the
//! decode that follows is what the user sees as "typing", and it costs the same
//! either way.
//!
//! The KV state itself lives in llama.cpp; this module only decides *which*
//! remembered prefix applies and *how many* to keep. Keeping that decision here
//! is what makes it testable without a model: the match is a prefix comparison
//! over token ids, and eviction is a budget over token counts.
//!
//! Enabled by default, and verified by the `prefix_check` example — see
//! `LoadConfig::prefix_cache`.

/// Shortest prompt worth remembering.
///
/// Saving a state is a copy of the KV cache, so a prompt this small costs more
/// to remember than it costs to encode again.
const MIN_TOKENS: usize = 32;

/// One encoded prompt and the KV state that goes with it.
#[derive(Debug, Clone)]
pub struct Cached {
    /// Token ids, in order. The match is against these.
    pub tokens: Vec<i32>,
    /// Serialised KV state for sequence 0 at `tokens.len()`.
    pub state: Vec<u8>,
    /// Serialised draft-context state, when a drafter was loaded.
    ///
    /// A draft head keeps its own cache and position state beside the target's,
    /// so restoring the target alone would leave the two disagreeing about where
    /// in the conversation they are — which is why reuse used to be switched off
    /// whenever anything was drafting. Capturing both sides together and putting
    /// them back together keeps them aligned, so reuse stays on.
    pub draft: Option<Vec<u8>>,
}

/// Which remembered prefixes apply, and which to keep.
#[derive(Debug)]
pub struct PromptCache {
    /// Most entries to hold. Zero disables the cache outright.
    max_entries: usize,
    /// Ceiling on the total state bytes held, which is what bounds the memory.
    ///
    /// Bytes, not tokens, and deliberately so: a remembered prompt costs its
    /// serialised KV cache, which for a large model is hundreds of kilobytes
    /// per token. A budget counted in tokens would look small while holding
    /// gigabytes.
    max_bytes: usize,
    /// Least-recently-used first.
    entries: Vec<Cached>,
}

impl PromptCache {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            entries: Vec::new(),
        }
    }

    /// A cache with no room is a cache that never matches, which is how the
    /// feature is switched off without special-casing every call site.
    pub fn disabled() -> Self {
        Self::new(0, 0)
    }

    pub fn is_enabled(&self) -> bool {
        self.max_entries > 0
    }

    /// How many leading tokens of `tokens` are already encoded.
    ///
    /// Returns the length of the longest remembered prompt that is a prefix of
    /// this one. Not the entry itself: the caller only needs the offset, and
    /// handing out a borrow here would tie the caller to this struct's
    /// lifetime for the whole request.
    ///
    /// `need_draft` selects which kind of snapshot may match: an entry saved
    /// while nothing was drafting carries no draft-side state, and restoring it
    /// under a loaded drafter would desynchronise the two caches. Entries of the
    /// wrong kind are skipped entirely rather than restored and repaired.
    pub fn longest_prefix(&mut self, tokens: &[i32], need_draft: bool) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.draft.is_some() != need_draft {
                continue;
            }
            if e.tokens.len() > tokens.len() {
                continue;
            }
            if tokens[..e.tokens.len()] != e.tokens[..] {
                continue;
            }
            if best.is_none_or(|(len, _)| e.tokens.len() > len) {
                best = Some((e.tokens.len(), i));
            }
        }
        let (len, i) = best?;
        // A hit is a use: move it to the back so eviction spares it.
        let entry = self.entries.remove(i);
        self.entries.push(entry);
        Some(len)
    }

    /// Take the snapshot belonging to the prompt just matched, if there is one.
    ///
    /// Callers call this right after [`Self::longest_prefix`] and before
    /// encoding anything, so the entry is the one at the back.
    pub fn snapshot_for(&self, len: usize) -> Option<&Cached> {
        self.entries
            .last()
            .filter(|e| e.tokens.len() == len)
    }

    /// Remember a prompt, its target KV state, and the draft-side state when a
    /// drafter was loaded for that turn.
    pub fn insert(&mut self, tokens: Vec<i32>, state: Vec<u8>, draft: Option<Vec<u8>>) {
        if !self.is_enabled() || tokens.len() < MIN_TOKENS || state.is_empty() {
            return;
        }
        // Re-remembering the same prompt replaces it rather than duplicating.
        self.entries.retain(|e| e.tokens != tokens);

        self.entries.push(Cached {
            tokens,
            state,
            draft,
        });

        while self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
        // Stops at one entry, however far over budget that entry is. Evicting
        // everything would leave a cache that can never hit, and a prompt
        // bigger than the whole budget is better served by caching it slowly
        // than by never caching it at all.
        while self.entries.len() > 1 && self.held_bytes() > self.max_bytes {
            self.entries.remove(0);
        }
    }

    /// Total serialised state held, target and draft sides together.
    ///
    /// The budget is what bounds memory, and a drafting snapshot costs the draft
    /// context's bytes on top of the target's, so both count.
    fn held_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.state.len() + e.draft.as_ref().map_or(0, Vec::len))
            .sum()
    }

    /// Forget everything.
    ///
    /// A context that has been rebuilt invalidates every state held, since the
    /// saved bytes describe one particular KV cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Tokens currently remembered, for reporting.
    pub fn held_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.tokens.len()).sum()
    }

    /// Prompts currently remembered, for reporting.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> PromptCache {
        PromptCache::new(3, 100)
    }

    /// A token sequence long enough to be worth caching.
    ///
    /// Anything shorter than [`MIN_TOKENS`] is refused on purpose, so tests have
    /// to clear that bar before matching is meaningful at all.
    fn seq(head: &[i32]) -> Vec<i32> {
        let mut v = head.to_vec();
        while v.len() < MIN_TOKENS {
            v.push(0);
        }
        v
    }

    #[test]
    fn a_disabled_cache_never_matches() {
        let mut c = PromptCache::disabled();
        assert!(!c.is_enabled());
        assert_eq!(c.longest_prefix(&[1, 2, 3], false), None);
        c.insert(vec![1, 2, 3], vec![9, 9], None);
        assert!(c.is_empty(), "a disabled cache must not accumulate");
    }

    #[test]
    fn a_continuation_matches_the_turn_before_it() {
        let mut c = PromptCache::new(3, 1_000);
        // Turn one.
        let turn_one = seq(&[1, 2, 3, 4]);
        c.insert(turn_one.clone(), vec![0; MIN_TOKENS], None);
        // Turn two: the same prompt, one turn longer.
        let mut turn_two = turn_one.clone();
        turn_two.extend([5, 6]);
        assert_eq!(c.longest_prefix(&turn_two, false), Some(turn_one.len()));
    }

    #[test]
    fn a_different_conversation_does_not_match() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS], None);
        // Same opening, different third token: not a prefix of what is cached.
        assert_eq!(c.longest_prefix(&seq(&[1, 2, 9, 9]), false), None);
    }

    #[test]
    fn a_drafting_snapshot_only_matches_a_drafting_lookup() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS], Some(vec![1; 8]));
        let mut asked = seq(&[1, 2, 3, 4]);
        asked.push(5);
        // Restoring this entry without putting the draft side back would leave
        // the head behind the target, so a plain lookup must not see it.
        assert_eq!(c.longest_prefix(&asked, false), None);
        assert_eq!(c.longest_prefix(&asked, true), Some(seq(&[1, 2, 3, 4]).len()));
    }

    #[test]
    fn a_plain_snapshot_only_matches_a_plain_lookup() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS], None);
        let mut asked = seq(&[1, 2, 3, 4]);
        asked.push(5);
        assert_eq!(c.longest_prefix(&asked, true), None);
        assert_eq!(c.longest_prefix(&asked, false), Some(seq(&[1, 2, 3, 4]).len()));
    }

    #[test]
    fn the_longest_matching_prefix_wins() {
        let mut c = PromptCache::new(3, 1_000);
        let short = seq(&[1, 2]);
        let mut long = seq(&[1, 2, 3, 4]);
        long.push(8);
        let state = vec![7u8; MIN_TOKENS];

        c.insert(short.clone(), vec![0; MIN_TOKENS], None);
        c.insert(long.clone(), state.clone(), None);

        let mut asked = long.clone();
        asked.push(5);
        assert_eq!(c.longest_prefix(&asked, false), Some(long.len()));
        assert_eq!(
            c.snapshot_for(long.len()).map(|e| e.state.as_slice()),
            Some(state.as_slice())
        );
    }

    #[test]
    fn a_prompt_longer_than_the_cache_cannot_match() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS], None);
        // What is remembered is longer than what was asked, so nothing applies.
        assert_eq!(c.longest_prefix(&[1, 2], false), None);
    }

    #[test]
    fn eviction_respects_both_limits() {
        let mut c = PromptCache::new(2, 100);
        for i in 0..4 {
            c.insert(vec![i; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        }
        assert_eq!(c.len(), 2, "entry cap must hold");

        // A budget smaller than one entry still holds one: evicting everything
        // would leave a cache that can never hit.
        let mut tight = PromptCache::new(8, MIN_TOKENS / 2);
        tight.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        assert_eq!(tight.len(), 1);
    }

    #[test]
    fn the_budget_is_bytes_of_state_not_tokens() {
        let mut c = PromptCache::new(8, 100);
        c.insert(vec![1; MIN_TOKENS], vec![0u8; 60], None);
        c.insert(vec![2; MIN_TOKENS], vec![0u8; 60], None);
        // Two entries hold 120 bytes of state against a 100-byte budget: the
        // oldest goes, whatever its token count.
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn the_draft_side_counts_against_the_budget() {
        let mut c = PromptCache::new(8, 100);
        // 60 target bytes plus 60 draft bytes is over a 100-byte budget, even
        // though neither side alone would be.
        c.insert(vec![1; MIN_TOKENS], vec![0u8; 60], Some(vec![0u8; 60]));
        c.insert(vec![2; MIN_TOKENS], vec![0u8; 60], Some(vec![0u8; 60]));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn reinserting_replaces_rather_than_duplicating() {
        let mut c = cache();
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn a_hit_moves_to_the_back_so_eviction_spares_it() {
        let mut c = PromptCache::new(2, 100);
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        c.insert(vec![2; MIN_TOKENS], vec![2; MIN_TOKENS], None);
        // Touch the older one.
        c.longest_prefix(&[1; MIN_TOKENS], false);
        c.insert(vec![3; MIN_TOKENS], vec![3; MIN_TOKENS], None);
        // The touched entry survives; the untouched one goes.
        assert_eq!(c.longest_prefix(&[1; MIN_TOKENS], false), Some(MIN_TOKENS));
        assert_eq!(c.longest_prefix(&[2; MIN_TOKENS], false), None);
    }

    #[test]
    fn clear_forgets_everything() {
        let mut c = cache();
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS], None);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.held_tokens(), 0);
    }
}
