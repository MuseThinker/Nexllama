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
//! Nothing here is enabled by default — see `LoadConfig::prefix_cache`.

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
}

/// Which remembered prefixes apply, and which to keep.
#[derive(Debug)]
pub struct PromptCache {
    /// Most entries to hold. Zero disables the cache outright.
    max_entries: usize,
    /// Ceiling on the total tokens held, which is what bounds the memory.
    max_tokens: usize,
    /// Least-recently-used first.
    entries: Vec<Cached>,
}

impl PromptCache {
    pub fn new(max_entries: usize, max_tokens: usize) -> Self {
        Self {
            max_entries,
            max_tokens,
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
    pub fn longest_prefix(&mut self, tokens: &[i32]) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.tokens.len() > tokens.len() {
                continue;
            }
            if tokens[..e.tokens.len()] != e.tokens[..] {
                continue;
            }
            if best.map_or(true, |(len, _)| e.tokens.len() > len) {
                best = Some((e.tokens.len(), i));
            }
        }
        let (len, i) = best?;
        // A hit is a use: move it to the back so eviction spares it.
        let entry = self.entries.remove(i);
        self.entries.push(entry);
        Some(len)
    }

    /// Take the state belonging to the prompt just matched, if there is one.
    ///
    /// Callers call this right after [`Self::longest_prefix`] and before
    /// encoding anything, so the entry is the one at the back.
    pub fn state_for(&self, len: usize) -> Option<&[u8]> {
        self.entries
            .last()
            .filter(|e| e.tokens.len() == len)
            .map(|e| e.state.as_slice())
    }

    /// Remember a prompt and its state.
    pub fn insert(&mut self, tokens: Vec<i32>, state: Vec<u8>) {
        if !self.is_enabled() || tokens.len() < MIN_TOKENS || state.is_empty() {
            return;
        }
        // Re-remembering the same prompt replaces it rather than duplicating.
        self.entries.retain(|e| e.tokens != tokens);

        self.entries.push(Cached { tokens, state });

        while self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
        // Stops at one entry, however far over budget that entry is. Evicting
        // everything would leave a cache that can never hit, and a prompt
        // bigger than the whole budget is better served by caching it slowly
        // than by never caching it at all.
        while self.entries.len() > 1
            && self.entries.iter().map(|e| e.tokens.len()).sum::<usize>() > self.max_tokens
        {
            self.entries.remove(0);
        }
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
        assert_eq!(c.longest_prefix(&[1, 2, 3]), None);
        c.insert(vec![1, 2, 3], vec![9, 9]);
        assert!(c.is_empty(), "a disabled cache must not accumulate");
    }

    #[test]
    fn a_continuation_matches_the_turn_before_it() {
        let mut c = PromptCache::new(3, 1_000);
        // Turn one.
        let turn_one = seq(&[1, 2, 3, 4]);
        c.insert(turn_one.clone(), vec![0; MIN_TOKENS]);
        // Turn two: the same prompt, one turn longer.
        let mut turn_two = turn_one.clone();
        turn_two.extend([5, 6]);
        assert_eq!(c.longest_prefix(&turn_two), Some(turn_one.len()));
    }

    #[test]
    fn a_different_conversation_does_not_match() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS]);
        // Same opening, different third token: not a prefix of what is cached.
        assert_eq!(c.longest_prefix(&seq(&[1, 2, 9, 9])), None);
    }

    #[test]
    fn the_longest_matching_prefix_wins() {
        let mut c = PromptCache::new(3, 1_000);
        let short = seq(&[1, 2]);
        let mut long = seq(&[1, 2, 3, 4]);
        long.push(8);
        let state = vec![7u8; MIN_TOKENS];

        c.insert(short.clone(), vec![0; MIN_TOKENS]);
        c.insert(long.clone(), state.clone());

        let mut asked = long.clone();
        asked.push(5);
        assert_eq!(c.longest_prefix(&asked), Some(long.len()));
        assert_eq!(c.state_for(long.len()), Some(state.as_slice()));
    }

    #[test]
    fn a_prompt_longer_than_the_cache_cannot_match() {
        let mut c = PromptCache::new(3, 1_000);
        c.insert(seq(&[1, 2, 3, 4]), vec![0; MIN_TOKENS]);
        // What is remembered is longer than what was asked, so nothing applies.
        assert_eq!(c.longest_prefix(&[1, 2]), None);
    }

    #[test]
    fn eviction_respects_both_limits() {
        let mut c = PromptCache::new(2, 100);
        for i in 0..4 {
            c.insert(vec![i; MIN_TOKENS], vec![1; MIN_TOKENS]);
        }
        assert_eq!(c.len(), 2, "entry cap must hold");

        // A budget smaller than one entry still holds one: evicting everything
        // would leave a cache that can never hit.
        let mut tight = PromptCache::new(8, MIN_TOKENS / 2);
        tight.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS]);
        assert_eq!(tight.len(), 1);
    }

    #[test]
    fn reinserting_replaces_rather_than_duplicating() {
        let mut c = cache();
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS]);
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS]);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn a_hit_moves_to_the_back_so_eviction_spares_it() {
        let mut c = PromptCache::new(2, 100);
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS]);
        c.insert(vec![2; MIN_TOKENS], vec![2; MIN_TOKENS]);
        // Touch the older one.
        c.longest_prefix(&[1; MIN_TOKENS]);
        c.insert(vec![3; MIN_TOKENS], vec![3; MIN_TOKENS]);
        // The touched entry survives; the untouched one goes.
        assert_eq!(c.longest_prefix(&[1; MIN_TOKENS]), Some(MIN_TOKENS));
        assert_eq!(c.longest_prefix(&[2; MIN_TOKENS]), None);
    }

    #[test]
    fn clear_forgets_everything() {
        let mut c = cache();
        c.insert(vec![1; MIN_TOKENS], vec![1; MIN_TOKENS]);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.held_tokens(), 0);
    }
}
