//! Whether the prompt prefix cache changes what the model actually says.
//!
//! A prefix cache that restores the wrong KV state does not fail — it answers
//! from the wrong point in the conversation, and the reply just quietly stops
//! making sense. Nothing returns an error, so the only way to catch it is to
//! run the same conversation twice, once with the cache on and once off, with
//! sampling made deterministic, and compare what came out.
//!
//! The two runs get their own engine, because the cache lives with the model.
//!
//! `--mtp=on` repeats the check with the bundled MTP head drafting. A drafting
//! turn remembers the draft side of its cache alongside the target's, and that
//! pair is what this is really here to check: a restored target state with a
//! stale draft head answers from the wrong point just as quietly.
//!
//! usage: prefix_check <model.gguf> [n_gpu_layers] [--mtp=on|off]

use infer_core::{ChatMsg, Engine, LoadConfig, SamplerParams};

/// Greedy decoding.
///
/// Any randomness would make the two runs differ for reasons that have nothing
/// to do with the cache, and there would be nothing left to compare.
fn params(speculative: bool) -> SamplerParams {
    SamplerParams {
        temperature: 0.0,
        max_tokens: 24,
        speculative,
        // Straight answers, no reasoning block: the second turn has to feed this
        // reply back as an assistant message verbatim, and a template that drops
        // `<think>…</think>` from history would stop the prompt from being a
        // continuation of the cache.
        thinking: Some(false),
        ..Default::default()
    }
}

/// The first turn: a long shared system prompt and one question.
fn first_turn() -> Vec<ChatMsg> {
    let system = "You are a careful and literal assistant. ".repeat(60);
    vec![
        ChatMsg::new("system", &system),
        ChatMsg::new("user", "Name one property of water."),
    ]
}

/// One pass over the conversation. Returns the replies and how long each took
/// to encode, which is where the cache should show up.
fn run(
    path: &str,
    n_gpu_layers: u32,
    prefix_cache: bool,
    mtp: bool,
) -> anyhow::Result<(Vec<String>, Vec<f64>)> {
    let mut cfg = LoadConfig::new(path);
    cfg.n_gpu_layers = n_gpu_layers;
    cfg.prefix_cache = prefix_cache;
    cfg.prefix_cache_entries = 4;
    cfg.prefix_cache_max_mb = 512;
    // Only a fixed draft length, and no adaptive gate: the acceptance rate must
    // not depend on which run this is, or the two would differ for a reason that
    // has nothing to do with the cache.
    cfg.adaptive_draft = false;
    cfg.load_mtp = mtp;
    cfg.mtp_n_max = if mtp { 2 } else { 0 };

    let engine = Engine::load(cfg, |_| {})?;
    let mut replies = Vec::new();
    let mut prompt_ms = Vec::new();

    let first = first_turn();
    let mut text = String::new();
    // Only the encode is measured. Loading the model is a fixed cost, and
    // including it would bury the difference being looked for.
    let stats = engine.generate(&first, &params(mtp), |t| {
        text.push_str(t);
        true
    })?;
    prompt_ms.push(stats.prompt_ms);
    replies.push(text.clone());

    // The second turn carries the first reply back as an assistant message, so
    // its prompt is the first turn's prompt plus that reply plus the new user
    // message — exactly the shape every chat turn takes, and the shape live
    // reuse is built for.
    let mut head = first;
    head.push(ChatMsg::new("assistant", &text));
    head.push(ChatMsg::new("user", "Name one more property."));
    let mut text2 = String::new();
    let stats = engine.generate(&head, &params(mtp), |t| {
        text2.push_str(t);
        true
    })?;
    prompt_ms.push(stats.prompt_ms);
    replies.push(text2);

    Ok((replies, prompt_ms))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: prefix_check <model.gguf> [n_gpu_layers] [--mtp=on|off]");
    let n_gpu_layers: u32 = args
        .next()
        .filter(|s| !s.starts_with("--"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mtp = args.any(|a| a == "--mtp=on" || a == "--mtp");

    let (without, off_prompt) = run(&path, n_gpu_layers, false, mtp)?;
    let (with, on_prompt) = run(&path, n_gpu_layers, true, mtp)?;

    println!("encode time per turn (ms), off vs on (mtp={mtp})");
    for (i, (off, on)) in off_prompt.iter().zip(on_prompt.iter()).enumerate() {
        let change = if *off > 0.0 {
            (*on - *off) / *off * 100.0
        } else {
            0.0
        };
        println!("  turn {i}: {off:8.1} -> {on:8.1}  ({change:+.0}%)");
    }

    let mut ok = true;
    for (i, (a, b)) in without.iter().zip(with.iter()).enumerate() {
        if a == b {
            println!("turn {i}: identical — {:?}", a.trim());
        } else {
            ok = false;
            println!("turn {i}: MISMATCH\n  without: {a:?}\n  with:    {b:?}");
        }
    }

    if !ok {
        anyhow::bail!("the prefix cache changed what the model said");
    }
    println!("ok: identical replies, so the restored state is the right state");
    Ok(())
}
