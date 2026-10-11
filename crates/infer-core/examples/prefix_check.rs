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
//! usage: prefix_check <model.gguf> [n_gpu_layers]

use infer_core::{ChatMsg, Engine, LoadConfig, SamplerParams};

/// Greedy decoding.
///
/// Any randomness would make the two runs differ for reasons that have nothing
/// to do with the cache, and there would be nothing left to compare.
fn params() -> SamplerParams {
    SamplerParams {
        temperature: 0.0,
        max_tokens: 24,
        ..Default::default()
    }
}

/// A two-turn conversation under a long system prompt.
///
/// The system prompt is what makes the saving visible: it is shared by both
/// turns, and encoding it is most of the wait before the first token. With a
/// short prompt the cache would be correct but there would be nothing to see.
///
/// Turn two contains turn one verbatim as its prefix, which is exactly the shape
/// the cache exists for — and the shape where being wrong is invisible unless
/// it is checked.
fn conversation() -> Vec<Vec<ChatMsg>> {
    let system = "You are a careful and literal assistant. ".repeat(60);
    vec![
        vec![
            ChatMsg::new("system", &system),
            ChatMsg::new("user", "Name one property of water."),
        ],
        vec![
            ChatMsg::new("system", &system),
            ChatMsg::new("user", "Name one property of water."),
            ChatMsg::new("assistant", "It is liquid at room temperature."),
            ChatMsg::new("user", "Name one more property."),
        ],
    ]
}

/// One pass over the conversation. Returns the replies and how long each took
/// to encode, which is where the cache should show up.
fn run(
    path: &str,
    n_gpu_layers: u32,
    prefix_cache: bool,
) -> anyhow::Result<(Vec<String>, Vec<f64>)> {
    let mut cfg = LoadConfig::new(path);
    cfg.n_gpu_layers = n_gpu_layers;
    cfg.prefix_cache = prefix_cache;
    cfg.prefix_cache_entries = 4;
    cfg.prefix_cache_max_mb = 512;
    // Drafting is off: it changes what gets sampled, and this is checking
    // whether the *cache* changes what gets sampled.
    cfg.adaptive_draft = false;

    let engine = Engine::load(cfg, |_| {})?;
    let mut replies = Vec::new();
    let mut prompt_ms = Vec::new();
    for turn in conversation() {
        let mut text = String::new();
        // Only the encode is measured. Loading the model is a fixed cost, and
        // including it would bury the difference being looked for.
        let stats = engine.generate(&turn, &params(), |t| {
            text.push_str(t);
            true
        })?;
        prompt_ms.push(stats.prompt_ms);
        replies.push(text);
    }
    Ok((replies, prompt_ms))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: prefix_check <model.gguf> [n_gpu_layers]");
    let n_gpu_layers: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let (without, off_prompt) = run(&path, n_gpu_layers, false)?;
    let (with, on_prompt) = run(&path, n_gpu_layers, true)?;

    println!("encode time per turn (ms), off vs on");
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
