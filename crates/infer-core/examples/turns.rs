//! Per-turn cost of a growing conversation, with the optimisations switched
//! one at a time.
//!
//! A single-prompt benchmark cannot see the two things that make a chat feel
//! slow: the answer of the previous turn has to be encoded again, and the prefix
//! cache pays a copy of the KV state every turn. Native llama.cpp does neither —
//! it simply keeps the sequence it already has. This runs the same conversation
//! several times over with the switches moved, so the difference can be measured
//! on the machine and model that actually matter, rather than argued about.
//!
//! ```text
//! cargo run --release -p infer-core --example turns -- \
//!     D:\models\model.gguf 999 4 --prefix=off --mtp=off
//! ```
//!
//! Read the `prompt` and `TTFT` columns. `prompt` is what a turn pays to put the
//! conversation into the cache; `TTFT` is what the user waits before the first
//! word. `draft` is the speculative acceptance rate, as drafted/kept tokens.

use std::time::Instant;

use infer_core::{ChatMsg, Engine, FlashAttn, LoadConfig, SamplerParams};

struct Opts {
    path: String,
    n_gpu_layers: u32,
    turns: usize,
    n_ctx: u32,
    prefix: bool,
    mtp: bool,
    mtp_n_max: u32,
    thinking: bool,
    speculative: bool,
    draft_model: Option<String>,
    draft_n_max: u32,
}

fn parse() -> Opts {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: turns <model.gguf> [n_gpu_layers] [turns] [--prefix=on|off] [--mtp=on|off] [--draft=path] [--ctx=N] [--thinking=on|off] [--spec=on|off]");
    // `take_while` swallows the first item that fails its predicate, which ate
    // the leading `--ctx=` of every run. Partitioning the whole tail instead
    // keeps the flags intact.
    let rest: Vec<String> = args.collect();
    let (positional, flags): (Vec<String>, Vec<String>) =
        rest.into_iter().partition(|a| !a.starts_with("--"));
    let n_gpu_layers = positional.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let turns = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut o = Opts {
        path,
        n_gpu_layers,
        turns,
        n_ctx: 8192,
        prefix: true,
        mtp: false,
        mtp_n_max: 0,
        thinking: false,
        speculative: false,
        draft_model: None,
        draft_n_max: 0,
    };

    for a in flags {
        let Some(body) = a.strip_prefix("--") else {
            continue;
        };
        let (k, v) = match body.split_once('=') {
            Some((k, v)) => (k, v),
            None => {
                eprintln!("[warn] expected --{body}=on|off, ignored");
                continue;
            }
        };
        match k {
            "prefix" => o.prefix = on(v, "--prefix"),
            "mtp" => {
                o.mtp = on(v, "--mtp");
                // An MTP head with no draft length drafts nothing: llama.cpp
                // needs `n_max >= 1` to propose at all, so an opted-in head gets
                // the engine's own default rather than silently no-ops.
                o.mtp_n_max = if o.mtp { 2 } else { 0 };
            }
            "mtp-n-max" => o.mtp_n_max = v.parse().unwrap_or(2),
            "draft" => o.draft_model = (!v.is_empty()).then(|| v.to_string()),
            "draft-n-max" => o.draft_n_max = v.parse().unwrap_or(0),
            "ctx" => o.n_ctx = v.parse().unwrap_or(8192),
            "thinking" => o.thinking = on(v, "--thinking"),
            "spec" => o.speculative = on(v, "--spec"),
            other => eprintln!("[warn] unknown flag --{other}"),
        }
    }
    o
}

/// `on|true|1` and `off|false|0`, with the flag named in the complaint.
fn on(v: &str, flag: &str) -> bool {
    match v {
        "on" | "true" | "1" => true,
        "off" | "false" | "0" => false,
        other => {
            eprintln!("[warn] {flag}={other} is neither on nor off; using off");
            false
        }
    }
}

/// A question long enough to be worth caching, and one reply per turn.
fn user_line(i: usize) -> String {
    format!(
        "Turn {i}: keep the running notes. We are writing a short log of a sea \
         voyage; reply with one sentence that adds to what has already been said."
    )
}

fn main() -> anyhow::Result<()> {
    let o = parse();

    let mut cfg = LoadConfig::new(&o.path);
    cfg.n_gpu_layers = o.n_gpu_layers;
    cfg.n_ctx = o.n_ctx;
    cfg.prefix_cache = o.prefix;
    cfg.flash_attn = FlashAttn::Auto;
    cfg.load_mtp = o.mtp;
    cfg.mtp_n_max = o.mtp_n_max;
    cfg.draft_model = o.draft_model.as_deref().map(std::path::PathBuf::from);
    cfg.draft_n_max = o.draft_n_max;

    eprintln!(
        "[cfg] prefix={} mtp={} (n_max={}) spec={} thinking={} ctx={} ngl={}",
        o.prefix, o.mtp, o.mtp_n_max, o.speculative, o.thinking, o.n_ctx, o.n_gpu_layers
    );

    let t_load = Instant::now();
    let engine = Engine::load(cfg, |ev| eprintln!("[load] {ev:?}"))?;
    eprintln!(
        "[load] {:.0} s | {:?}",
        t_load.elapsed().as_secs_f64(),
        engine.info().model_size
    );

    let mut msgs: Vec<ChatMsg> = vec![ChatMsg::new("system", "You keep a running log. Be brief.")];
    let params = SamplerParams {
        max_tokens: 96,
        thinking: Some(o.thinking),
        speculative: o.speculative,
        temperature: 0.0,
        ..Default::default()
    };

    for i in 0..o.turns {
        msgs.push(ChatMsg::new("user", user_line(i + 1)));
        let mut reply = String::new();
        let stats = engine.generate(&msgs, &params, |t| {
            reply.push_str(t);
            true
        })?;
        msgs.push(ChatMsg::new("assistant", reply));

        let draft = if stats.draft_tokens > 0 {
            format!(
                " | draft {}/{}",
                stats.accepted_draft_tokens, stats.draft_tokens
            )
        } else {
            String::new()
        };
        // The number to watch across runs: `prompt` tokens grow every turn, and
        // `prompt ms` says what that costs with the switches as set. Two runs
        // with `--prefix=off` and `--prefix=on` differ only by the KV copy.
        println!(
            "turn {}: prompt {} tok / {:.0} ms ({:.1} tok/s) | TTFT {:.0} ms | gen {:.1} tok/s{}",
            i + 1,
            stats.prompt_tokens,
            stats.prompt_ms,
            stats.prompt_tokens.saturating_mul(1000) as f64 / (stats.prompt_ms.max(1.0)),
            stats.first_token_ms,
            stats.tokens_per_second(),
            draft,
        );
    }

    Ok(())
}
