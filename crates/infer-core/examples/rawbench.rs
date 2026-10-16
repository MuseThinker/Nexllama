//! A decode loop with nothing around it, to separate the bundled llama.cpp
//! from the engine that drives it.
//!
//! `turns` measures what a chat turn actually costs, but it goes through
//! `Engine` — prompt templates, the sampler chain, detokenising and streaming
//! every token. When its numbers disagree with `llama-bench` on the same
//! machine, the disagreer has to be found: is the bundled llama.cpp slow, or
//! is the engine on top of it? This decodes with the same context parameters
//! and the same greedy sampler and does nothing else, so a number close to
//! `llama-bench` clears the core and points at the engine.
//!
//! ```text
//! cargo run --release -p infer-core --example rawbench -- model.gguf 128 4096
//! ```

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use llama_cpp_4::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: rawbench <model.gguf> [n_gen] [ctx]");
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let n_ctx: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(
        &backend,
        &path,
        &LlamaModelParams::default().with_n_gpu_layers(999),
    )?;

    // The same numbers the engine's LoadConfig defaults to, so a difference is
    // the driver and not the context.
    let mut cp = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(2048)
        .with_n_ubatch(512);
    if let Some(t) = std::env::var("RAW_THREADS").ok().and_then(|s| s.parse::<i32>().ok()) {
        if t > 0 {
            cp = cp.with_n_threads(t).with_n_threads_batch(t);
        }
    }
    let mut ctx = model.new_context(&backend, cp)?;

    let pp_target: usize = std::env::var("RAW_PP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let base = "The history of the sea is long and full of tales. Please continue, at length: ";
    let mut prompt = String::from(base);
    if pp_target > 0 {
        while model.str_to_token(&prompt, AddBos::Always)?.len() < pp_target {
            prompt.push_str(base);
        }
    }
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let n_prompt = tokens.len();

    let mut batch = LlamaBatch::new(n_prompt.max(1), 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], i == n_prompt - 1)?;
    }
    let pp_start = Instant::now();
    ctx.decode(&mut batch)?;
    let pp_secs = pp_start.elapsed().as_secs_f64();
    println!(
        "rawbench: pp {n_prompt} tok in {pp_secs:.3}s = {:.1} tok/s",
        n_prompt as f64 / pp_secs
    );

    let sampler = LlamaSampler::greedy();
    let mut pos = n_prompt as i32;

    // Prime the first token outside the timed region: it is sampled from the
    // prompt's logits, not from a decode of this loop's own.
    let mut tok = sampler.sample(&ctx, -1);
    batch.clear();
    batch.add(tok, pos, &[0], true)?;
    pos += 1;
    ctx.decode(&mut batch)?;

    let no_logits = std::env::var("RAW_NOLOGITS").is_ok();
    let start = Instant::now();
    let mut generated = 0usize;
    let mut t_sample = Duration::ZERO;
    let mut t_batch = Duration::ZERO;
    let mut t_decode = Duration::ZERO;
    for _ in 0..n {
        if no_logits {
            generated += 1;
            let t = Instant::now();
            batch.clear();
            batch.add(tok, pos, &[0], false)?;
            pos += 1;
            ctx.decode(&mut batch)?;
            t_decode += t.elapsed();
            continue;
        }
        let t = Instant::now();
        tok = sampler.sample(&ctx, -1);
        t_sample += t.elapsed();
        if model.is_eog_token(tok) {
            break;
        }
        generated += 1;
        let t = Instant::now();
        batch.clear();
        batch.add(tok, pos, &[0], true)?;
        pos += 1;
        t_batch += t.elapsed();
        let t = Instant::now();
        ctx.decode(&mut batch)?;
        t_decode += t.elapsed();
    }
    let secs = start.elapsed().as_secs_f64();

    println!(
        "rawbench: gen {generated} tok in {secs:.3}s = {:.1} tok/s (ctx {n_ctx}, prompt {n_prompt} tok)",
        generated as f64 / secs
    );
    let per = |d: Duration| d.as_secs_f64() * 1000.0 / generated as f64;
    println!(
        "rawbench: per-token ms: sample {:.3}, batch {:.3}, decode {:.3}",
        per(t_sample),
        per(t_batch),
        per(t_decode)
    );
    Ok(())
}
