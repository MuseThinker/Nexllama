//! Transcribe one audio file with an ASR bundle.
//!
//! Answers the question the design depends on: does the audio projector load
//! and does the transcript come out right on this machine?
//!
//! ```text
//! cargo run --release -p infer-core --example transcribe_check -- \
//!     models/asr/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf \
//!     models/asr/qwen3-asr-0.6b/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
//!     speech.wav --lang=Chinese
//! ```
//!
//! `--lang` is the model's own canonical name (`Chinese`, `English`, …);
//! omitting it lets the model detect the language. `--gpu=0` forces the CPU,
//! which is the interesting number for a laptop without a discrete card.

use std::time::Instant;

use infer_core::{AsrOptions, Engine, LoadConfig};

struct Opts {
    model: String,
    mmproj: String,
    audio: String,
    language: Option<String>,
    gpu_layers: u32,
    n_ctx: u32,
}

fn parse() -> Opts {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: transcribe_check <model.gguf> <mmproj.gguf> <audio.wav|mp3|flac> [--lang=Chinese] [--gpu=N] [--ctx=N]");
    let mmproj = args.next().expect("missing mmproj");
    let audio = args.next().expect("missing audio file");
    let mut o = Opts {
        model,
        mmproj,
        audio,
        language: None,
        gpu_layers: 999,
        n_ctx: 8192,
    };
    for arg in args {
        if let Some(v) = arg.strip_prefix("--lang=") {
            o.language = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--gpu=") {
            o.gpu_layers = v.parse().expect("--gpu needs a number");
        } else if let Some(v) = arg.strip_prefix("--ctx=") {
            o.n_ctx = v.parse().expect("--ctx needs a number");
        } else {
            eprintln!("ignoring unknown argument {arg}");
        }
    }
    o
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let o = parse();
    let audio = std::fs::read(&o.audio)?;
    println!(
        "audio: {} ({:.1} KiB)",
        o.audio,
        audio.len() as f64 / 1024.0
    );

    let mut cfg = LoadConfig::new(&o.model);
    cfg.mmproj = Some(o.mmproj.clone().into());
    cfg.n_gpu_layers = o.gpu_layers;
    cfg.n_ctx = o.n_ctx;

    let t_load = Instant::now();
    let engine = Engine::load(cfg, |ev| {
        if let infer_core::LoadEvent::Phase(p) = ev {
            eprintln!("load: {p}");
        }
    })?;
    println!(
        "loaded in {:.1}s: class={} vision={} audio={} ctx={}",
        t_load.elapsed().as_secs_f64(),
        engine.info().model_class,
        engine.info().vision,
        engine.info().audio,
        engine.info().n_ctx,
    );
    if !engine.info().audio {
        return Err("the projector reports no audio encoder".into());
    }

    let t = Instant::now();
    let first = engine.transcribe(
        audio.clone(),
        AsrOptions {
            language: o.language.clone(),
            ..AsrOptions::default()
        },
    )?;
    let first_wall = t.elapsed().as_secs_f64();

    // The first call pays for the audio encoder's graph capture; the second is
    // the number that matters for a transcription queue.
    let t = Instant::now();
    let result = engine.transcribe(
        audio,
        AsrOptions {
            language: o.language.clone(),
            ..AsrOptions::default()
        },
    )?;
    let wall = t.elapsed().as_secs_f64();

    println!("first call: {first_wall:.2}s -> {:?}", first.language);
    println!("language: {:?}", result.language);
    println!("text: {}", result.text);
    println!(
        "generated {} tokens in {:.2}s ({:.1} tok/s), first token {:.0} ms",
        result.stats.generated_tokens,
        wall,
        result.stats.tokens_per_second(),
        result.stats.first_token_ms,
    );
    println!(
        "prompt: {} tokens in {:.0} ms",
        result.stats.prompt_tokens, result.stats.prompt_ms
    );
    Ok(())
}
