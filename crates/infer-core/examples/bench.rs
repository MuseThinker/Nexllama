use std::time::Instant;

use infer_core::{ChatMsg, Engine, LoadConfig, SamplerParams};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: bench <model.gguf> [n_gpu_layers] [prompt]");
    let n_gpu_layers: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let prompt = args
        .next()
        .unwrap_or_else(|| "Write one sentence about the sea.".to_string());

    let mut cfg = LoadConfig::new(&path);
    cfg.n_gpu_layers = n_gpu_layers;

    let t_load = Instant::now();
    let engine = Engine::load(cfg, |ev| eprintln!("[load] {ev:?}"))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    eprintln!("[load] finished in {load_ms:.0} ms");
    eprintln!("[info] {:?}", engine.info());

    let msgs = vec![ChatMsg::new("user", prompt)];
    let params = SamplerParams {
        max_tokens: 128,
        ..Default::default()
    };

    let stats = engine.generate(&msgs, &params, |t| {
        print!("{t}");
        // One write per token, and stdout is line-buffered: without the flush
        // the reply arrives in one lump at the end, which is the opposite of
        // what a streaming benchmark is meant to show.
        let _ = std::io::Write::flush(&mut std::io::stdout());
        true
    })?;
    println!();

    eprintln!(
        "prompt {} tok / {:.0} ms ({:.1} tok/s) | gen {} tok / {:.0} ms ({:.1} tok/s) | TTFT {:.0} ms",
        stats.prompt_tokens,
        stats.prompt_ms,
        stats.prompt_tokens_per_second(),
        stats.generated_tokens,
        stats.gen_ms,
        stats.tokens_per_second(),
        stats.first_token_ms,
    );

    Ok(())
}
