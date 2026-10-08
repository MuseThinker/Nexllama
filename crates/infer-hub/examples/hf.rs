use std::io::Write;

use infer_hub::HubClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: hf <search|list|download> ...");
        std::process::exit(2);
    }

    let client = HubClient::new();
    eprintln!("endpoint: {}", client.endpoint());

    match args[0].as_str() {
        "search" => {
            let query = args.get(1).map(|s| s.as_str()).unwrap_or("");
            let limit = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            for m in client.search(query, limit).await? {
                println!("{:<58} {:>12} dl {:>7} likes", m.id, m.downloads, m.likes);
            }
        }
        "list" => {
            let repo = args.get(1).expect("usage: hf list <repo>");
            for f in client.list_gguf(repo).await? {
                println!("{:<58} {:>9.1} MB", f.filename, f.size as f64 / 1e6);
            }
        }
        "download" => {
            let repo = args
                .get(1)
                .expect("usage: hf download <repo> <filename> [dir]");
            let file = args
                .get(2)
                .expect("usage: hf download <repo> <filename> [dir]");
            let dir = args.get(3).map(|s| s.as_str()).unwrap_or("models");
            let mut last_bucket = u64::MAX;
            let path = client
                .download(repo, file, dir, |p| {
                    let bucket = p.downloaded / 10_000_000;
                    if bucket != last_bucket {
                        last_bucket = bucket;
                        let pct = p
                            .percent()
                            .map(|v| format!("{v:.1}%"))
                            .unwrap_or_else(|| "?".into());
                        let total = p.total.map(|t| t / 1_000_000).unwrap_or(0);
                        eprint!("\r{pct:>7}  {}/{} MB", p.downloaded / 1_000_000, total);
                        let _ = std::io::stderr().flush();
                    }
                })
                .await?;
            eprintln!();
            println!("{}", path.display());
        }
        other => {
            eprintln!("unknown command: {other}");
            std::process::exit(2);
        }
    }

    Ok(())
}
