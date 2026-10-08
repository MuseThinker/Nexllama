//! Print the family facts of a GGUF file.
//!
//! Handy when a model lands in the wrong class:
//!
//! ```text
//! cargo run -p gguf-probe --example probe -- path/to/model.gguf
//! ```

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: probe <model.gguf>");
        std::process::exit(2);
    };

    match gguf_probe::probe(&path) {
        Ok(p) => {
            println!("file         : {path}");
            println!("architecture : {:?}", p.architecture);
            println!("name         : {:?}", p.name);
            println!("rerank_heads : {:?}", p.rerank_heads);
            println!("pooling_type : {:?}", p.pooling_type);
            println!("class        : {}", p.classify().as_str());
        }
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }

    if std::env::args().any(|a| a == "--keys") {
        match gguf_probe::keys(&path) {
            Ok(keys) => {
                println!("\n--- keys ---");
                for key in keys {
                    println!("{key}");
                }
            }
            Err(e) => eprintln!("{path}: {e}"),
        }
    }
}
