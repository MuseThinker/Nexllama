/// Sign a backend update manifest in place.
///
/// ```powershell
/// cargo run -p nexllama-app --example sign-manifest -- manifest.json <private-hex>
/// ```
///
/// Writes the signature back into the manifest's `signature` field. The bytes
/// signed are the manifest's own payload — version, llama.cpp build and every
/// backend's URL and hash — so changing any of them afterwards invalidates it.
use ed25519_dalek::{Signer, SigningKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: sign-manifest <manifest.json> <private-hex>");
    let private_hex = args
        .next()
        .expect("usage: sign-manifest <manifest.json> <private-hex>");

    let mut key_bytes = [0u8; 32];
    hex::decode_to_slice(private_hex.trim(), &mut key_bytes)?;
    let signing = SigningKey::from_bytes(&key_bytes);

    let text = std::fs::read_to_string(&path)?;
    let mut manifest: serde_json::Value = serde_json::from_str(&text)?;

    // The same payload the app rebuilds when it verifies, so the two cannot
    // drift: see `manifest_payload` in crates/app/src/lib.rs.
    let payload = manifest_payload(&manifest);
    let signature = signing.sign(payload.as_bytes());

    manifest["signature"] = serde_json::Value::String(hex::encode(signature.to_bytes()));
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)?;
    println!("signed {path}");
    Ok(())
}

/// The bytes a signature covers, rebuilt from the parsed document.
///
/// Signing the parsed structure rather than the raw text keeps the check
/// independent of key order and formatting, which a text signature would not
/// survive a re-serialisation.
fn manifest_payload(m: &serde_json::Value) -> String {
    let mut parts = vec![
        format!(
            "version={}",
            m.get("version")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
        ),
        format!(
            "llama_cpp={}",
            m.get("llama_cpp")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
        ),
    ];
    let mut names: Vec<String> = m
        .get("backends")
        .and_then(|b| b.as_object())
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    names.sort();
    for name in names {
        let e = &m["backends"][&name];
        parts.push(format!(
            "backend={name};url={};sha256={}",
            e.get("url").and_then(|v| v.as_str()).unwrap_or_default(),
            e.get("sha256")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_ascii_lowercase()
        ));
    }
    parts.join("\n")
}
