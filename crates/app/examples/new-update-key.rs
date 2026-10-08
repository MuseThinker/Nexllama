/// Generate the Ed25519 keypair that signs backend update manifests.
///
/// ```powershell
/// cargo run -p nexllama-app --example new-update-key
/// ```
///
/// Put the public half in Settings → Backend updates (or in the
/// `NEXLLAMA_UPDATE_KEY` environment variable at build time). Keep the private
/// half somewhere the release script can reach and nowhere else — it is what
/// makes an update yours.
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

fn main() {
    let mut rng = OsRng;
    let signing = SigningKey::generate(&mut rng);
    println!(
        "public:  {}",
        hex::encode(signing.verifying_key().to_bytes())
    );
    println!("private: {}", hex::encode(signing.to_bytes()));
    eprintln!(
        "\npublic  -> Settings → Backend updates (or NEXLLAMA_UPDATE_KEY at build time)\n\
         private -> keep secret; sign-manifest needs it"
    );
}
