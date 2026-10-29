fn main() {
    // The updater target differs between the full and lite installer builds.
    println!("cargo:rerun-if-env-changed=NEXLLAMA_EDITION");
    tauri_build::build()
}
