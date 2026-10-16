//! Which GGUF files belong together.
//!
//! The desktop library and the daemon's `GET /admin/models` ask the same
//! question — a folder of weights with a projector and a drafter beside them is
//! one model, a folder of unrelated files is several — and they have to answer
//! it identically. The rules live here so they are written once, and so a head
//! that reads the models folder without a GUI (a container, a test) gets the
//! same list the app shows.
//!
//! Every row is a `serde_json::Value`: these are shapes the frontend reads, and
//! renaming a field is an API change rather than a refactor.

use std::path::Path;

/// The role a file plays in its folder, from its name alone.
pub fn role_of(filename: &str) -> &'static str {
    let n = filename.to_ascii_lowercase();
    if n.contains("mmproj") || n.contains("projector") {
        "projector"
    } else if n.contains("draft")
        || n.contains("eagle")
        || n.contains("mtp")
        || n.contains("dflash")
    {
        "draft"
    } else if n.contains("lora") || n.contains("adapter") {
        "lora"
    } else if n.contains("vocoder") {
        "vocoder"
    } else {
        "main"
    }
}

/// A short label for the drafter a side-mounted file implements, from its name.
///
/// The engine makes the same call from the checkpoint header when it loads;
/// this is the list's guess from the filename, so the choice is readable
/// before anything has been loaded.
pub fn draft_kind_label(filename: &str) -> &'static str {
    let n = filename.to_ascii_lowercase();
    if n.contains("dflash") {
        "DFlash"
    } else if n.contains("eagle") {
        "EAGLE-3"
    } else if n.contains("mtp") || n.contains("nextn") {
        "MTP"
    } else {
        "draft"
    }
}

/// Whether a directory name marks it as a drafter's folder.
///
/// A HuggingFace download of an MTP head (or an EAGLE / DFlash checkpoint)
/// usually lands in its own folder named after what it is — `MTP/` — sitting
/// beside the weights it drafts for. That folder belongs to the model next to
/// it, not to a model of its own.
pub fn is_drafter_dir(name: &str) -> bool {
    role_of(name) == "draft"
}

/// The row's name is what a file is called without its extension.
pub fn stem_of(filename: Option<&str>) -> String {
    let filename = filename.unwrap_or("model");
    Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
        .to_string()
}

/// Sorting by filename rather than by whatever `read_dir` felt like, so the
/// same folder keeps yielding the same list.
pub fn sort_files(files: &mut [serde_json::Value]) {
    files.sort_by(|a, b| {
        a["filename"]
            .as_str()
            .unwrap_or("")
            .cmp(b["filename"].as_str().unwrap_or(""))
    });
}

/// The GGUF files sitting directly in `dir`.
///
/// Deliberately one level. A model folder is a model: weights with its
/// projector and draft beside them, and nothing else. Descending further makes a
/// library whose entries cannot be told apart — two folders of files from
/// different repos merge into one list, and a stray backup copy becomes another
/// candidate for loading. A file in a subfolder is reached by opening that
/// folder instead.
pub fn collect_gguf_files(dir: &Path, base: &Path, out: &mut Vec<serde_json::Value>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // The one exception, and it is a naming convention rather than a
            // nesting level: a folder called `MTP/` beside the weights holds
            // that model's drafter, not a model of its own.
            let is_drafter = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_drafter_dir);
            if is_drafter {
                collect_gguf_files(&path, base, out);
            }
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        let rel = path.strip_prefix(base).unwrap_or(&path);
        let filename = rel.to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(serde_json::json!({
            "filename": filename,
            "role": role_of(&filename),
            "size": size,
            "path": path.to_string_lossy(),
        }));
    }
}

/// One row in the library: a name, the files that belong to it, and the picks
/// the load button needs (main weights, projector, draft, LoRAs).
pub fn bundle(name: &str, dir: &str, files: Vec<serde_json::Value>) -> serde_json::Value {
    let pick = |role: &str| {
        files
            .iter()
            .find(|f| f["role"] == role)
            .map(|f| f["path"].clone())
    };
    // The name is a hint; the header is the verdict. Main weights get named
    // after the family they serve — "…-MTP", "…-EAGLE" — exactly the way draft
    // heads are, so a folder can hold two files whose names both say "draft"
    // while only one is a full model. Among the candidates, the largest one
    // whose header can actually be read is the main file: a stripped head is
    // never the biggest GGUF in its own folder, and loading the second-biggest
    // file as the model is the one outcome worse than offering nothing.
    let main = {
        let mut candidates: Vec<&serde_json::Value> = files
            .iter()
            .filter(|f| matches!(f["role"].as_str(), Some("main") | Some("draft")))
            .collect();
        candidates.sort_by_key(|f| std::cmp::Reverse(f["size"].as_u64().unwrap_or(0)));
        candidates
            .iter()
            .find(|f| {
                f["path"]
                    .as_str()
                    .map(|p| gguf_probe::probe(p).is_ok())
                    .unwrap_or(false)
            })
            // Nothing could be read — a listing taken before the files existed,
            // or unreadable headers. Fall back to the largest file named as a
            // main model so a bundle still names a model rather than none.
            .or_else(|| candidates.iter().find(|f| f["role"] == "main"))
            .map(|f| f["path"].clone())
    };
    let loras: Vec<serde_json::Value> = files
        .iter()
        .filter(|f| f["role"] == "lora")
        .map(|f| f["path"].clone())
        .collect();
    let size: u64 = files.iter().filter_map(|f| f["size"].as_u64()).sum();
    // Read the family out of the main file's header so the library can group it
    // and chat can refuse the models that cannot chat.
    let probe = main
        .as_ref()
        .and_then(|p| p.as_str())
        .and_then(|p| gguf_probe::probe(p).ok());
    let class = probe
        .as_ref()
        .map(|p| p.classify().as_str().to_string())
        .unwrap_or_else(|| gguf_probe::ModelClass::Llm.as_str().to_string());
    let mtp = probe.as_ref().is_some_and(gguf_probe::Probe::has_mtp);
    serde_json::json!({
        "name": name,
        "dir": dir,
        "size": size,
        "main": main,
        "projector": pick("projector"),
        "draft": pick("draft"),
        "loras": loras,
        "class": class,
        "mtp": mtp,
        "draft_kind": pick("draft").and_then(|d| d.as_str().map(draft_kind_label)),
        "kind": if pick("projector").is_some() {
            "vision"
        } else if pick("draft").is_some() {
            "speculative"
        } else {
            class.as_str()
        },
        "files": files,
    })
}

/// Turns the loose `.gguf` files sitting in the models folder itself into rows.
///
/// Exactly one of them being the main file means the rest are that model's own
/// files, so they belong in one row. Two or more main files means a folder of
/// unrelated models with no way to tell whose projector is whose, so each file
/// keeps a row of its own.
pub fn root_bundles(files: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let mut mains = files.iter().filter(|f| f["role"] == "main");
    let (Some(main), None) = (mains.next(), mains.next()) else {
        return files
            .into_iter()
            .map(|f| {
                let name = stem_of(f["filename"].as_str());
                bundle(&name, "", vec![f])
            })
            .collect();
    };
    let name = stem_of(main["filename"].as_str());
    vec![bundle(&name, "", files)]
}

/// The models in `dir`, one row per model, sorted by name.
///
/// Unreadable folders are skipped (a library of what can be read beats a
/// failure that shows nothing); an unreadable `dir` itself is an error, because
/// that is the configuration being wrong rather than a stray file in it.
pub fn scan(dir: &Path) -> Result<Vec<serde_json::Value>, String> {
    let mut bundles = Vec::new();
    let mut loose: Vec<serde_json::Value> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            let mut files = Vec::new();
            collect_gguf_files(&path, &path, &mut files);
            sort_files(&mut files);
            if files.is_empty() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model")
                .to_string();
            // A folder named for a drafter — `MTP/`, where a HuggingFace
            // download drops a model's MTP head — is that model's drafter, not
            // a model of its own. Its files join the root so they pair with the
            // weights sitting beside the folder, instead of showing up as a
            // second thing to "load".
            if is_drafter_dir(&name) {
                for mut f in files {
                    f["role"] = serde_json::json!("draft");
                    loose.push(f);
                }
            } else {
                bundles.push(bundle(&name, &name, files));
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model")
                .to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            loose.push(serde_json::json!({
                "filename": filename,
                "role": role_of(&filename),
                "size": size,
                "path": path.to_string_lossy(),
            }));
        }
    }
    // The folder the user pointed the app at is often a single model's own
    // folder — projector, draft checkpoint and bundled head beside the weights.
    // Listing those as five separate models asked which one to load, and every
    // answer except the weights was wrong.
    sort_files(&mut loose);
    bundles.extend(root_bundles(loose));
    bundles.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
    });
    Ok(bundles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_folder_names_are_recognised() {
        for name in ["MTP", "mtp", "draft", "EAGLE", "dflash", "DFlash2"] {
            assert!(
                is_drafter_dir(name),
                "{name} should read as a drafter folder"
            );
        }
        for name in ["models", "Qwen3.8-27B", "bge-m3"] {
            assert!(!is_drafter_dir(name), "{name} should not read as a drafter");
        }
    }

    /// One model plus its own files is one row in the library; a folder of
    /// unrelated models keeps one row each.
    #[test]
    fn loose_files_in_the_models_folder_form_one_row() {
        let entry = |filename: &str| {
            serde_json::json!({
                "filename": filename,
                "role": role_of(filename),
                "size": 1u64,
                "path": format!("C:/models/{filename}"),
            })
        };

        let one = root_bundles(vec![
            entry("mmproj-x.gguf"),
            entry("base-model.gguf"),
            entry("dflash-x.gguf"),
        ]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0]["name"], "base-model");
        assert_eq!(
            one[0]["main"].as_str().and_then(|p| p.rsplit('/').next()),
            Some("base-model.gguf")
        );
        assert!(one[0]["projector"].is_string());
        assert!(one[0]["draft"].is_string());
        assert_eq!(one[0]["files"].as_array().map(Vec::len), Some(3));

        let many = root_bundles(vec![entry("a-model.gguf"), entry("b-model.gguf")]);
        assert_eq!(many.len(), 2);
        assert_eq!(many[0]["name"], "a-model");
        assert_eq!(many[1]["name"], "b-model");
    }

    /// A subfolder named for a drafter is scanned as part of the models next to
    /// it; a subfolder of another model is not.
    #[test]
    fn scan_pairs_a_drafter_folder_with_the_root() {
        let root = std::env::temp_dir().join(format!("nexllama-scan-{}", std::process::id()));
        let mtp = root.join("MTP");
        let other = root.join("other-model");
        std::fs::create_dir_all(&mtp).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(root.join("weights.gguf"), b"not a gguf").unwrap();
        std::fs::write(mtp.join("mtp-head.gguf"), b"not a gguf").unwrap();
        std::fs::write(other.join("other.gguf"), b"not a gguf").unwrap();

        let rows = scan(&root).unwrap();
        assert_eq!(rows.len(), 2, "the drafter folder is not a model of its own");
        let names: Vec<&str> = rows.iter().filter_map(|r| r["name"].as_str()).collect();
        assert_eq!(names, ["other-model", "weights"]);
        let weights = &rows[1];
        assert_eq!(
            weights["draft"]
                .as_str()
                .and_then(|p| p.rsplit(['/', '\\']).next()),
            Some("mtp-head.gguf")
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
