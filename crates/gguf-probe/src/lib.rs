//! What kind of model does a GGUF file hold?
//!
//! llama.cpp runs chat decoders, embedding encoders and rankers through
//! different code paths, so a launcher that treats them alike hands users a model
//! that cannot do what they asked. This crate reads just enough of the GGUF
//! header to tell the families apart — the KV section, streamed key by key, never
//! the tensors — and leaves anything unrecognised as a generation model, the
//! more useful default.

use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

/// The families llama.cpp runs through different decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelClass {
    /// Text generation (vision models too: the projector wraps one of these).
    Llm,
    /// Encoder models meant to produce embeddings (BERT and friends).
    Embedding,
    /// Encoder models with a ranking head.
    Rerank,
}

impl ModelClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelClass::Llm => "llm",
            ModelClass::Embedding => "embedding",
            ModelClass::Rerank => "rerank",
        }
    }
}

/// GGUF metadata value types, numbered as in llama.cpp's `<gguf.h>`.
///
/// Values beyond what the on-disk spec defines abort rather than guessing a
/// layout; that keeps a corrupt header from being read as plausible junk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => GgufType::U8,
            1 => GgufType::I8,
            2 => GgufType::U16,
            3 => GgufType::I16,
            4 => GgufType::U32,
            5 => GgufType::I32,
            6 => GgufType::F32,
            7 => GgufType::Bool,
            8 => GgufType::String,
            9 => GgufType::Array,
            10 => GgufType::U64,
            11 => GgufType::I64,
            12 => GgufType::F64,
            _ => return None,
        })
    }

    /// Byte width when every element is the same size, which is what makes
    /// skipping an array a single seek.
    fn scalar_size(self) -> Option<u64> {
        Some(match self {
            GgufType::U8 | GgufType::I8 | GgufType::Bool => 1,
            GgufType::U16 | GgufType::I16 => 2,
            GgufType::U32 | GgufType::I32 | GgufType::F32 => 4,
            GgufType::U64 | GgufType::I64 | GgufType::F64 => 8,
            GgufType::String | GgufType::Array => return None,
        })
    }
}

/// Error from reading a GGUF header.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// No `GGUF` magic at offset 0.
    NotGguf,
    /// The file cut off before the interesting keys.
    Truncated,
    /// A value type, string length or version this reader cannot make sense of.
    Malformed,
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ProbeError::NotGguf => "not a GGUF file",
            ProbeError::Truncated => "GGUF header is truncated",
            ProbeError::Malformed => "GGUF header is malformed",
        })
    }
}

impl std::error::Error for ProbeError {}

impl From<std::io::Error> for ProbeError {
    fn from(_: std::io::Error) -> Self {
        ProbeError::Truncated
    }
}

/// The facts about a GGUF file that decide its family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// `general.architecture`, e.g. `qwen2` or `bert`.
    pub architecture: Option<String>,
    /// `general.name`, free text, only ever used as a hint.
    pub name: Option<String>,
    /// `rerank.head_count` when present; a ranker writes one, a plain encoder does not.
    pub rerank_heads: Option<u64>,
    /// `pooling_type`, numbered as `llama_pooling_type`. `4` is rank, which is how
    /// a reranker that reuses a generation architecture announces itself.
    pub pooling_type: Option<u64>,
    /// Whether the tensors include llama.cpp's `cls` classification head.
    ///
    /// The decisive signal for the BERT rerankers: bge-reranker-v2-m3 and bge-m3
    /// share an architecture, a name and a KV section, and differ only by the
    /// head, which shows up as `cls.weight` / `cls.output.*` tensors.
    pub cls_head: bool,
}

impl Probe {
    /// Pick the family, with the rules written down:
    ///
    /// * a ranking head, rank pooling, or a `cls` tensor makes it a reranker
    ///   whatever the architecture says — Qwen3 rerankers keep `qwen3` as their
    ///   architecture and mark themselves with `pooling_type = rank`, while
    ///   BERT rerankers only differ from their embedding siblings by the head;
    /// * known encoder architectures, and any architecture name that mentions
    ///   embeddings (gte lists `pooling_type` but keeps `qwen2` as its
    ///   architecture, so the name is what identifies it), are embedding models;
    /// * everything else generates text — an optimistic default, but the useful
    ///   one, since generation is the only class this app can actually run.
    pub fn classify(&self) -> ModelClass {
        const POOLING_RANK: u64 = 4;
        if self.rerank_heads.unwrap_or(0) > 0
            || self.pooling_type == Some(POOLING_RANK)
            || self.cls_head
        {
            return ModelClass::Rerank;
        }
        if let Some(arch) = self.architecture.as_deref() {
            let arch = arch.to_ascii_lowercase();
            if matches!(
                arch.as_str(),
                "bert" | "nomic-bert" | "jina-bert-v2" | "jina-bert-eot" | "t5encoder"
            ) || arch.contains("embed")
            {
                return ModelClass::Embedding;
            }
        }
        ModelClass::Llm
    }
}

/// Read the header facts from a file on disk.
pub fn probe(path: impl AsRef<Path>) -> Result<Probe, ProbeError> {
    let file = std::fs::File::open(path)?;
    read_header(BufReader::new(file))
}

/// Every metadata key in the header, in file order.
///
/// Diagnostic only: when a model lands in the wrong class, this is how you find
/// out what it *does* say about itself.
pub fn keys(path: impl AsRef<Path>) -> Result<Vec<String>, ProbeError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let (tensor_count, kv_count) = read_preamble(&mut reader)?;

    let mut keys = Vec::new();
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let value_type = GgufType::from_u32(read_u32(&mut reader)?).ok_or(ProbeError::Malformed)?;
        keys.push(key);
        skip_value(&mut reader, value_type)?;
    }

    // Tensor descriptors follow the KV section, and the KV section is what the
    // skipping above lands us at the end of.
    let mut names = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let n_dims = read_u32(&mut reader)?;
        for _ in 0..n_dims {
            read_u8x8(&mut reader)?;
        }
        read_u32(&mut reader)?; // ggml type
        read_u8x8(&mut reader)?; // offset
        names.push(format!("tensor {name}"));
    }
    keys.extend(names);
    Ok(keys)
}

/// One-call convenience for callers that only want the family.
///
/// Anything unreadable counts as a generation model — a corrupt GGUF is going to
/// fail loudly in the loader anyway, and `unknown` would give the UI nothing to do.
pub fn probe_class(path: impl AsRef<Path>) -> String {
    probe(path)
        .map(|p| p.classify().as_str().to_string())
        .unwrap_or_else(|_| ModelClass::Llm.as_str().to_string())
}

/// Consume magic, version and the two counts, leaving the reader at the first key.
///
/// Returns `(tensor_count, kv_count)`.
fn read_preamble<R: BufRead>(reader: &mut R) -> Result<(u64, u64), ProbeError> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(ProbeError::NotGguf);
    }
    let version = read_u32(reader)?;
    if version != 2 && version != 3 {
        return Err(ProbeError::Malformed);
    }
    let tensor_count = u64::from_le_bytes(read_u8x8(reader)?);
    let kv_count = u64::from_le_bytes(read_u8x8(reader)?);
    Ok((tensor_count, kv_count))
}

/// Stream the KV section, keeping only the keys this family decision needs.
///
/// The interesting keys sit near the front, and the tokenizer takes megabytes
/// later in the same list, so array values are seeked past instead of read.
fn read_header<R: BufRead + Seek>(mut reader: R) -> Result<Probe, ProbeError> {
    let (tensor_count, kv_count) = read_preamble(&mut reader)?;

    let mut found = Probe {
        architecture: None,
        name: None,
        rerank_heads: None,
        pooling_type: None,
        cls_head: false,
    };

    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let value_type = GgufType::from_u32(read_u32(&mut reader)?).ok_or(ProbeError::Malformed)?;

        match (key.as_str(), value_type) {
            ("general.architecture", GgufType::String) => {
                found.architecture = Some(read_string(&mut reader)?);
            }
            ("general.name", GgufType::String) => {
                found.name = Some(read_string(&mut reader)?);
            }
            ("rerank.head_count", GgufType::U32) => {
                found.rerank_heads = Some(u64::from(read_u32(&mut reader)?));
            }
            ("rerank.head_count", GgufType::U64) => {
                found.rerank_heads = Some(u64::from_le_bytes(read_u8x8(&mut reader)?));
            }
            ("pooling_type", GgufType::U32) => {
                found.pooling_type = Some(u64::from(read_u32(&mut reader)?));
            }
            ("pooling_type", GgufType::U64) => {
                found.pooling_type = Some(u64::from_le_bytes(read_u8x8(&mut reader)?));
            }
            (_, value_type) => skip_value(&mut reader, value_type)?,
        }
    }

    found.cls_head = has_cls_head(&mut reader, tensor_count)?;
    Ok(found)
}

/// Walk the tensor descriptors looking for llama.cpp's classification head.
///
/// The head is what separates a BERT reranker from the embedding model it shares
/// everything else with, and it is not in the KV section, so this is the only
/// place the distinction exists.
fn has_cls_head<R: BufRead + Seek>(reader: &mut R, tensor_count: u64) -> Result<bool, ProbeError> {
    let mut found = false;
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        if name == "cls" || name.starts_with("cls.") {
            found = true;
        }
        let n_dims = read_u32(reader)?;
        for _ in 0..n_dims {
            read_u8x8(reader)?;
        }
        read_u32(reader)?; // ggml type
        read_u8x8(reader)?; // offset
    }
    Ok(found)
}

fn read_u32<R: BufRead>(reader: &mut R) -> Result<u32, ProbeError> {
    let mut b = [0u8; 4];
    reader
        .read_exact(&mut b)
        .map_err(|_| ProbeError::Truncated)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u8x8<R: BufRead>(reader: &mut R) -> Result<[u8; 8], ProbeError> {
    let mut b = [0u8; 8];
    reader
        .read_exact(&mut b)
        .map_err(|_| ProbeError::Truncated)?;
    Ok(b)
}

/// A header string with a length prefix, rejected when absurdly long.
fn read_string<R: BufRead>(reader: &mut R) -> Result<String, ProbeError> {
    let len = u64::from_le_bytes(read_u8x8(reader)?);
    // No GGUF header string is ever a megabyte; refusing here keeps a corrupt
    // length from making this crate allocate absurdly.
    if len > 1 << 16 {
        return Err(ProbeError::Malformed);
    }
    let mut bytes = vec![0u8; len as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| ProbeError::Truncated)?;
    String::from_utf8(bytes).map_err(|_| ProbeError::Malformed)
}

fn skip_value<R: BufRead + Seek>(reader: &mut R, value_type: GgufType) -> Result<(), ProbeError> {
    // Reborrow once so nested calls pass `&mut R` itself, not a new layer of it —
    // `&mut &mut R` also satisfies the bounds and would recurse types forever.
    let reader = &mut *reader;
    match value_type {
        GgufType::String => {
            read_string(reader)?;
        }
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(reader)?).ok_or(ProbeError::Malformed)?;
            let count = u64::from_le_bytes(read_u8x8(reader)?);
            if let Some(width) = elem_type.scalar_size() {
                let bytes = count
                    .checked_mul(width)
                    .filter(|b| *b <= i64::MAX as u64)
                    .ok_or(ProbeError::Malformed)?;
                reader
                    .seek(SeekFrom::Current(bytes as i64))
                    .map_err(|_| ProbeError::Truncated)?;
            } else {
                for _ in 0..count {
                    skip_value(reader, elem_type)?;
                }
            }
        }
        scalar => {
            let width = scalar
                .scalar_size()
                .expect("matched a non-array, non-string");
            reader
                .seek(SeekFrom::Current(width as i64))
                .map_err(|_| ProbeError::Truncated)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
