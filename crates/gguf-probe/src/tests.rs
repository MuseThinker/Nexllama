use super::*;
use std::io::Cursor;

type Result<T> = std::result::Result<T, ProbeError>;

/// Minimal GGUF writer — everything this crate parses, nothing llama.cpp would
/// accept as a model. That separation is deliberate: these fixtures test the
/// reader, not llama.cpp's loader.
fn gguf(kvs: Vec<(&'static str, u32, KvValue)>) -> Vec<u8> {
    gguf_with_tensors(kvs, &[])
}

/// Same, with tensor descriptors after the KV section. Only the names matter to
/// the reader, so dims/type/offset are written as fixed dummies.
fn gguf_with_tensors(kvs: Vec<(&'static str, u32, KvValue)>, tensors: &[&str]) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(b"GGUF");
    w.extend_from_slice(&2u32.to_le_bytes());
    w.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
    w.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, tag, value) in kvs {
        w.extend_from_slice(&(key.len() as u64).to_le_bytes());
        w.extend_from_slice(key.as_bytes());
        w.extend_from_slice(&tag.to_le_bytes());
        match value {
            KvValue::U32(v) => w.extend_from_slice(&v.to_le_bytes()),
            KvValue::U64(v) => w.extend_from_slice(&v.to_le_bytes()),
            KvValue::String(s) => {
                w.extend_from_slice(&(s.len() as u64).to_le_bytes());
                w.extend_from_slice(s.as_bytes());
            }
            // A token vocabulary: the bulky case `skip_value` exists for.
            KvValue::StringArray(items) => {
                w.extend_from_slice(&STRING_TAG.to_le_bytes());
                w.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    w.extend_from_slice(&(item.len() as u64).to_le_bytes());
                    w.extend_from_slice(item.as_bytes());
                }
            }
        }
    }
    for name in tensors {
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name.as_bytes());
        w.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        w.extend_from_slice(&4u64.to_le_bytes());
        w.extend_from_slice(&4u64.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes()); // ggml type
        w.extend_from_slice(&0u64.to_le_bytes()); // offset
    }
    w
}

enum KvValue {
    U32(u32),
    U64(u64),
    String(&'static str),
    StringArray(&'static [&'static str]),
}

const STRING_TAG: u32 = 8;
const ARRAY_TAG: u32 = 9;
const U32_TAG: u32 = 4;

fn read(bytes: Vec<u8>) -> Result<Probe> {
    read_header(Cursor::new(bytes))
}

fn class_of(kvs: Vec<(&'static str, u32, KvValue)>) -> ModelClass {
    read(gguf(kvs)).expect("parses").classify()
}

#[test]
fn generation_architectures_classify_as_llm() {
    for arch in ["llama", "qwen2", "qwen3", "gemma3", "mistral"] {
        assert_eq!(
            class_of(vec![(
                "general.architecture",
                STRING_TAG,
                KvValue::String(arch)
            )]),
            ModelClass::Llm,
            "{arch}"
        );
    }
}

#[test]
fn bert_family_classifies_as_embedding() {
    for arch in ["bert", "nomic-bert", "jina-bert-v2", "t5encoder"] {
        assert_eq!(
            class_of(vec![(
                "general.architecture",
                STRING_TAG,
                KvValue::String(arch)
            )]),
            ModelClass::Embedding,
            "{arch}"
        );
    }
}

#[test]
fn a_rerank_head_beats_the_architecture() {
    assert_eq!(
        class_of(vec![
            ("general.architecture", STRING_TAG, KvValue::String("bert")),
            ("rerank.head_count", 4, KvValue::U32(1)),
        ]),
        ModelClass::Rerank
    );
    // And the 64-bit spelling of the same key.
    assert_eq!(
        class_of(vec![("rerank.head_count", 10, KvValue::U64(1))]),
        ModelClass::Rerank
    );
}

#[test]
fn zero_heads_does_not_make_a_ranker() {
    assert_eq!(
        class_of(vec![
            ("general.architecture", STRING_TAG, KvValue::String("bert")),
            ("rerank.head_count", 4, KvValue::U32(0)),
        ]),
        ModelClass::Embedding
    );
}

#[test]
fn a_cls_tensor_is_what_separates_a_bert_reranker_from_its_embedding_sibling() {
    // bge-reranker-v2-m3 and bge-m3 agree on architecture, name and every KV key;
    // only the classification head tells them apart.
    let bert = || {
        vec![
            ("general.architecture", STRING_TAG, KvValue::String("bert")),
            ("general.name", STRING_TAG, KvValue::String("Bge M3")),
        ]
    };
    assert_eq!(
        read(gguf_with_tensors(
            bert(),
            &["token_embd.weight", "blk.0.attn_q.weight"]
        ))
        .unwrap()
        .classify(),
        ModelClass::Embedding,
        "no head => embedding"
    );
    assert_eq!(
        read(gguf_with_tensors(
            bert(),
            &[
                "token_embd.weight",
                "cls.weight",
                "cls.bias",
                "cls.output.weight"
            ]
        ))
        .unwrap()
        .classify(),
        ModelClass::Rerank,
        "a cls head => reranker"
    );
}

#[test]
fn rank_pooling_marks_a_reranker_whatever_the_architecture() {
    // Qwen3 rerankers keep `qwen3` as their architecture and say so with pooling.
    assert_eq!(
        class_of(vec![
            ("general.architecture", STRING_TAG, KvValue::String("qwen3")),
            ("pooling_type", 4, KvValue::U32(4)),
        ]),
        ModelClass::Rerank
    );
    // Any other pooling value leaves the architecture to decide.
    assert_eq!(
        class_of(vec![
            ("general.architecture", STRING_TAG, KvValue::String("qwen3")),
            ("pooling_type", 4, KvValue::U32(1)),
        ]),
        ModelClass::Llm
    );
}

#[test]
fn an_embedder_on_a_generation_arch_is_named_by_its_architecture() {
    // gte-Qwen2 keeps `qwen2` as its architecture and only carries `pooling_type`,
    // so nothing in this file says embedding except the architecture name itself;
    // the honest guess stays generation, which is what the UI will offer.
    assert_eq!(
        class_of(vec![
            ("general.architecture", STRING_TAG, KvValue::String("qwen2")),
            ("pooling_type", 4, KvValue::U32(1)),
        ]),
        ModelClass::Llm
    );
    assert_eq!(
        class_of(vec![(
            "general.architecture",
            STRING_TAG,
            KvValue::String("qwen2embed")
        )]),
        ModelClass::Embedding
    );
}

#[test]
fn huge_arrays_are_seeked_past_without_reading() {
    assert_eq!(
        class_of(vec![
            (
                "tokenizer.ggml.tokens",
                ARRAY_TAG,
                KvValue::StringArray(&["a"; 2048])
            ),
            ("general.architecture", STRING_TAG, KvValue::String("llama")),
        ]),
        ModelClass::Llm
    );
}

#[test]
fn keys_can_come_in_any_order() {
    assert_eq!(
        class_of(vec![
            (
                "tokenizer.ggml.merges",
                ARRAY_TAG,
                KvValue::StringArray(&["x"; 64])
            ),
            ("rerank.head_count", 10, KvValue::U64(1)),
        ]),
        ModelClass::Rerank
    );
}

#[test]
fn non_gguf_files_are_refused_by_name() {
    assert_eq!(read(b"nope".to_vec()), Err(ProbeError::NotGguf));
}

#[test]
fn truncated_files_are_truncated() {
    let full = gguf(vec![(
        "general.architecture",
        STRING_TAG,
        KvValue::String("llama"),
    )]);
    let cut = &full[..full.len() - 2];
    assert_eq!(read(cut.to_vec()), Err(ProbeError::Truncated));
}

#[test]
fn unsupported_value_types_are_malformed() {
    let full = gguf(vec![(
        "general.architecture",
        STRING_TAG,
        KvValue::String("llama"),
    )]);
    let tag_at = full
        .windows(4)
        .rposition(|w| w == 8u32.to_le_bytes())
        .expect("string tag present");
    let mut bytes = full;
    bytes[tag_at..tag_at + 4].copy_from_slice(&13u32.to_le_bytes());
    assert_eq!(read(bytes), Err(ProbeError::Malformed));
}

#[test]
fn a_key_of_nonsense_length_is_malformed() {
    let mut w = b"GGUF".to_vec();
    w.extend_from_slice(&2u32.to_le_bytes());
    w.extend_from_slice(&0u64.to_le_bytes());
    w.extend_from_slice(&1u64.to_le_bytes()); // one key
    w.extend_from_slice(&u64::MAX.to_le_bytes()); // corrupt length
    assert_eq!(read(w), Err(ProbeError::Malformed));
}

#[test]
fn probe_from_disk_matches_in_memory() {
    let dir = std::env::temp_dir().join("nexllama-gguf-probe");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.gguf");
    std::fs::write(
        &path,
        gguf(vec![(
            "general.architecture",
            STRING_TAG,
            KvValue::String("bert"),
        )]),
    )
    .unwrap();
    assert_eq!(probe(&path).unwrap().classify(), ModelClass::Embedding);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_files_name_the_problem() {
    let result = probe("certainly/not/here.gguf");
    assert!(result.is_err());
}
#[test]
fn mtp_is_recognised_from_the_nextn_layer_count() {
    // `<arch>.nextn_predict_layers` is what llama.cpp reads into n_layer_nextn.
    assert_eq!(
        class_of(vec![
            (
                "general.architecture",
                STRING_TAG,
                KvValue::String("qwen35moe")
            ),
            ("qwen35moe.nextn_predict_layers", 4, KvValue::U32(1)),
        ]),
        ModelClass::Llm
    );
    let p = read(gguf(vec![
        (
            "general.architecture",
            STRING_TAG,
            KvValue::String("qwen35moe"),
        ),
        ("qwen35moe.nextn_predict_layers", 4, KvValue::U32(1)),
    ]))
    .unwrap();
    assert!(p.has_mtp(), "a positive nextn layer count means MTP");
    assert_eq!(p.nextn_layers, Some(1));
}

#[test]
fn mtp_is_recognised_from_tensor_names_when_the_key_is_absent() {
    let p = read(gguf_with_tensors(
        vec![(
            "general.architecture",
            STRING_TAG,
            KvValue::String("qwen35moe"),
        )],
        &[
            "token_embd.weight",
            "blk.48.nextn.eh_proj.weight",
            "blk.48.nextn.hnorm.weight",
        ],
    ))
    .unwrap();
    assert!(p.has_mtp(), "nextn tensors are the fallback signal");
    assert_eq!(p.nextn_layers, None);
    assert!(p.nextn_tensors);
}

#[test]
fn zero_nextn_layers_without_tensors_is_not_mtp() {
    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("qwen2")),
        ("qwen2.nextn_predict_layers", 4, KvValue::U32(0)),
    ]))
    .unwrap();
    assert!(!p.has_mtp());
}

#[test]
fn an_ordinary_model_has_no_mtp() {
    let p = read(gguf_with_tensors(
        vec![("general.architecture", STRING_TAG, KvValue::String("qwen2"))],
        &["token_embd.weight", "blk.0.attn_q.weight"],
    ))
    .unwrap();
    assert!(!p.has_mtp());
}

/// `<arch>.context_length` is the trained context, and it arrives as either a
/// u32 or a u64 depending on the converter — both have to be read.
#[test]
fn the_training_context_is_read_from_the_architecture_key() {
    const U32_TAG: u32 = 4;
    const U64_TAG: u32 = 10;
    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("qwen2")),
        ("qwen2.context_length", U32_TAG, KvValue::U32(32768)),
    ]))
    .unwrap();
    assert_eq!(p.context_length, Some(32768));

    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("llama")),
        ("llama.context_length", U64_TAG, KvValue::U64(131072)),
    ]))
    .unwrap();
    assert_eq!(p.context_length, Some(131072));

    // A header that never states one reports nothing rather than a guess.
    let p = read(gguf(vec![(
        "general.architecture",
        STRING_TAG,
        KvValue::String("qwen2"),
    )]))
    .unwrap();
    assert_eq!(p.context_length, None);
}

/// The KV cache is what a context costs, and what decides whether it fits. The
/// shape comes from the header, and a header that does not state it says so
/// rather than reporting zero — a caller that read zero as "free" would size a
/// cache it cannot hold.
#[test]
fn the_kv_cache_shape_is_read_when_the_header_states_it() {
    // Grouped-query attention: 8 key/value heads against 40 attention heads, an
    // explicit key width, 48 layers. K and V, two bytes each.
    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("qwen3")),
        ("qwen3.block_count", U32_TAG, KvValue::U32(48)),
        ("qwen3.attention.head_count", U32_TAG, KvValue::U32(40)),
        ("qwen3.attention.head_count_kv", U32_TAG, KvValue::U32(8)),
        ("qwen3.attention.key_length", U32_TAG, KvValue::U32(128)),
    ]))
    .unwrap();
    assert_eq!(p.kv_bytes_per_token(), Some(48 * 8 * 128 * 2 * 2));

    // No GQA: the attention head count *is* the key/value head count, and the
    // head width is the embedding width divided by it — what llama.cpp derives
    // when the header leaves `key_length` out.
    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("llama")),
        ("llama.block_count", U32_TAG, KvValue::U32(32)),
        ("llama.attention.head_count", U32_TAG, KvValue::U32(32)),
        ("llama.embedding_length", U32_TAG, KvValue::U32(4096)),
    ]))
    .unwrap();
    assert_eq!(p.kv_bytes_per_token(), Some(32 * 32 * 128 * 2 * 2));

    // An encoder has no KV cache to size, and a header that says nothing
    // answers nothing.
    let p = read(gguf(vec![
        ("general.architecture", STRING_TAG, KvValue::String("bert")),
        ("bert.embedding_length", U32_TAG, KvValue::U32(768)),
    ]))
    .unwrap();
    assert_eq!(p.kv_bytes_per_token(), None);
}
