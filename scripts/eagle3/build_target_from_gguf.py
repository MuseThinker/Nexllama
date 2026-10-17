# Reconstructs the HuggingFace target directory that `convert_hf_to_gguf.py
# --target-model-dir` asks for, out of a GGUF that already carries the tokenizer.
#
# usage: python build_target_from_gguf.py <model.gguf> <out-dir>
import json
import sys
from pathlib import Path

try:
    from gguf import GGUFReader
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "gguf-py"))
    from gguf import GGUFReader

gguf_path, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

r = GGUFReader(gguf_path)
fd = r.fields

def strings(name):
    f = fd[name]
    return [bytes(f.parts[i]).decode("utf-8", "surrogateescape") for i in f.data]

def ints(name):
    f = fd[name]
    if len(f.parts) == 1:
        return [int(x) for x in f.parts[0]]
    return [int(f.parts[i][0]) for i in f.data]

def scalar_str(name):
    return bytes(fd[name].parts[fd[name].data[0]]).decode("utf-8", "replace")

def scalar_int(name):
    return int(list(fd[name].parts[fd[name].data[0]])[0])

tokens = strings("tokenizer.ggml.tokens")
merges = strings("tokenizer.ggml.merges")
types = ints("tokenizer.ggml.token_type")
eos = scalar_int("tokenizer.ggml.eos_token_id") if "tokenizer.ggml.eos_token_id" in fd else None
n_layer = scalar_int(f"{scalar_str('general.architecture')}.block_count")

cfg = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": eos,
    "eos_token_id": eos,
    "head_dim": None,
    "hidden_act": "silu",
    "hidden_size": None,
    "initializer_range": 0.02,
    "intermediate_size": None,
    "num_attention_heads": None,
    "num_hidden_layers": n_layer,
    "num_key_value_heads": None,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": len(tokens),
}
arch = scalar_str("general.architecture")
fields = {
    "hidden_size": "embedding_length",
    "intermediate_size": "feed_forward_length",
    "num_attention_heads": "attention.head_count",
    "num_key_value_heads": "attention.head_count_kv",
    "head_dim": "attention.key_length",
}
for k, suffix in fields.items():
    key = f"{arch}.{suffix}"
    if key in fd:
        cfg[k] = scalar_int(key)

(out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

bpe = models.BPE(vocab={t: i for i, t in enumerate(tokens)},
                 merges=[tuple(m.split(" ", 1)) for m in merges])
tok = Tokenizer(bpe)
# Qwen2's rule-based pre-split followed by byte-level folding; picked so the
# token id sequence matches the one the model was quantized against. The hash
# table in the converter's `get_vocab_base_pre` still misses, so its fallback
# decides the `pre` name - which the engine re-resolves against the target anyway.
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(
        Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        behavior="isolated",
    ),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()
tok.add_special_tokens([t for i, t in enumerate(tokens) if types[i] == 3])
tok.save(str(out_dir / "tokenizer.json"))

(out_dir / "tokenizer_config.json").write_text(json.dumps({
    "tokenizer_class": "PreTrainedTokenizerFast",
    "add_bos_token": False,
    "add_eos_token": False,
}, indent=2), encoding="utf-8")

print("wrote", out_dir)
