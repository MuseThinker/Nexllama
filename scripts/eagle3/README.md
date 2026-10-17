# EAGLE-3 draft checkpoint → GGUF

An EAGLE-3 head (`LlamaForCausalLMEagle3`, e.g. Qwen3-14B) shares its tokenizer
and token embedding with the target model, so its HuggingFace checkpoint alone
is not enough to load: the converter wants a *target directory* holding a
`config.json` and a tokenizer, purely to copy `vocab_size`, the layer count and
the tokenizer from. When only the target's GGUF is on disk, `build_target_from_gguf.py`
reconstructs that directory from it, no download needed.

```text
# one-time: an upstream llama.cpp clone (the vendored copy inside the sys crate
# was pruned down to 312 lines and has no Eagle3 conversion class)
git clone https://github.com/ggml-org/llama.cpp llama-upstream

python build_target_from_gguf.py Qwen3-14B-UD-Q4_K_XL.gguf target-dir
cd llama-upstream
python convert_hf_to_gguf.py <draft-dir> --target-model-dir target-dir \
    --outfile eagle3-Qwen3-14B.gguf
```

Two cuts stand in the way of a stock conversion:

- the head's `config.json` must be named exactly that (a download that names it
  `config .json` silently breaks `load_hparams`), and
- the tokenizer's `pre` detection hashes `tokenizer.encode` output against a
  hardcoded table (built by `convert_hf_to_gguf_update.py`), which a
  reconstructed tokenizer can never match. The engine resolves the draft's
  `pre` against the target at load time, so the value written here is cosmetic.
  Patching `get_vocab_base_pre` in the clone to fall back to `"qwen2"` clears it.

## What the finished GGUF exposes

`general.architecture = eagle3` plus `eagle3.target_layers` (three ids, low/mid/high),
`eagle3.target_hidden_size`, `norm_before_residual` and `norm_before_fc`.
`d2t` is the draft→target id map; `t2d` is not required, the C++ side builds
nothing from it.

## `patch_gguf_meta.py`

Rewrites those four KVs in place — no weight pass needed, the array has three
i32 slots and the flags are single bytes:

```text
python patch_gguf_meta.py eagle3-Qwen3-14B.gguf --layers 2 20 37
python patch_gguf_meta.py eagle3-Qwen3-14B.gguf --set norm_before_residual=false
```

## What was tried on Qwen3-14B (RTX 6000, Q4_K_XL target, 8 turns)

| variant                | acceptance | result                                    |
| ---------------------- | ---------- | ----------------------------------------- |
| layers `[2, 20, 37]`   | ~29%       | +28% gen (45–57 vs 32–44 baseline)        |
| layers `[2, 20, 39]`   | ~22%       | noise                                     |
| layers reversed        | ~1%        | order matters: low, mid, high, not reverse |
| `norm_before_residual` | 0%         | must be false                             |

An acceptance far below the MTP/DFlash2 heads (~72%) on the same class of
machine: the head is barely matched to its target, and the guessed extract
layers are one of the few levers still unverified — the config carries no
`eagle_aux_hidden_state_layer_ids`, so `[2, 20, 37]` is the converter's
`[2, n/2, n-3]` default rather than what training used.
