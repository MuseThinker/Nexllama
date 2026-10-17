# Rewrites EAGLE-3 KVs of a converted draft GGUF in place. The layer array is
# three i32 slots and the flags are single bytes, so no weight pass is needed
# to try out load-order or norm-convention variants.
#
# usage: python patch_gguf_meta.py <draft.gguf> [--layers LOW MID HIGH] [--set mirror_before=false]
import sys
from pathlib import Path

# The gguf package ships with llama.cpp's gguf-py;Point sys.path at an existing
# checkout when the standalone install is absent.
try:
    from gguf import GGUFReader
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "gguf-py"))
    from gguf import GGUFReader

import numpy as np

gguf_path = sys.argv[1]

layers = None
sets: list[tuple[str, bool]] = []
args = sys.argv[2:]
i = 0
while i < len(args):
    if args[i] == "--layers":
        layers = [int(x) for x in args[i + 1:i + 4]]
        assert len(layers) == 3, "--layers takes LOW MID HIGH"
        i += 4
    elif args[i] == "--set":
        name, raw = args[i + 1].split("=", 1)
        assert raw in ("true", "false"), "value must be true|false"
        sets.append((name, raw == "true"))
        i += 2
    else:
        raise SystemExit(f"unexpected argument {args[i]}")

if layers is None and not sets:
    r = GGUFReader(gguf_path)
    for k in r.fields:
        if k.startswith("eagle3."):
            f = r.fields[k]
            if not f.types or f.types[0] != 9:  # skip arrays like target_layers unless asked
                if len(f.data) == 1:
                    print(f"{k}: {f.parts[f.data[0]][0]}")
    f = r.fields["eagle3.target_layers"]
    print(f"eagle3.target_layers: {[f.parts[i][0] for i in f.data]}")
    sys.exit(0)

if layers is not None:
    r = GGUFReader(gguf_path, "r+")
    f = r.fields["eagle3.target_layers"]
    before = [f.parts[slot][0] for slot in f.data]
    for slot, idx in enumerate(f.data):
        part = f.parts[idx]
        assert part.dtype == np.int32 and part.shape == (1,), (part.dtype, part.shape)
        part[0] = layers[slot]
        part.flush()
    check = GGUFReader(gguf_path).fields["eagle3.target_layers"]
    after = [check.parts[slot][0] for slot in check.data]
    print(f"eagle3.target_layers: {before} -> {after}")

for name, val in sets:
    key = name if name.startswith("eagle3.") else f"eagle3.{name}"
    r = GGUFReader(gguf_path, "r+")
    f = r.fields[key]
    part = f.parts[f.data[0]]
    assert part.dtype == np.bool_, (name, part.dtype)
    before = bool(part[0])
    part[0] = val
    part.flush()
    check = GGUFReader(gguf_path).fields[key]
    after = bool(check.parts[check.data[0]][0])
    print(f"{key}: {before} -> {after}")
