#!/usr/bin/env bash
#
# Stage the inference daemon, and the shared libraries it needs, where the
# desktop bundle picks them up. The Linux/macOS counterpart of
# build-daemon.ps1.
#
#   ./scripts/build-daemon.sh [cpu|vulkan|cuda|metal] [--native] [--release]
#
# Each backend gets its own target directory, because switching features
# otherwise leaves a stale library behind in a shared one.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: build-daemon.sh [cpu|vulkan|cuda|metal] [--native] [--release]
EOF
    exit 2
}

BACKEND=cpu
NATIVE=""
PROFILE=debug
for arg in "$@"; do
    case "$arg" in
        cpu | vulkan | cuda | metal) BACKEND="$arg" ;;
        --native) NATIVE=native ;;
        --release) PROFILE=release ;;
        -h | --help) usage ;;
        *) usage ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OS="$(uname -s)"

case "$OS" in
    Darwin)
        LIB_PREFIX=lib
        LIB_SUFFIX=.dylib
        ;;
    Linux)
        LIB_PREFIX=lib
        LIB_SUFFIX=.so
        ;;
    *)
        echo "build-daemon.sh: unsupported OS '$OS'" >&2
        exit 1
        ;;
esac

export CARGO_TARGET_DIR="$ROOT/target/backend-$BACKEND"
cd "$ROOT"

FEATURES=()
[ "$BACKEND" != cpu ] && FEATURES+=("$BACKEND")
[ -n "$NATIVE" ] && FEATURES+=("$NATIVE")

ARGS=(build -p infer-daemon)
[ "$PROFILE" = release ] && ARGS+=(--release)
if [ "${#FEATURES[@]}" -gt 0 ]; then
    joined="$(IFS=, ; echo "${FEATURES[*]}")"
    ARGS+=(--features "$joined")
fi

echo "==> backend: $BACKEND (features: ${joined:-none})"
cargo "${ARGS[@]}"

OUT="$CARGO_TARGET_DIR/$PROFILE"
SRC="$OUT/infer-daemon"
if [ ! -f "$SRC" ]; then
    echo "build-daemon.sh: daemon binary not found at $SRC" >&2
    exit 1
fi

DST="$ROOT/crates/app/backends/$BACKEND"
rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC" "$DST/infer-daemon"

# llama-common carries common_sampler (speculative verification) and the n-gram
# drafters; mtmd carries the image projector. Both are called at runtime, so
# neither is dead weight.
LIBS=(llama llama-common mtmd ggml ggml-base ggml-cpu)
case "$BACKEND" in
    cuda) LIBS+=(ggml-cuda) ;;
    vulkan) LIBS+=(ggml-vulkan) ;;
esac

for name in "${LIBS[@]}"; do
    lib="$OUT/${LIB_PREFIX}${name}${LIB_SUFFIX}"
    if [ ! -f "$lib" ]; then
        echo "warning: missing runtime library: $(basename "$lib")" >&2
        continue
    fi
    if [ "$OS" = Linux ]; then
        # The build tree holds the same file under three names - `libllama.so`,
        # `libllama.so.0`, `libllama.so.0.4.0` - and the loader asks for the
        # *SONAME*, which is the middle one. Copying only the linker name (what
        # this script used to do) leaves `libllama.so.0 => not found` on any
        # machine without llama.cpp installed. Ship the name that is requested.
        soname="$(readelf -d "$lib" 2>/dev/null | sed -n 's/.*(SONAME).*\[\(.*\)\].*/\1/p')"
        [ -n "$soname" ] || soname="$(basename "$lib")"
        cp -f "$lib" "$DST/$soname"
    else
        cp -f "$lib" "$DST/"
    fi
done

# Metal keeps its shaders as a resource beside the binary.
if [ "$BACKEND" = metal ] && [ -f "$OUT/ggml-metal.metal" ]; then
    cp "$OUT/ggml-metal.metal" "$DST/"
fi

if [ "$OS" = Darwin ]; then
    # Without this the loader searches the paths the libraries were built with,
    # which exist only on the machine that built them.
    install_name_tool -add_rpath @loader_path "$DST/infer-daemon" 2>/dev/null || true
fi

if [ "$OS" = Linux ]; then
    # The same problem, in ELF form. The daemon and every llama.cpp library were
    # linked against absolute paths into the build directory, and the libraries
    # also refer to *each other*, so patching only the binary would leave
    # `libllama.so` unable to find `libggml.so`. `$ORIGIN` closes the chain:
    # every object resolves its neighbours in the same directory.
    if ! command -v patchelf >/dev/null 2>&1; then
        echo "build-daemon.sh: patchelf is required on Linux (apt install patchelf)" >&2
        exit 1
    fi
    for f in "$DST/infer-daemon" "$DST"/lib*.so*; do
        [ -f "$f" ] || continue
        patchelf --set-rpath '$ORIGIN' "$f"
    done
    chmod +x "$DST/infer-daemon"

    # Every staged library has to resolve to a file beside the binary. `not
    # found` means a name or the RPATH is wrong; a path outside $DST means the
    # build machine's own copy is standing in - an ABI this build never linked
    # against, which is where the crashes on other machines come from.
    bad="$(ldd "$DST/infer-daemon" 2>/dev/null \
        | grep -E 'lib(llama|ggml|mtmd)' | grep -vF "$DST/" || true)"
    if [ -n "$bad" ]; then
        echo "build-daemon.sh: staged libraries do not resolve to $DST:" >&2
        printf '%s\n' "$bad" >&2
        exit 1
    fi
fi

echo "backend ready: $DST"
