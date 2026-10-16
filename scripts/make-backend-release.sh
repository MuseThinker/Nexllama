#!/usr/bin/env bash
#
# Pack each staged backend into a zip and write the manifest the desktop app
# fetches updates from. The PowerShell twin is make-backend-release.ps1; keep
# the two in step (same file names, same manifest fields).
#
#   ./scripts/make-backend-release.sh --version 0.1.0 \
#       [--base-url URL] [--llama-cpp REF] [--backends-dir DIR] [--out-dir DIR]
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: make-backend-release.sh --version VERSION [options]

  --version VERSION     release version, required (names the zip files)
  --base-url URL        download prefix; default is the GitHub release URL
  --llama-cpp REF       llama.cpp revision, recorded in the manifest
  --backends-dir DIR    staged backends; default crates/app/backends
  --out-dir DIR         output directory; default dist/backends
EOF
    exit 2
}

VERSION=""
BASE_URL=""
LLAMA_CPP=""
BACKENDS_DIR=""
OUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
        --base-url) BASE_URL="${2:?--base-url needs a value}"; shift 2 ;;
        --llama-cpp) LLAMA_CPP="${2:?--llama-cpp needs a value}"; shift 2 ;;
        --backends-dir) BACKENDS_DIR="${2:?--backends-dir needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        -h | --help) usage ;;
        *) echo "make-backend-release.sh: unknown argument '$1'" >&2; usage ;;
    esac
done

[ -n "$VERSION" ] || { echo "make-backend-release.sh: --version is required" >&2; usage; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -n "$BACKENDS_DIR" ] || BACKENDS_DIR="$ROOT/crates/app/backends"
[ -n "$OUT_DIR" ] || OUT_DIR="$ROOT/dist/backends"
[ -n "$BASE_URL" ] || BASE_URL="https://github.com/OWNER/REPO/releases/download/v$VERSION"

mkdir -p "$OUT_DIR"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Zip the directory *contents* so files sit at the archive root, exactly as
# Compress-Archive does in the PowerShell version.
zip_dir() {
    local src="$1" dst="$2"
    rm -f "$dst"
    if command -v zip >/dev/null 2>&1; then
        (cd "$src" && zip -q -X -r "$dst" .)
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "$src" "$dst" <<'PY'
import os, sys, zipfile

src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for name in sorted(files):
            path = os.path.join(root, name)
            z.write(path, os.path.relpath(path, src))
PY
    else
        echo "make-backend-release.sh: need 'zip' or 'python3' to pack archives" >&2
        exit 1
    fi
}

entries=()
shopt -s nullglob
for dir in "$BACKENDS_DIR"/*/; do
    name="$(basename "$dir")"
    if [ -f "$dir/infer-daemon.exe" ]; then
        :
    elif [ -f "$dir/infer-daemon" ]; then
        :
    else
        echo "make-backend-release.sh: warning: skipping $name (no infer-daemon)" >&2
        continue
    fi

    zip_name="nexllama-backend-$name-$VERSION.zip"
    zip_path="$OUT_DIR/$zip_name"
    zip_dir "$dir" "$zip_path"

    hash="$(sha256_of "$zip_path")"
    size="$(wc -c <"$zip_path")"
    mb="$(awk -v b="$size" 'BEGIN { printf "%.1f", b / 1048576 }')"
    printf 'packed %-8s -> %s (%s MB, sha256 %s)\n' "$name" "$zip_name" "$mb" "${hash:0:12}"
    entries+=("$name $zip_name $hash")
done
shopt -u nullglob

if [ "${#entries[@]}" -eq 0 ]; then
    echo "make-backend-release.sh: nothing to pack under $BACKENDS_DIR" >&2
    exit 1
fi

# Written by hand rather than through jq/python so the script runs on a bare
# build machine. The values are URLs, hex digests and file names - none of
# which need JSON escaping - so quoting them directly is safe.
manifest="$OUT_DIR/manifest.json"
{
    printf '{\n'
    printf '  "version": "%s",\n' "$VERSION"
    printf '  "llama_cpp": "%s",\n' "$LLAMA_CPP"
    printf '  "backends": {\n'
    last=$((${#entries[@]} - 1))
    for i in "${!entries[@]}"; do
        read -r name zip_name hash <<<"${entries[$i]}"
        printf '    "%s": { "url": "%s/%s", "sha256": "%s" }' \
            "$name" "$BASE_URL" "$zip_name" "$hash"
        [ "$i" -eq "$last" ] || printf ','
        printf '\n'
    done
    printf '  }\n'
    printf '}\n'
} >"$manifest"

echo "manifest: $manifest"
echo "upload manifest.json + the .zip files to: $BASE_URL"
