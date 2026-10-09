#!/usr/bin/env bash
#
# Build the desktop installer for this platform: stage the daemon backends,
# then let the Tauri CLI bundle the app.
#
#   ./scripts/package.sh               # cpu, release, native
#   ./scripts/package.sh metal         # Apple GPU backend
#   ./scripts/package.sh cpu vulkan    # several at once
#   ./scripts/package.sh cpu --debug   # debug build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON_ARGS=(--release --native)
BACKENDS=()

for arg in "$@"; do
    case "$arg" in
        --debug) DAEMON_ARGS=(--debug) ;;
        --release) ;;
        cpu | vulkan | cuda | metal) BACKENDS+=("$arg") ;;
        *)
            echo "package.sh: unknown argument '$arg'" >&2
            exit 2
            ;;
    esac
done
[ "${#BACKENDS[@]}" -eq 0 ] && BACKENDS=(cpu)

for backend in "${BACKENDS[@]}"; do
    "$ROOT/scripts/build-daemon.sh" "$backend" "${DAEMON_ARGS[@]}"
done

# build-daemon.sh exports CARGO_TARGET_DIR per backend; the desktop app builds
# into its own workspace target directory instead.
unset CARGO_TARGET_DIR

TAURI="$ROOT/frontend/node_modules/.bin/tauri"
if [ ! -x "$TAURI" ]; then
    echo "package.sh: Tauri CLI not found. Run: npm --prefix frontend install" >&2
    exit 1
fi

cd "$ROOT/crates/app"
"$TAURI" build

BUNDLE="$ROOT/crates/app/target/release/bundle"
echo "bundle output:"
find "$BUNDLE" -maxdepth 2 -type f \( -name '*.dmg' -o -name '*.deb' -o -name '*.AppImage' -o -name '*.app' \) -print 2>/dev/null || true
