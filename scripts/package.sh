#!/usr/bin/env bash
#
# Build the desktop installer for this platform: stage the daemon backends,
# then let the Tauri CLI bundle the app.
#
#   ./scripts/package.sh               # cpu backend, release, portable CPU
#   ./scripts/package.sh metal         # Apple GPU backend
#   ./scripts/package.sh cpu vulkan    # several at once
#   ./scripts/package.sh cpu --debug   # debug build
#   ./scripts/package.sh cpu --native  # build for this machine only
#
# Outputs: .deb + .rpm + .AppImage on Linux, .dmg on macOS. The Windows
# installer comes from package.ps1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The same baseline package.ps1 uses: a shipped backend runs on any CPU, so it
# is built without `native` - that flag bakes this machine's instructions in and
# crashes on the next one. Pass --native for a personal, same-machine build.
DAEMON_ARGS=(--release)
NATIVE=0
BACKENDS=()

for arg in "$@"; do
    case "$arg" in
        --debug) DAEMON_ARGS=(--debug) ;;
        --release) DAEMON_ARGS=(--release) ;;
        --native) NATIVE=1 ;;
        cpu | vulkan | cuda | metal) BACKENDS+=("$arg") ;;
        *)
            echo "package.sh: unknown argument '$arg'" >&2
            exit 2
            ;;
    esac
done
# Appended last so the flag survives any order the arguments came in.
[ "$NATIVE" = 1 ] && DAEMON_ARGS+=(--native)
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

# tauri.conf.json pins `nsis` for a plain Windows build, and the platform
# configs (tauri.linux.conf.json, tauri.macos.conf.json) carry the rest. Spell
# the Linux list out on the command line too: it is what the artifacts are
# judged by, and a flag cannot silently disagree with a config nobody re-reads.
case "$(uname -s)" in
    Linux | Darwin) ;;
    *)
        echo "package.sh: unsupported OS '$(uname -s)'" >&2
        exit 1
        ;;
esac

cd "$ROOT/crates/app"
if [ "$(uname -s)" = Linux ]; then
    "$TAURI" build --bundles deb,rpm,appimage
else
    "$TAURI" build
fi

BUNDLE="$ROOT/crates/app/target/release/bundle"
echo "bundle output:"
find "$BUNDLE" -maxdepth 2 -type f \
    \( -name '*.dmg' -o -name '*.deb' -o -name '*.rpm' -o -name '*.AppImage' -o -name '*.app' \) \
    -print 2>/dev/null || true
