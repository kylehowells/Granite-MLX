#!/bin/sh

set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <archive> <expected-version>" >&2
    exit 2
fi

ARCHIVE="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename "$1")"
EXPECTED_VERSION="$2"

if [ ! -f "$ARCHIVE" ]; then
    echo "error: release archive does not exist: $ARCHIVE" >&2
    exit 1
fi
if [ ! -f "$ARCHIVE.sha256" ]; then
    echo "error: release checksum does not exist: $ARCHIVE.sha256" >&2
    exit 1
fi

(
    cd -- "$(dirname -- "$ARCHIVE")"
    shasum -a 256 -c "$(basename "$ARCHIVE").sha256"
)

TEMPORARY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/granite-mlx-archive-test.XXXXXX")"
cleanup() {
    rm -rf "$TEMPORARY_ROOT"
}
trap cleanup EXIT

tar -C "$TEMPORARY_ROOT" -xzf "$ARCHIVE"
PACKAGE_DIRECTORY="$TEMPORARY_ROOT/granite-mlx-$EXPECTED_VERSION-macos-arm64"
BINARY="$PACKAGE_DIRECTORY/granite-mlx"

for required in granite-mlx mlx.metallib LICENSE LICENSE-APACHE LICENSE-MIT README.md; do
    if [ ! -e "$PACKAGE_DIRECTORY/$required" ]; then
        echo "error: release archive is missing $required" >&2
        exit 1
    fi
done

if [ ! -x "$BINARY" ]; then
    echo "error: archived granite-mlx is not executable" >&2
    exit 1
fi
if ! file "$BINARY" | grep -q 'arm64'; then
    echo "error: archived executable is not an ARM64 Mach-O binary" >&2
    file "$BINARY" >&2
    exit 1
fi

UNEXPECTED_LIBRARIES="$(otool -L "$BINARY" | tail -n +2 | awk '{print $1}' \
    | grep -Ev '^(/System/Library/|/usr/lib/)' || true)"
if [ -n "$UNEXPECTED_LIBRARIES" ]; then
    echo "error: archived executable has non-system dynamic-library dependencies:" >&2
    echo "$UNEXPECTED_LIBRARIES" >&2
    exit 1
fi

ISOLATED_CACHE="$TEMPORARY_ROOT/model-cache"
ISOLATED_COREML_CACHE="$TEMPORARY_ROOT/coreml-cache"
ISOLATED_CONFIG="$TEMPORARY_ROOT/config.json"
export GRANITE_MLX_HUB_DIRECTORY="$ISOLATED_CACHE"
export GRANITE_MLX_COREML_CACHE_DIRECTORY="$ISOLATED_COREML_CACHE"
export GRANITE_MLX_TEST_CONFIG_PATH="$ISOLATED_CONFIG"

if [ "$($BINARY --version)" != "$EXPECTED_VERSION" ]; then
    echo "error: archived executable reports the wrong version" >&2
    exit 1
fi
"$BINARY" --help >/dev/null
"$BINARY" transcribe --help >/dev/null
"$BINARY" models list --json >/dev/null
"$BINARY" config show >/dev/null

if find "$ISOLATED_CACHE" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "error: model-free archive checks unexpectedly downloaded model data" >&2
    exit 1
fi

echo "Release archive passed model-free validation: $ARCHIVE"
