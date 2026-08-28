#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <version> [output-directory]" >&2
    exit 2
fi

VERSION="$1"
case "$VERSION" in
    ''|*[!0-9A-Za-z.-]*)
        echo "error: version contains unsupported characters: $VERSION" >&2
        exit 2
        ;;
esac

unset CDPATH
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd)"
OUTPUT_DIRECTORY="${2:-$REPOSITORY_ROOT/dist}"
ARCHIVE_NAME="granite-mlx-$VERSION-macos-arm64"

cd "$REPOSITORY_ROOT"
swift build -c release
Scripts/build_mlx_metallib.sh release

BUILD_DIRECTORY="$(swift build -c release --show-bin-path)"
BINARY="$BUILD_DIRECTORY/granite-mlx"
METAL_LIBRARY="$BUILD_DIRECTORY/mlx.metallib"

if [ ! -x "$BINARY" ]; then
    echo "error: release executable was not produced at $BINARY" >&2
    exit 1
fi
if [ ! -f "$METAL_LIBRARY" ]; then
    echo "error: MLX Metal library was not produced at $METAL_LIBRARY" >&2
    exit 1
fi

REPORTED_VERSION="$($BINARY --version)"
if [ "$REPORTED_VERSION" != "$VERSION" ]; then
    echo "error: release version mismatch; requested=$VERSION executable=$REPORTED_VERSION" >&2
    exit 1
fi

STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/granite-mlx-release.XXXXXX")"
cleanup() {
    rm -rf "$STAGING_ROOT"
}
trap cleanup EXIT

PACKAGE_DIRECTORY="$STAGING_ROOT/$ARCHIVE_NAME"
mkdir -p "$PACKAGE_DIRECTORY" "$OUTPUT_DIRECTORY"
OUTPUT_DIRECTORY="$(cd -- "$OUTPUT_DIRECTORY" && pwd)"
cp "$BINARY" "$PACKAGE_DIRECTORY/"
cp "$METAL_LIBRARY" "$PACKAGE_DIRECTORY/"
cp LICENSE LICENSE-APACHE LICENSE-MIT README.md "$PACKAGE_DIRECTORY/"

ARCHIVE="$OUTPUT_DIRECTORY/$ARCHIVE_NAME.tar.gz"
COPYFILE_DISABLE=1 tar -C "$STAGING_ROOT" -czf "$ARCHIVE" "$ARCHIVE_NAME"
(
    cd "$OUTPUT_DIRECTORY"
    shasum -a 256 "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)

echo "Created $ARCHIVE"
echo "Created $ARCHIVE.sha256"
