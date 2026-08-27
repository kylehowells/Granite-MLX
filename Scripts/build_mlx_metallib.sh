#!/bin/sh
set -eu

CONFIG="${1:-debug}"
if [ "$CONFIG" != "debug" ] && [ "$CONFIG" != "release" ]; then
  echo "usage: $0 [debug|release]" >&2
  exit 2
fi

unset CDPATH
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
BUILD_DIR="${GRANITE_BUILD_DIR:-$ROOT/.build}"
MLX_SWIFT_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

if [ ! -d "$KERNELS_DIR" ]; then
  echo "error: MLX Metal kernels not found at $KERNELS_DIR" >&2
  echo "run swift build first, or set GRANITE_BUILD_DIR to the SwiftPM scratch path" >&2
  exit 1
fi

OUT_DIR="$BUILD_DIR/$CONFIG"
if [ ! -d "$OUT_DIR" ]; then
  OUT_DIR="$(find "$BUILD_DIR" -maxdepth 5 -type d -path "*/$CONFIG" | head -n 1 || true)"
fi
if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
  echo "error: SwiftPM output directory not found for $CONFIG" >&2
  exit 1
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/granite-mlx-metallib.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

find "$KERNELS_DIR" -type f -name '*.metal' ! -name '*_nax.metal' \
  | LC_ALL=C sort \
  | while IFS= read -r source; do
  key="$(printf '%s' "${source#"$KERNELS_DIR/"}" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
  output="$TMP/$key.air"
  xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
    -Wno-c++17-extensions -Wno-c++20-extensions \
    -I"$KERNELS_DIR" -I"$MLX_SWIFT_DIR/Source/Cmlx/mlx" \
    -c "$source" -o "$output"
done

set -- "$TMP"/*.air
if [ ! -e "$1" ]; then
  echo "error: no MLX Metal kernels were compiled" >&2
  exit 1
fi
xcrun -sdk macosx metallib "$@" -o "$OUT_DIR/mlx.metallib"
# SwiftPM's executable/test bundle normally lives below an architecture
# directory. Keep a copy at the architecture-specific output too when the
# fallback above selected a flat debug/release directory.
ARCH_OUTPUT="$BUILD_DIR/arm64-apple-macosx/$CONFIG/mlx.metallib"
if [ "${OUT_DIR%/arm64-apple-macosx/"$CONFIG"}" = "$OUT_DIR" ] \
  && [ -d "$(dirname "$ARCH_OUTPUT")" ] \
  && [ ! "$OUT_DIR/mlx.metallib" -ef "$ARCH_OUTPUT" ]; then
  cp "$OUT_DIR/mlx.metallib" "$ARCH_OUTPUT"
fi
echo "Wrote $OUT_DIR/mlx.metallib"
