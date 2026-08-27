#!/bin/sh

set -eu

unset CDPATH
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TEMPORARY_OUTPUT="$(mktemp -d "${TMPDIR:-/tmp}/granite-mlx-docc.XXXXXX")"

cleanup() {
    rm -rf "$TEMPORARY_OUTPUT"
}
trap cleanup EXIT

cd "$REPOSITORY_ROOT"

swift package dump-symbol-graph \
    --pretty-print \
    --skip-synthesized-members \
    --minimum-access-level public

SYMBOL_GRAPH_FILE="$(find .build -type f -name 'GraniteMLX.symbols.json' -print -quit)"
if [ -z "$SYMBOL_GRAPH_FILE" ]; then
    echo "Documentation check failed: SwiftPM did not produce a symbol graph." >&2
    exit 1
fi

xcrun swift Scripts/check_public_api_documentation.swift "$SYMBOL_GRAPH_FILE"
SYMBOL_GRAPH_DIRECTORY="$TEMPORARY_OUTPUT/symbolgraphs"
mkdir -p "$SYMBOL_GRAPH_DIRECTORY"
cp "$SYMBOL_GRAPH_FILE" "$SYMBOL_GRAPH_DIRECTORY/"

xcrun docc convert Sources/GraniteMLX/GraniteMLX.docc \
    --additional-symbol-graph-dir "$SYMBOL_GRAPH_DIRECTORY" \
    --output-path "$TEMPORARY_OUTPUT/GraniteMLX.doccarchive" \
    --fallback-display-name GraniteMLX \
    --fallback-bundle-identifier com.kylehowells.GraniteMLX \
    --fallback-default-module-kind Library \
    --analyze \
    --warnings-as-errors \
    --experimental-documentation-coverage \
    --coverage-summary-level brief
