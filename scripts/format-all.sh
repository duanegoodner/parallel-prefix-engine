#!/bin/bash
# -----------------------------------------------------------------------------
# format-all.sh
#
# Usage:
#   ./scripts/format-all.sh           # Format all source files in-place
#   ./scripts/format-all.sh --check   # Check for formatting issues (CI-safe)
#   ./scripts/format-all.sh --diff    # Show formatted diff (non-destructive)
# -----------------------------------------------------------------------------

set -e

MODE=${1:-format}
CLANG_FORMAT=clang-format

# File matcher
FILES=$(find . -type f \( -name '*.cpp' -o -name '*.hpp' \) \
  -not -path './build/*' \
  -not -path './_deps/*')

if [ "$MODE" == "--check" ]; then
  echo "[format] Checking for formatting issues..."
  FAIL=0
  for file in $FILES; do
    if ! diff -u "$file" <($CLANG_FORMAT "$file"); then
      echo "⛔ Formatting issue: $file"
      FAIL=1
    fi
  done
  exit $FAIL

elif [ "$MODE" == "--diff" ]; then
  echo "[format] Showing diffs for formatted changes..."
  for file in $FILES; do
    diff -u "$file" <($CLANG_FORMAT "$file") || true
  done
else
  echo "[format] Formatting files in-place..."
  echo "$FILES" | xargs $CLANG_FORMAT -i
  echo "[format] Done."
fi
