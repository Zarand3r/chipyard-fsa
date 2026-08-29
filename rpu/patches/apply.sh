#!/usr/bin/env bash
# Apply RPU experiment patches to the FSA submodule. Idempotent-ish: re-applying a
# patch that is already in place is reported and skipped.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY/generators/fsa" || exit 1
for p in "$CY"/rpu/patches/*.patch; do
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "  already applied: $(basename "$p")"
  elif git apply "$p" 2>/dev/null; then
    echo "  applied: $(basename "$p")"
  else
    echo "  FAILED: $(basename "$p")"; exit 1
  fi
done
