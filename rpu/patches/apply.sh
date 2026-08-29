#!/usr/bin/env bash
# Apply RPU experiment patches to the FSA submodule.
#
# Takes patch numbers: `apply.sh 02` applies only 02. With no argument it applies
# NOTHING and lists what is available -- applying everything by default once silently
# re-applied patch 01, which D-130 keeps reverted because it demonstrably buys nothing.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY/generators/fsa" || exit 1
if [ $# -eq 0 ]; then
  echo "usage: apply.sh <patch-number> [...]   e.g. apply.sh 02"
  echo "available:"
  for p in "$CY"/rpu/patches/*.patch; do echo "  $(basename "$p")"; done
  exit 2
fi
for n in "$@"; do
  p=$(ls "$CY"/rpu/patches/"$n"-*.patch 2>/dev/null | head -1)
  [ -n "$p" ] || { echo "  no patch numbered $n"; exit 1; }
  if git apply --reverse --check "$p" 2>/dev/null; then echo "  already applied: $(basename "$p")"
  elif git apply "$p" 2>/dev/null; then echo "  applied: $(basename "$p")"
  else echo "  FAILED: $(basename "$p")"; exit 1; fi
done
