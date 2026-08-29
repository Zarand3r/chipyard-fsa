#!/usr/bin/env bash
# Restore the FSA submodule to its pinned state, whatever patches are applied.
#
# An earlier version hardcoded AXI4FSA.scala and silently left patch 02's changes to
# FSA.scala in place. Reverting the whole submodule is both simpler and correct: it is
# pinned, and nothing of ours is supposed to live inside it (D-106).
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
git -C "$CY/generators/fsa" checkout -- .
echo "  submodule restored to $(git -C "$CY/generators/fsa" rev-parse --short HEAD)"
git -C "$CY/generators/fsa" status --short | grep -q . && echo "  WARNING: still dirty" || echo "  clean"
