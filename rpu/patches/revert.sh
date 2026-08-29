#!/usr/bin/env bash
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
git -C "$CY/generators/fsa" checkout -- src/main/scala/fsa/AXI4FSA.scala
echo "  submodule reverted to pinned state"
