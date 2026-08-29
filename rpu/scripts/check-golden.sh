#!/usr/bin/env bash
# Phase 3 unit vectors for the RPU numerical golden model (GOLDEN_MODEL_SPEC §10).
#
# Pure software: no RTL, no array, no FPGA. Runs in seconds and is the fastest gate in
# the program, so there is no excuse for not running it.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-$HOME/rpu-simulation/reference/dit-venv/bin/python}
fail=0
for t in test_reduce test_formats test_datapath test_state; do
  printf '\n== %s ==\n' "$t"
  "$PY" "$CY/rpu/golden/$t.py" || fail=1
done
echo
[ $fail -eq 0 ] && echo "GOLDEN: all unit vectors PASSED" || echo "GOLDEN: FAILED"
exit $fail
