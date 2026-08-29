#!/usr/bin/env bash
# Phase 5: one DiT block through the verification chain.
#
#   PyTorch functional golden  --(tolerance)-->  numerical golden  <--(exact)-->  RTL
#
# The FPGA leg is SKIP (D-102), so the chain is closed to Verilator only and this gate
# does NOT close phase 5 as the roadmap words it.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY" || exit 1
export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/condabin:$HOME/.local/bin:$PATH"
set +u; source env.sh || exit 1; set -u
[ -d workloads/dit/build/dit_xl2_block0 ] || {
  echo "  regenerating the phase-1 trace (gitignored per D-107)"
  "$HOME/rpu-simulation/reference/dit-venv/bin/python" \
    workloads/dit/trace_block.py --out workloads/dit/build/dit_xl2_block0 || exit 1
}
cd generators/fsa/python || exit 1
uv run --no-sync ../../../rpu/experiments/phase5_chain.py --config "${CONFIG:-RpuGemm16X16Fp16Config}"
