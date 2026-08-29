#!/usr/bin/env bash
# Gate B -- prove FSA can become the RPU array.
#
#   PyTorch GEMM  <->  Verilator  <->  U55C FPGA
#
# The FPGA leg cannot run here (D-102); it is reported SKIP, never PASS, and this gate
# is therefore NOT fully closed even when every case below passes.
#
# Runs the tiled-GEMM regression for both function codes so the comparison D-110 asks
# for -- does GemmExecPlan earn its place over plain ATTN_VALUE? -- stays visible.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY" || exit 1
CONFIG=${CONFIG:-RpuGemm4X4Fp16Config}

export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/condabin:$HOME/.local/bin:$PATH"
# chipyard's activate-riscv-tools.sh dereferences $RISCV before setting it.
set +u
# shellcheck disable=SC1091
source env.sh || { echo "  FAIL  env.sh"; exit 1; }
set -u

fail=0
cd generators/fsa/python || exit 1
for func in 2 5; do
  printf '\n== function code %s (%s) ==\n' "$func" \
    "$([ "$func" = 2 ] && echo 'ATTN_VALUE, upstream' || echo 'GemmExecPlan, ours')"
  uv run ../../../rpu/experiments/gate_b_test.py --config "$CONFIG" --func "$func" || fail=1
done

printf '\n== U55C FPGA ==\n  SKIP  no Vivado install, no Xilinx PCIe device, XDMA rescan needs root\n'
echo
if [ $fail -eq 0 ]; then
  echo "GATE B: simulation legs PASSED; FPGA leg SKIPPED -- gate is NOT fully closed"
else
  echo "GATE B: FAILED"
fi
exit $fail
