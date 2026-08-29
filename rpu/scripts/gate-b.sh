#!/usr/bin/env bash
# Gate B -- prove FSA can become the RPU array.
#
#   PyTorch GEMM  <->  Verilator  <->  U55C FPGA
#
# The FPGA leg cannot run here (D-102): SKIP, never PASS, so this gate is not fully
# closed even when everything below passes.
#
# Three deliberate axes, each one a lesson paid for (see rpu/PARANOIA.md):
#   configs  -- D-112 and D-110 were both wrong because a result at 4x4 was generalised
#               to the design. Never report a single-config result as a property.
#   vseeds   -- D-113 was uninitialised RTL state hiding behind a fixed $random seed.
#   asserts  -- Verilator runs with --assert; a firing assertion fails the case.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY" || exit 1

CONFIGS=${CONFIGS:-"RpuGemm4X4Fp16Config RpuGemm8X8Fp16Config RpuGemm16X16Fp16Config"}
VSEEDS=${VSEEDS:-"1,7,12345"}
FUNC=${FUNC:-5}

export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/condabin:$HOME/.local/bin:$PATH"
set +u
# shellcheck disable=SC1091
source env.sh || { echo "  FAIL  env.sh"; exit 1; }
set -u

fail=0
cd generators/fsa/python || exit 1
for cfg in $CONFIGS; do
  printf '\n== %s, func=%s, vseeds=%s ==\n' "$cfg" "$FUNC" "$VSEEDS"
  uv run ../../../rpu/experiments/gate_b_test.py \
      --config "$cfg" --func "$FUNC" --vseeds "$VSEEDS" || fail=1
done

printf '\n== U55C FPGA ==\n  SKIP  no Vivado install, no Xilinx PCIe device, XDMA rescan needs root\n'
echo
if [ $fail -eq 0 ]; then
  echo "GATE B: simulation legs PASSED across all configs and seeds;"
  echo "        FPGA leg SKIPPED -- gate is NOT fully closed"
else
  echo "GATE B: FAILED"
fi
exit $fail
