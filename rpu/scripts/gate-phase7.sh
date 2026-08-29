#!/usr/bin/env bash
# Phase 7: a complete small one-step DiT block on the array, with FUSED attention.
#
# The model is synthetic and tile-aligned by construction (D-125). This gate tests
# PIPELINE COMPLETENESS, not fidelity -- phase 5 owns fidelity, on the real checkpoint.
# No number from here may be quoted about DiT-XL/2 or about model quality.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY" || exit 1
export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/condabin:$HOME/.local/bin:$PATH"
set +u; source env.sh || exit 1; set -u
cd generators/fsa/python || exit 1
uv run --no-sync ../../../rpu/experiments/phase7_dit.py --config "${CONFIG:-RpuGemm16X16Fp16Config}"
