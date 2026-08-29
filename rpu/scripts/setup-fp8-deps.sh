#!/usr/bin/env bash
# Install ml_dtypes into FSA's uv environment, WITHOUT touching the submodule.
#
# The FP8 shims in rpu/experiments/rpu_gemm.py need ml_dtypes for the OCP-conformant
# numpy dtypes (D-122). `uv add ml_dtypes` would work, but it writes pyproject.toml and
# uv.lock inside generators/fsa -- a separate git repository our fork does not track, so
# the change is invisible to a fresh clone and is lost on a submodule update (D-106).
#
# `uv pip install` targets the environment only, leaving project files alone.
set -euo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV="$CY/generators/fsa/python/.venv/bin/python"
[ -x "$VENV" ] || { echo "FSA venv not found at $VENV -- run a kernel once to create it"; exit 1; }
"$HOME/.local/bin/uv" pip install --python "$VENV" ml_dtypes
"$VENV" -c "import ml_dtypes; print('ml_dtypes', ml_dtypes.__version__, 'ready')"
