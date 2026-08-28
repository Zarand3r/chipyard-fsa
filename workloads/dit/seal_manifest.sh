#!/usr/bin/env bash
# Create or deliberately update the committed golden contract (rpu/DECISIONS.md D-107).
#
# Not part of any gate. Running this is an explicit act: it says "the current golden is
# the one we stand behind". If check_determinism.sh is failing against the committed
# manifest, find out why before you run this.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-$HOME/rpu-simulation/reference/dit-venv/bin/python}
OUT=${OUT:-$HERE/build/dit_xl2_block0}

"$PY" "$HERE/trace_block.py" --out "$OUT"
mkdir -p "$HERE/manifests"
cp "$OUT/manifest.json" "$HERE/manifests/dit_xl2_block0.json"
echo "sealed $HERE/manifests/dit_xl2_block0.json"
