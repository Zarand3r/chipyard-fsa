#!/usr/bin/env bash
# Phase 1's correctness criterion: the golden reproduces bit-for-bit.
#
# Two independent traces, in separate processes, must produce identical sha256s for
# every tensor. A dump that cannot be reproduced is a sample, not a golden model.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-$HOME/rpu-simulation/reference/dit-venv/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for run in a b; do
  echo "== trace run $run =="
  "$PY" "$HERE/trace_block.py" --out "$TMP/$run" || exit 1
done

# Compare tensor checksums only. If a committed manifest exists for this pin, compare
# against it too -- that is what extends determinism from "twice on this machine" to
# "the same as the run that produced the contract" (D-107).
COMMITTED="$HERE/manifests/dit_xl2_block0.json"
"$PY" - "$TMP/a/manifest.json" "$TMP/b/manifest.json" "$COMMITTED" <<'PYEOF'
import json, os, sys

def tensors(path):
    return json.load(open(path))["tensors"]

a, b = tensors(sys.argv[1]), tensors(sys.argv[2])
if a.keys() != b.keys():
    print("FAIL: the two runs produced different tensor sets"); raise SystemExit(1)
bad = [k for k in a if a[k]["sha256"] != b[k]["sha256"]]
if bad:
    print(f"FAIL: {len(bad)} tensors differ between runs: {bad[:5]}"); raise SystemExit(1)
print(f"PASS: {len(a)} tensors identical across two independent traces")

# D-107: the committed manifest is the contract. Disagreeing with it is a failure to
# investigate, not a manifest to refresh.
committed = sys.argv[3]
if not os.path.exists(committed):
    print("NOTE: no committed manifest yet; run seal_manifest.sh to create the contract")
    raise SystemExit(0)
c = tensors(committed)
if a.keys() != c.keys():
    print("FAIL: tensor set differs from the committed manifest"); raise SystemExit(1)
drift = [k for k in a if a[k]["sha256"] != c[k]["sha256"]]
if drift:
    print(f"FAIL: {len(drift)} tensors differ from the committed manifest: {drift[:5]}")
    print("      investigate before updating it -- see rpu/DECISIONS.md D-107")
    raise SystemExit(1)
print(f"PASS: {len(a)} tensors match the committed manifest")
PYEOF
