"""Does GemmExecPlan earn its place? Cycle comparison against plain ATTN_VALUE.

D-110 required the new plan to justify itself on measurements rather than necessity,
and D-111 showed correctness is identical. This measures the only remaining claim:
that dropping the online-softmax declarations makes the instruction shorter.

    uv run ../../rpu/experiments/gemm_cycles.py --config RpuGemm4X4Fp16Config

Run from generators/fsa/python.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import pathlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G                                                  # noqa: E402

# fsa.engine calls subprocess.run(sim_cmd, check=True) with no capture, so the
# performance counters go to the terminal and are unreachable from Python. Wrap it
# here rather than patching the submodule (D-106).
_real_run = subprocess.run
_captured: list[str] = []


def _capturing_run(*a, **kw):
    if "capture_output" not in kw and "stdout" not in kw:
        kw["capture_output"] = True
        kw["text"] = True
        r = _real_run(*a, **kw)
        _captured.append((r.stdout or "") + (r.stderr or ""))
        return r
    return _real_run(*a, **kw)


subprocess.run = _capturing_run

COUNTERS = ("execTime", "mxActive", "mxBubble", "mxInst", "dmaActive", "dmaInst", "rawInst")


def counters(text: str) -> dict[str, int]:
    out = {}
    for name in COUNTERS:
        m = re.findall(rf"FSA:\s*{name}\s*=\s*(\d+)", text)
        if m:
            out[name] = int(m[-1])
    return out


def measure(config: str, m: int, n: int, k: int, func: int, seed: int):
    G.reset()
    rows, cols = G.load_config(config)
    engine = G.make_engine(config)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, k)).astype(np.float16)
    B = rng.normal(size=(k, n)).astype(np.float16)
    _captured.clear()
    got = G.run(engine, A, B, rows, cols, func=func)
    ref = A.astype(np.float32) @ B.astype(np.float32)
    rel = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()
                / max(np.abs(ref).max(), 1e-30))
    text = "\n".join(_captured)
    if os.environ.get("GEMM_CYCLES_DEBUG"):
        pathlib.Path("/tmp/gemm_cycles_capture.txt").write_text(text)
        print(f"[debug] captured {len(_captured)} chunks, {len(text)} chars")
    return counters(text), rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, cols = G.load_config(args.config)
    R, C = rows, cols
    shapes = [
        ("single tile",       C,     R,     R),
        ("k x4",              C,     R, 4 * R),
        ("m x n x k",     2 * C, 2 * R, 2 * R),
    ]

    print(f"{args.config}: {rows}x{cols}\n")
    print(f"{'shape':<14} {'func':>5} {'execTime':>9} {'mxActive':>9} {'mxBubble':>9} "
          f"{'mxInst':>7} {'rel err':>11}")
    verdict = []
    for name, m, n, k in shapes:
        row = {}
        for func in (G.ATTN_VALUE_FUNC, G.GEMM_FUNC):
            c, rel = measure(args.config, m, n, k, func, args.seed)
            row[func] = c.get("execTime", -1)
            print(f"{name:<14} {func:>5} {c.get('execTime',-1):>9} {c.get('mxActive',-1):>9} "
                  f"{c.get('mxBubble',-1):>9} {c.get('mxInst',-1):>7} {rel:>11.3e}")
        a, g = row[G.ATTN_VALUE_FUNC], row[G.GEMM_FUNC]
        verdict.append((name, a, g))

    print("\nverdict (D-110: GemmExecPlan must justify itself on measurements):")
    any_better = False
    for name, a, g in verdict:
        if a <= 0 or g <= 0:
            print(f"  {name:<14} counters unavailable")
            continue
        d = g - a
        any_better |= d < 0
        print(f"  {name:<14} ATTN_VALUE {a:>6}  GemmExecPlan {g:>6}  "
              f"{'saves' if d < 0 else 'costs' if d > 0 else 'identical'} "
              f"{abs(d)} cycles ({100.0 * d / a:+.1f}%)")
    if not any_better:
        print("\n  GemmExecPlan is not faster on any shape measured. On this evidence it\n"
              "  should be deleted rather than kept: it adds a function code, a plan and a\n"
              "  config for no correctness and no speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
