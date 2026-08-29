"""Gate B regression suite: tiled GEMM on the FSA array, checked against numpy.

Every case runs on the real Verilator RTL. Cases are ordered so a failure names the
smallest broken thing: one tile, then k-accumulation, then m/n tiling, then the shapes
the phase-1 DiT trace actually contains.

    uv run ../../rpu/experiments/gate_b_test.py --config RpuGemm4X4Fp16Config
    uv run ../../rpu/experiments/gate_b_test.py --config RpuGemm16X16Fp16Config --func 5

Run from generators/fsa/python.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rpu_gemm as G                                                  # noqa: E402


def check(name: str, config: str, rows: int, cols: int, m: int, n: int, k: int,
          func: int, seed: int, tol: float) -> bool:
    # Fresh FSA state per case: allocations are never freed and a raised kernel leaves
    # the kernel context set. See rpu_gemm.reset().
    G.reset()
    G.load_config(config)
    engine = G.make_engine(config)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, k)).astype(np.float16)
    B = rng.normal(size=(k, n)).astype(np.float16)
    try:
        got = G.run(engine, A, B, rows, cols, func=func)
    except Exception as e:                                   # noqa: BLE001
        print(f"  FAIL  {name:<34} raised {type(e).__name__}: {e}")
        return False
    ref = A.astype(np.float32) @ B.astype(np.float32)
    rel = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()
                / max(np.abs(ref).max(), 1e-30))
    s = G.Shape(m, n, k, rows, cols)
    ok = rel < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} "
          f"[{m}x{k}]@[{k}x{n}]  {s.stationary_loads:>5} loads  rel {rel:.3e}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
    ap.add_argument("--func", type=int, default=G.ATTN_VALUE_FUNC,
                    help="2 = ATTN_VALUE (upstream), 5 = GemmExecPlan (ours)")
    ap.add_argument("--seed", type=int, default=0)
    # fp16 operands into an fp32 accumulator. Error grows with K, so the tolerance is
    # derived from the contraction length rather than picked to make cases pass.
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--full", action="store_true",
                    help="include the large phase-1 DiT shapes (slow)")
    args = ap.parse_args()

    rows, cols = G.load_config(args.config)
    print(f"config {args.config}: {rows} rows x {cols} cols, func={args.func}")

    R, C = rows, cols
    cases = [
        ("single tile",                C,     R,     R),
        ("k accumulation x2",          C,     R,   2 * R),
        ("k accumulation x4",          C,     R,   4 * R),
        ("n tiling x2",                C, 2 * R,     R),
        ("m tiling x2",            2 * C,     R,     R),
        ("m x n tiling",           2 * C, 2 * R,     R),
        ("m x n x k tiling",       2 * C, 2 * R,   2 * R),
    ]
    if args.full:
        # Real DiT-XL/2 block shapes, rounded to the tiling constraint.
        cases += [
            ("dit qkv_proj-like",   256, 3 * 384, 1152 // R * R),
        ]

    ok = True
    for name, m, n, k in cases:
        tol = args.tol if args.tol is not None else max(1e-3, 3e-4 * (k / R) ** 0.5)
        ok &= check(name, args.config, rows, cols, m, n, k, args.func, args.seed, tol)

    print("\nGATE B:", "all cases PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
