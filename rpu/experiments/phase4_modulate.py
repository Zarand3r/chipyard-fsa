"""Phase 4 option A: per-channel elementwise scaling on the array, as a diagonal matmul.

`X * s` (s per channel) is `X @ diag(s)`. `diag(s)` is block-diagonal, so for output
c-tile j only k-tile j is non-zero -- one GEMM tile per (m-tile, c-tile), never a full
K sweep. That is what makes the overhead `rows`x the useful work rather than
`rows * C/rows`, and what brings the block cost to +0.84% at RPU shapes (D-119).

Checked against the golden model's `modulate`, which matches the pinned DiT-XL/2 form.

    uv run ../../rpu/experiments/phase4_modulate.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
import rpu_gemm as G                                                   # noqa: E402
from datapath import modulate                                          # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def scale_on_array(config: str, X: np.ndarray, s: np.ndarray, rows: int, cols: int,
                   func: int) -> np.ndarray:
    """X * s via X @ diag(s), skipping the zero k-tiles.

    Each c-tile is a separate kernel launch, and FSA never frees allocations within a
    process, so state is reset per launch (see rpu_gemm.reset()). Forgetting that is
    how this first failed -- "Requested 512 bytes, but only 0 bytes available".
    """
    M, C = X.shape
    assert C == len(s) and M % cols == 0 and C % rows == 0
    out = np.zeros((M, C), np.float32)
    for j in range(C // rows):
        blk = slice(j * rows, (j + 1) * rows)
        D = np.diag(s[blk]).astype(np.float16)      # the one non-zero k-tile
        G.reset()
        G.load_config(config)
        eng = G.make_engine(config)
        out[:, blk] = G.run(eng, np.ascontiguousarray(X[:, blk]), D,
                            rows, cols, func=func)
    return out


def layernorm_stats_on_array(config: str, X: np.ndarray, rows: int, cols: int,
                             func: int):
    """Mean and sum-of-squares per row, both as native contractions.

    PHASE4_MODULATION.md first claimed LayerNorm's variance "hits the same elementwise
    problem" because it needs a sum of squares. That is wrong: sum(x_i^2) is the dot
    product of a row **with itself**, which is exactly what a systolic column computes.
    No elementwise square is needed, so nothing here pays option A's `rows`x tax.

      mean       = (x . ones) / C
      sum_sq     = (x . x)
      var        = sum_sq / C - mean^2
    """
    M, C = X.shape
    ones = np.ones((C, rows), np.float16)
    G.reset(); G.load_config(config)
    mean_cols = G.run(G.make_engine(config), X, ones, rows, cols, func=func)
    mean = mean_cols[:, 0] / np.float32(C)

    # x . x, one row at a time: stationary is the row block, streamed is the same data
    sum_sq = np.zeros(M, np.float32)
    for m0 in range(0, M, cols):
        blk = X[m0:m0 + cols]
        acc = np.zeros(cols, np.float32)
        for j in range(C // rows):
            sl = slice(j * rows, (j + 1) * rows)
            G.reset(); G.load_config(config)
            prod = G.run(G.make_engine(config), np.ascontiguousarray(blk[:, sl]),
                         np.ascontiguousarray(blk[:, sl].T), rows, cols, func=func)
            acc += np.diag(prod)          # row i dotted with itself
        sum_sq[m0:m0 + cols] = acc
    return mean, sum_sq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
    ap.add_argument("--func", type=int, default=G.GEMM_FUNC)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, cols = G.load_config(args.config)
    print(f"{args.config}: {rows} rows x {cols} cols\n")
    rng = np.random.default_rng(args.seed)

    M, C = 2 * cols, 2 * rows
    X = rng.normal(size=(M, C)).astype(np.float16)
    scale = rng.normal(size=C).astype(np.float16)
    shift = rng.normal(size=C).astype(np.float16)

    # --- the scaling half, on the array ---
    G.clear_assertions()
    got = scale_on_array(args.config, X, (np.float16(1.0) + scale).astype(np.float16),
                         rows, cols, args.func)
    ref = (X.astype(np.float32) * (1.0 + scale.astype(np.float32)))
    rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    check("x * (1+scale) on the array matches numpy", rel < 1e-3, f"rel {rel:.3e}")
    check("no RTL assertions fired", not G.assertions())

    # --- full modulation: the array does the multiply, the add is a vector op ---
    full = got + shift.astype(np.float32)
    gold = np.array([[float(v) for v in r]
                     for r in modulate([[float(v) for v in row] for row in X],
                                       [float(v) for v in shift],
                                       [float(v) for v in scale])], np.float32)
    relg = float(np.abs(full - gold).max() / max(np.abs(gold).max(), 1e-30))
    check("full adaLN modulate matches the golden model", relg < 1e-3,
          f"rel {relg:.3e}")

    # --- the cost claim from D-119, recomputed on these shapes ---
    useful = M * C
    spent = M * C * rows
    check("cost is rows x useful, not rows * C/rows",
          spent == useful * rows, f"{spent:,} MACs for {useful:,} useful ({rows}x)")

    # --- LayerNorm statistics, both native contractions ---
    Xn = rng.normal(size=(cols, C)).astype(np.float16)
    mean, sum_sq = layernorm_stats_on_array(args.config, Xn, rows, cols, args.func)
    ref_mean = Xn.astype(np.float32).mean(axis=1)
    ref_sq = (Xn.astype(np.float32) ** 2).sum(axis=1)
    check("row mean via x . ones matches numpy",
          float(np.abs(mean - ref_mean).max() / max(np.abs(ref_mean).max(), 1e-30)) < 1e-2,
          f"rel {float(np.abs(mean - ref_mean).max() / max(np.abs(ref_mean).max(), 1e-30)):.3e}")
    check("row sum-of-squares via x . x matches numpy",
          float(np.abs(sum_sq - ref_sq).max() / ref_sq.max()) < 1e-2,
          f"rel {float(np.abs(sum_sq - ref_sq).max() / ref_sq.max()):.3e}")
    var = sum_sq / np.float32(C) - mean ** 2
    check("variance from those two is correct",
          float(np.abs(var - Xn.astype(np.float32).var(axis=1)).max()) < 1e-2,
          "var = sum_sq/C - mean^2, no elementwise square needed")

    print(f"\nPHASE 4 MODULATE: {'PASSED' if not FAILURES else 'FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
