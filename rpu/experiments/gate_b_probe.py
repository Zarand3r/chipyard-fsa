"""Gate B probe -- KEPT AS A RECORD OF A WRONG ANSWER. Do not trust its conclusion.

This probe reported that the existing instructions cannot compute A @ B. That is false:
its stationary operand used `reverse(dim=0)`, copied from the attention kernel, when the
array wants `rev_both` (rows AND columns). With the layout fixed, func 2 and func 5 both
compute the product at rel err ~2.5e-08. See DECISIONS.md D-110 for the retraction and
rpu_gemm.py for the working implementation; gate_b_test.py is the real suite.

Original description follows.

Gate B probe: is a single-tile GEMM already expressible with existing FSA instructions?

rpu/GATE_B_FEASIBILITY.md argued back and forth about this. The Scala plans suggested
ATTN_VALUE was already a GEMM; the Python kernel suggested it was not, because its left
operand is the softmax result P. Reading PE.scala settles the mechanism and leaves one
question that only an experiment answers:

  ATTN_VALUE multiplies `reg` -- the stationary register -- by the streamed operand.
  Its execution plan contains no `load_reg_*` and no `update_reg`, so it does not touch
  `reg` at all. In the attention kernel `reg` happens to hold P, because ATTN_SCORE
  overwrote it. Issue LOAD_STATIONARY(A) then ATTN_VALUE(B) with no score step in
  between, and `reg` still holds A.

If that is right, `C = A @ B` for one tile needs *no RTL change whatsoever*, and Gate B
collapses to tiling in Python. This probe tests exactly that claim on the real Verilator
RTL, at the smallest config, against numpy.

    uv run ../../rpu/experiments/gate_b_probe.py --config FSA4X4Fp16Config

Run it from generators/fsa/python so the fsa package and its uv env resolve.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# This file lives under rpu/ rather than inside generators/fsa/ so that our code stays
# separable from upstream (D-106). uv run does not put the invocation directory on
# sys.path, so add it explicitly.
sys.path.insert(0, os.getcwd())

import fsa as F                                                       # noqa: E402
from fsa.tensor import MTile                                          # noqa: E402


@F.kernel
def gemm_one_tile(A: MTile, B_t: MTile) -> MTile:
    """C^T = (A @ B)^T for a single tile, using only existing FSA instructions.

    Shapes follow the attention kernel's mapping with d = sa_rows and br = sa_cols:
    the stationary operand is (cols, rows), the streamed operand is (rows, rows), and
    the accumulator tile is (rows, cols) -- i.e. C transposed.
    """
    M, K = A.shape
    Kb, N = B_t.shape
    assert K == Kb, f"inner dimensions disagree: A is {A.shape}, B^T is {B_t.shape}"

    C_t = F.alloc_mem((K, M), F.fp32)
    A_tile = F.alloc_spad((M, K))
    B_t_tile = F.alloc_spad((K, N))
    C_t_tile = F.alloc_accumulator((K, M))

    sem_a = F.Semaphore(id=0, n=2)
    sem_b = F.Semaphore(id=1, n=2)
    sem_c = F.Semaphore(id=2, n=2)

    F.load_tile(A, A_tile, sem_a)
    # The attention kernel reverses the stationary tile along dim 0 to match the
    # systolic skew; do the same rather than inventing a different convention.
    F.mx_load_stationary(A_tile.reverse(dim=0), sem_a)

    F.load_tile(B_t, B_t_tile, sem_b)
    # accumulate=False -> MatrixInstrucionAcc.zero, so the accumulator is seeded rather
    # than added to. This is the k-tile seed case.
    F.mx_attn_value(B_t_tile, C_t_tile, False, sem_b)

    F.store_tile(C_t_tile, C_t, sem_c)
    F.fence(mx=True, dma=True, stop=True)
    return C_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="FSA4X4Fp16Config")
    ap.add_argument("--build_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build = args.build_dir or f"../../../sims/verilator/generated-src/chipyard.harness.TestHarness.{args.config}"
    cfg_file = os.path.join(build, f"chipyard.harness.TestHarness.{args.config}.FSAConfig.json")
    if not os.path.isfile(cfg_file):
        print(f"config not found: {cfg_file}")
        return 2
    print(f"Loading config from: {cfg_file}")
    F.init(cfg_file)
    cfg = F.get_config()
    rows, cols = cfg.sa_rows, cfg.sa_cols
    print(f"array: {rows} rows x {cols} cols")

    rng = np.random.default_rng(args.seed)
    A_np = rng.normal(size=(cols, rows)).astype(np.float16)
    B_np = rng.normal(size=(rows, rows)).astype(np.float16)

    sim_bin = f"../../../sims/verilator/simulator-chipyard.harness-{args.config}"
    if not os.path.isfile(sim_bin):
        print(f"simulator binary not found: {sim_bin}")
        return 2
    engine = F.VerilatorSimulator(sim_bin, max_cycles=0)
    A = F.from_numpy(A_np)
    B_t = F.from_numpy(np.ascontiguousarray(B_np.T))
    C_t = engine.execute(gemm_one_tile(A, B_t))
    C_fsa = F.to_numpy(C_t).T

    C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32))

    err = np.abs(C_fsa.astype(np.float64) - C_ref.astype(np.float64))
    rel = err.max() / max(np.abs(C_ref).max(), 1e-30)
    print("\nA @ B via LOAD_STATIONARY + ATTN_VALUE, no RTL change:")
    print(f"  max abs err : {err.max():.6e}")
    print(f"  max rel err : {rel:.6e}")
    print(f"  fsa[0,:4]   : {C_fsa.reshape(-1)[:4]}")
    print(f"  ref[0,:4]   : {C_ref.reshape(-1)[:4]}")

    # fp16 inputs into an fp32 accumulator over K=rows terms. Anything near 1e-3 is
    # arithmetic; a wrong dataflow shows up orders of magnitude larger, or as garbage.
    if rel < 1e-2:
        print("\nRESULT: existing instructions already compute a single-tile GEMM")
        return 0

    # A mismatch alone does not say WHY. Before concluding the mechanism is wrong,
    # rule out the boring explanation: that the operands are laid out differently than
    # assumed. If the hardware computed some transpose or operand order, that is a
    # layout bug in this probe, not evidence about ATTN_VALUE.
    a32, b32 = A_np.astype(np.float32), B_np.astype(np.float32)
    raw = F.to_numpy(C_t)
    candidates = {
        "A @ B":        a32 @ b32,
        "(A @ B).T":    (a32 @ b32).T,
        "A @ B.T":      a32 @ b32.T,
        "A.T @ B":      a32.T @ b32,
        "B @ A":        b32 @ a32,
        "(B @ A).T":    (b32 @ a32).T,
        "B.T @ A":      b32.T @ a32,
        "A.T @ B.T":    a32.T @ b32.T,
    }
    print("\n  candidate layouts, against both the raw accumulator and its transpose:")
    best = (None, float("inf"))
    for label, cand in candidates.items():
        for what, got in (("C_t raw", raw), ("C_t.T", raw.T)):
            if got.shape != cand.shape:
                continue
            r = np.abs(got.astype(np.float64) - cand.astype(np.float64)).max() / max(
                np.abs(cand).max(), 1e-30)
            flag = "  <-- MATCH" if r < 1e-2 else ""
            print(f"    {label:<12} vs {what:<8} rel {r:.3e}{flag}")
            if r < best[1]:
                best = (f"{label} vs {what}", r)
    print(f"\n  closest: {best[0]} at rel {best[1]:.3e}")

    if best[1] < 1e-2:
        print("\nRESULT: the instructions DO compute a GEMM -- this probe had the "
              "operand layout wrong. Fix the layout, not the RTL.")
        return 0
    print("\nRESULT: no operand layout reproduces the output, so ATTN_VALUE is not a "
          "general GEMM. A GemmExecPlan is genuinely required.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
