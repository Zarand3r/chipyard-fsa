"""Decisive check: is the GEMM output simply row-reversed?

gemm_identity.py showed A = I returns B with its rows reversed. That is a skew
convention, not a broken dataflow -- and it means the sixteen-layout sweep in
gate_b_probe.py missed the answer, because it tested transposes and operand orders but
never a row reversal.

This script tests one instruction sequence (selected by --func) against A @ B both
directly and row-reversed, so the two probes can be compared on equal terms.

    uv run ../../rpu/experiments/gemm_reversal_check.py --func gemm
    uv run ../../rpu/experiments/gemm_reversal_check.py --func attn_value --config FSA4X4Fp16Config
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import fsa as F                                                       # noqa: E402
from fsa.instructions import (                                        # noqa: E402
    MatrixInstruction, MatrixInstructionSpad, MatrixInstrucionAcc,
)
from fsa.tensor import ATile, MTile, STile                            # noqa: E402

_K = importlib.import_module("fsa.kernel")
GEMM_FUNC = 5
ATTN_VALUE_FUNC = 2


def _emit(func: int, b_t: STile, c_t: ATile, accumulate: bool, sem) -> None:
    ctx = getattr(_K, "__g_kernel_ctx")
    header = _K.build_matrix_instruction_header(func, False, sem, True, True)
    spad = MatrixInstructionSpad(ctx.tile_row_addr(b_t), ctx.tile_stride(b_t), True, False, True)
    acc = MatrixInstrucionAcc(ctx.tile_row_addr(c_t), ctx.tile_stride(c_t), not accumulate)
    ctx.push(MatrixInstruction(header, spad, acc))


def build_kernel(func: int, reverse_stationary: bool):
    @F.kernel
    def k(A: MTile, B_t: MTile) -> MTile:
        M, K = A.shape
        N, _ = B_t.shape
        C_t = F.alloc_mem((N, M), F.fp32)
        A_tile = F.alloc_spad((M, K))
        B_t_tile = F.alloc_spad((N, K))
        C_t_tile = F.alloc_accumulator((N, M))
        sem_a, sem_b, sem_c = (F.Semaphore(id=i, n=2) for i in range(3))
        F.load_tile(A, A_tile, sem_a)
        F.mx_load_stationary(A_tile.reverse(dim=0) if reverse_stationary else A_tile, sem_a)
        F.load_tile(B_t, B_t_tile, sem_b)
        _emit(func, B_t_tile, C_t_tile, False, sem_b)
        F.store_tile(C_t_tile, C_t, sem_c)
        F.fence(mx=True, dma=True, stop=True)
        return C_t
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", choices=["gemm", "attn_value"], default="gemm")
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
    ap.add_argument("--no-reverse", action="store_true",
                    help="do not reverse the stationary tile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build = f"../../../sims/verilator/generated-src/chipyard.harness.TestHarness.{args.config}"
    cfg_file = os.path.join(build, f"chipyard.harness.TestHarness.{args.config}.FSAConfig.json")
    if not os.path.isfile(cfg_file):
        print(f"config not found: {cfg_file}")
        return 2
    F.init(cfg_file)
    cfg = F.get_config()
    rows, cols = cfg.sa_rows, cfg.sa_cols

    sim_bin = f"../../../sims/verilator/simulator-chipyard.harness-{args.config}"
    engine = F.VerilatorSimulator(sim_bin, max_cycles=0)

    rng = np.random.default_rng(args.seed)
    A_np = rng.normal(size=(cols, rows)).astype(np.float16)
    B_np = rng.normal(size=(rows, rows)).astype(np.float16)

    func = GEMM_FUNC if args.func == "gemm" else ATTN_VALUE_FUNC
    kern = build_kernel(func, reverse_stationary=not args.no_reverse)
    C_t = engine.execute(kern(F.from_numpy(A_np),
                              F.from_numpy(np.ascontiguousarray(B_np.T))))
    got = F.to_numpy(C_t).T
    ref = A_np.astype(np.float32) @ B_np.astype(np.float32)

    def rel(a, b):
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
                     / max(np.abs(b).max(), 1e-30))

    direct = rel(got, ref)
    rowrev = rel(got, ref[::-1, :])
    colrev = rel(got, ref[:, ::-1])
    print(f"\nfunc={args.func} ({func}) config={args.config} "
          f"reverse_stationary={not args.no_reverse}")
    print(f"  vs A@B                : rel {direct:.6e}")
    print(f"  vs (A@B) row-reversed : rel {rowrev:.6e}")
    print(f"  vs (A@B) col-reversed : rel {colrev:.6e}")

    best = min(direct, rowrev, colrev)
    if best < 1e-2:
        which = {direct: "A@B", rowrev: "(A@B) row-reversed",
                 colrev: "(A@B) col-reversed"}[best]
        print(f"\nRESULT: computes {which}")
        return 0
    print("\nRESULT: matches none of them")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
