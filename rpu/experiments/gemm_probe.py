"""Gate B: does `GemmExecPlan` actually compute C = A @ B on the real RTL?

Companion to gate_b_probe.py, which established that the existing instructions do NOT
(D-109). This one issues the new GEMM function code against a config built from
RpuConfigs.gemm*, and checks a single tile against numpy.

    uv run ../../rpu/experiments/gemm_probe.py --config RpuGemm4X4Fp16Config

Run from generators/fsa/python.
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

# fsa.kernel keeps its context in a module global, and `fsa.kernel` as an attribute of
# the package resolves to the decorator function rather than the module, so reach the
# module explicitly. Kept here rather than added to the submodule: generators/fsa is a
# separate git repo and anything written there would be untracked by our fork (D-106).
_K = importlib.import_module("fsa.kernel")

# ISA.MX_FUNC_BITS is 5 (32 codes); upstream uses 0-4. Must match RpuMxFunc.GEMM in
# generators/chipyard/src/main/scala/rpu/GemmExecPlan.scala.
GEMM_FUNC = 5


def _ctx():
    ctx = getattr(_K, "__g_kernel_ctx")
    assert ctx is not None, "mx_gemm must be called inside an @F.kernel function"
    return ctx


def mx_gemm(b_t: STile, c_t: ATile, accumulate: bool, sem, aq: bool = True, rl: bool = True) -> None:
    """Emit one GEMM instruction: C (+)= reg @ B, where reg is the stationary tile.

    Descriptor flags mirror mx_attn_value, whose dataflow GemmExecPlan reuses:
    revInput=True, revOutput=False, delayOutput=True. `accumulate=False` sets
    MatrixInstrucionAcc.zero, seeding the accumulator instead of adding to it.
    """
    ctx = _ctx()
    header = _K.build_matrix_instruction_header(GEMM_FUNC, False, sem, aq, rl)
    spad = MatrixInstructionSpad(ctx.tile_row_addr(b_t), ctx.tile_stride(b_t), True, False, True)
    acc = MatrixInstrucionAcc(ctx.tile_row_addr(c_t), ctx.tile_stride(c_t), not accumulate)
    ctx.push(MatrixInstruction(header, spad, acc))


@F.kernel
def gemm_one_tile(A: MTile, B_t: MTile) -> MTile:
    """C^T = (A @ B)^T for one tile.

    Shapes follow the attention mapping: stationary A is (cols, rows), the streamed
    operand is B^T at (rows, rows), and the accumulator holds C^T at (rows, cols).
    """
    M, K = A.shape
    N, Kb = B_t.shape
    assert K == Kb, f"inner dimensions disagree: A {A.shape}, B^T {B_t.shape}"

    C_t = F.alloc_mem((N, M), F.fp32)
    A_tile = F.alloc_spad((M, K))
    B_t_tile = F.alloc_spad((N, K))
    C_t_tile = F.alloc_accumulator((N, M))

    sem_a = F.Semaphore(id=0, n=2)
    sem_b = F.Semaphore(id=1, n=2)
    sem_c = F.Semaphore(id=2, n=2)

    F.load_tile(A, A_tile, sem_a)
    F.mx_load_stationary(A_tile.reverse(dim=0), sem_a)
    F.load_tile(B_t, B_t_tile, sem_b)
    mx_gemm(B_t_tile, C_t_tile, False, sem_b)
    F.store_tile(C_t_tile, C_t, sem_c)
    F.fence(mx=True, dma=True, stop=True)
    return C_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
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
    print(f"config {args.config}: {rows} rows x {cols} cols")

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

    C_ref = A_np.astype(np.float32) @ B_np.astype(np.float32)
    err = np.abs(C_fsa.astype(np.float64) - C_ref.astype(np.float64))
    rel = err.max() / max(np.abs(C_ref).max(), 1e-30)

    print(f"\nC = A @ B  [{cols}x{rows}] @ [{rows}x{rows}] via GemmExecPlan:")
    print(f"  max abs err : {err.max():.6e}")
    print(f"  max rel err : {rel:.6e}")
    print(f"  fsa[0,:4]   : {C_fsa.reshape(-1)[:4]}")
    print(f"  ref[0,:4]   : {C_ref.reshape(-1)[:4]}")

    # fp16 operands into an fp32 accumulator over K=rows terms.
    if rel < 1e-2:
        print("\nRESULT: GemmExecPlan computes a general single-tile GEMM")
        return 0
    print("\nRESULT: GemmExecPlan does not yet compute A @ B")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
