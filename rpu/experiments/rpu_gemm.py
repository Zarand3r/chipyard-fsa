"""Tiled GEMM on the FSA systolic array.

`C = A @ B` for arbitrary shapes, built from FSA instructions. Roadmap phase 2 / Gate B.

Two facts this module exists to encapsulate, both established by experiment and
recorded in rpu/DECISIONS.md D-110:

1. The array computes ``C = rev_both(S) @ B``, where ``S`` is the stationary tile as
   loaded and ``rev_both`` reverses both rows and columns. Callers pass a normal ``A``;
   the reversal is applied here so no caller has to remember it.

2. ``ATTN_VALUE`` (func 2) already does this. ``GemmExecPlan`` (func 5) is selectable
   with ``func=`` so the two can be measured against each other -- D-110 requires the
   new plan to justify itself on numbers rather than on necessity.

Tile mapping, inherited from the attention kernel's use of the array:

    stationary A tile : (cols, rows)      -- M_tile x K_tile
    streamed B^T tile : (rows, rows)      -- N_tile x K_tile
    accumulator C^T   : (rows, cols)      -- N_tile x M_tile

so M tiles are ``cols`` wide and both K and N tiles are ``rows`` wide.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.getcwd())

import fsa as F                                                       # noqa: E402
from fsa.instructions import (                                        # noqa: E402
    MatrixInstruction, MatrixInstructionSpad, MatrixInstrucionAcc,
)
from fsa.tensor import ATile, MTile, STile                            # noqa: E402

# `fsa.kernel` as a package attribute resolves to the decorator, not the module.
_K = importlib.import_module("fsa.kernel")
_C = importlib.import_module("fsa.config")

ATTN_VALUE_FUNC = 2
GEMM_FUNC = 5          # must match RpuMxFunc.GEMM in rpu/GemmExecPlan.scala
SET_ACC_SCALE_FUNC = 6 # must match RpuMxFunc.SET_ACC_SCALE


def mx_gemm(func: int, b_t: STile, c_t: ATile, accumulate: bool, sem, aq=True, rl=True) -> None:
    """One matrix instruction: C = reg @ B, or C += reg @ B when `accumulate`.

    `waitPrevAcc` is set whenever this instruction accumulates. Back-to-back k-tiles
    read-modify-write the same accumulator row, and without the interlock the second
    instruction's `readAccRAM` at cycle `rows + cols - 1` races the first instruction's
    still-draining writeback -- producing accumulator garbage (~1e36) while every
    non-accumulating case passes at ~3e-8. The attention kernel gets away with
    `waitPrevAcc=False` because an ATTN_SCORE always separates two ATTN_VALUEs; a GEMM
    k-loop has no such gap.
    """
    ctx = getattr(_K, "__g_kernel_ctx")
    assert ctx is not None, "mx_gemm must be called inside an @F.kernel function"
    header = _K.build_matrix_instruction_header(func, accumulate, sem, aq, rl)
    spad = MatrixInstructionSpad(ctx.tile_row_addr(b_t), ctx.tile_stride(b_t), True, False, True)
    acc = MatrixInstrucionAcc(ctx.tile_row_addr(c_t), ctx.tile_stride(c_t), not accumulate)
    ctx.push(MatrixInstruction(header, spad, acc))


def ones_operands(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Operands whose product is exactly all-ones, for priming the accumulator scale.

    `A[:, 0] = 1` and `B[0, :] = 1` gives `A @ B = ones`, with every value exactly
    representable in fp16 so the primed scale is exactly 1.0 and not 1.0 +/- an ulp.
    """
    a = np.zeros((cols, rows), np.float16); a[:, 0] = 1
    b = np.zeros((rows, rows), np.float16); b[0, :] = 1
    return a, b


def mx_set_acc_scale(src: ATile, sem) -> None:
    """scale <- one row of AccRAM[src], which must already hold 1.0.

    Required before any accumulating GEMM: `ACC_SA` computes
    `out = scale * sram_in + sa_in`, and `scale` is an unreset register carrying
    FlashAttention's rescale factor. See rpu/GemmExecPlan.scala SetAccScale and
    DECISIONS.md D-111.

    The 1.0 row is produced *on the array* rather than written by DMA: `isAccum` is
    declared in the DMA instruction bundle but is read nowhere in the RTL, so a
    DMA into accumulator SRAM never completes and deadlocks the semaphore.
    """
    ctx = getattr(_K, "__g_kernel_ctx")
    header = _K.build_matrix_instruction_header(SET_ACC_SCALE_FUNC, True, sem, True, True)
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(ctx.tile_row_addr(src), ctx.tile_stride(src), False)
    ctx.push(MatrixInstruction(header, spad, acc))


def stationary(a_tile: np.ndarray) -> np.ndarray:
    """Convert a plain (M_tile, K_tile) block into the layout the array expects."""
    return np.ascontiguousarray(a_tile[::-1, ::-1])


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    rows: int
    cols: int

    @property
    def mt(self) -> int: return self.m // self.cols
    @property
    def nt(self) -> int: return self.n // self.rows
    @property
    def kt(self) -> int: return self.k // self.rows

    def check(self) -> None:
        if self.m % self.cols or self.n % self.rows or self.k % self.rows:
            raise ValueError(
                f"({self.m}x{self.k}) @ ({self.k}x{self.n}) does not tile on a "
                f"{self.rows}x{self.cols} array: M must be a multiple of {self.cols}, "
                f"N and K multiples of {self.rows}"
            )

    @property
    def stationary_loads(self) -> int:
        return self.mt * self.nt * self.kt


def build_kernel(func: int, shape: Shape):
    """Return an @F.kernel that computes every C tile of A @ B.

    Semaphores and scratchpad tiles are allocated ONCE, outside the loops, and
    double-buffered by parity. Allocating them per iteration is a real bug and was one:
    `Semaphore` carries a mutable `value` advanced by `inc()`, so a fresh object each
    iteration restarts the acquire/release sequence and the engine reads a partially
    written tile.
    """
    s = shape

    @F.kernel
    def gemm(A_tiles: list[MTile], B_tiles: list[MTile],
             ones_a: MTile, ones_b: MTile) -> list[MTile]:
        c_out = [F.alloc_mem((s.rows, s.cols), F.fp32) for _ in range(s.mt * s.nt)]
        a_buf = [F.alloc_spad((s.cols, s.rows)) for _ in range(2)]
        b_buf = [F.alloc_spad((s.rows, s.rows)) for _ in range(2)]
        c_acc = F.alloc_accumulator((s.rows, s.cols))

        sem_a = [F.Semaphore(id=0, n=2), F.Semaphore(id=1, n=2)]
        sem_b = [F.Semaphore(id=2, n=2), F.Semaphore(id=3, n=2)]
        sem_c = F.Semaphore(id=4, n=2)
        sem_s = F.Semaphore(id=5, n=2)

        # Prime the accumulator scale to exactly 1.0, once. `scale` is a register and
        # nothing below writes it, so one prologue covers every tile.
        #
        # The 1.0 row is computed into `c_acc` by a GEMM whose product is exactly
        # all-ones, then read back by SET_SCALE. Using `c_acc` rather than a dedicated
        # row matters: accRows is 1 + rows, which does not fit two (rows, cols) tiles.
        # The first real k-tile below sets zero=True and overwrites it.
        if s.kt > 1:
            F.load_tile(ones_a, a_buf[0], sem_a[0])
            F.mx_load_stationary(a_buf[0], sem_a[0])
            F.load_tile(ones_b, b_buf[0], sem_b[0])
            mx_gemm(func, b_buf[0], c_acc, False, sem_b[0])
            mx_set_acc_scale(c_acc, sem_s)

        for mi in range(s.mt):
            for ni in range(s.nt):
                for ki in range(s.kt):
                    buf = ki % 2
                    a_mem = A_tiles[mi * s.kt + ki]
                    b_mem = B_tiles[ki * s.nt + ni]
                    F.load_tile(a_mem, a_buf[buf], sem_a[buf])
                    F.mx_load_stationary(a_buf[buf], sem_a[buf])
                    F.load_tile(b_mem, b_buf[buf], sem_b[buf])
                    mx_gemm(func, b_buf[buf], c_acc, ki > 0, sem_b[buf])
                # Drain the matrix engine before reading the accumulator out.
                #
                # A matrix instruction carries ONE semaphore, and mx_gemm spends it on
                # its scratchpad dependency, so there is no semaphore left to order the
                # store against the accumulation. Without this fence the DMA races the
                # writeback: at 4x4 it happened to land late enough and every case
                # passed, at 16x16 the readout contained ~1e14 garbage in scattered
                # elements. Upstream never needs it because AttentionLseNorm is a
                # blocking instruction that sits between the last ATTN_VALUE and the
                # store, acting as the barrier.
                F.fence(mx=True, dma=False, stop=False)
                F.store_tile(c_acc, c_out[mi * s.nt + ni], sem_c)
        F.fence(mx=True, dma=True, stop=True)
        return c_out

    return gemm


def run(engine, A: np.ndarray, B: np.ndarray, rows: int, cols: int,
        func: int = GEMM_FUNC) -> np.ndarray:
    """Execute A @ B on the array and return C as a normal (M, N) fp32 array."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"inner dimensions disagree: {A.shape} @ {B.shape}"
    s = Shape(m, n, k, rows, cols)
    s.check()

    a_tiles = [F.from_numpy(stationary(A[mi * cols:(mi + 1) * cols,
                                         ki * rows:(ki + 1) * rows]))
               for mi in range(s.mt) for ki in range(s.kt)]
    b_tiles = [F.from_numpy(np.ascontiguousarray(
                   B[ki * rows:(ki + 1) * rows, ni * rows:(ni + 1) * rows].T))
               for ki in range(s.kt) for ni in range(s.nt)]

    oa, ob = ones_operands(rows, cols)
    outs = engine.execute(build_kernel(func, s)(
        a_tiles, b_tiles,
        F.from_numpy(stationary(oa)), F.from_numpy(np.ascontiguousarray(ob.T))))
    if not isinstance(outs, list):
        outs = [outs]

    C = np.zeros((m, n), dtype=np.float32)
    for mi in range(s.mt):
        for ni in range(s.nt):
            # each tile comes back as C^T of shape (N_tile, M_tile)
            blk = F.to_numpy(outs[mi * s.nt + ni]).T
            C[mi * cols:(mi + 1) * cols, ni * rows:(ni + 1) * rows] = blk
    return C


def reset() -> None:
    """Drop all FSA global state so another kernel can be built in this process.

    Two separate leaks make this necessary:

    * `fsa.config` holds the memory manager in a module global, and scratchpad,
      accumulator and DRAM allocations accumulate for the life of the process. A second
      `run()` in the same process hits "not enough memory" even though each kernel fits.
    * `fsa.kernel`'s `@kernel` decorator sets its context on entry and clears it on
      normal return only. If a kernel body raises, the context stays set and every
      later kernel fails with "Nested kernels are not supported yet".

    Neither is our bug to fix upstream, but both must be handled to run more than one
    case per process.
    """
    setattr(_C, "__global_vars", None)
    setattr(_K, "__g_kernel_ctx", None)


def _install_fp8_shims() -> None:
    """Teach FSA's Python side the FP8 element formats.

    The Scala side emits `"e_type": "e4m3"` into FSAConfig.json and `fsa/config.py`
    does `eval(cfg["e_type"])`, but `fsa/dtype.py` only defines the name `fp8` -- so an
    FP8 config fails with `NameError: name 'e4m3' is not defined`. Its numpy
    conversions cover fp32 and fp16 only, for the same reason: no FP8 config existed
    upstream to exercise them.

    Both are patched here rather than in the submodule (D-106). `ml_dtypes` supplies the
    OCP-conformant numpy dtypes, and is the same reference the golden model's format
    tests check against.
    """
    import ml_dtypes
    _dt = importlib.import_module("fsa.dtype")
    _cf = importlib.import_module("fsa.config")

    for name in ("e4m3", "e5m2"):
        setattr(_dt, name, _dt.fp8)
        setattr(_cf, name, _dt.fp8)      # config.py evals in its own globals

    _np_of = {_dt.fp32: np.float32, _dt.fp16: np.float16,
              _dt.fp8: ml_dtypes.float8_e4m3fn}
    _dt.to_numpy_dtype = lambda t: _np_of[t]
    for mod in ("fsa.tensor", "fsa.mem", "fsa", "fsa.engine"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "to_numpy_dtype"):
            m.to_numpy_dtype = _dt.to_numpy_dtype

    _orig_from = _dt.from_numpy_dtype

    def from_numpy_dtype(n_type):
        if np.dtype(n_type) == np.dtype(ml_dtypes.float8_e4m3fn):
            return _dt.fp8
        if np.dtype(n_type) == np.dtype(ml_dtypes.float8_e5m2):
            return _dt.fp8
        return _orig_from(n_type)

    _dt.from_numpy_dtype = from_numpy_dtype
    for mod in ("fsa.tensor", "fsa.mem", "fsa", "fsa.engine"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "from_numpy_dtype"):
            m.from_numpy_dtype = from_numpy_dtype

    # fsa.from_numpy calls np.finfo(array.dtype) directly, which rejects ml_dtypes
    # ("data type not inexact"), so patch the entry point rather than the helper.
    _fsa = importlib.import_module("fsa")
    _orig_from_numpy = _fsa.from_numpy

    def from_numpy(array: np.ndarray):
        if np.dtype(array.dtype) in (np.dtype(ml_dtypes.float8_e4m3fn),
                                     np.dtype(ml_dtypes.float8_e5m2)):
            tile = _fsa.get_mem_manager().alloc_mem(array.shape, dtype=_dt.fp8)
            tile.data = array.tobytes(order="C")
            return tile
        return _orig_from_numpy(array)

    _fsa.from_numpy = from_numpy
    globals()["F"].from_numpy = from_numpy


def load_config(config: str) -> tuple[int, int]:
    _install_fp8_shims()
    build = f"../../../sims/verilator/generated-src/chipyard.harness.TestHarness.{config}"
    cfg_file = os.path.join(build, f"chipyard.harness.TestHarness.{config}.FSAConfig.json")
    if not os.path.isfile(cfg_file):
        raise SystemExit(f"config not found: {cfg_file}")
    F.init(cfg_file)
    cfg = F.get_config()
    return cfg.sa_rows, cfg.sa_cols


# --- paranoia: RTL random-seed injection -------------------------------------------
# D-113 was uninitialised state hiding behind a fixed $random seed. Varying the seed is
# the general detector for that whole class, so the harness can force one.
_real_run = subprocess.run
_asserts: list[str] = []


def _wrap_subprocess(vseed: int | None) -> None:
    def runner(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd and "simulator-chipyard" in str(cmd[0]):
            if vseed is not None:
                cmd = list(cmd) + [f"+verilator+seed+{vseed}"]
            kw.setdefault("capture_output", True)
            kw.setdefault("text", True)
            r = _real_run(cmd, *a, **kw)
            out = (r.stdout or "") + (r.stderr or "")
            # Verilator is built with --assert; a firing assertion is a hardware
            # contract violation and must never be scrolled past.
            for line in out.splitlines():
                if "ssertion" in line or "%Error" in line:
                    _asserts.append(line.strip())
            return r
        return _real_run(cmd, *a, **kw)
    subprocess.run = runner


def assertions() -> list[str]:
    return list(_asserts)


def clear_assertions() -> None:
    _asserts.clear()


def make_engine(config: str, vseed: int | None = None):
    _wrap_subprocess(vseed)
    sim = f"../../../sims/verilator/simulator-chipyard.harness-{config}"
    if not os.path.isfile(sim):
        raise SystemExit(f"simulator binary not found: {sim}")
    # Bounded, not unlimited: a semaphore deadlock (e.g. DMA into accumulator SRAM,
    # which the RTL never completes) otherwise hangs the run forever instead of failing.
    return F.VerilatorSimulator(sim, max_cycles=2_000_000)
