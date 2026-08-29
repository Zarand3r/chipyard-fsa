"""Bit-accurate software reference for the tiled GEMM.

Gate B's claim is an RTL <-> golden claim, and the roadmap's verification chain marks
that arrow `<-->` -- **bit-exact**, not "within a tolerance". Comparing against a float32
numpy matmul cannot support that: numpy's reduction order and rounding are not the
array's, so agreement is only ever approximate and the tolerance is an envelope somebody
guessed (D-115 caveat).

This reproduces the arithmetic the hardware actually performs, using PyEasyFloat -- the
same library FSA's own reference uses and the same one `main.py --diff` checks attention
against:

  * fp16 multiply into an fp32 accumulator (`mul_ew=5, mul_mw=10`, `acc_ew=8, acc_mw=23`),
    matching `Configs.fp16MulFp32AddArithmeticImpl`;
  * the contraction inside a tile walks k in **reversed** order, matching `fa_ref.py`'s
    `__mul_pv` (`for i in reversed(range(bc))`), which is the systolic drain order;
  * k-tiles accumulate into the running total in issue order, as the accumulator does.

Slow -- it is a scalar FMA per MAC -- so it is meant for small tiles, which is exactly
where a bit-exactness claim needs to be established.
"""
from __future__ import annotations

import numpy as np
from pyeasyfloat.backend import PyEasyFloatBackend

# Reuse FSA's own conversion helpers rather than reimplementing bit packing -- a second
# implementation of float encoding is exactly the kind of thing that silently disagrees.
from fa_ref import fp_to_np, np_to_fp                                 # noqa: E402

MUL_EW, MUL_MW = 5, 10      # fp16 operands
ACC_EW, ACC_MW = 8, 23      # fp32 accumulator

_to_fp = np_to_fp
_from_fp = fp_to_np


def gemm_bitexact(A: np.ndarray, B: np.ndarray, k_tile: int) -> np.ndarray:
    """C = A @ B in the hardware's arithmetic and reduction order.

    A is (M, K) fp16, B is (K, N) fp16; result is fp32. `k_tile` is the array's `rows`,
    since the in-array contraction spans one tile and k-tiles accumulate across
    instructions.
    """
    assert A.dtype == np.float16 and B.dtype == np.float16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and K % k_tile == 0

    be = PyEasyFloatBackend()
    zero = _to_fp(0.0, ACC_EW, ACC_MW)
    Afp = [[_to_fp(v, MUL_EW, MUL_MW) for v in row] for row in A]
    Bfp = [[_to_fp(v, MUL_EW, MUL_MW) for v in row] for row in B]

    one = _to_fp(1.0, ACC_EW, ACC_MW)
    out = np.zeros((M, N), np.float32)
    for r in range(M):
        for c in range(N):
            acc = zero
            for kt in range(K // k_tile):
                base = kt * k_tile
                # Each k-tile contracts inside the array from ZERO -- the partial sum
                # flows down the column through `rows` PEs -- and only then is merged
                # into accumulator SRAM. Carrying one continuous chain across tiles
                # instead was wrong by 1 ulp and showed up immediately as a bit-exact
                # mismatch on exactly the k-accumulating cases.
                tile = zero
                for i in reversed(range(k_tile)):   # matches fa_ref.py's __mul_pv order
                    tile = be.fma(Afp[r][base + i], Bfp[base + i][c], tile,
                                  ACC_EW, ACC_MW)
                # ACC_SA is `out = scale * sram_in + sa_in`, one fused rounding, with
                # scale primed to 1.0 by SetAccScale (D-111).
                acc = be.fma(one, acc, tile, ACC_EW, ACC_MW)
            out[r, c] = _from_fp(acc)
    return out
