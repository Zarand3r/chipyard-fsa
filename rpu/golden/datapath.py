"""§5 datapath blocks of GOLDEN_MODEL_SPEC.

Implements 5.1 (dequant row), 5.2 (matmul), the 5.3 online-softmax streamer, 5.4 vector
ops and the 5.5 FLOP-share checksum. Everything routes its arithmetic through
`reduce.dot` so the §4 reduction contract is the single place reduction order lives.

Open decisions appear as flags, never as silent behaviour: DECIDE-5 (`exp`), DECIDE-6
(probability precision), and the §5.3 softmax variant. §5.3 states the golden's default
`exp` is correctly-rounded FP32 and that the hardware approximation switches in once
gate 1 specifies it to the bit -- so selecting `HARDWARE_APPROX` raises rather than
guessing an approximation.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from config import (ExpImpl, Fp8Format, NumericConfig, ProbPrecision,
                    SoftmaxVariant, WeightProfile)
from formats import BLOCK_SIZE, dequant_block, quantize_fp8
from reduce import dot


# --- 5.1 dequant row ---------------------------------------------------------------

def dequant_row(codes: list[int], scales: list[int], cfg: NumericConfig) -> list[Fraction]:
    """A full weight row from packed blocks. Exact; §5.1's 'effective multiplicand'."""
    n = BLOCK_SIZE[cfg.weight_profile]
    if len(codes) != len(scales) * n:
        raise ValueError(f"{len(codes)} codes needs {len(codes) // n} scales, "
                         f"got {len(scales)}")
    out: list[Fraction] = []
    for b, s in enumerate(scales):
        out.extend(dequant_block(codes[b * n:(b + 1) * n], s, cfg.weight_profile))
    return out


# --- 5.2 matmul --------------------------------------------------------------------

def matmul_fp8(X: list[list[Fraction]], W: list[list[Fraction]],
               cfg: NumericConfig, requantize: bool = True):
    """Y = X @ W. X is FP8-valued, W dequantized per §5.1, Y FP32 per §4.

    §5.2 re-quantizes Y to FP8 at the tensor boundary; `requantize=False` exposes the
    raw FP32 accumulator, which §9(c) names as an observation point.
    """
    M, K = len(X), len(X[0])
    N = len(W[0])
    out = [[None] * N for _ in range(M)]
    for r in range(M):
        for c in range(N):
            acc = dot([X[r][k] * W[k][c] for k in range(K)], cfg)
            out[r][c] = (quantize_fp8(Fraction(float(acc)), cfg.activation_fp8)
                         if requantize else acc)
    return out


# --- 5.3 online-softmax streamer ---------------------------------------------------

def _exp_fp32(x: float, cfg: NumericConfig) -> np.float32:
    if cfg.exp_impl is ExpImpl.FP32_CORRECTLY_ROUNDED:
        return np.float32(math.exp(x))
    raise NotImplementedError(
        "DECIDE-5: the hardware `exp` approximation is not specified yet. §5.3 requires "
        "it bit-specified at gate 1 before the golden model can switch to it; "
        "approximating it here would silently invent an architectural commitment."
    )


def online_softmax(tiles: list[list[float]], cfg: NumericConfig,
                   descending: bool = False):
    """§5.3 streamer: running max and sum over k-tiles in **ascending** tile order.

    The spec says "the recurrence order is part of the spec", so the order is a
    contract, not an implementation detail. `descending=True` is the §10 mutant.

    Returns (probabilities, running_sum) with probabilities normalised at the end.
    """
    if cfg.softmax is not SoftmaxVariant.ONLINE:
        raise NotImplementedError(
            f"§5.3 variant {cfg.softmax} is a modelled alternative that is not "
            "implemented yet; only ONLINE is."
        )
    order = list(reversed(tiles)) if descending else tiles
    m = np.float32(-np.inf)
    s = np.float32(0.0)
    scaled: list[list[np.float32]] = []
    for tile in order:
        tmax = np.float32(max(tile)) if tile else np.float32(-np.inf)
        m_new = np.float32(max(float(m), float(tmax)))
        if np.isfinite(m):
            # Rescale onto the new max: the running sum AND every value already
            # emitted. Rescaling only the sum leaves earlier tiles on the old max and
            # the probabilities stop summing to 1 -- which is exactly how this was
            # first written and what the sum-to-1 check caught.
            corr = _exp_fp32(float(m) - float(m_new), cfg)
            s = np.float32(s * corr)
            scaled = [[np.float32(v * corr) for v in row] for row in scaled]
        row = [_exp_fp32(float(v) - float(m_new), cfg) for v in tile]
        for v in row:
            s = np.float32(s + v)
        scaled.append(row)
        m = m_new
    if descending:
        scaled = list(reversed(scaled))
    probs = [[np.float32(v / s) for v in row] for row in scaled]
    return probs, s


def quantize_probs(probs, cfg: NumericConfig):
    """DECIDE-6: FP8 into pass 2 (baseline) versus keeping FP16 through AV."""
    if cfg.prob_precision is ProbPrecision.FP8:
        return [[quantize_fp8(Fraction(float(v)), cfg.activation_fp8) for v in row]
                for row in probs]
    return [[Fraction(float(np.float16(v))) for v in row] for row in probs]


# --- 5.4 vector unit ---------------------------------------------------------------

def layernorm_fp32(x: list[float], eps: float) -> list[np.float32]:
    """§5.4, FP32 internal. DECIDE-7 picks RMSNorm vs LayerNorm and its placement;
    this is the LayerNorm arm, which is what the phase-1 DiT-XL/2 trace pins for the
    bring-up workload (elementwise_affine=False, eps=1e-6)."""
    n = len(x)
    mean = np.float32(sum(np.float32(v) for v in x) / np.float32(n))
    var = np.float32(sum(np.float32((np.float32(v) - mean) ** 2) for v in x)
                     / np.float32(n))
    inv = np.float32(1.0 / math.sqrt(float(var) + eps))
    return [np.float32((np.float32(v) - mean) * inv) for v in x]


def gelu_tanh_fp32(x: float) -> np.float32:
    """GELU, tanh approximation -- the form the pinned DiT-XL/2 checkpoint uses."""
    c = math.sqrt(2.0 / math.pi)
    return np.float32(0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3))))


# --- 5.5 FLOP-share checksum -------------------------------------------------------

# §5.5, at contract shapes. The spec calls this "checksum for the model".
FLOP_SHARES = {
    "self_attention": 0.407,
    "ffn": 0.300,
    "qkv": 0.167,
    "cross_attention": 0.070,
    "output_projection": 0.056,
}


def flop_shares(d: int, d_ff: int, n_ctx: int, n_new: int, n_text: int) -> dict:
    """Recompute §5.5's shares from shapes, as a self-check on the model's geometry."""
    qkv = 3 * n_new * d * d
    self_attn = 2 * n_new * n_ctx * d          # QK^T and PV
    out_proj = n_new * d * d
    cross = 2 * n_new * n_text * d + n_new * d * d
    ffn = 2 * n_new * d * d_ff
    total = qkv + self_attn + out_proj + cross + ffn
    return {"qkv": qkv / total, "self_attention": self_attn / total,
            "output_projection": out_proj / total, "cross_attention": cross / total,
            "ffn": ffn / total}


# --- 5.4/5.5 conditioning: modulation and gated residual ----------------------------
#
# These are the ops roadmap phase 4 adds. Numerically they are trivial; the whole
# difficulty is where they run (see rpu/PHASE4_MODULATION.md). Having them here in the
# golden means the RTL mapping, whichever option is chosen, has something exact to be
# checked against.

def modulate(x: list[list[float]], shift: list[float], scale: list[float]):
    """adaLN-Zero: `x * (1 + scale) + shift`, per channel, FP32 internal (§5.4).

    Matches the pinned DiT-XL/2 form exactly: `models.py` defines
    `modulate(x, shift, scale) = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)`.
    """
    C = len(x[0])
    if len(shift) != C or len(scale) != C:
        raise ValueError(f"per-channel params must be length {C}")
    return [[np.float32(np.float32(v) * np.float32(1.0 + scale[c]) + np.float32(shift[c]))
             for c, v in enumerate(row)] for row in x]


def gated_residual(x: list[list[float]], branch: list[list[float]],
                   gate: list[float]):
    """adaLN-Zero gated residual: `x + gate * branch`, gate per channel (§5.5)."""
    C = len(x[0])
    if len(gate) != C:
        raise ValueError(f"gate must be length {C}")
    return [[np.float32(np.float32(xv) + np.float32(gate[c]) * np.float32(bv))
             for c, (xv, bv) in enumerate(zip(xr, br))]
            for xr, br in zip(x, branch)]


def fold_scale_into_weights(W: list[list[Fraction]], scale: list[float],
                            side: str) -> list[list[Fraction]]:
    """Option B of the phase-4 mapping: fold a per-channel scale into a weight matrix.

    `(x * s) @ W == x @ (diag(s) @ W)` -- row-scaling, `side="in"`.
    `g * (c @ W) == c @ (W @ diag(g))`  -- column-scaling, `side="out"`.

    Exact, and costs zero array operations. Its problem is architectural, not
    numerical: the RPU streams 4-bit weights from DRAM and these scales are recomputed
    per step from the conditioning vector, so folding rewrites the weight stream every
    step. See rpu/PHASE4_MODULATION.md.
    """
    if side == "in":
        if len(scale) != len(W):
            raise ValueError("input-side scale must match W's rows")
        return [[Fraction(float(scale[k])) * v for v in row] for k, row in enumerate(W)]
    if side == "out":
        if len(scale) != len(W[0]):
            raise ValueError("output-side scale must match W's columns")
        return [[Fraction(float(scale[c])) * v for c, v in enumerate(row)] for row in W]
    raise ValueError(f"side must be 'in' or 'out', got {side!r}")
