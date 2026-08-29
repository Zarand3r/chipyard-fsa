"""§4 of GOLDEN_MODEL_SPEC: the deterministic reduction.

This is called "the bit-exactness backbone" in the spec, and it is what makes
golden-vs-RTL comparison equality rather than tolerance. The contract, verbatim in
structure:

1. dot products tile in k-blocks of `k_block` (one systolic tile traversal);
2. within a k-block, products reduce through a fixed `tree_width`-input adder tree in
   ascending k -- for width 8, `((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))` -- evaluated
   exactly, with no intermediate rounding (DECIDE-3 selects exact vs per-node rounding);
3. tree outputs accumulate into the FP32 accumulator in ascending order, one RNE-rounded
   FP32 add per tree output;
4. accumulators for a given output tile are independent -- no cross-output reduction
   exists anywhere in the datapath.

Exactness is *enforced*, not assumed. The tree is evaluated over `Fraction`, and the
result is checked to be representable in float64 before the single rounding to float32,
so a double-rounding error cannot slip through silently.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from config import NumericConfig, TreeNodeRounding


# float32 (§3 accumulator format): 24-bit significand, min normal exponent -126,
# denormals to 2**-149, max finite (2 - 2**-23) * 2**127.
_F32_MANT_BITS = 24
_F32_MIN_EXP = -126
_F32_DENORM_EXP = _F32_MIN_EXP - (_F32_MANT_BITS - 1)     # -149
_F32_MAX = Fraction(2 ** 128 - 2 ** 104)                  # largest finite float32


def _rne_f32(x: Fraction) -> np.float32:
    """Round an exact Fraction to float32, round-to-nearest-even. **One** rounding.

    Implemented from the Fraction rather than via float64, deliberately. The obvious
    shortcut -- `np.float32(float(x))` -- rounds twice, and for an exact sum needing
    more than 53 bits it can land one ulp off. That is not hypothetical here: an
    8-product tree with a realistic exponent spread routinely exceeds 53 bits, and an
    earlier version of this function asserted float64-representability and fired on
    ordinary random vectors.

    §3 specifies "round-to-nearest-even at every format boundary", and this is that
    boundary, so it is worth doing exactly rather than approximately.
    """
    if x == 0:
        return np.float32(0.0)
    neg = x < 0
    a = -x if neg else x

    # Binade: largest e with 2**e <= a.
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1

    # Quantum exponent: normal numbers keep 24 significand bits; denormals are pinned
    # to the fixed 2**-149 grid.
    q = max(e - (_F32_MANT_BITS - 1), _F32_DENORM_EXP)
    scaled = a / Fraction(2) ** q                  # exact
    fl = scaled.numerator // scaled.denominator
    rem = scaled - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1                                    # nearest, ties to even

    val = Fraction(fl) * Fraction(2) ** q
    if val > _F32_MAX:                             # overflow -> inf, per IEEE RNE
        return np.float32(-np.inf if neg else np.inf)
    out = np.float32(-val if neg else val)
    # float() on a Fraction whose value is exactly a float32 is itself exact, so this
    # conversion cannot re-round; assert it rather than trust it.
    assert Fraction(float(out)) == (-val if neg else val), "rounding grid escaped"
    return out


def _tree_exact(products: list[Fraction]) -> Fraction:
    """Balanced binary tree over an exact type. Ascending k, pairwise, no rounding."""
    level = list(products)
    while len(level) > 1:
        level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
    return level[0]


def _tree_rounded(products: list[Fraction]) -> np.float32:
    """Same shape, but RNE to fp32 at every node (the DECIDE-3 alternative)."""
    level = [_rne_f32(p) for p in products]
    while len(level) > 1:
        level = [np.float32(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def dot(products: list[Fraction], cfg: NumericConfig) -> np.float32:
    """Reduce one output element's dequantized products under the §4 contract.

    `products` are the exact dequantized MAC products in ascending k. §3 states MAC
    products are exact ("formats are narrow enough that the product is representable
    before reduction"), so they arrive as Fractions and no rounding has happened yet.
    """
    n = len(products)
    if n % cfg.tree_width:
        raise ValueError(
            f"{n} products is not a multiple of tree_width {cfg.tree_width}; "
            "the spec's reduction is defined on whole trees"
        )
    if n % cfg.k_block and n > cfg.k_block:
        raise ValueError(f"{n} products must tile into k-blocks of {cfg.k_block}")

    acc = np.float32(0.0)
    # Ascending k-block order (§4.3), and ascending tree order within a block.
    for base in range(0, n, cfg.tree_width):
        group = products[base:base + cfg.tree_width]
        if cfg.tree_rounding is TreeNodeRounding.EXACT:
            out = _rne_f32(_tree_exact(group))          # exact tree, then one rounding
        else:
            out = _tree_rounded(group)
        acc = np.float32(acc + out)                      # one RNE fp32 add per tree out
    return acc


def matmul(A_exact, B_exact, cfg: NumericConfig) -> np.ndarray:
    """C = A @ B under the §4 contract, with independent per-output accumulators (§4.4).

    A and B hold exact dequantized values (Fraction), shaped (M, K) and (K, N).
    """
    M, K = len(A_exact), len(A_exact[0])
    N = len(B_exact[0])
    out = np.zeros((M, N), np.float32)
    for r in range(M):
        for c in range(N):
            out[r, c] = dot([A_exact[r][k] * B_exact[k][c] for k in range(K)], cfg)
    return out


# --- mutants that MUST fail (§10) -------------------------------------------------

def dot_linear_order(products: list[Fraction], cfg: NumericConfig) -> np.float32:
    """MUTANT: linear-order accumulation instead of the adder tree.

    §10 names this as a mutant the test suite must be able to catch. It is kept beside
    the real implementation on purpose: a must-fail test that lives in a different file
    from the thing it mutates tends to rot into a test of nothing.
    """
    acc = np.float32(0.0)
    for p in products:
        acc = np.float32(acc + _rne_f32(p))
    return acc
