"""§10 unit vectors for the §4 reduction, including the mutants that MUST fail.

Run: python rpu/golden/test_reduce.py
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NumericConfig, TreeNodeRounding, working_assumption   # noqa: E402
from reduce import dot, dot_linear_order                                 # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_open_decides_refused() -> None:
    print("\n§11 open decisions have no defaults")
    try:
        NumericConfig()
        check("constructing without DECIDEs raises", False)
    except ValueError:
        check("constructing without DECIDEs raises", True)
    try:
        NumericConfig(activation_fp8="e4m3", tree_width=7,
                      tree_rounding=TreeNodeRounding.EXACT, weight_profile="mxfp4")
        check("tree_width outside DECIDE-4's (4,8,16) raises", False)
    except ValueError:
        check("tree_width outside DECIDE-4's (4,8,16) raises", True)


def test_tree_shape() -> None:
    """§4.2: ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)), ascending k."""
    print("\n§4.2 adder-tree association")
    cfg = working_assumption()
    # Values chosen so the tree order is observable: pairwise sums are exact, but a
    # linear walk loses the small terms against the large leading one.
    big = Fraction(2 ** 24)
    ps = [big] + [Fraction(1)] * 7
    tree = dot(ps, cfg)
    linear = dot_linear_order(ps, cfg)
    # 2**24 + 7 is NOT representable in float32: the ulp at 2**24 is 2, so the exact
    # value sits halfway between 16777222 and 16777224 and RNE takes the even one.
    # Expecting 16777223 here was a bug in this test, not in the model.
    want = np.float32(16777224.0)
    check("exact tree carries the small terms into the rounding", tree == want,
          f"tree={float(tree):.1f} want={float(want):.1f}")
    check("linear order drops them entirely (mutant differs)",
          linear == np.float32(2.0 ** 24), f"linear={float(linear):.1f}")


def test_mutant_detected() -> None:
    """§10: 'Mutants that must fail: linear-order accumulation instead of the tree.'"""
    print("\n§10 must-fail mutant: linear-order accumulation")
    cfg = working_assumption()
    rng = np.random.default_rng(0)
    caught = 0
    trials = 200
    for _ in range(trials):
        # Catastrophic-cancellation flavoured: wide exponent spread within one tree.
        vals = [Fraction(float(rng.normal() * 2.0 ** int(rng.integers(-12, 12))))
                for _ in range(cfg.tree_width)]
        if dot(vals, cfg) != dot_linear_order(vals, cfg):
            caught += 1
    check("mutant distinguishable on random wide-exponent vectors", caught > 0,
          f"{caught}/{trials} vectors distinguish tree from linear order")


def test_decide3_flag_is_real() -> None:
    """DECIDE-3: exact tree vs FP32-rounded nodes must be a real behavioural difference."""
    print("\nDECIDE-3 exact vs per-node-rounded tree")
    base = working_assumption()
    rounded = NumericConfig(activation_fp8=base.activation_fp8, tree_width=base.tree_width,
                            tree_rounding=TreeNodeRounding.ROUND_EACH_NODE,
                            weight_profile=base.weight_profile)
    rng = np.random.default_rng(1)
    diff = 0
    for _ in range(200):
        vals = [Fraction(float(rng.normal() * 2.0 ** int(rng.integers(-14, 14))))
                for _ in range(base.tree_width)]
        if dot(vals, base) != dot(vals, rounded):
            diff += 1
    check("the two DECIDE-3 behaviours are distinguishable", diff > 0,
          f"{diff}/200 vectors differ -- the flag is not cosmetic")


def test_k_block_order() -> None:
    """§4.3: tree outputs accumulate in ascending k-block order."""
    print("\n§4.3 ascending k-block accumulation order")
    cfg = working_assumption()
    # A known-answer construction, not random data. An earlier version of this test used
    # random vectors, failed to distinguish the orders, and so asserted nothing -- the
    # claim "order is load-bearing" needs a vector that actually demonstrates it.
    #
    # Group 0 holds 2**24; the other 15 groups hold 1 each.
    #   forward: acc = 2**24, then +1 fifteen times. Each 2**24 + 1 is an exact tie and
    #            RNE keeps the even value, so every one of them is swallowed -> 2**24.
    #   reversed: the fifteen 1s accumulate to 15 first, then + 2**24 -> 2**24 + 15,
    #            which rounds to 2**24 + 16.
    ngroups = cfg.k_block // cfg.tree_width
    pad = [Fraction(0)] * (cfg.tree_width - 1)
    groups = [[Fraction(2 ** 24)] + pad] + [[Fraction(1)] + pad
                                            for _ in range(ngroups - 1)]
    fwd = dot([v for g in groups for v in g], cfg)
    rev = dot([v for g in reversed(groups) for v in g], cfg)
    check("forward order swallows the small terms", fwd == np.float32(2.0 ** 24),
          f"fwd={float(fwd):.1f}")
    check("reversed order keeps them -- order is load-bearing",
          rev == np.float32(2 ** 24 + 16), f"rev={float(rev):.1f}")
    check("forward order is reproducible",
          dot([v for g in groups for v in g], cfg) == fwd)


def test_rne_correctness() -> None:
    """RNE from an exact Fraction, including the cases float64 cannot mediate."""
    print("\n§3 round-to-nearest-even at the fp32 boundary")
    cfg = working_assumption()
    pad = [Fraction(0)] * (cfg.tree_width - 1)

    # A third: not representable in binary at any precision. Must still round, not raise.
    third = dot([Fraction(1, 3)] + pad, cfg)
    check("1/3 rounds to the nearest float32", third == np.float32(1.0 / 3.0),
          f"got {third!r}")

    # Exact tie at the fp32 grid: 2**24 + 1 sits halfway; RNE takes the even neighbour.
    tie = dot([Fraction(2 ** 24 + 1)] + pad, cfg)
    check("exact tie rounds to even", tie == np.float32(2.0 ** 24),
          f"got {float(tie):.1f}")
    tie3 = dot([Fraction(2 ** 24 + 3)] + pad, cfg)
    check("the other exact tie also rounds to even", tie3 == np.float32(2 ** 24 + 4),
          f"got {float(tie3):.1f}")

    # Denormal grid and overflow.
    den = dot([Fraction(1, 2 ** 149)] + pad, cfg)
    check("smallest denormal survives", float(den) == float(np.float32(2.0 ** -149)))
    huge = dot([Fraction(2 ** 200)] + pad, cfg)
    check("overflow becomes inf", np.isinf(huge))

    # Against numpy on values float64 represents exactly, where both must agree.
    rng = np.random.default_rng(7)
    vals = rng.normal(size=500).astype(np.float64)
    agree = all(dot([Fraction(float(v))] + pad, cfg) == np.float32(v) for v in vals)
    check("agrees with numpy on float64-exact inputs", agree)


for t in (test_open_decides_refused, test_tree_shape, test_mutant_detected,
          test_decide3_flag_is_real, test_k_block_order, test_rne_correctness):
    t()

print(f"\n{'ALL PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
