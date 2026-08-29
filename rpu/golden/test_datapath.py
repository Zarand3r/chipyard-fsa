"""§10 vectors for the §5 datapath blocks, including the softmax-order mutant."""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ExpImpl, ProbPrecision, working_assumption          # noqa: E402
from datapath import (FLOP_SHARES, dequant_row, flop_shares,           # noqa: E402
                      gelu_tanh_fp32, layernorm_fp32, matmul_fp8,
                      online_softmax, quantize_probs)
from formats import encode_fp8                                          # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_dequant_row() -> None:
    print("\n§5.1 dequant row")
    cfg = working_assumption()          # MXFP4, 32-element blocks
    codes = ([7] + [0] * 31) + ([2] + [0] * 31)      # 6, then 1
    row = dequant_row(codes, [127, 128], cfg)        # scales 1.0, 2.0
    check("two blocks dequantize independently",
          row[0] == Fraction(6) and row[32] == Fraction(2),
          f"{float(row[0])}, {float(row[32])}")
    check("row length is blocks x block size", len(row) == 64)
    try:
        dequant_row(codes, [127], cfg); check("scale-count mismatch rejected", False)
    except ValueError:
        check("scale-count mismatch rejected", True)


def test_matmul() -> None:
    print("\n§5.2 matmul")
    cfg = working_assumption()
    X = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]
    W = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    # tree_width is 8, so pad the contraction to a whole tree
    Xp = [row + [Fraction(0)] * 6 for row in X]
    Wp = W + [[Fraction(0), Fraction(0)]] * 6
    raw = matmul_fp8(Xp, Wp, cfg, requantize=False)
    check("identity W reproduces X (FP32 accumulator)",
          [[float(v) for v in r] for r in raw] == [[1.0, 2.0], [3.0, 4.0]])
    q = matmul_fp8(Xp, Wp, cfg, requantize=True)
    check("re-quantized output is FP8-representable",
          all(quantize_probs([[v]], cfg)[0][0] == v for r in q for v in r))


def test_softmax_order_is_contract() -> None:
    """§5.3: 'the recurrence order is part of the spec'. §10 mutant: descending order."""
    print("\n§5.3 online softmax, ascending tile order")
    cfg = working_assumption()
    tiles = [[0.0, 1.0], [50.0, 2.0], [-30.0, 3.0]]
    probs, s = online_softmax(tiles, cfg)
    flat = [float(v) for row in probs for v in row]
    check("probabilities sum to 1", abs(sum(flat) - 1.0) < 1e-6, f"sum={sum(flat):.9f}")
    ref = np.exp(np.array([v for t in tiles for v in t], dtype=np.float64) - 50.0)
    ref = ref / ref.sum()
    check("matches a direct stable softmax", np.allclose(flat, ref, rtol=1e-6))

    mut = online_softmax(tiles, cfg, descending=True)[0]
    mutflat = [float(v) for row in mut for v in row]
    check("descending-order mutant is detectable", mutflat != flat,
          "running max updated in descending tile order")


def test_decide5_refuses_to_guess() -> None:
    print("\nDECIDE-5: hardware exp must be specified, not invented")
    cfg = replace(working_assumption(), exp_impl=ExpImpl.HARDWARE_APPROX)
    try:
        online_softmax([[0.0, 1.0]], cfg)
        check("selecting an unspecified hardware exp raises", False)
    except NotImplementedError:
        check("selecting an unspecified hardware exp raises", True)


def test_decide6_flag_is_real() -> None:
    print("\nDECIDE-6: probability precision into pass 2")
    base = working_assumption()
    fp16 = replace(base, prob_precision=ProbPrecision.FP16)
    probs = [[np.float32(0.3333333), np.float32(0.1666667)]]
    a = quantize_probs(probs, base)
    b = quantize_probs(probs, fp16)
    check("FP8 and FP16 paths differ", a != b,
          f"fp8={float(a[0][0]):.6f} fp16={float(b[0][0]):.6f}")


def test_vector_unit() -> None:
    print("\n§5.4 vector unit")
    x = [1.0, 2.0, 3.0, 4.0]
    ln = [float(v) for v in layernorm_fp32(x, 1e-6)]
    check("layernorm is zero-mean", abs(sum(ln)) < 1e-5, f"mean={sum(ln)/4:.2e}")
    check("layernorm is unit-variance", abs(sum(v * v for v in ln) / 4 - 1.0) < 1e-4)
    # GELU tanh against the closed form
    check("gelu(0) == 0", float(gelu_tanh_fp32(0.0)) == 0.0)
    # GELU is NOT monotone -- it dips below zero and has a minimum near x = -0.75.
    # Asserting monotonicity was a bug in this test, not in the model.
    xs = np.arange(-4.0, 4.0, 0.01)
    ys = [float(gelu_tanh_fp32(float(a))) for a in xs]
    argmin = float(xs[int(np.argmin(ys))])
    check("gelu has its minimum near -0.75", abs(argmin + 0.75) < 0.05,
          f"argmin={argmin:.3f}")
    check("gelu is monotone increasing above its minimum",
          all(ys[i] <= ys[i + 1] + 1e-7 for i in range(int(np.argmin(ys)), len(ys) - 1)))
    check("gelu(x) -> x for large positive", abs(float(gelu_tanh_fp32(8.0)) - 8.0) < 1e-4)
    check("gelu(x) -> 0 for large negative", abs(float(gelu_tanh_fp32(-8.0))) < 1e-6)


def test_flop_share_checksum() -> None:
    """§5.5 calls these shares a 'checksum for the model'."""
    print("\n§5.5 FLOP-share checksum at contract shapes (§2)")
    got = flop_shares(d=5120, d_ff=13824, n_ctx=18720, n_new=3120, n_text=256)
    check("shares sum to 1", abs(sum(got.values()) - 1.0) < 1e-9)
    worst = max(abs(got[k] - FLOP_SHARES[k]) for k in FLOP_SHARES)
    for k in sorted(FLOP_SHARES):
        print(f"        {k:<18} spec {FLOP_SHARES[k]:.3f}   computed {got[k]:.3f}")
    # The spec's shares are tagged [S] (simulated/derived), so this is a consistency
    # check on the model's geometry, not a bit-exact claim.
    check("computed shares match §5.5 within 2 points", worst < 0.02,
          f"worst deviation {worst:.4f}")


for t in (test_dequant_row, test_matmul, test_softmax_order_is_contract,
          test_decide5_refuses_to_guess, test_decide6_flag_is_real,
          test_vector_unit, test_flop_share_checksum):
    t()

print(f"\n{'ALL PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
