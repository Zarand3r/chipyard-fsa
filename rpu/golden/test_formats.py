"""§10 unit vectors for §3: dequant (both profiles, denormals, scale extremes),
RNE/saturation edges. Checked against ml_dtypes where it is available, which is the
reference implementation of the OCP FP8 formats.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Fp8Format, WeightProfile                              # noqa: E402
from formats import (BLOCK_SIZE, E2M1_MAX, decode_e2m1, decode_e8m0,     # noqa: E402
                     decode_fp8, dequant_block, encode_fp8, fp8_max,
                     quantize_fp8)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_e2m1() -> None:
    print("\nE2M1 (FP4) element decode")
    want = [0, .5, 1, 1.5, 2, 3, 4, 6]
    got = [float(decode_e2m1(c)) for c in range(8)]
    check("positive codes give the OCP value set", got == want, str(got))
    check("sign bit negates", [float(decode_e2m1(c | 8)) for c in range(8)]
          == [-v for v in want])
    check("denormal 0.5 is present (code 1)", decode_e2m1(1) == Fraction(1, 2))
    check("max is 6", E2M1_MAX == Fraction(6))


def test_e8m0() -> None:
    print("\nE8M0 scale decode")
    check("bias 127: code 127 is 1.0", decode_e8m0(127) == Fraction(1))
    check("code 128 is 2.0", decode_e8m0(128) == Fraction(2))
    check("smallest scale is 2**-127", decode_e8m0(0) == Fraction(1, 2 ** 127))
    check("largest finite scale is 2**127", decode_e8m0(254) == Fraction(2) ** 127)
    try:
        decode_e8m0(0xFF); check("0xFF (NaN) is rejected", False)
    except ValueError:
        check("0xFF (NaN) is rejected", True)


def test_fp8_against_reference() -> None:
    print("\nFP8 decode vs ml_dtypes (OCP reference)")
    try:
        import ml_dtypes
    except ImportError:
        print("  SKIP  ml_dtypes not installed")
        return
    for fmt, dt in ((Fp8Format.E4M3, ml_dtypes.float8_e4m3fn),
                    (Fp8Format.E5M2, ml_dtypes.float8_e5m2)):
        bad = []
        for code in range(256):
            ref = np.frombuffer(bytes([code]), dtype=dt)[0]
            if not np.isfinite(ref):
                continue                       # inf/NaN codes are rejected by design
            if Fraction(float(ref)) != decode_fp8(code, fmt):
                bad.append(code)
        check(f"{fmt}: all finite codes match ml_dtypes", not bad,
              f"{len(bad)} mismatches" if bad else "256 codes")
        check(f"{fmt}: max matches", float(fp8_max(fmt)) == float(ml_dtypes.finfo(dt).max),
              f"{float(fp8_max(fmt))} vs {float(ml_dtypes.finfo(dt).max)}")


def test_fp8_roundtrip_and_saturation() -> None:
    print("\nFP8 encode: RNE and saturating casts (§3)")
    for fmt in (Fp8Format.E4M3, Fp8Format.E5M2):
        hi = fp8_max(fmt)
        check(f"{fmt}: huge value saturates, never inf",
              quantize_fp8(Fraction(10 ** 6), fmt) == hi, f"-> {float(hi)}")
        check(f"{fmt}: negative huge saturates",
              quantize_fp8(Fraction(-10 ** 6), fmt) == -hi)
        check(f"{fmt}: zero round-trips", quantize_fp8(Fraction(0), fmt) == 0)
        # Every representable value must survive a round trip exactly.
        bad = []
        for code in range(256):
            try:
                v = decode_fp8(code, fmt)
            except ValueError:
                continue
            if quantize_fp8(v, fmt) != v:
                bad.append(code)
        check(f"{fmt}: every finite code round-trips exactly", not bad,
              f"{len(bad)} failed" if bad else "")
        # Midpoint between two representables must round to even.
        a, b = decode_fp8(0x20, fmt), decode_fp8(0x21, fmt)
        mid = (a + b) / 2
        r = quantize_fp8(mid, fmt)
        check(f"{fmt}: exact midpoint rounds to even", r in (a, b) and
              encode_fp8(r, fmt) % 2 == 0, f"mid {float(mid)} -> {float(r)}")


def test_blocks() -> None:
    print("\nmicroscaling blocks (§3)")
    check("MXFP4 blocks are 32 elements", BLOCK_SIZE[WeightProfile.MXFP4] == 32)
    check("NVFP4 blocks are 16 elements", BLOCK_SIZE[WeightProfile.NVFP4] == 16)

    # MXFP4: scale is a pure power of two, so dequant is exact and multiplier-free.
    els = list(range(8)) + [0] * 24
    out = dequant_block(els, 128, WeightProfile.MXFP4)      # scale 2.0
    check("MXFP4 dequant is an exponent add",
          [float(v) for v in out[:8]] == [0, 1, 2, 3, 4, 6, 8, 12], str([float(v) for v in out[:8]]))

    # Scale extremes must not lose exactness -- these are Fractions, not floats.
    # E2M1 code 7 is the value 6; code 6 is the value 4. Getting that backwards was a
    # bug in this test, and exactly the kind a hand-written expectation invites.
    assert decode_e2m1(7) == Fraction(6) and decode_e2m1(6) == Fraction(4)
    tiny = dequant_block([7] + [0] * 31, 0, WeightProfile.MXFP4)
    check("smallest scale stays exact", tiny[0] == Fraction(6, 2 ** 127),
          f"{tiny[0]}")
    huge = dequant_block([7] + [0] * 31, 254, WeightProfile.MXFP4)
    check("largest scale stays exact", huge[0] == Fraction(6) * Fraction(2) ** 127)
    check("extremes span 2**254 without loss",
          huge[0] / tiny[0] == Fraction(2) ** 254)

    # NVFP4: E4M3 scale, a real multiply.
    nv = dequant_block([4] + [0] * 15, encode_fp8(Fraction(3), Fp8Format.E4M3),
                       WeightProfile.NVFP4)
    check("NVFP4 dequant multiplies by an E4M3 scale", nv[0] == Fraction(6),
          f"2 * 3 = {float(nv[0])}")

    try:
        dequant_block([0] * 16, 127, WeightProfile.MXFP4)
        check("wrong block length is rejected", False)
    except ValueError:
        check("wrong block length is rejected", True)


for t in (test_e2m1, test_e8m0, test_fp8_against_reference,
          test_fp8_roundtrip_and_saturation, test_blocks):
    t()

print(f"\n{'ALL PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
