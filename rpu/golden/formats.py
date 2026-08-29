"""§3 of GOLDEN_MODEL_SPEC: number formats, dequantization and rounding.

Everything here returns **exact** values (`Fraction`), because §3 states MAC products
are exact -- "formats are narrow enough that the product is representable before
reduction" -- and §4 then reduces them. Rounding happens at format boundaries only, and
`reduce.py` owns the one at the FP32 accumulator.

Scope, per §3: **weight quantization is out of scope.** The golden model consumes a
prepared weight image and never quantizes weights itself, so weight paths here decode
only. Activations do need an encoder, because §3 puts FP8 at tensor boundaries.

Formats, all from the OCP Microscaling spec:

  E2M1   FP4 element, bias 1, values {0, .5, 1, 1.5, 2, 3, 4, 6}, denormal 0.5
  E8M0   exponent-only scale, bias 127, value 2**(e-127), 0xFF is NaN
  E4M3   FP8, bias 7, max 448, **no infinities**, NaN only at S.1111.111
  E5M2   FP8, bias 15, max 57344, IEEE-like with inf and NaN

MXFP4 (default profile) pairs 32 E2M1 elements with one E8M0 scale, so dequant is an
exponent add with no multiplier. NVFP4 pairs 16 elements with an E4M3 scale, costing one
FP8 multiply per block.
"""

from __future__ import annotations

from fractions import Fraction

from config import Fp8Format, WeightProfile

# --- E2M1 (FP4) --------------------------------------------------------------------
# Index by the 3 magnitude bits; sign is bit 3. Denormal (exp 0, mant 1) is 0.5.
_E2M1 = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
         Fraction(2), Fraction(3), Fraction(4), Fraction(6)]
E2M1_MAX = Fraction(6)


def decode_e2m1(code: int) -> Fraction:
    """One FP4 element from its 4-bit code. Exact; E2M1 has no inf or NaN."""
    if not 0 <= code < 16:
        raise ValueError(f"e2m1 code out of range: {code}")
    v = _E2M1[code & 0b111]
    return -v if code & 0b1000 else v


def decode_e8m0(code: int) -> Fraction:
    """Power-of-two scale. 0xFF is NaN per the MX spec and is rejected here."""
    if not 0 <= code < 256:
        raise ValueError(f"e8m0 code out of range: {code}")
    if code == 0xFF:
        raise ValueError("e8m0 code 0xFF is NaN; a weight image must not contain it")
    return Fraction(2) ** (code - 127)


# --- FP8 ---------------------------------------------------------------------------
_FP8 = {
    #                ew mw bias  has_inf
    Fp8Format.E4M3: (4, 3, 7, False),
    Fp8Format.E5M2: (5, 2, 15, True),
}


def fp8_max(fmt: Fp8Format) -> Fraction:
    ew, mw, bias, has_inf = _FP8[fmt]
    max_exp = (2 ** ew - 1) - bias - (1 if has_inf else 0)
    # E4M3 reserves only S.1111.111 for NaN, so 1111.110 is finite and usable.
    max_mant = (2 ** mw - 1) - (0 if has_inf else 1)
    return Fraction(2) ** max_exp * (1 + Fraction(max_mant, 2 ** mw))


def decode_fp8(code: int, fmt: Fp8Format) -> Fraction:
    """One FP8 value, exact. Raises on NaN/Inf codes -- §3 forbids them in tensors."""
    ew, mw, bias, has_inf = _FP8[fmt]
    if not 0 <= code < 256:
        raise ValueError(f"fp8 code out of range: {code}")
    sign = -1 if code & 0x80 else 1
    exp = (code >> mw) & (2 ** ew - 1)
    mant = code & (2 ** mw - 1)

    if exp == 2 ** ew - 1:
        if has_inf:
            raise ValueError(f"{fmt} code {code:#04x} is {'NaN' if mant else 'Inf'}; "
                             "§3 forbids inf/NaN in FP8 tensors")
        if mant == 2 ** mw - 1:
            raise ValueError(f"{fmt} code {code:#04x} is NaN; §3 forbids it in tensors")

    if exp == 0:                                   # denormal (OCP MX: supported)
        return sign * Fraction(mant, 2 ** mw) * Fraction(2) ** (1 - bias)
    return sign * (1 + Fraction(mant, 2 ** mw)) * Fraction(2) ** (exp - bias)


def encode_fp8(x: Fraction, fmt: Fp8Format) -> int:
    """Exact value -> FP8 code. RNE, and **saturating**: §3 says no inf/NaN reaches an
    FP8 tensor, so an out-of-range magnitude clamps to the largest finite value rather
    than becoming infinity."""
    ew, mw, bias, _ = _FP8[fmt]
    neg = x < 0
    a = -x if neg else x
    sign_bit = 0x80 if neg else 0

    hi = fp8_max(fmt)
    if a >= hi:
        # Saturate only once genuinely past the top; values between the largest finite
        # and the saturation point still round normally below.
        pass

    # Quantum exponent: normal above 2**(1-bias), else the denormal grid.
    if a == 0:
        return sign_bit
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    q = max(e - mw, 1 - bias - mw)
    scaled = a / Fraction(2) ** q
    fl = scaled.numerator // scaled.denominator
    rem = scaled - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    val = Fraction(fl) * Fraction(2) ** q

    if val >= hi:
        val = hi                                    # saturating cast, never inf
    # Re-derive the code from the rounded value.
    if val < Fraction(2) ** (1 - bias):             # denormal
        mant = int(val / Fraction(2) ** (1 - bias) * 2 ** mw)
        return sign_bit | mant
    ev = val.numerator.bit_length() - val.denominator.bit_length()
    if Fraction(2) ** ev > val:
        ev -= 1
    mant = int((val / Fraction(2) ** ev - 1) * 2 ** mw)
    return sign_bit | ((ev + bias) << mw) | mant


def quantize_fp8(x: Fraction, fmt: Fp8Format) -> Fraction:
    """Round-trip an exact value through FP8. This is the §3 tensor boundary."""
    return decode_fp8(encode_fp8(x, fmt), fmt)


# --- microscaled weight blocks -----------------------------------------------------
BLOCK_SIZE = {WeightProfile.MXFP4: 32, WeightProfile.NVFP4: 16}


def dequant_block(elements: list[int], scale_code: int,
                  profile: WeightProfile) -> list[Fraction]:
    """Dequantize one microscaling block of weights. Exact; no rounding occurs.

    MXFP4's E8M0 scale makes this an exponent add -- the spec's stated reason for
    preferring it ("dequant = exponent add, no multiplier"). NVFP4's E4M3 scale is a
    real multiply, which is the cost of its finer 16-element blocking.
    """
    n = BLOCK_SIZE[profile]
    if len(elements) != n:
        raise ValueError(f"{profile} blocks hold {n} elements; got {len(elements)}")
    scale = (decode_e8m0(scale_code) if profile is WeightProfile.MXFP4
             else decode_fp8(scale_code, Fp8Format.E4M3))
    return [decode_e2m1(c) * scale for c in elements]
