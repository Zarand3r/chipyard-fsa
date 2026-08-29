"""Pre-RTL numerics study: how many PWL pieces does GELU need?

GOLDEN_MODEL_SPEC DECIDE-5 says the hardware `exp` approximation "must be specified to
the bit at gate 1". §5.4's GELU has the same character: FSA already evaluates `exp2` as
a piecewise-linear function inside the MAC (`exp2PwlIntercepts` / `exp2PwlSlopes`,
selected by `PROP_EXP2_INTERCEPTS`), so a GELU built on that path inherits a piece-count
choice that nobody has made.

This study does not make it. It measures what each choice costs, which is what a
"pre-RTL numerics study" is for. Inventing a piece count here would be exactly the
silent architectural commitment `config.py` refuses elsewhere.

Two candidate forms are measured:
  * tanh-GELU, the form the pinned DiT-XL/2 checkpoint uses;
  * PWL approximation of that curve at N pieces, uniformly spaced over the range where
    GELU is non-trivial.

Reported against the activation distribution the phase-1 trace actually produced, not
against a uniform grid -- accuracy on inputs the model never sees is not worth paying
for.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datapath import gelu_tanh_fp32                                    # noqa: E402

TRACE = Path.home() / "rpu-simulation/chipyard-fsa/workloads/dit/build/dit_xl2_block0"


def gelu_exact(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def pwl_gelu(x: np.ndarray, pieces: int, lo: float, hi: float) -> np.ndarray:
    """Uniform piecewise-linear fit, evaluated as slope*x + intercept per piece."""
    edges = np.linspace(lo, hi, pieces + 1)
    ys = gelu_exact(edges)
    out = np.empty_like(x, dtype=np.float64)
    below, above = x < lo, x > hi
    out[below] = 0.0                       # GELU -> 0 well below the range
    out[above] = x[above]                  # GELU -> x well above it
    mid = ~(below | above)
    idx = np.clip(np.searchsorted(edges, x[mid], side="right") - 1, 0, pieces - 1)
    x0, x1 = edges[idx], edges[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    out[mid] = y0 + (x[mid] - x0) * (y1 - y0) / (x1 - x0)
    return out


def main() -> int:
    src = TRACE / "mlp_fc1_out.npy"
    if not src.exists():
        print(f"trace not found: {src}\nrun workloads/dit/trace_block.py first")
        return 2
    acts = np.load(src).astype(np.float64).reshape(-1)
    print(f"GELU input distribution from the phase-1 trace ({src.name}, {acts.size:,} values)")
    print(f"  range [{acts.min():.3f}, {acts.max():.3f}]  "
          f"p0.1={np.percentile(acts,0.1):.3f}  p99.9={np.percentile(acts,99.9):.3f}\n")

    ref = gelu_exact(acts)
    denom = max(np.abs(ref).max(), 1e-30)
    lo, hi = -8.0, 8.0
    print(f"{'pieces':>7}  {'max abs err':>12}  {'max rel err':>12}  {'rms err':>12}")
    for pieces in (4, 8, 16, 32, 64, 128):
        approx = pwl_gelu(acts, pieces, lo, hi)
        err = np.abs(approx - ref)
        print(f"{pieces:>7}  {err.max():>12.3e}  {err.max()/denom:>12.3e}  "
              f"{np.sqrt((err**2).mean()):>12.3e}")

    fp32 = np.array([float(gelu_tanh_fp32(float(v))) for v in acts[:20000]])
    e = np.abs(fp32 - ref[:20000]).max()
    print(f"\n  golden fp32 tanh-GELU vs float64 reference: max abs err {e:.3e}")

    # The number that makes this actionable: GELU's output is re-quantized to FP8 at the
    # tensor boundary (§3), so PWL error below FP8's own quantization error is invisible.
    # E4M3 carries 3 explicit mantissa bits, so a half-ulp is 2**-5 relative.
    half_ulp = 2.0 ** -5
    print(f"\n  FP8 E4M3 half-ulp (relative): {half_ulp:.3e}")
    for pieces in (4, 8, 16, 32):
        rel = np.abs(pwl_gelu(acts, pieces, lo, hi) - ref).max() / denom
        verdict = "below FP8 noise" if rel < half_ulp else "above FP8 noise"
        print(f"    {pieces:>3} pieces -> rel {rel:.3e}   {verdict}")
    print("\n  So piece counts past ~16 buy accuracy the FP8 boundary immediately\n"
          "  discards. That bounds the choice; it does not make it.")

    print("\nNOT a decision. DECIDE-5's owner picks the piece count at gate 1; this is\n"
          "the cost curve that choice should be made against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
