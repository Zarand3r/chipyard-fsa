"""Pre-RTL numerics study for phase 8: what does MXFP4 cost on real DiT weights?

D-122 established the phase-8 split: FP8 is a config parameter and already works, while
**MXFP4 microscaling is the one genuine datapath addition** -- E2M1 elements with a
shared E8M0 scale per 32-element block, which FSA has no mechanism for.

Before building that, measure what it buys and what it costs, on the actual DiT-XL/2
weights from the phase-1 trace rather than on synthetic tensors.

Note on scope: §3 says "weight quantization is out of scope -- the golden model consumes
a prepared weight image and never quantizes weights itself". So the *quantizer* lives
here, in the offline tooling, not in rpu/golden/. The golden supplies only `dequant`,
which is what hardware does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
from config import Fp8Format, WeightProfile                              # noqa: E402
from formats import BLOCK_SIZE, E2M1_MAX, _E2M1, decode_e8m0, quantize_fp8  # noqa: E402
from fractions import Fraction                                            # noqa: E402

TRACE = Path(__file__).resolve().parents[2] / "workloads/dit/build/dit_xl2_block0"
_E2M1_F = np.array([float(v) for v in _E2M1])          # magnitudes, ascending


def quantize_mxfp4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offline MXFP4 quantizer: E2M1 elements + one E8M0 scale per 32-element block.

    The scale is the power of two that puts the block's largest magnitude at or below
    E2M1's max of 6 -- exactly the "dequant = exponent add" property §3 wants.
    """
    n = BLOCK_SIZE[WeightProfile.MXFP4]
    flat = w.reshape(-1)
    pad = (-flat.size) % n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
    blocks = flat.reshape(-1, n).astype(np.float64)

    amax = np.abs(blocks).max(axis=1)
    exp = np.where(amax > 0,
                   np.ceil(np.log2(np.maximum(amax, 1e-300) / float(E2M1_MAX))), 0.0)
    exp = np.clip(exp, -127, 127)
    scale = np.power(2.0, exp)

    scaled = blocks / scale[:, None]
    mag = np.abs(scaled)
    # nearest E2M1 magnitude (ties resolved by np.argmin, adequate for a study)
    idx = np.argmin(np.abs(mag[..., None] - _E2M1_F[None, None, :]), axis=-1)
    q = _E2M1_F[idx] * np.sign(scaled)
    deq = (q * scale[:, None]).reshape(-1)[: w.size].reshape(w.shape)
    return deq, (exp + 127).astype(np.int32)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def rms_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def main() -> int:
    if not TRACE.exists():
        print(f"trace missing: {TRACE}")
        return 2

    print("MXFP4 on real DiT-XL/2 weights (phase-1 trace)\n")
    print(f"{'tensor':<12} {'shape':<16} {'MXFP4 rms':>11} {'FP8 rms':>10} "
          f"{'fp16 rms':>10} {'bits/wt':>8}")
    total_fp16 = total_mx = 0
    for name in ("w_qkv", "w_proj", "w_fc1", "w_fc2"):
        W = np.load(TRACE / f"{name}.npy").astype(np.float64)
        deq, _ = quantize_mxfp4(W)
        f8 = np.array([float(quantize_fp8(Fraction(float(v)), Fp8Format.E4M3))
                       for v in W.reshape(-1)[:20000]]).reshape(-1)
        w16 = W.astype(np.float16).astype(np.float64)
        # 4 bits per element + 8 bits of scale per 32 = 4.25 bits/weight
        bits = 4 + 8 / BLOCK_SIZE[WeightProfile.MXFP4]
        print(f"{name:<12} {str(W.shape):<16} {rms_rel(deq, W):>11.4f} "
              f"{rms_rel(f8, W.reshape(-1)[:20000]):>10.4f} "
              f"{rms_rel(w16, W):>10.6f} {bits:>8.2f}")
        total_fp16 += W.size * 16
        total_mx += W.size * bits

    print(f"\nblock weight memory: fp16 {total_fp16/8/2**20:.1f} MiB  ->  "
          f"MXFP4 {total_mx/8/2**20:.1f} MiB   "
          f"({total_fp16/total_mx:.2f}x reduction)")
    print("\nWhy this is the lever: GOLDEN_MODEL_SPEC §2 puts 14B weights at 7.0 GB in\n"
          "4-bit and specifies weight streaming with no resident weights, so DRAM\n"
          "traffic scales directly with bits/weight. 16-bit weights would be 28 GB per\n"
          "step at the same shapes -- not a bandwidth budget that closes.")
    # --- what actually matters: error in the GEMM OUTPUT, not per weight -----------
    #
    # A dot product over K=1152 averages independent weight errors, so output error is
    # much smaller than per-weight error. Quoting the weight number alone would be the
    # D-119 mistake again -- a true ratio that does not answer the question.
    print("\nend-to-end: qkv projection on real activations, K = 1152")
    X = np.load(TRACE / "modulate1_out.npy")[0].astype(np.float64)
    W = np.load(TRACE / "w_qkv.npy").T.astype(np.float64)
    ref = X @ W
    deq, _ = quantize_mxfp4(W)
    out_mx = X @ deq
    out_16 = X.astype(np.float16).astype(np.float64) @ W.astype(np.float16).astype(np.float64)
    print(f"  weights fp16  -> output rms rel {rms_rel(out_16, ref):.5f}")
    print(f"  weights MXFP4 -> output rms rel {rms_rel(out_mx, ref):.5f}   "
          f"(per-weight rms was {rms_rel(deq, W):.4f})")
    print(f"  averaging factor: {rms_rel(deq, W)/max(rms_rel(out_mx, ref),1e-12):.1f}x "
          f"-- the contraction absorbs most of it")

    print("\nRead this as an UPPER bound on MXFP4 error, not a verdict. These weights\n"
          "were trained in fp32 and quantized round-to-nearest with no calibration.\n"
          "§3 specifies a weight image 'quantized offline', which in practice means\n"
          "calibration or quantization-aware training; the spec assumes 4-bit weights\n"
          "as a premise (§2: 14B params at 7.0 GB), not as something to be discovered.")

    print("\nNOT a decision: DECIDE-1/2's owner sets the FP8 flavours and requant points\n"
          "at the pre-RTL numerics study. This is input to that, on real weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
