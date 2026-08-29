"""Phase 5: one DiT block through the verification chain, stage by stage.

    PyTorch functional golden  ──►  numerical golden  ◄──►  RTL

The two arrow types are checked differently, deliberately (roadmap, EXECUTION_ROADMAP.md):

  ──►   derivation under a stated tolerance. fp32 PyTorch to the array's fp16/fp32
        arithmetic loses bits; the bound is quantization, and it is reported, not hidden.
  ◄──►  bit-exact. The numerical golden and the RTL run the *same* arithmetic, so this
        must be equality (D-116 established it for GEMM).

Stages follow §9's observation points: post-projection accumulators before
re-quantization, and post-block activations.

Slices, not full shapes: a full block is 83k-332k stationary loads per GEMM, which is
not a Verilator workload (D-115). Every number below carries its slice.

    uv run ../../rpu/experiments/phase5_chain.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
import rpu_gemm as G                                                   # noqa: E402
from gemm_golden import gemm_bitexact                                  # noqa: E402

TRACE = Path(__file__).resolve().parents[2] / "workloads/dit/build/dit_xl2_block0"
FAILURES: list[str] = []


def t(name: str) -> np.ndarray:
    return np.load(TRACE / f"{name}.npy")


def stage(label: str, got: np.ndarray, functional: np.ndarray,
          numerical: np.ndarray | None, tol: float) -> None:
    """Report both arrows for one stage."""
    f_rel = float(np.abs(got.astype(np.float64) - functional.astype(np.float64)).max()
                  / max(np.abs(functional).max(), 1e-30))
    ok_f = f_rel < tol
    line = f"  {'PASS' if ok_f else 'FAIL'}  {label:<22} vs functional golden  rel {f_rel:.3e}"
    if numerical is not None:
        exact = np.array_equal(got.astype(np.float32), numerical.astype(np.float32))
        line += f"   | vs numerical golden: {'BIT-EXACT' if exact else 'DIFFERS'}"
        if not exact:
            FAILURES.append(f"{label} not bit-exact")
    print(line)
    if not ok_f:
        FAILURES.append(f"{label} vs functional")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
    ap.add_argument("--tokens", type=int, default=16, help="token slice (multiple of cols)")
    ap.add_argument("--nout", type=int, default=16, help="output slice (multiple of rows)")
    args = ap.parse_args()

    if not TRACE.exists():
        print(f"trace missing: {TRACE}\nrun workloads/dit/trace_block.py first")
        return 2
    rows, cols = G.load_config(args.config)
    M, N = args.tokens, args.nout
    print(f"{args.config}: {rows}x{cols}   slice: {M} tokens, {N} output channels, "
          f"full K\n")

    # --- stage 1: adaLN modulation of norm1 output -----------------------------------
    n1 = t("norm1_out")[0][:M]                    # functional golden, fp32
    scale = t("scale_msa")[0]
    shift = t("shift_msa")[0]
    C = n1.shape[1]

    X = n1.astype(np.float16)
    got_mod = np.zeros((M, C), np.float32)
    for j in range(C // rows):
        sl = slice(j * rows, (j + 1) * rows)
        D = np.diag((np.float16(1.0) + scale[sl].astype(np.float16))).astype(np.float16)
        G.reset(); G.load_config(args.config)
        got_mod[:, sl] = G.run(G.make_engine(args.config),
                               np.ascontiguousarray(X[:, sl]), D, rows, cols)
    got_mod += shift.astype(np.float32)
    stage("adaLN modulate", got_mod, t("modulate1_out")[0][:M], None, 5e-3)

    # --- stage 2: QKV projection (§9c: accumulator before re-quantization) -----------
    A = t("modulate1_out")[0][:M].astype(np.float16)
    W = t("w_qkv").T[:, :N].astype(np.float16)      # (C, N) slice of the projection
    bias = t("b_qkv")[:N].astype(np.float32)
    G.reset(); G.load_config(args.config)
    got_qkv = G.run(G.make_engine(args.config), A, np.ascontiguousarray(W), rows, cols)
    num_qkv = gemm_bitexact(A, np.ascontiguousarray(W), rows)
    stage("qkv accumulator", got_qkv, t("qkv_proj_out")[0][:M, :N] - bias, num_qkv, 5e-3)
    stage("qkv + bias", got_qkv + bias, t("qkv_proj_out")[0][:M, :N], None, 5e-3)

    # --- stage 3: MLP fc1 ------------------------------------------------------------
    A2 = t("modulate2_out")[0][:M].astype(np.float16)
    W2 = t("w_fc1").T[:, :N].astype(np.float16)
    b2 = t("b_fc1")[:N].astype(np.float32)
    G.reset(); G.load_config(args.config)
    got_fc1 = G.run(G.make_engine(args.config), A2, np.ascontiguousarray(W2), rows, cols)
    num_fc1 = gemm_bitexact(A2, np.ascontiguousarray(W2), rows)
    stage("mlp_fc1 accumulator", got_fc1, t("mlp_fc1_out")[0][:M, :N] - b2, num_fc1, 5e-3)

    # --- stage 4: LayerNorm statistics, native contractions --------------------------
    xin = t("block_in_x")[0][:M].astype(np.float16)
    ones = np.ones((C, rows), np.float16)
    G.reset(); G.load_config(args.config)
    mean = G.run(G.make_engine(args.config), xin, ones, rows, cols)[:, 0] / np.float32(C)
    ref_mean = t("block_in_x")[0][:M].astype(np.float64).mean(axis=1)
    rel = float(np.abs(mean - ref_mean).max() / max(np.abs(ref_mean).max(), 1e-30))
    ok = rel < 5e-2
    print(f"  {'PASS' if ok else 'FAIL'}  {'layernorm mean':<22} vs functional golden  "
          f"rel {rel:.3e}   | native contraction, no option-A tax")
    if not ok:
        FAILURES.append("layernorm mean")

    print(f"\nPHASE 5 CHAIN: {'all stages PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    print("  FPGA leg: SKIP (D-102). The chain is closed to Verilator only.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
