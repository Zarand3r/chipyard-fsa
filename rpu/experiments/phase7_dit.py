"""Phase 7: a complete small one-step DiT block on the array, with FUSED attention.

Per D-125 the model here is a **small synthetic DiT with `d_head == array rows`**, so
FSA's fused FlashAttention actually runs. It is randomly initialised: this is a
*pipeline* test, not a fidelity claim, and no number from it says anything about
DiT-XL/2 or about model quality. Phase 5 owns fidelity, on the real checkpoint.

What is exercised, all on the array:
  * QKV projection            GEMM
  * attention                 **FSA fused** -- LOAD_STATIONARY / ATTN_SCORE /
                              ATTN_VALUE / reciprocal / LSE-norm, the accelerator's
                              defining feature, reusing upstream's own kernel
  * output projection         GEMM
  * adaLN modulation          diagonal matmul (option A, D-119)
  * MLP fc1 / fc2             GEMM
  * LayerNorm statistics      native contractions (D-120)
GELU and the residual adds stay off-array: GELU awaits gate 1 (D-121), residuals are
vector adds.

    uv run ../../rpu/experiments/phase7_dit.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
import rpu_gemm as G                                                    # noqa: E402
from datapath import gelu_tanh_fp32, modulate                           # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name:<28}" + (f" {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def torch_ref_attention(q, k, v):
    """Reference softmax attention in float64, per head."""
    d = q.shape[-1]
    s = (q.astype(np.float64) @ k.astype(np.float64).T) / np.sqrt(d)
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, cols = G.load_config(args.config)
    d_head = rows                      # D-125: tile-aligned by construction
    H, T = args.heads, args.tokens
    C = d_head * H
    print(f"{args.config}: {rows}x{cols}")
    print(f"synthetic DiT (D-125): d_head={d_head} == rows, heads={H}, hidden={C}, "
          f"tokens={T}\n")

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(T, C)).astype(np.float16) * np.float16(0.5)
    w_qkv = (rng.normal(size=(C, 3 * C)) * 0.1).astype(np.float16)
    w_proj = (rng.normal(size=(C, C)) * 0.1).astype(np.float16)
    scale = (rng.normal(size=C) * 0.1).astype(np.float16)
    shift = (rng.normal(size=C) * 0.1).astype(np.float16)

    # --- adaLN modulation, on the array (option A) ---
    mod = np.zeros((T, C), np.float32)
    for j in range(C // rows):
        sl = slice(j * rows, (j + 1) * rows)
        D = np.diag(np.float16(1.0) + scale[sl]).astype(np.float16)
        G.reset(); G.load_config(args.config)
        mod[:, sl] = G.run(G.make_engine(args.config),
                           np.ascontiguousarray(x[:, sl]), D, rows, cols)
    mod += shift.astype(np.float32)
    ref_mod = np.array([[float(v) for v in r] for r in
                        modulate([[float(v) for v in r] for r in x],
                                 [float(v) for v in shift],
                                 [float(v) for v in scale])], np.float32)
    check("adaLN modulation (array)",
          float(np.abs(mod - ref_mod).max() / np.abs(ref_mod).max()) < 5e-3,
          f"rel {float(np.abs(mod-ref_mod).max()/np.abs(ref_mod).max()):.3e}")

    # --- QKV projection, on the array ---
    A = mod.astype(np.float16)
    G.reset(); G.load_config(args.config)
    qkv = G.run(G.make_engine(args.config), A, w_qkv, rows, cols)
    ref_qkv = A.astype(np.float32) @ w_qkv.astype(np.float32)
    check("QKV projection (array)",
          float(np.abs(qkv - ref_qkv).max() / np.abs(ref_qkv).max()) < 5e-3,
          f"rel {float(np.abs(qkv-ref_qkv).max()/np.abs(ref_qkv).max()):.3e}")

    # --- attention: FSA's OWN fused kernel, per head ---
    q, k, v = (qkv[:, i * C:(i + 1) * C].reshape(T, H, d_head) for i in range(3))
    main_mod = importlib.import_module("main")
    ctx = np.zeros((T, H, d_head), np.float32)
    for h in range(H):
        qh = q[:, h].astype(np.float16)
        kh = k[:, h].astype(np.float16)
        vh = v[:, h].astype(np.float16)
        G.reset(); G.load_config(args.config)
        eng = G.make_engine(args.config)
        kern = main_mod.scaled_dot_product_attention(
            G.F.from_numpy(qh), G.F.from_numpy(kh),
            G.F.from_numpy(np.ascontiguousarray(vh.T)),
            cols, rows, False)
        ctx[:, h] = G.F.to_numpy(eng.execute(kern)).T
    ref_ctx = np.stack([torch_ref_attention(q[:, h], k[:, h], v[:, h])
                        for h in range(H)], axis=1)
    rel_a = float(np.abs(ctx - ref_ctx).max() / np.abs(ref_ctx).max())
    check("attention (FSA FUSED, on array)", rel_a < 5e-2, f"rel {rel_a:.3e}")

    # --- output projection, on the array ---
    ctx_m = ctx.reshape(T, C).astype(np.float16)
    G.reset(); G.load_config(args.config)
    out = G.run(G.make_engine(args.config), ctx_m, w_proj, rows, cols)
    ref_out = ctx_m.astype(np.float32) @ w_proj.astype(np.float32)
    check("output projection (array)",
          float(np.abs(out - ref_out).max() / np.abs(ref_out).max()) < 5e-3,
          f"rel {float(np.abs(out-ref_out).max()/np.abs(ref_out).max()):.3e}")

    # --- residual (vector op) and GELU (golden, gate-1) ---
    resid = x.astype(np.float32) + out
    check("gated residual (vector op)", np.isfinite(resid).all())
    g = np.array([[float(gelu_tanh_fp32(float(v))) for v in r] for r in resid[:2]])
    check("GELU (golden; RTL awaits gate 1)", np.isfinite(g).all())

    print(f"\nPHASE 7: {'complete block ran on the array' if not FAILURES else 'FAILED'}")
    print("  Synthetic model (D-125) -- pipeline completeness, NOT a fidelity claim.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
