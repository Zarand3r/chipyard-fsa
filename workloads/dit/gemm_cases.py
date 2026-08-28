"""Extract the GEMMs of one real DiT block from the phase-1 trace.

Gate B asks for `C = AB` validated on "a real transformer-shaped matrix". These are
those matrices -- pulled from the frozen workload rather than invented, so a Gate B
pass is evidence about the workload the program actually targets.

Emitted as plain .npy triples (A, B, C_golden) with a manifest. Deliberately
format-agnostic: phase 1 produces reference values, phase 2 decides how they map onto
the array. Keeping that boundary is what stops the golden from quietly acquiring the
hardware's opinions.

    python gemm_cases.py --trace build/golden/dit_xl2_block0 --out build/golden/gemm
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pin import PIN


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _load(trace: Path, name: str) -> np.ndarray:
    return np.load(trace / f"{name}.npy")


def build_cases(trace: Path) -> dict[str, dict]:
    """Every GEMM in the block, as (A, B, C) with C recomputed and checked.

    torch's nn.Linear holds weight as [out_features, in_features] and computes
    x @ W.T + b, so B here is the transposed weight -- an honest [K, N] operand rather
    than a layout the reader has to infer.
    """
    N, D, H, Dh = PIN.num_tokens, PIN.hidden_size, PIN.num_heads, PIN.head_dim
    mlp = 4 * D

    x1 = _load(trace, "modulate1_out")[0]        # (N, D)
    x2 = _load(trace, "modulate2_out")[0]        # (N, D)
    ctx = _load(trace, "attn_ctx")[0]            # (H, N, Dh)
    ctx_m = ctx.transpose(1, 0, 2).reshape(N, D)
    gelu = _load(trace, "mlp_gelu_out")[0]       # (N, mlp)
    q = _load(trace, "q")[0]                     # (H, N, Dh) -- already scaled? no: q, pre-scale
    k = _load(trace, "k")[0]
    v = _load(trace, "v")[0]
    probs = _load(trace, "attn_probs")[0]        # (H, N, N)

    cases: dict[str, dict] = {
        "qkv_proj": dict(
            A=x1, B=_load(trace, "w_qkv").T, C=_load(trace, "qkv_proj_out")[0],
            bias=_load(trace, "b_qkv"),
            note=f"fused QKV: [{N}x{D}] @ [{D}x{3*D}]",
        ),
        "attn_out_proj": dict(
            A=ctx_m, B=_load(trace, "w_proj").T, C=_load(trace, "attn_proj_out")[0],
            bias=_load(trace, "b_proj"),
            note=f"output projection: [{N}x{D}] @ [{D}x{D}]",
        ),
        "mlp_fc1": dict(
            A=x2, B=_load(trace, "w_fc1").T, C=_load(trace, "mlp_fc1_out")[0],
            bias=_load(trace, "b_fc1"),
            note=f"MLP up: [{N}x{D}] @ [{D}x{mlp}]",
        ),
        "mlp_fc2": dict(
            A=gelu, B=_load(trace, "w_fc2").T, C=_load(trace, "mlp_fc2_out")[0],
            bias=_load(trace, "b_fc2"),
            note=f"MLP down: [{N}x{mlp}] @ [{mlp}x{D}]",
        ),
        "adaln": dict(
            A=_load(trace, "adaln_in_silu"), B=_load(trace, "w_adaln").T,
            C=_load(trace, "adaln_out"), bias=_load(trace, "b_adaln"),
            note=f"adaLN-Zero: [{PIN.batch}x{D}] @ [{D}x{6*D}] -- M=1, the awkward one",
        ),
        # Head 0 only. The per-head GEMMs are where d_head=72 bites (D-104): these are
        # the shapes that have to land on an array whose row count is the head dim.
        # Scale is folded into Q before the GEMM, exactly as timm does it, so the
        # case is a plain matmul with no fused scalar hiding in it.
        "attn_scores_h0": dict(
            A=q[0] * _load(trace, "attn_scale"), B=k[0].T,
            C=_load(trace, "attn_scores")[0][0],
            note=f"S = (Q*scale) @ K^T for one head: [{N}x{Dh}] @ [{Dh}x{N}]",
        ),
        "attn_ctx_h0": dict(
            A=probs[0], B=v[0], C=ctx[0],
            note=f"O = P @ V for one head: [{N}x{N}] @ [{N}x{Dh}]",
        ),
    }
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.trace)
    manifest: dict[str, dict] = {}
    fail = 0

    for name, c in cases.items():
        A, B = c["A"], c["B"]
        entry = {"note": c["note"]}

        if A is None:
            print(f"  SKIP  {name}: no A operand dumped -- case cannot be validated")
            fail = 1
        else:
            np.save(args.out / f"{name}.A.npy", A)
            entry["A"] = {"shape": list(A.shape), "sha256": _sha(A)}
            np.save(args.out / f"{name}.B.npy", B)
            entry["B"] = {"shape": list(B.shape), "sha256": _sha(B)}

            # Recompute in float64 and compare against the traced result. This is the
            # tolerance arrow of the verification chain, not the bit-exact one: fp32
            # accumulation order differs from float64, so agreement is bounded, not
            # exact. Stating the bound is the point.
            ref = A.astype(np.float64) @ B.astype(np.float64)
            if c.get("bias") is not None:
                ref = ref + c["bias"].astype(np.float64)
            if c["C"] is not None:
                C = c["C"]
                np.save(args.out / f"{name}.C.npy", C)
                err = np.abs(ref - C.astype(np.float64))
                rel = err.max() / max(np.abs(ref).max(), 1e-30)
                entry["C"] = {"shape": list(C.shape), "sha256": _sha(C)}
                entry["fp64_recompute"] = {
                    "max_abs_err": float(err.max()),
                    "max_rel_err": float(rel),
                }
                # fp32 over K up to 4608 terms; 1e-5 relative is generous but catches a
                # wrong transpose or a mis-sliced operand, which is what this guards.
                if rel > 1e-5:
                    print(f"  FAIL  {name}: rel err {rel:.3e} -- operand layout is wrong")
                    fail = 1
                else:
                    print(f"  ok    {name}: {c['note']}  rel err {rel:.2e}")
            else:
                print(f"  ok    {name}: {c['note']}  (no traced C to compare)")
        manifest[name] = entry

    (args.out / "manifest.json").write_text(
        json.dumps({"pin": PIN.as_dict(), "cases": manifest}, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nwrote {len(cases)} GEMM cases to {args.out}")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
