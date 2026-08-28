"""Phase 1: freeze the DiT workload and dump the golden functional model.

Runs the real pretrained DiT-XL/2 on a pinned input, captures the inputs to one real
transformer block, then recomputes every stage of that block explicitly from its own
weights and asserts the recomputation reproduces the block bit-for-bit.

That assertion is the point. A dump of hooked intermediates tells you what came out; a
recomputation that matches the module exactly tells you the stage decomposition is
*correct*, which is what the RTL will be verified against stage by stage.

Output: one .npy per tensor plus a manifest.json carrying sha256 of every array, the
pin, and the environment. See rpu/DECISIONS.md D-104 (workload) and D-105 (determinism).

    python trace_block.py --out build/golden/dit_xl2_block0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinism import disable_fused_attention, lock_down          # noqa: E402
from pin import CHECKPOINT, DIT_REPO, PIN                           # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Dump:
    """Collects named tensors and writes them with a checksum manifest."""

    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}

    def add(self, name: str, t: torch.Tensor, note: str = "") -> None:
        a = t.detach().cpu().numpy()
        np.save(self.out / f"{name}.npy", a)
        self.entries[name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": sha256_array(a),
            "note": note,
        }

    def write_manifest(self, extra: dict) -> Path:
        path = self.out / "manifest.json"
        payload = {**extra, "tensors": self.entries}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


def build_model(smoke: bool = False):
    sys.path.insert(0, str(DIT_REPO))
    from models import DiT_models                                    # noqa: E402

    model = DiT_models[PIN.model](
        input_size=PIN.latent_size,
        num_classes=PIN.num_classes,
    )
    if not smoke:
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval().to(torch.float32)
    return model


def pinned_input(model):
    """The frozen input. Generated from PIN.seed, never sampled at run time."""
    g = torch.Generator(device="cpu").manual_seed(PIN.seed)
    x = torch.randn(
        PIN.batch, model.in_channels, PIN.latent_size, PIN.latent_size,
        generator=g, dtype=torch.float32,
    )
    t = torch.full((PIN.batch,), PIN.timestep, dtype=torch.long)
    y = torch.full((PIN.batch,), PIN.class_label, dtype=torch.long)
    return x, t, y


def trace(out: Path, smoke: bool = False) -> int:
    lock_down(PIN.seed)
    model = build_model(smoke)
    n_unfused = disable_fused_attention(model)

    x_lat, t, y = pinned_input(model)
    block = model.blocks[PIN.block_index]

    # Capture the real inputs to the block rather than reconstructing the stem.
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_mod, args):
        captured["x"] = args[0].detach().clone()
        captured["c"] = args[1].detach().clone()

    def post_hook(_mod, _args, output):
        captured["y_ref"] = output.detach().clone()

    h1 = block.register_forward_pre_hook(pre_hook)
    h2 = block.register_forward_hook(post_hook)
    model(x_lat, t, y)
    h1.remove()
    h2.remove()

    x, c, y_ref = captured["x"], captured["c"], captured["y_ref"]

    d = Dump(out)
    d.add("block_in_x", x, "activations entering the block: patch-embed + pos-embed")
    d.add("block_in_c", c, "conditioning vector: t_embed + y_embed")

    # ---- stage-by-stage recomputation from the block's own weights ----
    B, N, C = x.shape
    H, Dh = PIN.num_heads, PIN.head_dim
    attn, mlp = block.attn, block.mlp

    # Dump the SiLU output too: it is the A operand of the adaLN GEMM, and without it
    # that case cannot be checked the way the other six are.
    c_silu = block.adaLN_modulation[0](c)
    d.add("adaln_in_silu", c_silu, "SiLU(c) -- the A operand of the adaLN projection")
    six = block.adaLN_modulation(c)
    d.add("adaln_out", six, "SiLU -> Linear(hidden, 6*hidden); adaLN-Zero parameters")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = six.chunk(6, dim=1)
    for nm, v in [
        ("shift_msa", shift_msa), ("scale_msa", scale_msa), ("gate_msa", gate_msa),
        ("shift_mlp", shift_mlp), ("scale_mlp", scale_mlp), ("gate_mlp", gate_mlp),
    ]:
        d.add(nm, v, "adaLN-Zero per-channel parameter")

    n1 = block.norm1(x)
    d.add("norm1_out", n1, "LayerNorm, elementwise_affine=False, eps=1e-6")
    h = n1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    d.add("modulate1_out", h, "x*(1+scale)+shift -- per-channel affine")

    qkv = attn.qkv(h)
    d.add("qkv_proj_out", qkv, "fused QKV projection, bias=True")
    q, k, v = qkv.reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)   # Identity in this checkpoint
    d.add("q", q, "(B,H,N,Dh)")
    d.add("k", k, "(B,H,N,Dh)")
    d.add("v", v, "(B,H,N,Dh)")

    d.add("attn_scale", torch.tensor(attn.scale, dtype=torch.float32),
          "1/sqrt(d_head); folded into Q before the score GEMM, as timm does it")
    scores = (q * attn.scale) @ k.transpose(-2, -1)
    d.add("attn_scores", scores, "S = (Q*scale) @ K^T, matches timm's manual path")
    probs = scores.softmax(dim=-1)
    d.add("attn_probs", probs, "P = softmax(S), row-wise")
    ctx = probs @ v
    d.add("attn_ctx", ctx, "O = P @ V, (B,H,N,Dh)")
    ctx_m = ctx.transpose(1, 2).reshape(B, N, C)
    attn_out = attn.proj(ctx_m)
    d.add("attn_proj_out", attn_out, "output projection")

    x1 = x + gate_msa.unsqueeze(1) * attn_out
    d.add("residual1_out", x1, "gated residual: x + gate_msa * attn_out")

    n2 = block.norm2(x1)
    d.add("norm2_out", n2, "LayerNorm, elementwise_affine=False, eps=1e-6")
    h2 = n2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    d.add("modulate2_out", h2, "x*(1+scale)+shift")

    fc1 = mlp.fc1(h2)
    d.add("mlp_fc1_out", fc1, "hidden -> 4*hidden")
    act = mlp.act(fc1)
    d.add("mlp_gelu_out", act, "GELU, approximate='tanh'")
    fc2 = mlp.fc2(act)
    d.add("mlp_fc2_out", fc2, "4*hidden -> hidden")

    x2 = x1 + gate_mlp.unsqueeze(1) * fc2
    d.add("block_out", x2, "gated residual: x1 + gate_mlp * mlp_out")
    d.add("block_out_reference", y_ref, "the block module's own output, for comparison")

    # ---- weights ----
    for nm, p in [
        ("w_qkv", attn.qkv.weight), ("b_qkv", attn.qkv.bias),
        ("w_proj", attn.proj.weight), ("b_proj", attn.proj.bias),
        ("w_fc1", mlp.fc1.weight), ("b_fc1", mlp.fc1.bias),
        ("w_fc2", mlp.fc2.weight), ("b_fc2", mlp.fc2.bias),
        ("w_adaln", block.adaLN_modulation[1].weight),
        ("b_adaln", block.adaLN_modulation[1].bias),
    ]:
        d.add(nm, p, "block weight")

    # ---- the assertion that makes this a decomposition and not a guess ----
    delta = (x2 - y_ref).abs().max().item()
    exact = torch.equal(x2, y_ref)
    print(f"recomputed vs module output: max|delta| = {delta:.3e}  bit-exact = {exact}")

    if smoke:
        # Random weights: the stage decomposition is still exercised, but nothing here
        # is a golden. Refuse to leave a manifest that could be mistaken for one.
        for f in out.glob("*.npy"):
            f.unlink()
        print("SMOKE: decomposition exercised on random weights; no manifest written")
        return 0 if exact else 1

    manifest = d.write_manifest({
        "pin": PIN.as_dict(),
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "fused_attention_disabled_in_modules": n_unfused,
        },
        "recomputation": {"max_abs_delta": delta, "bit_exact": exact},
    })
    print(f"wrote {len(d.entries)} tensors + {manifest}")

    if not exact:
        print("FAIL: stage decomposition does not reproduce the block bit-for-bit")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="run with random weights to exercise the decomposition; "
                         "writes no manifest and produces no golden")
    a = ap.parse_args()
    raise SystemExit(trace(a.out, a.smoke))
