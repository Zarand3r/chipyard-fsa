"""Determinism controls for the functional golden. See rpu/DECISIONS.md D-105.

The golden must reproduce bit-for-bit on any x86-64 machine with the pinned torch
version. That rules out the GPU as the source of the tensors -- TF32, SDPA backend
dispatch and cuBLAS split-k all make the low bits a function of the device. These
controls remove the class of variation rather than mitigating it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def lock_down(seed: int) -> None:
    """Make a forward pass reproducible. Call before constructing the model."""
    # cuBLAS workspace must be fixed before the first CUDA context, even on a CPU run,
    # or torch.use_deterministic_algorithms raises when CUDA is later touched.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)

    # TF32 silently truncates fp32 matmul significands to 10 bits on Ampere and later.
    # Off, unconditionally: a golden that depends on the card is not a golden.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul"):
        # torch >= 2.9 spells the same thing this way
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        except (AttributeError, RuntimeError):
            pass

    torch.set_grad_enabled(False)


def disable_fused_attention(module: torch.nn.Module) -> int:
    """Force timm's explicit attention path everywhere.

    timm's `Attention.forward` branches on `self.fused_attn` into
    `F.scaled_dot_product_attention`, which dispatches across flash / mem-efficient /
    math backends by shape and device and does not expose the score matrix. The manual
    path is both observable and stable, and it is what `trace_block.py` recomputes
    against. Returns how many modules were switched.
    """
    n = 0
    for m in module.modules():
        if getattr(m, "fused_attn", False):
            m.fused_attn = False
            n += 1
    return n
