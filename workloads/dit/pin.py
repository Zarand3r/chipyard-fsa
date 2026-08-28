"""The frozen workload definition. Everything phase 1 pins lives here and nowhere else.

Changing any value in this file invalidates every golden tensor downstream of it, so a
change here must come with a new entry in rpu/DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

# Reference clone and checkpoint live OUTSIDE the chipyard-fsa fork: the roadmap keeps
# the PyTorch model repo off the FPGA build path.
REFERENCE_ROOT = Path.home() / "rpu-simulation" / "reference"
DIT_REPO = REFERENCE_ROOT / "DiT"
CHECKPOINT = REFERENCE_ROOT / "checkpoints" / "DiT-XL-2-256x256.pt"


@dataclass(frozen=True)
class WorkloadPin:
    """See rpu/DECISIONS.md D-104 for why DiT-XL/2 and not a tile-aligned variant."""

    model: str = "DiT-XL/2"
    image_size: int = 256
    # DiT-XL/2, from models.py: depth=28, hidden_size=1152, patch_size=2, num_heads=16
    depth: int = 28
    hidden_size: int = 1152
    patch_size: int = 2
    num_heads: int = 16
    num_classes: int = 1000

    # The pinned input. Fixed, not sampled at run time.
    seed: int = 0
    timestep: int = 500
    class_label: int = 207          # golden retriever, an ImageNet class DiT was trained on
    batch: int = 1

    # Which block is traced. Block 0 sees the raw patch-embed + pos-embed activations.
    block_index: int = 0

    @property
    def head_dim(self) -> int:
        # 1152 / 16 = 72. Not a multiple of 16, and FSA binds d_head to sa_rows
        # (generators/fsa/python/main.py: `d=cfg.sa_rows`). The mapping question this
        # opens is deliberately deferred to phases 2/4 -- see D-104.
        return self.hidden_size // self.num_heads

    @property
    def latent_size(self) -> int:
        # DiT operates in the SD VAE latent space: 256/8 = 32
        return self.image_size // 8

    @property
    def num_tokens(self) -> int:
        return (self.latent_size // self.patch_size) ** 2   # (32/2)^2 = 256

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            head_dim=self.head_dim,
            latent_size=self.latent_size,
            num_tokens=self.num_tokens,
        )
        return d


PIN = WorkloadPin()
