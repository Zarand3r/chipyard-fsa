"""Configuration for the RPU numerical golden model.

`docs/GOLDEN_MODEL_SPEC.md` §11 lists thirteen open decisions and is explicit that where
a parameter is open it is "marked **DECIDE** with its owning gate, never silently
invented". This module keeps that discipline mechanical: an open decision has **no
default**. Constructing a config without naming it raises.

That is deliberate friction. A silent default is how a gate-1 numerics question turns
into an accidental architectural commitment that nobody remembers making.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WeightProfile(StrEnum):
    """§3. Both are E2M1 elements; they differ in block size and scale format."""
    MXFP4 = "mxfp4"      # 32-element blocks, shared E8M0 power-of-two scale
    NVFP4 = "nvfp4"      # 16-element blocks, FP8-E4M3 scale


class Fp8Format(StrEnum):
    E4M3 = "e4m3"
    E5M2 = "e5m2"


class TreeNodeRounding(StrEnum):
    """DECIDE-3, gate-1: exact tree vs FP32-rounded tree nodes.

    §4.2 says the golden model implements both behind a flag, default exact-then-round.
    """
    EXACT = "exact"              # evaluate the whole tree exactly, round once
    ROUND_EACH_NODE = "rounded"  # RNE to fp32 at every tree node


@dataclass(frozen=True)
class NumericConfig:
    """One configuration. §4.5: bit-exactness claims are always *per configuration*."""

    # --- fixed by the spec, not open ---
    k_block: int = 128                 # §4.1, one systolic tile traversal
    accum_format: str = "fp32"         # §3, default profile

    # --- open decisions; no defaults on purpose ---
    activation_fp8: Fp8Format = None        # DECIDE-1, pre-RTL numerics study
    tree_width: int = None                  # DECIDE-4, gate-1 (4/8/16)
    tree_rounding: TreeNodeRounding = None  # DECIDE-3, gate-1
    weight_profile: WeightProfile = None    # §3 profile is a ucode field

    def __post_init__(self) -> None:
        missing = [n for n in ("activation_fp8", "tree_width", "tree_rounding",
                               "weight_profile") if getattr(self, n) is None]
        if missing:
            raise ValueError(
                "open DECIDE parameters must be set explicitly, never defaulted: "
                + ", ".join(missing)
                + ". See docs/GOLDEN_MODEL_SPEC.md §11."
            )
        if self.tree_width not in (4, 8, 16):
            raise ValueError(f"DECIDE-4 allows tree_width in (4, 8, 16); got {self.tree_width}")
        if self.k_block % self.tree_width:
            raise ValueError(f"k_block {self.k_block} must be a multiple of tree_width")


def working_assumption() -> NumericConfig:
    """The spec's own stated working assumptions, for tests and exploration.

    §3: "working assumption: E4M3 everywhere; E5M2 nowhere until a range analysis
    demands it". §4.2 default exact-then-round. §4.5 tree width 8 is the described
    baseline. This is **not** a decision — it is a labelled placeholder so tests can run
    before gate 1, and every result derived from it must say so.
    """
    return NumericConfig(
        activation_fp8=Fp8Format.E4M3,
        tree_width=8,
        tree_rounding=TreeNodeRounding.EXACT,
        weight_profile=WeightProfile.MXFP4,
    )
