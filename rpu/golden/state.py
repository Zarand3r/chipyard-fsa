"""§7: the named persistent state objects, and §6's memory semantics.

§1 defines three conformance levels. `reduce.py` and `datapath.py` serve **L1**, value
conformance. This module serves **L2** (the memory access trace) and **L3** (persistent
state exposed for inspection between chunks) -- the two the golden model had no
implementation of at all.

The spec is unusually specific about *how* two of these must be built, because the
implementation choice is itself the thing under test:

  * §6 KV ring: "No copies, no remapping -- pointer arithmetic only, and the golden
    model must implement it as such so wraparound addressing is exercised."
  * §6 conveyor invariant: "One read per weight block per step (F2)", i.e. the CFG pair
    shares every fetched block.

§10 names both of their negations as mutants the suite must catch, so both are
implemented here beside the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Mode(StrEnum):
    """§7 MODE_REG. §2: 1/2/3 diffusion steps, mode-selected."""
    QUALITY = "quality"
    BALANCED = "balanced"
    DEADLINE = "deadline"


STEPS = {Mode.QUALITY: 3, Mode.BALANCED: 2, Mode.DEADLINE: 1}
# §2: CFG pair in Quality and Balanced; DECIDE-9 for Deadline, so it is not assumed.
BRANCHES = {Mode.QUALITY: 2, Mode.BALANCED: 2}


class Region(StrEnum):
    """§6 address map. Layout is normative; base addresses are configuration."""
    WEIGHT = "weight"
    KV_RING = "kv_ring"
    ACTIVATION_SPILL = "spill"
    SCHEDULE = "schedule"


@dataclass(frozen=True)
class Read:
    """One entry of the L2 trace: region, address, length. Order is significant."""
    region: Region
    addr: int
    length: int


@dataclass
class KVRing:
    """§6 KV ring. Pointer arithmetic only -- never copies, never remaps.

    Holds `n_ctx` token slots per layer. At chunk end the head advances by `n_new`,
    which evicts the oldest `n_new` implicitly. `slot()` is the only address mapping,
    and it wraps -- which is what makes wraparound addressing testable.
    """
    n_ctx: int
    n_layers: int
    head: int = 0                       # index of the oldest live token
    filled: int = 0

    def slot(self, layer: int, age: int) -> int:
        """Physical slot for the token `age` positions after the oldest. Wraps."""
        if not 0 <= age < self.n_ctx:
            raise IndexError(f"age {age} outside the {self.n_ctx}-token window")
        return layer * self.n_ctx + (self.head + age) % self.n_ctx

    def advance(self, n_new: int) -> None:
        """Chunk-end pointer advance. No data moves."""
        if n_new > self.n_ctx:
            raise ValueError("cannot evict more tokens than the window holds")
        self.head = (self.head + n_new) % self.n_ctx
        self.filled = min(self.filled + n_new, self.n_ctx)

    def live_slots(self, layer: int) -> list[int]:
        return [self.slot(layer, a) for a in range(min(self.filled, self.n_ctx))]


@dataclass
class RingAsMemcpy(KVRing):
    """MUTANT (§10): "ring implemented as memcpy (passes values, fails wraparound
    addressing vectors)".

    Keeps the oldest token pinned at slot 0 and pretends the data was shifted down. It
    returns the same *values* a caller would expect, so any test that only checks values
    passes -- and the slot sequence never wraps, which is exactly what a wraparound
    vector detects.
    """

    def slot(self, layer: int, age: int) -> int:
        if not 0 <= age < self.n_ctx:
            raise IndexError(f"age {age} outside the {self.n_ctx}-token window")
        return layer * self.n_ctx + age          # never wraps

    def advance(self, n_new: int) -> None:
        self.filled = min(self.filled + n_new, self.n_ctx)   # head never moves


@dataclass
class RPUState:
    """§7's named objects. A chunk is a pure function of (state, tokens, mode)."""
    n_layers: int
    n_ctx: int
    n_text: int
    mode: Mode
    weight_blocks_per_layer: int
    kv_ring: KVRing = field(init=False)
    text_kv: dict[int, object] = field(default_factory=dict)     # TEXT_KV[L], per chunk
    diffusion_latent: object = None
    action_buffer: list = field(default_factory=list)
    schedule_image: dict = field(default_factory=dict)           # µcode ROM

    def __post_init__(self) -> None:
        self.kv_ring = KVRing(self.n_ctx, self.n_layers)

    def fingerprint(self) -> tuple:
        """L3 inspection point: everything that persists across chunks."""
        return (self.kv_ring.head, self.kv_ring.filled, self.mode,
                len(self.action_buffer), tuple(sorted(self.text_kv)))


def weight_trace(state: RPUState, double_fetch_cfg: bool = False) -> list[Read]:
    """§6 conveyor invariant: the weight-region read trace for ONE diffusion step.

    Layer-major, then per-layer operator order (QKV, attn-out, FFN-in, FFN-out,
    cross-attn). DECIDE-11 leaves the intra-layer order open; this is the spec's stated
    listing and is the parameter that decision will fix.

    `double_fetch_cfg=True` is the §10 mutant: it fetches every block once per CFG
    branch, breaking F2 ("one read per weight block per step regardless of branch
    count"). Values are unaffected -- only the trace changes -- so nothing but an L2
    check can catch it.
    """
    per_block = 1
    reads: list[Read] = []
    branches = BRANCHES.get(state.mode, 1) if double_fetch_cfg else 1
    for layer in range(state.n_layers):
        for blk in range(state.weight_blocks_per_layer):
            addr = (layer * state.weight_blocks_per_layer + blk) * per_block
            for _ in range(branches):
                reads.append(Read(Region.WEIGHT, addr, per_block))
    return reads
