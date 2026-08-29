"""§10 vectors for §6/§7: L2 trace and L3 state, with both remaining must-fail mutants."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state import (BRANCHES, KVRing, Mode, RPUState, Read, Region,       # noqa: E402
                   RingAsMemcpy, STEPS, weight_trace)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_conveyor_invariant() -> None:
    """§6 L2: every step issues the IDENTICAL weight-region trace."""
    print("\n§6 conveyor invariant (L2)")
    st = RPUState(n_layers=4, n_ctx=64, n_text=256, mode=Mode.BALANCED,
                  weight_blocks_per_layer=5)
    traces = [weight_trace(st) for _ in range(STEPS[st.mode])]
    check("every step issues an identical trace", all(t == traces[0] for t in traces),
          f"{STEPS[st.mode]} steps, {len(traces[0])} reads each")

    counts: dict[int, int] = {}
    for r in traces[0]:
        counts[r.addr] = counts.get(r.addr, 0) + 1
    check("F2: exactly one read per weight block per step",
          set(counts.values()) == {1}, f"{len(counts)} distinct blocks")
    check("order is layer-major", [r.addr for r in traces[0]]
          == sorted(r.addr for r in traces[0]))


def test_cfg_double_fetch_mutant() -> None:
    """§10 mutant: "double-fetched CFG weights (breaks L2)"."""
    print("\n§10 must-fail mutant: double-fetched CFG weights")
    st = RPUState(n_layers=4, n_ctx=64, n_text=256, mode=Mode.BALANCED,
                  weight_blocks_per_layer=5)
    good = weight_trace(st)
    bad = weight_trace(st, double_fetch_cfg=True)
    check("the mutant produces a different trace", good != bad,
          f"{len(good)} reads vs {len(bad)}")
    counts: dict[int, int] = {}
    for r in bad:
        counts[r.addr] = counts.get(r.addr, 0) + 1
    check("mutant fetches each block once per CFG branch",
          set(counts.values()) == {BRANCHES[Mode.BALANCED]})
    check("a VALUE check would not catch it (same blocks, same order)",
          sorted({r.addr for r in bad}) == sorted({r.addr for r in good}),
          "only the trace differs -- L2 is the only detector")


def test_ring_is_pointer_arithmetic() -> None:
    """§6: "No copies, no remapping -- pointer arithmetic only"."""
    print("\n§6 KV ring: pointer arithmetic, wraparound exercised")
    n_ctx = 8
    ring = KVRing(n_ctx=n_ctx, n_layers=2)
    ring.filled = n_ctx
    before = ring.live_slots(0)
    check("a full window covers every slot exactly once",
          sorted(s % n_ctx for s in before) == list(range(n_ctx)))

    ring.advance(3)
    after = ring.live_slots(0)
    check("advancing moves the head, not the data", ring.head == 3, f"head={ring.head}")
    check("the window wraps", after[-1] % n_ctx < after[0] % n_ctx,
          f"first={after[0] % n_ctx} last={after[-1] % n_ctx}")
    check("still covers every slot exactly once after wrap",
          sorted(s % n_ctx for s in after) == list(range(n_ctx)))

    ring.advance(n_ctx)
    check("advancing a full window returns to the start", ring.head == 3)
    try:
        ring.advance(n_ctx + 1); check("over-eviction is rejected", False)
    except ValueError:
        check("over-eviction is rejected", True)


def test_ring_memcpy_mutant() -> None:
    """§10 mutant: "ring implemented as memcpy (passes values, fails wraparound)"."""
    print("\n§10 must-fail mutant: ring as memcpy")
    real, mut = KVRing(8, 2), RingAsMemcpy(8, 2)
    real.filled = mut.filled = 8
    check("before any advance the two agree",
          real.live_slots(0) == mut.live_slots(0),
          "a value-only test would pass here and stop")
    real.advance(3); mut.advance(3)
    check("after advancing, the mutant's slots never wrap",
          mut.live_slots(0) != real.live_slots(0),
          f"real head={real.head}, mutant head={mut.head}")
    check("the mutant's slot sequence is monotonic (the tell)",
          mut.live_slots(0) == sorted(mut.live_slots(0)))
    check("the real ring's is not", real.live_slots(0) != sorted(real.live_slots(0)))


def test_l3_and_chunk_purity() -> None:
    """§7: "a chunk is a pure function of (state, fresh tokens, mode) -- that property
    is itself a conformance test"."""
    print("\n§7 L3 state and chunk purity")
    def fresh() -> RPUState:
        return RPUState(n_layers=2, n_ctx=16, n_text=256, mode=Mode.QUALITY,
                        weight_blocks_per_layer=3)
    a, b = fresh(), fresh()
    check("identical inputs give identical L3 fingerprints",
          a.fingerprint() == b.fingerprint())
    a.kv_ring.advance(4); b.kv_ring.advance(4)
    check("identical chunk-end advances stay identical",
          a.fingerprint() == b.fingerprint())
    b.kv_ring.advance(1)
    check("a different advance is visible in the fingerprint",
          a.fingerprint() != b.fingerprint())
    check("MODE_REG selects the step count",
          [STEPS[m] for m in (Mode.QUALITY, Mode.BALANCED, Mode.DEADLINE)] == [3, 2, 1])
    check("DECIDE-9 is not assumed: Deadline has no branch count",
          Mode.DEADLINE not in BRANCHES)


for t in (test_conveyor_invariant, test_cfg_double_fetch_mutant,
          test_ring_is_pointer_arithmetic, test_ring_memcpy_mutant,
          test_l3_and_chunk_purity):
    t()

print(f"\n{'ALL PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
