"""§10 vectors for §8: chunk purity, the F2 trace across steps, open-DECIDE guards."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk import NotYetSpecified, encode_tokens, run_chunk, update_engine  # noqa: E402
from state import Mode, RPUState, Region, STEPS                              # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def fresh(mode=Mode.BALANCED) -> RPUState:
    return RPUState(n_layers=3, n_ctx=32, n_text=256, mode=mode,
                    weight_blocks_per_layer=4)


def test_chunk_purity() -> None:
    """§7: "a chunk is a pure function of (state, fresh tokens, mode)"."""
    print("\n§7 chunk purity")
    st = fresh()
    before = st.fingerprint()
    a = run_chunk(st, n_new=8)
    check("the input state is not mutated", st.fingerprint() == before)
    b = run_chunk(st, n_new=8)
    check("same inputs give the same output state",
          a.state.fingerprint() == b.state.fingerprint())
    check("same inputs give the same trace", a.trace == b.trace,
          f"{len(a.trace)} reads")
    c = run_chunk(st, n_new=9)
    check("different tokens give a different state",
          c.state.fingerprint() != a.state.fingerprint())


def test_step_and_trace_structure() -> None:
    """§8.3 with §6's F2 rule."""
    print("\n§8.3 steps and the F2-shared weight stream")
    for mode in (Mode.QUALITY, Mode.BALANCED):
        r = run_chunk(fresh(mode), n_new=4)
        check(f"{mode}: runs S(mode) = {STEPS[mode]} steps", r.steps_run == STEPS[mode])
        w = [x for x in r.trace if x.region is Region.WEIGHT]
        blocks = 3 * 4                       # layers x blocks per layer
        check(f"{mode}: weight reads = steps x blocks, not x branches",
              len(w) == STEPS[mode] * blocks,
              f"{len(w)} reads, branches={r.branches}, F2 holds")
        per_step = len(w) // STEPS[mode]
        check(f"{mode}: every step's weight trace is identical",
              all(w[i * per_step:(i + 1) * per_step] == w[:per_step]
                  for i in range(STEPS[mode])))


def test_commit_advances_the_ring() -> None:
    print("\n§8.5 commit")
    st = fresh()
    r = run_chunk(st, n_new=8)
    check("ring head advanced by n_new", r.state.kv_ring.head == 8)
    r2 = run_chunk(r.state, n_new=8)
    check("a second chunk advances again", r2.state.kv_ring.head == 16)
    check("ACTION_BUFFER accumulates across chunks",
          len(r2.state.action_buffer) == 2)
    try:
        run_chunk(st, n_new=st.n_ctx + 1)
        check("over-sized chunk is rejected", False)
    except ValueError:
        check("over-sized chunk is rejected", True)


def test_open_decides_refuse() -> None:
    """Open decisions must raise, not guess -- the config.py discipline, in §8."""
    print("\nopen DECIDEs refuse rather than invent")
    for name, fn in (("DECIDE-12 VAE encoder scope", lambda: encode_tokens(None, fresh())),
                     ("DECIDE-10 update engine", lambda: update_engine(None, fresh()))):
        try:
            fn(); check(f"{name} raises", False)
        except NotYetSpecified:
            check(f"{name} raises", True)
    try:
        run_chunk(fresh(Mode.DEADLINE), n_new=4)
        check("DECIDE-9 Deadline guidance raises", False)
    except NotYetSpecified:
        check("DECIDE-9 Deadline guidance raises", True,
              "branch count unknown until the mode freeze")


for t in (test_chunk_purity, test_step_and_trace_structure,
          test_commit_advances_the_ring, test_open_decides_refuse):
    t()

print(f"\n{'ALL PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
