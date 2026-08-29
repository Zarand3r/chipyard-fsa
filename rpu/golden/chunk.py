"""§8: chunk execution -- the superloop, one iteration.

The step sequence is normative:

  1. ingest fresh tokens; encode  (DECIDE-12: working assumption UPSTREAM, so this
                                   model receives tokens already encoded)
  2. compute fresh-token K/V; prefill attention primes against the ring window
  3. for s = 1..S(mode): full DiT forward -- CFG pair with F2-shared weight stream
     where the mode has guidance; update engine advances the latent
  4. action head fills ACTION_BUFFER   (DECIDE-13: shape and trajectory format)
  5. commit: ring pointers advance, fresh K/V appended, latent updated

This module owns the *ordering and the trace*, not the arithmetic -- `datapath.py`
supplies the values, and a chunk here emits the L2 trace `state.py` defines. §7 says a
chunk is a pure function of (state, fresh tokens, mode) "and that property is itself a
conformance test", so `run_chunk` takes state explicitly and returns the new state
rather than mutating a global.

Two steps are deliberately not implemented and raise instead of guessing:
`DECIDE-12` (VAE encoder scope) and `DECIDE-10` (§5.6 update engine, whose ISA document
does not exist).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from state import BRANCHES, Mode, RPUState, Read, Region, STEPS, weight_trace


class NotYetSpecified(NotImplementedError):
    """Raised where an open DECIDE would otherwise be silently invented."""


@dataclass(frozen=True)
class ChunkResult:
    state: RPUState
    trace: list[Read]
    steps_run: int
    branches: int


def encode_tokens(raw, state: RPUState):
    """§8.1. DECIDE-12's working assumption is that encoding happens upstream."""
    raise NotYetSpecified(
        "DECIDE-12: VAE-encoder scope is open. §8.1's working assumption is that tokens "
        "arrive already encoded, so run_chunk() takes encoded tokens; call this only if "
        "the decision moves in-scope."
    )


def update_engine(latent, state: RPUState):
    """§8.3 / §5.6. Blocked on DECIDE-10."""
    raise NotYetSpecified(
        "DECIDE-10: the update engine's ISA document does not exist. §5.6 gives L1 over "
        "a reference Euler integrator only, and the shipped integrator and CEM variant "
        "are unchosen."
    )


def run_chunk(state: RPUState, n_new: int, encoded_tokens=None,
              emit_trace: bool = True) -> ChunkResult:
    """One superloop iteration. Pure in (state, tokens, mode).

    Returns the new state; the caller's state is not mutated, so chunk purity (§7) is
    testable by running the same chunk twice from the same state.
    """
    if n_new > state.n_ctx:
        raise ValueError(f"{n_new} fresh tokens exceeds the {state.n_ctx}-token window")

    steps = STEPS[state.mode]
    branches = BRANCHES.get(state.mode)
    if branches is None:
        raise NotYetSpecified(
            f"DECIDE-9: guidance on/off for {state.mode} mode is open, so the branch "
            "count is unknown. Quality and Balanced carry the CFG pair."
        )

    # Work on a copy: §7's purity property is only testable if the input survives.
    new = RPUState(state.n_layers, state.n_ctx, state.n_text, state.mode,
                   state.weight_blocks_per_layer)
    new.kv_ring.head = state.kv_ring.head
    new.kv_ring.filled = state.kv_ring.filled
    new.text_kv = dict(state.text_kv)
    new.diffusion_latent = state.diffusion_latent
    new.action_buffer = list(state.action_buffer)
    new.schedule_image = dict(state.schedule_image)

    trace: list[Read] = []

    # §8.2 -- fresh-token K/V written into the ring's incoming slots. Addresses come
    # from the ring's own mapping, so they wrap exactly as §6 requires.
    for layer in range(new.n_layers):
        for i in range(n_new):
            age = min(new.kv_ring.filled + i, new.n_ctx - 1)
            if emit_trace:
                trace.append(Read(Region.KV_RING, new.kv_ring.slot(layer, age), 1))

    # §8.3 -- S steps, each issuing the identical weight trace. F2: one read per block
    # per step regardless of branch count, so the branch loop must NOT re-fetch.
    for _ in range(steps):
        if emit_trace:
            trace.extend(weight_trace(new))

    # §8.4 -- action head. DECIDE-13 leaves the format open, so record the event only.
    new.action_buffer.append({"chunk_tokens": n_new, "steps": steps})

    # §8.5 -- commit. Pointer arithmetic only (§6).
    new.kv_ring.advance(n_new)

    return ChunkResult(new, trace, steps, branches)
