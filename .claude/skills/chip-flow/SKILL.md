---
name: chip-flow
description: Use when taking a hardware accelerator from a chip spec all the way to FPGA — planning the whole flow, deciding what stage you are in, or when an artifact from one stage (execution model, golden model, compiler schedule, HLS kernel, RTL, bitstream) disagrees with another. Routes to the per-stage skills.
---

# Chip Flow

## Overview

An accelerator is built as a chain of models, each one more concrete than the last
and each one obligated to agree with its predecessor. The value is not in any single
artifact; it is in the **agreement between adjacent artifacts**, checked mechanically
on the same stimulus. A flow without those checks is six independent designs that
happen to share a name.

**Core principle:** every stage consumes the stage above it as its oracle and produces
a frozen artifact the stage below can check against. You are never "writing RTL"; you
are making RTL agree with a golden model on a corpus you froze before you started.

## The Stages

| # | Stage | Produces | Checked against | Skill |
|---|-------|----------|-----------------|-------|
| 1 | Spec → execution model | Analytical perf/area model, design-space pick | Spec's own budgets | `execution-model-from-spec` |
| 2 | Golden functional model | Bit-exact reference + frozen vectors | Spec semantics, real workload | `golden-model-first` |
| 3 | Compiler / schedule | Tiling, loop order, instruction/config stream | Execution model's cycle estimate | `accelerator-scheduling` |
| 4 | HLS | C++ kernel, `.xo`, II/latency reports | Golden model (csim), schedule (cosim cycles) | `hls-dataflow-tapa` |
| 5 | RTL | SystemVerilog, lint-clean, sim-clean | Golden model (cosim), HLS output | `sv-verification-stack` |
| 6 | FPGA | Bitstream, timing report, measured latency | Execution model's predicted cycles | `fpga-synthesis-fit`, `fpga-bringup` |

Stage 2 does not depend on stage 1 and can be written in parallel. Everything else
is a chain.

## The Gate Between Stages

Do not start stage N+1 until stage N has:

1. **A frozen artifact** — a file, committed, with a version or hash. Not a notebook.
2. **A mechanical check** — a script that exits nonzero on disagreement. Not a
   sentence in a README saying it matched.
3. **A recorded number** — the cycle count, the resource total, the max abs error.
   Written down where stage N+1 can read it, so a regression is visible.

If you cannot state the check as a command, the stage is not done.

## Where Flows Actually Break

**The model and the hardware drift apart silently.** The execution model predicts
40k cycles; RTL comes in at 95k; nobody notices because the model was never re-run.
Re-run stage 1 against the measured number at every stage that produces one, and
treat a >20% gap as a bug in the model *or* the design — you must say which.

**The golden model absorbs the DUT's bugs.** You hit a mismatch, you "fix" the
reference, the test passes. The reference is now a transcript of the implementation
and verifies nothing. When they disagree, decide which side is wrong **before**
editing either, and write down why. See `differential-verification`.

**Tolerance creep.** A quantized datapath legitimately needs a tolerance. It does not
need a tolerance that grows every time a test fails. Fix the tolerance at stage 2
from the arithmetic (what the quantization scheme can actually produce), and treat a
later widening as a design change requiring a reason.

**The schedule is never validated.** The compiler emits a tiling; nobody checks the
resulting cycle count against what the execution model said that tiling would cost.
Stage 3's output must be fed back into stage 1.

**Different stages see different stimulus.** Stage 4's csim uses a random tensor,
stage 5's testbench uses a different random tensor, the FPGA runs a third. Freeze one
vector corpus at stage 2 and drive **every** stage from it.

## Sequencing Advice

- **Vertical slice first.** One small end-to-end path (one layer, one tile, minimal
  dims) through all six stages beats a complete stage-2 model with nothing below it.
  You learn where the flow leaks only by getting to the bottom once.
- **Bring the hard part forward.** If the risk is timing closure or an HLS II you
  may not hit, build a throwaway stage-4/5/6 spike on the critical kernel before
  investing in stages 1–3.
- **Keep the golden model slow and obvious.** Its job is to be right, not fast. The
  moment it gets optimized, it starts sharing bugs with the implementation.

## Red Flags

| Smell | Do instead |
|-------|------------|
| "The RTL matches" with no command that proves it | Make the check a script with an exit code |
| Golden model edited to make a test pass | Localize first; decide which side is wrong and say why |
| Tolerance widened after a failure | Derive tolerance from the arithmetic at stage 2 and hold it |
| Execution model never revisited after stage 1 | Re-run against measured cycles at stages 3, 4, and 6 |
| Each stage generates its own random stimulus | One frozen corpus from stage 2 drives all stages |
| Perf claim from the execution model alone | Label predicted vs measured; never publish predicted as measured |
| Starting RTL before the golden model exists | Stage 2 is the oracle; without it you are guessing |
| Skipping csim to "save time" and debugging in cosim | csim is minutes, cosim is hours; fix functional bugs first |
