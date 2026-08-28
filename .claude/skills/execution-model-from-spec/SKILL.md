---
name: execution-model-from-spec
description: Use when turning a chip or accelerator spec into an executable performance/resource model — roofline and bandwidth analysis, cycle estimates, DSP/BRAM/URAM/LUT budgets, tiling and parallelism design-space exploration, or deciding whether a design fits a target part before any RTL exists
---

# Execution Model From Spec

## Overview

Before any RTL, you need a program that answers "how many cycles, how much memory,
does it fit?" for a *candidate* design point. That program is the execution model.
It is the cheapest place to be wrong, and the only place you can explore a thousand
configurations in a minute.

**Core principle:** the model is code that takes a design point and returns cycles
and resources, not a spreadsheet and not a paragraph. If you cannot sweep it in a
loop, it is not an execution model.

## What The Model Must Return

For each design point, return a struct, not a single number:

- **Latency in cycles**, broken down per stage (the breakdown is what makes it
  actionable — a single total tells you nothing about what to fix).
- **Resources**: DSP, LUT (logic vs LUTRAM separately), BRAM, URAM, and on-chip
  bytes. Keep memory *banks* separate from memory *bytes*; designs fail on banks
  and ports far more often than on capacity.
- **Off-chip traffic in bytes**, and the latency that implies at the part's
  bandwidth. Report `max(compute_cycles/f, offchip_time)` and say which side binds.
- **A fit verdict** per resource against the target part, with the percentage.

Return the breakdown even when the caller only wants the total. Ninety percent of
the model's value is "QKV projection is 70% of your cycles."

## Build It In This Order

1. **Count the work.** Pure arithmetic from the spec: MACs, bytes moved, elements
   normalized. No hardware in this step. This number is a fact and is your denominator
   for every efficiency claim later.
2. **Model one engine.** Take the dominant kernel. Express its cycles as a function
   of the parallelism knobs (vector length, PE counts, banking, unroll factor).
3. **Model the memory.** On-chip footprint per engine and off-chip traffic per layer.
   This is where designs actually die.
4. **Compose.** Sum or max across engines depending on whether they run temporally
   (sequential) or spatially (overlapped). **Be explicit about which** — it is the
   single most common modeling error, and it is worth a comment in the code at the
   point of composition.
5. **Sweep.** Only now, loop over design points and plot.

## Being Honest About Fidelity

An analytical model is systematically optimistic. Name the things you left out, in
the code, near the return:

- Pipeline fill/drain per stage (matters enormously at short sequence lengths)
- FIFO stalls and imperfect overlap between producer and consumer
- Control overhead, reconfiguration, weight reload between layers
- DRAM efficiency below peak (row conflicts, refresh) — assuming 100% of spec
  bandwidth is a fiction; carry an efficiency factor and expose it

State the expected error band up front ("±20%, optimistic"). Then **calibrate**:
the first time you get a real cycle count from cosim or hardware, compare and record
the ratio. A model that has never been calibrated is a hypothesis.

## Design-Space Exploration

- Sweep the knobs that trade the same resource against each other (parallelism vs
  banking, tile size vs on-chip capacity). Sweeping independent knobs one at a time
  hides the interaction that actually binds.
- **Filter to feasible first**, then rank by latency. A point that does not fit is
  not a data point.
- Plot resource utilization alongside latency. The interesting design points sit
  right at a resource cliff, and you want to see the cliff.
- Record the chosen point and *why* it beat its neighbors. Six months later the
  constraint that ruled out the obvious choice will be invisible.

## Keeping It Honest Downstream

The model's numbers are **predictions**. Label them that way in every artifact that
leaves the project. When cosim or hardware gives you a real number:

1. Put predicted and measured side by side.
2. If the gap is over ~20%, that is a finding — either the model missed an effect
   or the implementation left performance on the table. Say which, with evidence.
3. Update the model. An uncalibrated model silently rots into a marketing number.

## Red Flags

| Smell | Do instead |
|-------|------------|
| Model returns one scalar | Return a per-stage breakdown and a resource struct |
| Spreadsheet instead of code | Make it sweepable in a loop |
| Temporal vs spatial composition left implicit | Comment it at the point of sum-vs-max |
| Assumes 100% of datasheet bandwidth | Carry an explicit DRAM efficiency factor |
| Memory modeled as bytes only | Model banks and ports too; that is what fails |
| Predicted latency reported as achieved | Label predicted vs measured everywhere |
| Model never compared to a real cycle count | Calibrate at the first cosim result |
| Infeasible points ranked alongside feasible ones | Filter to fitting designs, then rank |
| Only the total FLOPs quoted as "efficiency" | Divide by the honest work count from step 1 |
