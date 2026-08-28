---
name: accelerator-scheduling
description: Use when mapping a workload onto fixed accelerator hardware — choosing tiling and loop order, allocating on-chip buffers, double buffering, deciding what runs temporally vs spatially, or emitting and validating an instruction/config stream from a compiler pass. The stage between the execution model and HLS/RTL.
---

# Accelerator Scheduling

## Overview

The hardware is a fixed set of engines, buffers, and ports. The schedule decides
what data sits where, in what order work runs, and when transfers overlap compute.
This stage is where most of the achievable performance is won or lost — a good
schedule on modest hardware beats a bad schedule on generous hardware, routinely.

**Core principle:** a schedule is only valid if it is *feasible* (fits the buffers,
respects the ports and dependences) and only good if its predicted cost matches the
execution model. Emit both facts with the schedule; a schedule without a cost is a
guess.

## Decide In This Order

1. **Dataflow / stationarity.** What stays resident across the inner loop — weights,
   activations, or partial sums? This choice determines which operand pays the
   off-chip traffic and dominates everything downstream. Pick it from the
   arithmetic intensity of the actual layer shapes, not from a paper's default.
2. **Tiling.** Tile sizes must satisfy: working set ≤ on-chip capacity, tile shape
   divides (or cleanly pads) the problem dims, and the innermost tile is large
   enough to amortize pipeline fill.
3. **Loop order.** Given tiling, the permutation sets reuse and reload counts.
   Compute the reload count per operand explicitly for each candidate order.
4. **Buffer allocation and banking.** Assign tiles to physical banks. Check *ports*,
   not just capacity — a buffer that fits but needs three concurrent reads from a
   dual-port RAM is infeasible.
5. **Overlap.** Double buffering, prefetch depth, which stages run concurrently.

Doing these out of order means redoing them. Tiling chosen before stationarity is
almost always thrown away.

## Feasibility Is A Checkable Property

Write the checks as code that runs on every emitted schedule:

- Working set per buffer ≤ its capacity, per tile, at every loop level.
- Concurrent accesses per bank per cycle ≤ physical ports.
- Every read is dominated by the write that produces it (dependences respected).
- Double-buffered regions are actually 2× allocated, not aspirationally so.
- Padding for non-dividing dims is explicit, and the padded lanes are masked out of
  the result rather than quietly summed in.

A scheduler that can emit an infeasible schedule will emit one, and you will find out
in cosim, hours later. Make infeasibility a hard error at compile time.

## Validate Against The Execution Model

This is the step that gets skipped, and skipping it disconnects stages 1 and 3.

- For the chosen schedule, compute predicted cycles **from the schedule** (tiles ×
  per-tile cost + fill/drain + non-overlapped transfer).
- Compare to what `execution-model-from-spec` predicted for that design point. They
  should agree closely — they are modeling the same thing. **A disagreement means one
  of them is wrong, and you must say which.**
- Later, compare both to the measured cosim/hardware number and record all three.

## Emitting The Instruction / Config Stream

If the accelerator is programmed rather than hardwired:

- **Version the encoding.** Hardware and compiler will drift; a version field in the
  stream turns a silent misdecode into a clean error.
- **Generate the decoder and the encoder from one description.** Hand-writing both
  sides of a bit layout guarantees they disagree eventually. If you must hand-write,
  add a round-trip test: encode → decode → compare.
- **Emit a human-readable disassembly alongside the binary.** You will read it
  constantly while debugging, and a diff of two disassemblies localizes a compiler
  regression in seconds.
- **Check the stream against the golden model** by running an interpreter over it.
  An interpreted schedule that produces the wrong answer is a compiler bug you can
  find in seconds; the same bug found in RTL sim costs a day.

## Red Flags

| Smell | Do instead |
|-------|------------|
| Tiling picked before stationarity | Choose the dataflow first; tiling follows |
| Buffer capacity checked, ports ignored | Check concurrent accesses per bank per cycle |
| Schedule emitted with no cost estimate | Emit predicted cycles with every schedule |
| Predicted cost never compared to the execution model | Reconcile stage 1 and stage 3; explain any gap |
| Infeasible schedules caught in cosim | Make feasibility a compile-time hard error |
| Encoder and decoder hand-written separately | Generate from one description, or round-trip test |
| Padding lanes summed into the result | Mask explicitly |
| Double buffering assumed, not allocated | Assert the 2× allocation |
| No disassembly output | Emit readable text alongside the binary |
| Tile sizes tuned by trying numbers in RTL sim | Tune against the model; RTL confirms, it doesn't search |
