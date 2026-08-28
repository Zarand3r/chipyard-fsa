---
name: hls-dataflow-tapa
description: Use when writing or debugging AMD Vitis HLS or TAPA dataflow kernels — task graphs and streams, initiation interval (II), array partitioning and banking, csim/cosim/synth/impl ladder, reading HLS reports honestly, or diagnosing II violations, deadlocks, and II-vs-resource tradeoffs on Xilinx/AMD parts
---

# HLS Dataflow (Vitis HLS / TAPA)

## Overview

HLS turns a restricted C++ dialect into a dataflow circuit. The productive mental
model is not "compile my C++" but "describe a network of concurrent tasks connected
by FIFOs, and let the tool schedule each one." Code that reads like software
synthesizes like bad hardware.

**Core principle:** you are designing the task graph and the memory layout. The C++
is notation for that graph. When HLS disappoints, the fix is almost always a change
to the graph or the layout, not a pragma.

## The Ladder — Climb It In Order

Each rung is 10–100× slower than the one above. Never skip a rung to "save time."

| Rung | Command | Catches | Cost |
|------|---------|---------|------|
| 1 csim | `tapa g++` / `csim_design` | Functional bugs vs golden model | seconds–minutes |
| 2 synth | `tapa compile` / `csynth_design` | II violations, resource blowup, unsynthesizable constructs | minutes–hours |
| 3 cosim | `--bitstream=*.xo`, fast-cosim | Real cycle counts, deadlocks, FIFO depth | hours |
| 4 impl | `v++ --link`, Vivado | Timing closure, routing, real Fmax | hours–a day |

A functional bug found at rung 3 costs a day and teaches you nothing rung 1 wouldn't
have. Get csim matching the golden model bit-for-bit before you ever run synthesis.

## Designing The Task Graph

- **One task, one job, one loop nest.** Tasks that do two unrelated things serialize
  their II to the slower one.
- **Balance the stages.** Throughput is set by the slowest task. A stage with II=2
  starves every II=1 stage around it; find it in the synthesis report and fix that
  one rather than optimizing anything else.
- **Depth the FIFOs for skew, not for comfort.** A too-shallow FIFO between stages
  with unequal latency deadlocks; a too-deep one silently eats BRAM. Size from the
  actual production/consumption skew, and re-check in cosim.
- **Keep the graph acyclic where you can.** Feedback loops bound II by the loop
  latency and are the usual reason a design will not pipeline.
- **Split wide reads at the source.** Read one wide word from memory, fan it out to
  narrow lanes in a dedicated splitter task. Interleaving wide and narrow access in
  one task creates ports you did not intend.

## II, Banking, And The Real Constraint

An II violation is nearly always a **memory port conflict** or a **loop-carried
dependence**, not a compute limit.

- Read the report's stated *reason* for the II. It names the resource or the
  dependence. Fix that. Adding `#pragma HLS pipeline II=1` to something the tool
  already knows it cannot pipeline just moves the error message.
- **Array partitioning is banking.** `cyclic factor=N` gives N banks; N concurrent
  accesses per cycle need N banks *and* the access pattern must actually hit
  different banks. Partitioning an array the loop reads sequentially buys nothing.
- `complete` partitioning turns an array into registers. Cheap for tens of elements,
  catastrophic for thousands — check the LUT/FF cost in the report, do not assume.
- Choose storage deliberately: `impl=LUTRAM` for small/wide, `BRAM` for medium,
  `URAM` for large and deep. Letting the tool choose is how you exhaust BRAM while
  URAM sits idle.
- A loop-carried dependence on an accumulator is fixed by restructuring (tree
  reduction, multiple partial accumulators), not by a pragma.

## TAPA Specifics

- Tasks are `void` functions taking `tapa::istream`/`ostream`; the top level is a
  `tapa::task().invoke(...)` graph. The graph is explicit — read it first when
  picking up an unfamiliar TAPA design; it is the architecture diagram.
- `tapa::mmap` / `async_mmap` for off-chip; `async_mmap` when you need to keep
  multiple reads in flight, which is most of the time for bandwidth-bound stages.
- `tapa g++` gives a fast software simulation of the whole graph — this is rung 1
  and it is genuinely fast. Use it as the inner development loop.
- TAPA needs Xilinx HLS headers (`XILINX_HLS`) even for csim. Synthesis additionally
  needs the real Vitis install (vendor GCC, `vitis_hls`) and a part license — csim
  works without them.
- `.invoke<tapa::join, N>` instantiates N copies; the arity of the stream bundle must
  match or you get an inscrutable tuple-index static assertion.

## Reading Reports Honestly

- Post-**synthesis** numbers are estimates. Post-**implementation** numbers are real.
  Never quote synthesis estimates as achieved results, and label which you are citing.
- Latency reported as a range with `?` means trip counts are unknown; add
  `#pragma HLS loop_tripcount` so the report is meaningful — but remember it changes
  only the *report*, never the hardware.
- Cosim cycle counts include fill and drain. For steady-state throughput, measure
  across enough iterations that fill amortizes, and say which you are reporting.
- Check II *per task* in the schedule viewer, not just the top-level latency.

## Red Flags

| Smell | Do instead |
|-------|------------|
| Debugging a functional bug in cosim | Fix it in csim; cosim is for cycles and deadlock |
| Adding pragmas until II drops | Read the reported reason; fix the port conflict or dependence |
| `complete` partition on a large array | Check the LUT/FF cost; use banking instead |
| Storage impl left to the tool | Choose LUTRAM/BRAM/URAM deliberately |
| FIFO depths bumped until it stops deadlocking | Size from real skew, then confirm in cosim |
| One task doing two jobs | Split; the graph serializes to the slowest stage |
| Optimizing a stage that isn't the bottleneck | Find the slowest II in the report first |
| Synthesis estimates quoted as results | Label estimate vs measured; impl numbers are the real ones |
| Accumulator dependence "fixed" with a pragma | Restructure: tree reduction or partial accumulators |
| Wide and narrow access mixed in one task | Dedicated splitter/merger tasks |
