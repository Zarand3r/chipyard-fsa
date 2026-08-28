---
name: sv-verification-stack
description: Use when writing or verifying SystemVerilog RTL with the open-source stack on this machine — Verible/slang lint, Verilator lint and simulation, cocotb testbenches, SymbiYosys formal, Yosys synthesis checks. Covers the tool ladder, what each catches, and SystemVerilog constructs that simulate but don't synthesize.
---

# SystemVerilog Verification Stack

## Overview

Open-source SV tooling is good enough to catch nearly every bug before a vendor tool
is involved — but only if you run the tools in the right order and believe them.
Each tool in the ladder catches a class the others structurally cannot.

**Core principle:** lint failures are bugs, not style. The overwhelming majority of
"weird simulation behavior" is a latch, a width mismatch, or an incomplete
sensitivity list that a linter named in the first five seconds.

## Environment

`source ~/.local/bin/eda-env.sh` puts the whole stack on PATH: Verilator 5.051,
Yosys 0.68, iverilog, SymbiYosys, slang, cocotb 2.1, nextpnr, GHDL, GTKWave, plus
Verible. All no-root, all self-contained.

Note: oss-cad-suite puts its own `python3` first on PATH. Use an explicit project
venv interpreter for anything needing numpy/torch.

## The Ladder

| Rung | Tool | Catches |
|------|------|---------|
| 1 | `verible-verilog-lint`, `slang` | Style, naming, parse errors, elaboration errors — instant |
| 2 | `verilator --lint-only -Wall` | Latches, width mismatches, unused/undriven, combinational loops |
| 3 | `verilator --binary` / cocotb | Functional behavior vs the golden model |
| 4 | `sby` (SymbiYosys) | Properties that hold for *all* inputs, not just your vectors |
| 5 | `yosys` synth + `nextpnr` | Synthesizability, area, Fmax |

Run 1 and 2 on every save; they cost nothing. Do not skip 2 because 3 passes —
a design with an inferred latch can simulate correctly and fail in silicon.

## Verilator Specifics

- `-Wall` is the useful setting. `UNOPTFLAT` usually means a real combinational
  loop; `WIDTHEXPAND`/`WIDTHTRUNC` catch the single most common RTL bug class.
- Waive a warning only with a targeted `/* verilator lint_off */` **at the site**
  plus a comment saying why it is safe. A blanket waiver file is how a real bug hides.
- Verilator is a 2-state simulator: it will **not** show you X propagation. Uninit
  registers read as 0, so a missing reset can look fine. Use iverilog (4-state) or
  formal for reset and X-propagation questions.
- `--trace` (VCD) or `--trace-fst` for waveforms; FST is far smaller for long runs.
- `--timing` is needed for testbenches using delays; prefer clock-edge-driven
  testbenches that don't need it.

## cocotb

Right choice when the reference model is Python — which it is, if you followed
`golden-model-first`. The testbench imports the golden model directly, so there is
no serialization step to get wrong between reference and DUT.

- Drive from the frozen corpus, not from freshly generated random data.
- Use a scoreboard that compares every transaction, not just the last one.
- Assert on data, never on the handshake alone. "Transaction completed" is not
  "transaction was correct."

## Formal With SymbiYosys

Underused and cheap for the properties that matter on an accelerator:

- **FIFO/stream properties**: never overflow, never underflow, no data loss.
  These are exactly where dataflow designs break, and bounded model checking finds
  the counterexample in seconds.
- **Handshake protocol**: valid stable until ready, no deadlock.
- **Arbiter fairness**: no starvation.
- Start with `mode bmc` and a shallow depth. A failing BMC gives you a concrete
  waveform, which is worth more than an unbounded proof you can't debug.
- `mode prove` (k-induction) once BMC is clean and you want the unbounded result.

## Constructs That Simulate But Don't Synthesize

- `initial` blocks for anything but FPGA block-RAM init
- Unbounded `while`, recursion, dynamic arrays, queues, `new`, classes in RTL
- Real/shortreal types
- Multiple drivers on a net, or assigning the same signal in two `always` blocks
- Blocking assignments in sequential logic — `<=` in `always_ff`, `=` in
  `always_comb`, always. Mixing them is undefined-in-practice behavior that simulates
  one way and synthesizes another.
- Use `always_ff` / `always_comb` / `always_latch` rather than bare `always`; the
  tool then checks your intent against what you wrote and errors when they disagree.

## Red Flags

| Smell | Do instead |
|-------|------------|
| Lint warnings triaged as style | Treat them as bugs; latches and width mismatches hide here |
| Blanket lint waiver file | Waive at the site, with a reason |
| Verilator-only, reset never checked | Verilator is 2-state; use iverilog or formal for X and reset |
| Testbench generates its own random data | Drive from the frozen corpus |
| Scoreboard checks the final value only | Compare every transaction |
| Assertion on handshake only | Assert the data |
| Bare `always` blocks | `always_ff` / `always_comb` so the tool checks intent |
| Blocking assignment in sequential logic | `<=` in `always_ff`, `=` in `always_comb` |
| Formal skipped as "too advanced" | BMC on FIFO overflow/underflow is minutes and finds real bugs |
| Synthesis deferred to the vendor tool | `yosys` synth early; unsynthesizable constructs surface immediately |
