---
name: golden-model-first
description: Use when authoring the bit-exact reference model and frozen test-vector corpus that an accelerator's compiler, HLS, RTL, and FPGA stages will all be checked against — including fixed-point/quantization semantics, tolerance policy, and vector serialization. Pairs with differential-verification, which covers running the comparison.
---

# Golden Model First

## Overview

The golden model is the definition of what your hardware computes. Every stage below
it — schedule, HLS, RTL, bitstream — is checked by replaying the same inputs and
comparing. Write it before the implementation, or it will quietly become a
transcription of whatever the implementation happens to do.

**Core principle:** the golden model is optimized for being obviously correct, not
for being fast. The moment you optimize it, it starts sharing bugs with the design.

This skill covers **authoring** the model and the corpus. For running comparisons,
coverage-guided stimulus, and localizing a divergence, see `differential-verification`.

## Two Models, Not One

Keep these separate; conflating them is the classic mistake:

- **Behavioral reference** — float64, textbook math, no hardware concepts.
  Answers "what is this layer supposed to compute?"
- **Bit-accurate reference** — the actual arithmetic the hardware will do:
  fixed-point widths, rounding mode, saturation, accumulator width, the exact
  quantization/codebook scheme, operation order where it is not associative.
  Answers "what will the hardware produce, exactly?"

The RTL must match the **bit-accurate** model exactly (bit-for-bit, no tolerance).
The bit-accurate model is compared to the behavioral one **once**, to characterize
quantization error, and that comparison is what justifies your tolerance number.

If you only build the behavioral model, you will spend the project arguing about
tolerances. If you only build the bit-accurate one, you cannot tell a quantization
effect from a bug.

## Getting The Arithmetic Right

These are the details that decide whether RTL ever matches:

- **Accumulator width and when it saturates vs wraps.** Write it down; both are
  legitimate, silently picking one is not.
- **Rounding mode.** Truncate, round-half-up, round-half-even. They differ on real
  data and the difference will show up as a "random" LSB mismatch.
- **Operation order.** Floating-point addition is not associative. If the hardware
  reduces a 64-element vector as a tree, the model must reduce it as *the same tree*,
  not with a sequential loop. This one causes more phantom mismatches than any other.
- **Fused vs separate operations.** An FMA and a multiply-then-add give different
  results. Match the hardware.
- **Denormals, NaN, and infinity policy.** Hardware often flushes; float64 does not.

Encode each of these as a named constant or enum in the model, not as an implicit
consequence of how you wrote the loop.

## The Frozen Corpus

One vector set drives every stage. Freeze it early.

- **Serialize in a format every stage can read.** Plain binary or hex text beats a
  pickle: HLS csim (C++), the RTL testbench (SV `$readmemh`), cocotb (Python), and
  the on-board host all need it. Write the loader once per language, and check the
  loaders agree on a known pattern before trusting any result.
- **Store inputs *and* expected outputs**, plus a hash of both. A corpus that
  regenerates itself each run is not frozen and will drift.
- **Cover the classes deliberately**, not just random: zeros, all-ones, max/min
  representable, values that straddle a rounding boundary, values that saturate the
  accumulator, and a real workload sample. Random uniform data exercises almost none
  of the interesting arithmetic.
- **Keep a tiny case.** One that is small enough to trace by hand and fast enough to
  run in every loop. You will use it a hundred times more than the big one.

## Tolerance Policy

- RTL vs bit-accurate model: **exact**. No tolerance. A tolerance here is hiding a bug.
- HLS csim vs bit-accurate model: exact if the C++ uses the same fixed-point types;
  tolerance only where csim legitimately uses float for speed — and then say so.
- Bit-accurate vs behavioral: a tolerance derived from the quantization scheme, fixed
  once, with the derivation written next to the number.
- **Report the distribution, not just pass/fail.** Max abs error, max rel error, and
  how many elements exceeded threshold. "12 of 262144 elements over tolerance, all in
  row 50" is a finding; "PASS" hides it.

A test that prints `NOTICE` and exits zero on out-of-tolerance elements is not a
passing test. Decide whether it passes, and make the exit code say so.

## Red Flags

| Smell | Do instead |
|-------|------------|
| One model doing double duty as behavioral and bit-accurate | Split them; compare once to set tolerance |
| Golden model written after the RTL | Write it first, or it transcribes the bug |
| Golden model optimized for speed | Keep it obvious; slow is fine |
| Reduction order differs from hardware | Mirror the hardware's tree exactly |
| Rounding/saturation left implicit | Name them as constants in the model |
| Corpus regenerated per run | Freeze it, hash it, commit it |
| Only random uniform stimulus | Add boundary, saturating, and real-workload cases |
| Tolerance widened when a test fails | Derive it from the arithmetic; a widening needs a reason |
| `NOTICE: 12 results don't match` with exit 0 | Make the verdict binary and the exit code honest |
| Each stage loads vectors its own way | One corpus, one loader per language, cross-checked |
