# DP16KD INITVAL Packing (ECP5)

How to instantiate a Lattice ECP5 DP16KD block RAM with initialized contents,
so a ROM/RAM maps to block RAM instead of a flop array. Derive the exact packing
from yosys's own `share/yosys/lattice/brams_map_16kd.v` (`init_slice`); do not
reinvent it. This file records the layout so you can sanity-check the result.

## x18 mode geometry

- 64 `INITVAL_xx` parameters, each 320 bits = 16 words.
- Word `i` sits at bits `[i*20 +: 18]` (18 data bits in a 20-bit slot; the top 2
  bits of each 20-bit slot are unused in x18 mode).
- 64 slices x 16 words = 1024 words deep.
- Width greater than 18 bits: use `ceil(W / 18)` DP16KD blocks side by side, each
  holding bits `[b*18 +: 18]` of every word.
- The x18 word address sits in `AD[13:4]`; the low 4 bits are tied to 0.

## Ports

- Port A = write.
- Port B = registered read (readLatency 1).
- The registered read IS your read-pipeline stage. Do not add a separate
  register stage after it, or you double-count latency.

## INITVAL Reads And Runtime Writes Are Different On Silicon

x18 mode is correct for a read-only ROM initialized by INITVAL. For a block RAM
you WRITE at runtime (a register file), x18 is risky: on at least one ECP5
(OrangeCrab LFE5U-25F) an x18 clocked runtime write never commits on silicon,
even though x18 INITVAL reads are perfect and x9 read/writes work. The variable
was the WIDTH MODE, not the port count (single-port x18 also failed), so it is
not a true-dual-port hazard.

Build a writable EBR in x9 single-port instead: width 9, word index in the upper
address bits, `ceil(64 / 9) = 8` blocks per read port, port A does both the
shared write and the read (`adA = mux(wrEn, wrAd, rdAd)`), port B disabled. It
costs more EBR blocks than x18 but it is the mode the silicon honors, and it
keeps the LUT-diet area win. Keep INITVAL ROMs (read-only) in x18; only the
writable regfile needs x9.

Localize this kind of quirk with a bare-primitive A/B ladder ON THE CHIP, not by
theorizing about ports: write a known value to the real primitive in each mode
and read it back out the UART. INITVAL-read, x9 write, and x18 write are three
separate probes. See the probe ladder in `fpga-bringup`.

## Simulation

You cannot simulate a DP16KD blackbox honoring INITVAL. Keep a flop-based model
for simulation at the SAME read latency (1) as the DP16KD, and verify against
that. Running the sim model at a faster latency than the real primitive verifies
the wrong hardware. See the RegisterFile read-latency note in `rohd-rtl-gotchas`.

Note the sim trap that hides this write-mode bug: a functional DP16KD model is
usually x18-only and shares one memory array, so x18 writes "work" in sim and
fail only on silicon. The yosys `ecp5/cells_sim.v` DP16KD is a pure stub with no
read/write logic at all, so a real-netlist sim reads every block as zero unless
you supply a functional model.

## Checklist

- Is the thing you think is a ROM actually flops? Generic yosys stat shows it as
  N `$sdffe`. If so, it has per-entry reset values and never became BRAM.
- Did you instantiate DP16KD explicitly with INITVAL, rather than relying on
  inference?
- Does the sim flop model run at read latency 1 to match port B?
- For width > 18, is each block sliced as `[b*18 +: 18]` and addressed in parallel?
