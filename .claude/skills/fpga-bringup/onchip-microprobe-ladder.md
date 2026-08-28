# On-chip micro-probe bring-up ladder (and silicon-only BRAM gotchas)

When every simulation says the design works but the chip is dead silent and you
cannot observe internals, do NOT keep guessing at the PLL or the core. Build a
ladder of TINY bitstreams that bit-bang ONE diagnostic byte out a known-good pin
(a UART TX, or any GPIO) from the RAW oscillator, each isolating ONE layer. Each
builds in ~2 min and routes in seconds.

## The ladder (generalizes to any board)
1. **raw-clk streamer**: bit-bang `0x55` ('U') straight from the oscillator pin,
   no PLL, no core. Proves osc + FPGA config + the pin + the host adapter.
2. **PLL LOCK probe**: emit 'L' if `LOCK` else 'U', bit-banged from the raw clk.
   Proves the PLL locks.
3. **PLL output probe**: bit-bang 'U' clocked FROM the PLL output (divide to the
   baud). Proves the post-PLL clock is clean at the expected frequency.
4. **reset-path probe**: replicate the POR + LOCK + sync-reset logic, emit a
   stage letter. Proves reset RELEASES (does not hold the core forever).
5. **per-memory probe**: instantiate the REAL memory primitive (SRAM/BRAM),
   write a known byte, read it back, stream it. Proves each memory read/writes on
   THIS silicon (see the BRAM gotcha below).
6. **ROM/INITVAL probe**: a BRAM with a known INITVAL pattern, address-walk and
   stream. Proves INITVAL contents read correctly on silicon.
7. **core-liveness heartbeat**: rewire the UART TX to bit-bang the core's own PC
   / a bus-address bit as a letter ('_' never fetched, 'S' stuck at first addr,
   'M' advancing). Localizes a wedged core.

Each probe splits the problem cleanly; when EVERY block passes in isolation, the
fault is in the assembled interaction (timing, an unmodeled primitive, a reset
race). Prefer a NON-LATCHING live indicator: a latching heartbeat made "slower
clock helps" look real when it was coincidence on a marginal design.

## Silicon-only BRAM write-width gotcha (Lattice DP16KD, reusable lesson)
A dual-port block RAM can behave differently per data-width mode on real silicon
even when sim and INITVAL-reads are fine. Concretely on one ECP5 OrangeCrab:
**x18-mode RUNTIME WRITES read back as zero**, while x9-mode writes work and x18
INITVAL reads work. A register file built on x18 wrote registers that read back 0,
so the core wedged on the first data-dependent branch, totally silent. The bare
single-primitive write probe (#5 above) is what nailed it: x9 SRAM wrote 'B',
x18 INITVAL read 'ABC', x18 runtime-write read '00'. Fix was to rebuild the RAM
in the proven width mode (x9). LESSON: when a memory is suspect, probe the bare
primitive in EACH width mode with a RUNTIME write+readback, not just an
INITVAL read or a sim model.

## Faithful-primitive sim beats the vendor stub
yosys's `ecp5/cells_sim.v` DP16KD is a non-functional STUB (declares INITVAL
params, uses none). A real-netlist sim with it reads ROMs as 0 and the core hangs
at instruction 0, a SIM ARTIFACT, not a hw bug. Write a functional primitive
model (unpack INITVAL exactly per the vendor's `brams_map` init slice; clocked
read/write) + behavioral PLL/USRMCLK stubs, and Verilate the REAL netlist. If the
banner streams in that faithful sim, the gateware is correct and the remaining
fault is analog (the PLL the stub can't model), which partitions the problem
decisively. A functional model that SHARES one array between ports will hide
true-dual-port and per-width write bugs, so model the ports honestly.
