# Area "optimizations" that backfire or no-op (measure post-pack to catch them)

Two measured surprises that contradict intuition. Both reinforce the core rule:
judge every area change by POST-PACK `TRELLIS_COMB`, not pre-pack LUT4 and not a
mental model.

## Making a multiply ITERATIVE can INCREASE area (config-dependent)
Replacing a single-cycle NxN multiply (DSP partial products + a LUT adder tree)
with an iterative shift-add (one small multiply + accumulator, multi-cycle) is the
textbook area lever, and it does free DSP tiles. But the iterative version adds an
FSM, an accumulator, sequencing, and handshake registers. Whether it is a net
post-pack LUT/TRELLIS_COMB WIN depends on the configuration:
- In one SoC config it saved ~540 post-pack TRELLIS_COMB.
- In a LEANER config of the SAME core it LOST ~1360 (94% -> 100%): the FSM/
  accumulator overhead exceeded the partial-product tree it removed.
The DSP count dropped both times (16 -> 8). So: iterative multiply is a DSP win,
not necessarily a LUT win. NEVER assume; build both, read post-pack TRELLIS_COMB
for the EXACT config you ship, keep the smaller. (Pre-pack LUT4 also lied here -
it went UP while the real saving, when present, was in PFUMX/L6MUX mux cells that
pack into slices.)

But the STRUCTURE of the iterative multiply decides whether it wins: that first
"backfire" was a LEFT-SHIFTING-MULTIPLICAND scheme with a 2*width (128-bit)
multiplicand register, a 128-bit accumulator, and a wide feedback adder -- that
extra state and adder cost more LUT than the partial-product tree it removed.
Rewriting it as a RIGHT-SHIFTING ACCUMULATOR (shift the multiplier, accumulate
`acc + a*chunk`, narrow the accumulator to width+1 bits via a proven carry bound,
shift finished low bits into the result) shrank the state FFs and the feedback
adder and turned the SAME lever into a real win: -798 post-pack TRELLIS_COMB and
12 freed DSPs on the same core. So "iterative multiply is bigger" is not a law --
a fat iterative is, a lean right-shifting one wins. When an iterative multiply
costs more than the tree, suspect a too-wide multiplicand shifter / accumulator
before abandoning the lever.

NOTE on "noise": post-pack TRELLIS_COMB is DETERMINISTIC across nextpnr SEEDS for
a fixed netlist (the packed cell count does not depend on the placement seed), so
a cell-count A/B between two RTL variants is directly trustworthy. What IS noisy
seed-to-seed is ROUTABILITY and timing (whether it converges, the overused-wire
count, Fmax). Do not confuse the two: trust the pack-count delta; re-run the
ROUTE across seeds. (A small ~tens-of-cells wobble can come from yosys/abc9
run-to-run nondeterminism between separate synth invocations; that is well inside
any real area win.)

## The "compute everything and select" hoist only wins on DISTINCT selectors
A microcoded datapath that computes an operand read per opcode-arm and muxes the
active arm looks like a huge waste. The real win (select-then-compute: mux the
small field SELECTOR by opcode first, then do ONE wide read) is real, BUT only
for call sites with DIFFERENT runtime selectors. yosys already CSEs source-level
duplicate reads with the SAME selector, so "deduplicating" those is a NO-OP. And
once the obvious per-side hoists are done (one ALU-A read, one ALU-B read, one
store-data read), the remaining per-arm muxes often FOLD anyway: e.g. a trap-target
mux over machine/supervisor mode collapses to a constant when the core has no
supervisor mode, so yosys already shares it. After the big hoists, the residual
bulk is the INHERENT interpreter datapath (the opcode Case, the shared ALU, the
multiply tree), not more operand-mux waste, so stop hunting operand muxes and
either attack a different structure (the multiply tree) or cut SCOPE.

## Corollary: know when you have hit the floor
If a core already had its area-diet passes (dead-arm trim, operand hoist, shared
ALU, EBR regfile, iterative divider), the next "obvious" mux-sharing change may
yield ~100 cells, not thousands. When per-module stats show the giant is the
inherent datapath and the cheap structural levers are spent, the honest options
are a genuinely different microarchitecture (pipeline/serialize a big structure),
dropping a feature, a smaller core tier, or a bigger device, not another round
of mux surgery. Report the measured floor; do not gut a verified core for
diminishing, risky returns.
