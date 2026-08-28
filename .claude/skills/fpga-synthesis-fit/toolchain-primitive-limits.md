# Toolchain primitive limits (what the open flow physically cannot emit)

Open FPGA flows (yosys plus nextpnr, openXC7) silently lack primitives you assume
exist from the vendor tools. Enumerate what your flow CANNOT emit before you
design around it, or you build a datapath that has no legal placement.

## Dead-primitive catalog
Check these before committing to a design that needs them:
- No true-differential output buffer (OBUFDS or DIFF_SSTL) on some bank types. A
  pseudo-differential workaround can route on one pin pair and be electrically
  dead on another with byte-identical config.
- Inert PLL or MMCM fine-phase output. A phase-shifted clock output (CLKOUT4, a
  CPHASE tap) can emit but do nothing on the silicon in the open flow, so a
  phase-sweep lever reads as completely inert.
- No output delay element (ODELAYE2) on high-range banks, so there is no sub-bit
  per-lane output delay knob.
- Missing site types. A primitive (BUFR, BUFIO, XADC) is unplaceable when the open
  chipdb has no site for it. nextpnr reports "no Bels remaining of type X". Some
  blocks (XADC) have NO bel at all, so the logic synthesizes but cannot place and
  the feature is unreachable without the vendor tool.

Read the package pinout too. A pin class can make a path physically impossible: a
7-series DQS-dedicated pin is not clock-capable, so a capture-clock route from it
cannot exist. Change only the pin to a clock-capable one to prove it.

## A netlist-identical clocking param can flip routability
A parameter baked into a clocking primitive can emit the IDENTICAL netlist yet
change physical routing near the IO. One value routes, another thrashes forever.
Diagnose "not converging" as CONGESTION, not timing or seed: if timing passes but
the router stays stuck with tens of thousands of arcs remaining even with timing
relaxed, it is physical congestion. Do not grind seeds against a routability cliff.
Sweep seeds only to tell routing noise from a hard structural floor. If the best
seed still cannot reach zero, the problem is structural. Stop the seed lottery.

## Floorplan the serdes off the _SING tiles with a pre-place file
The dominant DDR3 failure on openXC7 was the read/write `ISERDESE2` / `OSERDESE2`
landing on a `_SING` I/O tile (for example `X0Y149`), whose PIPs are missing from
the prjxray database, so nextpnr routes them wrong. The fix is a floorplan, not a
toolchain rebuild. openXC7 exposes `PNR_ARGS = --pre-place constraints.py --pre-route
show_bels.py`. In `constraints.py`, pin the serdes to good (non-_SING) BELs before
placement, for example `setAttr('BEL', 'ILOGIC_X0Y145/ISERDESE2')` and
`OLOGIC_X0Y145`. `show_bels.py` dumps the available BELs. That one constraint made
UberDDR3 calibrate on the exact same fork that otherwise failed. Pin the
timing-critical serdes and the rest of the placement becomes reproducible. A
pin-tied per-bit ISERDES cannot move this way, so re-pin its DQ IOB (the XDC) to
relocate it, not the serdes BEL.

## No static timing means clock routing is unprotected
An open flow with no STA does not protect the CLK-to-CLKDIV skew that a DDR read
capture depends on. Two builds from IDENTICAL RTL with bit-identical PLL config
(same tile, same phase and divide) can pass or fail calibration purely because the
GCLK spine and HCLK leaf routed through different tiles and pins, which changes the
skew at the ISERDESE2. Nothing protects that skew, so adding unrelated logic (even
a UART) perturbs placement and breaks the DDR cal. Read "it breaks when I add an
unrelated block" as evidence of a floorplan or STA gap, not a bug in the block you
added. Pin the critical serdes (above) and constrain the clock path to make it
reproducible.

## Pin your place-and-route engine, newer is not safer
A newer nextpnr commit can flip a routable design to unroutable. A 34%-LUT design
that an older engine routed failed on a "deterministic, cross-platform initial
placement" commit (`placer_heap.cc 1863fa0e`), which its own comment calls "the
butterfly that flips a routable placement into an unroutable one." Later tags carry
the open issue #97 (`Failed to route arc ... CEUSEDMUX_OUT`, a wire-reservation
regression). Pin the engine to a known-good commit and bracket regressions by
version. Reverting one commit may not fully restore routability, because local
libstdc++ will not byte-reproduce another platform's placement. A skip-failed-arcs
flag (`NEXTPNR_SKIP_FAILED_ARCS`) does NOT yield a usable bitstream. It silently
drops required connections. Treat it as a diagnostic, never a shippable path.

## Cross-vendor memory inference is not portable
A registered case-statement ROM that one vendor's synth maps to block RAM is
inferred as thousands of flops by another vendor's synth. One case-ROM became
116k flip-flops, 186% of the device. Emit an explicitly inferable memory array or
a vendor primitive, not a case pattern you hope the tool recognizes.

## A chipdb fork needs a matching binary
Adding a missing site type can be one meta file (copy the artix7
`site_type_BUFR.json` into spartan7, it is fabric-generic with the same md5 across
families) plus an engine pre-place line. But you MUST rebuild nextpnr from the SAME
fork and regenerate the chipdb. A prebuilt binary against a new chipdb throws
"internal IDs inconsistent with the supplied chip database", because the baked-in
constids differ. The binary and the chipdb must come from one source. See
`nix-eda-packaging` for pinning that build.

A ported clock buffer needs more than the meta file and a rename. Region-preplace
it like the other global buffers, or the placer drops it in a clock region
unreachable from its input pin. For BUFR that is one engine line in the preplace
loop (`if (ci->type == id_BUFR_BUFR) try_preplace(ci, id_I)`), plus, for BUFIO, the
`I -> O` pip in the previously-empty `site_type_BUFIO.json`. Regenerate the chipdb
with the FULL part name (`xc7s50csga324-1`), not the die family (`xc7s50`), or the
export regex does not match. Confirm the regen worked by size: the `.bin` grows by
exactly the added site (a few kB), and you can exonerate the database versus the
engine by diffing `.bin` sizes. Same size plus a routing failure means the engine
binary regressed, not the database.
