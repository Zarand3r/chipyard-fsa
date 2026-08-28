# Multiseed routing, measure-vs-route, and build timeouts

Operational companion to the metric-trap and seed notes in SKILL.md. These cost
real wall-clock and a hung agent if you get them wrong.

## Decouple MEASUREMENT from ROUTING
nextpnr prints the packed `Device utilisation: TRELLIS_COMB` line AFTER packing
but BEFORE routing finishes. So to MEASURE fit you do NOT need a full route:

```
timeout 150 nextpnr-ecp5 --25k --package <pkg> --json top.json --lpf top.lpf \
  --textcfg /dev/null --freq <f> --seed 1 2>&1 | grep -E "TRELLIS_COMB|DP16KD"
```

The utilisation prints in well under a minute; the `timeout` then kills the
routing attempt you do not care about yet. Use THIS in the inner area-reduction
loop (change one thing, re-measure util), and only attempt a real route once util
is already under the routable threshold. Running a full route to "measure" a
94%-util design wastes 30+ minutes thrashing and tells you nothing the util line
didn't.

## ALWAYS bound long EDA commands with `timeout`
At >~88% util the router thrashes and may never converge; an agent (or script)
that launches a route and waits is dead in the water. Wrap everything:
- synth: `timeout 600 yosys -s synth.tcl`
- util read: `timeout 150 nextpnr ... --textcfg /dev/null`
- a real seed route: `timeout 1800 nextpnr ... --textcfg out_sN.config`
- generation/tests: `timeout 300 <gen>`, `timeout 1200 <tests>`
If a command times out, report it and move on; never block indefinitely on a
non-converging route.

## Multiseed: RACE seeds, don't try them serially
On a congested design seed choice dominates wall-clock and you cannot predict the
winner. On a many-core box run one nextpnr per seed CONCURRENTLY (each router is
~single-threaded), so wall-clock = the FASTEST seed, not the sum. First seed to
print "Program finished normally" AND drop its `.config` wins; `pkill` the rest;
ecppack only the winner (ecppack lives in the trellis package, often a separate
shell, keep it out of the race). Scale seed count to difficulty: ~6 at a relaxed
freq target, 8+ when timing is tight.

CONVERGENCE SIGNAL, in order: (1) PLACEMENT anneals first (log shows "iteration
#N ... temp" with falling cost); zero overused here means nothing. (2) Then
ROUTING: watch the "overused" wire column trend toward 0. Stuck/flat at tens of
thousands = that seed won't converge; let the others race. 0 overused + "Program
finished" = done.

## "Fits" (places) != "routes"
A netlist UNDER 100% TRELLIS_COMB places, but it only ROUTES reliably under ~85%,
and churns (sometimes converging) in the ~88-91% band. 94%+ thrashes (flat at
50-80k overused) and will not close, no seed saves it. So "it placed at 94%" is
NOT a fit; the deliverable is a routed bitstream. If a design is over the device's
physical cell count (util reported as 100%+), no seed can ever place it. That is
hard over-capacity; stop seeding and cut area or scope.

## A reduced-clock bring-up bitstream must match its Fmax
If the logic only meets, say, 28 MHz, regenerate the WHOLE design at a clock <=
Fmax (PLL divide + UART baud + timer timebase all derive from it). Running a
faster-configured bitstream on slower silicon fails with setup violations AND a
wrong baud. Read the per-clock "Max frequency for clock" line to confirm each
domain (e.g. a 48 MHz USB domain and a 12 MHz core domain are timed separately).
