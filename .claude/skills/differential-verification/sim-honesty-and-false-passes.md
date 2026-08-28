# Sim honesty and false passes (when green is a lie)

A differential test is only as honest as the reference model and the signals it
compares. A model that drops a physical channel, or a test that checks only
liveness, prints a green pass over a real defect. These are the false-pass modes
that cost the most.

## A model that drops a physical channel gives a false pass
If the reference or sim memory ignores a real channel, any bug on that channel is
invisible.
- A sim memory that writes the FULL data bus and ignores byte-enables (SEL) hides
  every wrong-byte-select bug. An atomic doubleword that writes only 4 bytes
  passes in sim and fails on a SEL-respecting slave or on hardware. A directed
  test cannot regress it. Use a byte-enable-respecting memory model or hardware.
- A PHY test that leaves the DQ and DQS pins as X in sim can never reproduce a
  data bug below the pin boundary. You can exonerate every digital layer (cache,
  downsizer, both CDC bridges, the sequencer) and still have the fault live in the
  one analog layer sim cannot model. Below the PHY pins, bisection must move to
  hardware with a deterministic reproducer.

## ACK-liveness-only tests hide data bugs
A test that asserts only the handshake (`expect(done, n)`) passes while the data
is wrong. Assert the READ-BACK DATA, not just that the transaction completed.

## Variance is the discriminator: metastable vs logic bug
Read the pattern of failure before you pick a fix.
- Build-to-build or run-to-run variance at a FIXED config means analog, marginal,
  or metastable. Fix it with margin: a slower clock, a resync, retry.
- Deterministic wrong-but-stable means a LOGIC bug. Fix the RTL. Retry and margin
  will not help.
Do not apply a timing-margin fix to a deterministic error, or chase logic for a
metastable one.

## Reproduce on perfect memory to exonerate the hardware
Before you blame silicon, run the suspect workload on a golden emulator against
IDEAL memory. A long "DDR streaming write drop" saga was a compiler dead-code-
elimination bug: a register pinned across a loop back-edge had no forward use, so
DCE removed the pointer increment. The failing program reproduced the EXACT
signature (err = N-1, one survivor) on a reference core with perfect memory. A
reference core plus perfect memory failing means the PROGRAM is miscompiled, not
the hardware. Run the workload against the golden model with ideal memory first.

## Verify-retry only fixes UNCORRELATED errors
Write-then-read-until-match, or read-until-two-agree, robustly masks non-
deterministic transient errors and is a legitimate correctness filter. It fails
when errors are CORRELATED within a run (a per-boot locked-marginal phase), where
re-reads return the same wrong value. Confirm the error is uncorrelated before you
rely on voting. A runtime write with no retry still needs a clean eye.

## Byte-identical config that still fails is a physical difference
When two builds that "should be the same" behave differently, diff the actual
bitstream, not the RTL. `prjxray bit2fasm` (needs `bitread` on PATH) on both bits
localizes the divergence to real wires. Two same-RTL DDR builds had bit-identical
PLL config but routed the clock spine through different tiles, which changed the
capture skew. The FASM diff proved "same clocks, different wires" and even tied a
single SLICE move back to a suspect engine commit. The inverse is as useful: if the
FASM is byte-identical between a working instance and a dead one (two byte-groups of
the same DDR PHY), the fault is a physical or analog difference the tool cannot
express, not an RTL bug. Stop editing RTL and look at the pins.

## A deeply-investigated root cause can still be secondary
A plausible root cause you have chased for days can be secondary to a cheaper one.
A DDR marathon was spent convinced the fault was clock-spine routing in a toolchain
fork, but the dominant fault was serdes on bad `_SING` tiles, fixed by a one-line
pre-place constraint. Validate the fix EMPIRICALLY (the LEDs calibrated) before you
commit to a large toolchain rebuild. Try the cheap floorplan or constraint
experiment before the version swap and the revert war. And do not assume a fix
transfers to another design: re-verify the failure mode is actually present in the
new netlist or placement first. The `_SING` fix that saved one design did not apply
to another whose serdes placed fine, and nine apparent hits there were a false
positive (fabric LUTs, not IOB _SING). See `fpga-synthesis-fit`.

## A read-only oracle bisects read from write
A write-then-read test cannot separate a bad write from a bad read. Add a read-only
reference source: the DDR3 MPR (drives a fixed `0101` on all DQ) or a pre-written
known pattern. An all-zero MPR return across every tap means the read path delivers
nothing (a deep break), while a written-pattern read at 75% correct means marginal
per-bit skew (a tight eye). Use the read-only oracle to bisect read from write and
dead from marginal before you tune either side. See `fpga-bringup` for the DDR MPR
detail.

## A debug probe can lie: validate its own math
Do not trust a JTAG SBA or scan read-back tool until you validate its addressing.
One SBA adapter drove the raw byte address onto a 64-bit bus AND lane-shifted the
data by the same offset, so upper-word reads landed on the next word. The core was
fine. Only the debug path was wrong. Also: an SBA or scan probe is a PACED
inspection tool, not a streaming trigger. It cannot reproduce high-rate streaming
corruption, so do not conclude "no bug" from a probe that is too slow to see it.
