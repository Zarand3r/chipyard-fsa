# DDR3 read capture on the open flow (Arty S7, openXC7)

Real DDR3 read-path bring-up on a Spartan-7 (Arty S7) with the open toolchain
(yosys plus a forked nextpnr-xilinx, no vendor static timing). Each item is a
"looked like a timing eye, was something else" trap. The headline: on the open
flow, a reliable DDR3 read captures on CK plus calibration, not on a DQS strobe.

## Clock the read data on CK, then calibrate
The proven open-flow DDR3 controllers (UberDDR3, LiteDRAM s7ddrphy) clock the read
DATA `ISERDESE2` on the free-running memory clock, not on DQS. UberDDR3 uses
`.CLK(i_ddr3_clk) .CLKDIV(i_controller_clk)` with no BUFIO, BUFR, or DQS on the read
data clock. Reliability comes from the calibration around that capture, not from a
source-synchronous data strobe:

- IDELAYCTRL for absolute IDELAY tap delay (see below).
- Per-bit `IDELAYE2` eye-centering.
- MPR-based read alignment (see below).
- BitSlip training plus per-lane pipe alignment.
- Gate the consumer bus until a real-cadence read/write self-test passes.

Do not build a per-read DQS-as-data self-framing gate (feed DQS into the data
ISERDESE2 and open the window when its 8 beats show the burst pattern). It routes
and it seems elegant, but LiteDRAM never captures DQS for reads at all, so the
detour costs weeks. Read the working reference PHY before you invent a mechanism.

## DQS-phase-referenced training beats data-pattern training at high CK
Even with CK-based data capture, the TRAINING algorithm sets your frequency ceiling.
Two flavors, and the difference is why one controller does 333 MHz and another dies
at 300:

- **Data-pattern training** compares reads against known constants (0x00/FF/55/AA)
  and centers the lanes. It needs the reads already roughly framed, so it locks
  loose at 200 MHz but cannot find the eye at 300 MHz and up.
- **DQS-phase-referenced calibration** deserializes DQS (separately from the data
  path) to find the burst-start index (the `01_01_01_01_00` transition), then walks
  each bit's IDELAY to the eye center with a bitslip model and a read pipe. It
  locates the eye independently of framing and scales to 333 MHz.

So DQS is not the data capture clock, but the calibration references the deserialized
DQS phase to find where the burst starts. Choose the DQS-phase-referenced algorithm
when you need high-CK read margin.

## A DQS-dedicated pin is not clock-capable
DQS-synchronous read capture can be physically unreachable on a given pinout. On
the Arty S7 csga324 package, the DQS balls (K1/L1 = `IO_L3P/N_T0_DQS_34`) are
DQS-dedicated, NOT MRCC or SRCC clock-capable pins. A 7-series part routes a DQS
pin to the ISERDES capture clock only through the `PHASER_IN` / `IN_FIFO` MIG hard
block, which the open flow does not support. Prove the wall with a tiny bufiotest:
an MRCC pin (R2) routes `CCIO -> BUFIO` with zero errors, the DQS pin (K1) fails
`IOB -> BUFIO/I`. See `fpga-synthesis-fit` for the pin-class check.

## Do not underclock DDR3 below its slowest rated bin
CK-based capture needs the on-die DLL to keep DQ and DQS aligned to CK. The DLL is
characterized only at DDR3-800 and up (400 MHz CK and faster). Running at 200 MHz
CK is half of the slowest bin, so a `dllOn = ckMhz >= 125` gate turns the DLL on
well below spec. Beats near the CK edge then resolve metastably: non-deterministic,
byte-granular, tap-independent, static-clean but dynamic-fail. Either drive the
rated CK (400 to 800 MHz) or switch to DQS-strobed capture, which works with the
DLL off. Do not grind capture-framing fixes against an out-of-spec operating point.

## IDELAYCTRL is mandatory for absolute tap delay
Without `IDELAYCTRL`, `IDELAYE2` taps give only relative delay whose absolute value
drifts with temperature and voltage. UberDDR3 instantiates IDELAYCTRL at
`REFCLK_FREQUENCY` 200 MHz with a reset pulse of about 52 us. Only then does a tap
count map to a calibrated absolute delay. If your read timing relies on absolute
tap delay, IDELAYCTRL with the correct refclk and a long-enough reset is not
optional.

## Reads of exactly 0x00 are a strobe framing failure, not a tight eye
A lane that returns exactly `0x00` (not garbage) has a whole-strobe framing
failure, not a marginal eye. IDELAY-tap sweeps and beat-rotation sweeps are inert
on such a lane, which confirms it is not a fine-timing problem. Garbage says
marginal timing. A clean `0x00` says the strobe or a whole beat is missing.

## Use the DDR3 MPR to prove a lane present and read-perfect
The multi-purpose register (MR3 A2=1) drives a fixed `0101` pattern on every DQ. An
MPR read has no write and no real data, so it isolates the read datapath alone. On
the x16 part an MPR sweep returned `0xFFFF0000` on both beats across every IDELAY
tap, which proved all 16 bits (including lane1 `bits[31:24]`) come back with a wide
clean eye. That proved the x16 die is present (256 MB, not a smaller part) and the
lane1 read path is perfect, which redirected the whole hunt to the write side. Use
a read-only oracle (MPR or a pre-written known pattern) to bisect read from write
and dead from marginal before you tune anything. See `differential-verification`.

## A CK-free-running read path never exercises the write DQS strobe
Because the read path free-runs on CK, DQS is never toggled by an MPR read, a
loopback, or any read. A "healthy" read subsystem that shares no signal path with
the failing write tells you nothing about the write strobe. Watch for validation
coverage gaps where the exonerated block and the failing block have no common
signal.

## Per-lane write-leveling is for skewed DIMM traces, not a matched chip
A single x16 chip has short matched point-to-point traces with no per-lane write
skew, so both byte lanes write open-loop with identical timing. Write-leveling
exists for long-trace DIMMs. UberDDR3 on an HR-bank board (no ODELAY) skips write
calibration and lands both lanes open-loop. Treat a "we need per-lane write phase"
wall as a likely bug or a physical strobe fault, not a missing calibration feature.

## CL+1 is the only knob that shifts capture by a whole memory clock
Xilinx read data must sit one memory-clock phase INTO the ISERDESE2 CLKDIV word,
not at the word edge, or the 8-beat burst straddles the word boundary. LiteDRAM
gets this by programming DRAM `CL+1` in MR0 (address constant `14'h510` for CL5 to
`14'h520` for CL6), which makes the DRAM drive read data one full CK later. That is
the only lever that moves the capture phase by a whole CK: IDELAY is sub-CK,
BitSlip is post-capture, and a coarse window tap is 4-CK. A `readclextra` device
parameter (default 0, plumbed like every other device param) exposes it cleanly.
Confirm the failing case is actually static-capture-phase before you spend the
knob. On creek, CL5 static reads were already clean and CL+1 was a regression: the
real failure was dynamic instruction-plus-data reads, downstream of clean static
capture.

## Read calibration must never regress below the working default
A self-calibration routine must fall back to the known-good default rather than
commit a worse result, and it must gate the consumer bus until a real-cadence
self-test passes (UberDDR3 `final_calibration_done`). A read coin-flip was
root-caused to cal picking a worse-than-default IDELAY tap, which corrupted a base
pointer and jammed the bus. A closed loop that can pick a tap worse than its
starting point is worse than no loop. See `silicon-grade-discipline`.
