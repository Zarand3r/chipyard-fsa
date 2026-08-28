# Building a full-speed USB soft-PHY + DFU loader: the silicon-grade bugs

A full-speed (12 Mbps) USB device with a SOFT PHY (D+/D- straight to FPGA pins,
no PHY chip) + an EP0/DFU engine is a great FPGA bring-up vehicle, but it is dense
with "passes the happy-path test, dead or corrupt on silicon" traps. These are the
ones an adversarial, per-module RTL review caught. Run the PHY off a clock that is
4x the bit rate (48 MHz for FS) so you can 4x-oversample; cross to the slower
core/bus domain with an async FIFO.

## Clock recovery: sample at BIT CENTER, not the boundary
A 4x-oversample edge-aligned DLL re-centers a phase counter on each line
transition. If you re-center so the sample strobe fires on the SAME cycle as the
edge (the bit boundary, where the line is transitioning), you get ZERO jitter
margin and sample the in-flight transition. In ROHD/RTL where the strobe is
combinational on a REGISTERED phase, "re-center to 1, strobe at phase==1" fires at
the boundary. Re-center so the strobe lands ~2 oversamples after the boundary
(bit center) for symmetric margin. LATCH the sampled symbol at the strobe; do not
rely on a combinational alias the consumer must read at exactly the right cycle.

## EOP / SE0 detection: count CONSECUTIVE, not cumulative
End-of-packet is SE0 held for ~2 bit-times. If your SE0 counter increments on any
SE0 strobe and only clears at EOP, a single line glitch anywhere in the packet
arms a hair-trigger false EOP for the rest of the packet. CLEAR the counter on any
non-SE0 strobe so only CONSECUTIVE SE0s accumulate. Separately: a stray body bit
leaks on the first EOP SE0 (it produces a registered valid one cycle later while
still "active"); stop FORWARDING body data on the FIRST SE0, but only pulse EOP +
deassert active on the SECOND consecutive SE0.

## NRZI/bit-stuff timing
NRZI: data 1 = line HOLD, data 0 = TOGGLE. De-stuff: drop the bit after six
consecutive 1s. In RTL where outputs are combinational on registered state, a
`valid`/`data` pair fires on the SAME posedge the state register updates (no extra
latency); account for this when you reason about when a bit is emitted, and lock
test vectors to the actually-observed cadence rather than an assumed pipeline
delay. Bit-stuff on TX must insert the forced-toggle 0 and NOT advance the host's
bit pointer that cycle.

## Async FIFO across the PHY/core clock boundary
Two killers here, both silent:
1. A gray-pointer FIFO that tracks pointers but hardwires `rd_data` to 0 NEVER
   CARRIES DATA. Construction-only tests pass; the data path is dead. Always test
   the FIFO with a real multi-byte sequence read back, not just port construction.
2. An UNSTOPPABLE producer (PHY streams a buffered packet at 1 byte/clk) can
   silently OVERFLOW the FIFO -> dropped firmware bytes -> corrupt image, with no
   error. Add real BACK-PRESSURE (a `ready`/almost-full signal the producer
   honors; the margin must cover the producer->FIFO in-flight latency) AND a
   sticky `overflow` flag so a drop is observable. Do not rely on "the host leaves
   a gap between blocks" as an unverified timing guarantee.

## EP0 control-transfer FSM (enumeration)
- Apply SET_ADDRESS only AFTER the status stage ACK (the device must still answer
  at address 0 through the status stage). Applying it early is the classic
  enumeration-fails bug.
- DATA toggle: SETUP is DATA0; first data-stage packet DATA1 then toggles; status
  stage DATA1. On a lost ACK, RESEND with the SAME toggle (flip only after the ACK
  is confirmed), or the host rejects the duplicate and the retry never converges.
- ACCEPT a SETUP in ANY state. The host aborts a transfer by issuing a fresh
  SETUP; if only IDLE watches for it, a host abort/retry wedges EP0 forever. Add a
  global SETUP-token catch that restarts the transfer from any state, and/or a
  watchdog timeout (no RX-wait state should hang indefinitely).
- ZLP discipline: send a zero-length packet only when the data stage ended because
  the DEVICE ran out of data AND that length is an exact multiple of the max
  packet AND it was less than wLength. If it ended because you hit wLength
  (host-limited), DO NOT send a ZLP -> the host moves to status and your waiting-
  for-another-IN state wedges. Track device-response-length and wLength SEPARATELY.
- Size the EP0 receive buffer to wTransferSize + CRC (e.g. 64 + 2). A buffer
  smaller than the advertised wTransferSize silently zero-fills the tail of every
  full block -> corrupt firmware, no error.

## SPI-NOR flash write/erase engine (for DFU-to-flash)
A read-only XIP controller has no write path; provisioning needs WREN -> sector-
erase/page-program -> poll Read-Status WIP. Bricking-class bugs:
- Read/write ARBITRATION: only accept a write request when the read FSM is idle,
  and freeze the read FSM while writing, or a write seizes the shared SPI pins
  mid-read-frame (malformed CS-low transaction) and the read FSM acks garbage to
  the CPU. Hand the pins over only at a clean CS-high boundary.
- WREN needs CS to toggle (deassert) between WREN and the command, or real flash
  ignores the op.
- BOUND the WIP poll with a watchdog + an error output; an unbounded poll hangs
  forever if WIP never clears (stuck part / wrong MISO edge), locking the bus.
- Derive the address byte count from the part config (3 vs 4 byte); a hardcoded
  24-bit address erases/programs the WRONG sector on a 4-byte-address part.
- Enforce the page boundary (a program crossing a 256-byte page wraps on real NOR)
  and reject zero-length programs -> raise an error, do not silently corrupt.

## Process note
Every one of these survived a happy-path unit test and was caught only by an
adversarial reviewer that probed the corners (host abort, slow ack, back-to-back
packets, a line glitch, an exact-multiple wLength, a >max-packet transfer). Build
each layer standalone, round-trip it (TX->PHY->RX), then have a skeptic attack the
corners BEFORE silicon.
