# Bench and capture gotchas (silent no-ops that look like hangs)

Most "the board hung" reports on the bench are capture or tooling artifacts, not
the design. Rule these out before you debug the logic. Each one below produced a
long false chase.

## A multi-word command in a shell variable can silently not run
The interactive shell here is zsh. zsh does NOT word-split an unquoted multi-word
variable. So `OFL="nix shell nixpkgs#openfpgaloader --command openFPGALoader -b
arty"; $OFL -f file` fails inline with "command not found: nix shell ...", and the
flash or bitstream load NEVER runs. The symptom is a 0-byte UART capture that
looks exactly like a boot hang, but nothing was ever loaded. Put every multi-word
program invocation in a `.sh` file and run `bash script.sh`. bash word-splits.

## UART drops the first seconds right after reconfiguration
A USB-UART bridge (FT2232H and similar) drops the first few seconds of the stream
right after the FPGA reconfigures. Early boot prints (banner, first-stage setup)
capture unreliably. Two fixes:
- Store early diagnostics in globals and RE-PRINT them late, after config settles.
- Size the capture window to the FULL boot. A 2.3 MB SPI-XIP copy at a slow core
  takes minutes, so a window that ends at "read setup" is this artifact, not a
  hang. Make the window longer than the whole boot before you call it dead.

## Kill stale UART readers before every capture
Two readers on one tty split the byte stream between them, so every capture looks
empty. Kill all stale `cat /dev/ttyUSBx` before each capture.

## A transient JTAG state looks like dead RTL but is not
A config-JTAG or BSCAN tunnel can drop into a weird transient state where OpenOCD
reads a garbage `dtmcontrol` (for example 0x9, "Unsupported DTM version 9") even
on a KNOWN-GOOD bitstream with working JTAG. Do NOT conclude the RTL or the debug
module is broken from this alone. Recover it: reload the bitstream, retry OpenOCD,
or replug and power-cycle the adapter, then re-run. Suspect the debug module only
after the garbage SURVIVES a fresh bitstream reload plus a retry.

## Widen a marginal capture eye by slowing the clock
When the flow gives you no fine-phase actuator (no per-lane output delay, a dead
PLL phase output), you cannot CENTER a marginal capture eye at full rate. Slow the
interface clock instead. Halving the clock roughly doubles the timing margin and
turns random build-to-build garble into stable, tunable behavior. For a boot
device, correctness beats bandwidth, so a slower working link is the right call.

Counterpoint, know the floor: an analog block may be characterized only at its
rated speed. Running a DDR3 interface far below its slowest JEDEC bin runs the
on-die DLL below spec, and beats near the clock edge resolve metastably. "Widen by
slowing" has a floor set by analog validity, not by the digital logic. Read the
part's speed-bin table before you drop the clock past it.
