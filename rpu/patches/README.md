# Patches to the FSA submodule

`generators/fsa` is a separate git repository (`VCA-EPFL/FSA`) that this fork does not
track, so any edit there is invisible to a fresh clone and lost on a submodule update
(D-106). Changes we need for an experiment live here as patches instead: applied
deliberately, reverted deliberately, and visible in our history.

Apply with `rpu/patches/apply.sh`, revert with `rpu/patches/revert.sh`.

| patch | why |
|---|---|
| `01-dma-inst-queue-depth.patch` | D-130: `dmaInst` is `Queue(..., pipe = true)` with no `entries`, so it holds Chisel's default of **2**, while `mxInst` gets `mxInflight = 8`. The decoder is a single in-order splitter, so a full DMA queue stalls every instruction behind it and caps prefetch at 2 regardless of scratchpad size. This makes the depth a parameter so the hypothesis can be measured. |
