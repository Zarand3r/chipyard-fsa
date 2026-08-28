# DiT workload freeze and functional golden

RPU-owned. Roadmap **phase 1**: pin model, checkpoint and input, then dump
deterministic intermediate tensors from one real transformer block. This is the head of
the value-verification chain:

```
PyTorch functional golden  ──►  RPU numerical golden  ◄──►  RTL  ◄──►  FPGA
      (here)                       (rpu/golden/)
```

The `──►` is a derivation under stated tolerance, not a bit-exactness claim: this
golden is fp32 and defines *what* is computed. `rpu/golden/` narrows it to exact
FP4/FP8 arithmetic. Do not report agreement with this model as bit-exactness.

**Not gated behind Gate B.** Phase 1 needs a checkpoint, a machine to run it on, and
determinism — none of which depend on whether the FSA array takes general GEMM. It runs
in parallel with the phase-2 GEMM work.

## Correctness criterion

Determinism and provenance, not accuracy. Same checkpoint, same input, same tensors,
every time, on any machine. A dump that cannot be reproduced bit-for-bit is not a
golden model, it is a sample.

## Boundary

The official PyTorch implementation (`facebookresearch/DiT`) is a **reference clone kept
outside this repository** and must not enter the Chipyard build. Its only jobs are:
load the pretrained model, run it, trace one transformer block, and dump weights,
activations and golden intermediates. Everything downstream of that dump is ours.

```
PyTorch DiT → RPU exporter (here) → static execution plan / descriptors → FSA-derived RPU
```

## What phase 1 pins, and what it therefore unblocks

`rpu/docs/GOLDEN_MODEL_SPEC.md` DECIDE-7, DECIDE-8 and DECIDE-13 (norm type and
placement, positional + conditioning scheme, action-head format) share one
prerequisite: pinning the reference checkpoint's exact block structure. Tracing a real
block is what pins it. Record the resolution in `rpu/DECISIONS.md` when it lands.

adaLN-Zero's gated residual (`x = x + gate * attn_out`) needs a per-channel elementwise
multiply. Roadmap phase 4 lists modulation among the ops to add, so this is expected
work — but it is a datapath question, not a frontend one, and phase 1's trace is what
sizes it.
