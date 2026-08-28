# Decisions log

Every deviation from `EXECUTION_ROADMAP.md`, and every choice a future reader would
otherwise have to reverse-engineer.

> Decision / Reason / Expected effect / Baseline rerun / Measured effect / Keep-revert

Newest last.

---

## D-101 — Miniforge, not Miniconda, for the Chipyard conda environment

**Date:** 2026-08-28 · **Roadmap phase:** 0 · **Status:** adopted

**Decision.** The conda used to run `build-setup.sh` is Miniforge
(`~/miniforge3`, conda-forge channel only), not the Miniconda that FSA's README
suggests installing.

**Reason.** `build-setup.sh` failed at step 1 under Miniconda with
`CondaToSNonInteractiveError`: Anaconda now requires accepting Terms of Service for
`repo.anaconda.com/pkgs/main` and `/pkgs/r`. Accepting those terms is a licensing
decision with commercial-use implications, and it is not one this setup needs to make:
Chipyard's own `conda-reqs/chipyard-base.yaml` declares `channels: [ucb-bar,
conda-forge, litex-hub, nodefaults]` and its lockfiles resolve only against `ucb-bar`
and `conda-forge`. The default Anaconda channels are never used. Miniforge removes them
from the picture entirely.

**Expected effect.** None on the built environment — the lockfile pins exact package
versions and checksums, and the channels it names are unchanged.

**Baseline rerun.** Gate A, once it exists.

**Keep / revert.** Keep. If a future Chipyard bump introduces a `defaults`-channel
dependency, that is a decision to surface, not to auto-accept.

---

## D-102 — The FPGA legs of both gates cannot run on this machine

**Date:** 2026-08-28 · **Roadmap phase:** 0 · **Status:** accepted constraint, blocking phases 5+

**Observation.** The roadmap's Gate A is a three-way agreement (Python reference ↔
Verilator ↔ U55C FPGA) and Gate B repeats it. The third leg is not executable here:

- No Vivado install, and none of `/opt/Xilinx`, `/tools/Xilinx`, `~/Xilinx` exists.
  `fpga/ make bitstream` therefore cannot run at all.
- No Xilinx PCIe device is present (`lspci | grep -i xilinx` is empty), so there is no
  U55C to flash even given a bitstream.
- FSA's documented host procedure needs `echo 1 > /sys/class/pci_bus/.../remove`
  followed by a PCIe rescan. That is root, which is not available.

**Consequence**, against the revised phase list. Gates A and B can be closed on their
**simulation legs only**:

| Phase | Effect |
|---|---|
| 0 — reproduce Chipyard-FSA | FPGA leg blocked; Verilator + PyTorch reference legs run |
| 5 — one complete DiT block | final FPGA leg blocked; golden → golden → Verilator runs |
| 6 — measure FPGA | blocked outright |
| 9 — correlate cycle model | simulator ↔ RTL only; the FPGA correlation is blocked |
| 10 — Jetson Thor benchmark | blocked outright, no board |
| 11 — ASIC P&R + power | additionally needs a PDK |
| 12 — tapeout decision | blocked, depends on 6 and 10 |

Phases 1, 2, 3, 4, 7 and 8 are fully reachable here: they are golden-model,
Chisel/Verilator and architecture work. Note that the revision which split the golden
functional model (phase 1) and RPU numerical golden (phase 3) into their own phases
*increases* how much of the program runs on this machine, because both are pure
software artifacts.

**How it is handled.** `rpu/scripts/gate-a.sh` reports the FPGA leg as **SKIP**, never
as PASS. A skipped leg must never be summarised as a passing gate — the whole point of
the gate is the three-way agreement, and two-way agreement is a weaker claim that has
to be stated as such.

**Keep / revert.** Revisit the moment a U55C host with Vivado is available. Until then,
do not quote any number that requires the board.

---

## D-103 — `rpu_simulation_2` is superseded by this repository

**Date:** 2026-08-28 · **Roadmap phase:** 0 · **Status:** adopted

**Decision.** `Zarand3r/chipyard-fsa` (this fork) is the single top-level hardware
repository. `Zarand3r/rpu_simulation_2`, scaffolded earlier the same day, is not part
of the program.

**Reason.** The roadmap is explicit: "This is the only top-level hardware repository for
the project. Do not create a separate project that later tries to combine Chipyard,
FSA, and FPGA infrastructure." `rpu_simulation_2` is exactly such a separate project.
It predates the roadmap by about an hour.

**Expected effect.** Its useful content — the `spec/` snapshots and the project skills
— is carried into `rpu/docs/` and `.claude/skills/` here, so nothing is lost.

**Not done unilaterally.** The `rpu_simulation_2` GitHub repository has been left in
place, not deleted. Deleting it is the owner's call.

---

## D-104 — Phase 1 pins DiT-XL/2 256×256 with real pretrained weights, despite `d_head = 72`

**Date:** 2026-08-28 · **Roadmap phase:** 1 · **Status:** adopted

**Decision.** The frozen workload is **DiT-XL/2 at 256×256**, `facebookresearch/DiT`,
checkpoint `DiT-XL-2-256x256.pt` from `dl.fbaipublicfiles.com`. Shape, read from
`models.py`: `depth=28, hidden_size=1152, patch_size=2, num_heads=16`, so
`d_head = 1152 / 16 = 72`, 256 tokens at 256×256 input.

**The tension.** FSA binds the head dimension to the systolic array's row count —
`main.py:234` calls the kernel with `d=cfg.sa_rows, br=cfg.sa_cols, bc=cfg.sa_rows`.
So `d_head` *is* `sa_rows`. And 72 sits badly:

- The RPU's own target is `d_head = 128` (`docs/GOLDEN_MODEL_SPEC.md` §2: H=40, d_h=128),
  and `Configs.fsa128x128` already exists at that geometry.
- 72 neither divides nor tiles 128. Padding 72 → 128 masks 44% of the array's rows.
- `defaultFSAParams(rows, cols, memPorts)` takes any `Int`, so an `sa_rows = 72` array
  is legal Chisel — but it is a bring-up-only shape whose utilisation and energy
  numbers do not transfer to a 128-row RPU.

**Alternatives rejected.**

- *Pad 72 → 128 and move on.* This is the trap. It masks 44% of the rows and would
  systematically distort exactly the utilisation and J/block figures phase 6 exists to
  measure. A padded run is quotable only if the padding is quoted with it, and
  "temporary" padding in an energy comparison has a way of surviving to the results
  table.
- *Pin DiT-B/2 instead* (`hidden_size=768, num_heads=12`, so `d_head = 64`, which is
  exactly half of 128 and tiles cleanly). Rejected because **no pretrained DiT-B/2
  checkpoint exists** — Meta released only XL/2 at 256 and 512. Pinning B/2 would mean
  pinning random weights, and phase 5's "one real DiT block" would quietly become "one
  DiT-shaped block". Choosing a worse *workload* to make a later *mapping* convenient
  optimises the wrong phase.

**Reason for the decision taken.** Phase 1's job is to pin model, checkpoint and input
and to dump deterministic intermediates. Its correctness criterion is determinism and
provenance, not array fit — the array does not appear anywhere in phase 1. The
`d_head = 72` problem is a **mapping** question that belongs to phases 2 and 4, and
deciding it now, by degrading the workload, would trade a real pretrained reference for
a convenience that phase 1 does not need.

**Expected effect.** Phase 1 is unaffected and can proceed immediately. Phase 2's GEMM
work is unaffected — general GEMM does not care about `d_head`. Phase 4 and phase 5
inherit an open mapping question, recorded below.

**Consequence to carry forward — do not let this go quiet.** Before phase 5 runs a real
block, decide explicitly how `d_head = 72` maps onto the array, and record it as its own
decision. The three live options are: an `sa_rows = 72` bring-up config; padding to 128
with the masked fraction quoted alongside every number derived from it; or tiling 72 as
64 + 8. **Any phase 6 measurement taken on a padded array must state the padding in the
same breath as the number.**

**Keep / revert.** Keep. Revisit only if a pretrained tile-aligned DiT checkpoint
appears, which would remove the tension at its source.

---

## D-105 — The functional golden is dumped on CPU in fp32, not on the GPU

**Date:** 2026-08-28 · **Roadmap phase:** 1 · **Status:** adopted

**Decision.** Phase 1's reference tensors are produced by a CPU fp32 forward pass with
`torch.use_deterministic_algorithms(True)`, TF32 disabled, the SDPA math backend forced,
and fixed seeds. The GPU is used only as a cross-check under stated tolerance, never as
the source of the dumped tensors.

**Reason.** The correctness criterion for this artifact is that it reproduces
bit-for-bit on any machine, not that it runs fast. A GPU dump makes the golden a
function of the device and driver:

- TF32 is on by default for matmul on Ampere and later, silently reducing fp32 matmuls
  to 10 significand bits — on a Blackwell card this would quietly change the reference.
- `F.scaled_dot_product_attention`, which timm's `Attention` calls, dispatches across
  flash / mem-efficient / math backends by shape and device. The chosen backend changes
  the reduction order and therefore the low bits.
- cuBLAS split-k and atomic reductions vary with device and library version.

Each is separately controllable, but the point of a golden is that a reader can
regenerate it without reproducing our exact GPU, driver and library stack. CPU fp32
removes the whole class of variables rather than mitigating it. The cost is negligible:
one DiT-XL/2 block over 256 tokens at `d = 1152` is a fraction of a second on CPU.

**Expected effect.** The dump is reproducible on any x86-64 machine with the pinned
torch version. The GPU stays useful for phase 7's full one-step DiT, where a
tolerance-bounded comparison is the right instrument anyway.

**Measured effect.** Recorded 2026-08-28. `check_determinism.sh` runs two independent
traces in separate processes: **39/39 tensors identical**. The full trace of one
DiT-XL/2 block takes **6.9 s** on CPU, so the cost argument for using the GPU never
arises. `torch 2.11.0+cu128`, numpy 2.4.6,
python 3.11.15; checkpoint sha256
`9ec1876e4c03471b...`.

Separately, and more important: the tracer's stage-by-stage recomputation from the
block's own weights reproduces the module's output **bit-exactly**
(`max|delta| = 0.0`). The decomposition the RTL will be verified against is therefore
the block, not an approximation of it.

**Keep / revert.** Keep. If a later phase needs a workload too large for CPU, that is a
new decision with its own determinism argument, not an amendment to this one.

---

## D-106 — RPU work lives on an `rpu-main` branch; `msaga-main` stays an untouched mirror

**Date:** 2026-08-28 · **Roadmap phase:** 0 · **Status:** adopted

**Decision.** The fork's default branch `msaga-main` is kept identical to upstream
`VCA-EPFL/chipyard-fsa@fa8665b7`. All RPU work happens on `rpu-main`, branched from
that commit.

**Reason.** Phase 0 is "reproduce Chipyard-FSA unchanged", and the cheapest way to keep
that claim checkable forever is to keep a branch that *is* unchanged, rather than
asserting it about a branch we have been editing. `git diff msaga-main rpu-main` then
answers "what is ours" exactly, and rebasing onto a future upstream bump stays a
mechanical operation.

**Expected effect.** None on the build. Gate A can be re-run from `msaga-main` at any
time to confirm the backbone still reproduces without our changes in the picture —
which is the whole value of the gate.

**How to apply.** Additive changes only where possible: `rpu/`, `workloads/`,
`.claude/`, `CLAUDE.md`. When an upstream file genuinely must change, record why here,
because that is the thing that makes the next upstream merge expensive.

---

## D-107 — The golden is committed as a checksum manifest; the tensors are regenerated

**Date:** 2026-08-28 · **Roadmap phase:** 1 · **Status:** adopted

**Decision.** `workloads/dit/manifests/*.json` — the pin, the checkpoint sha256, the
environment, and a sha256 per tensor — is committed. The `.npy` tensors themselves are
written to a gitignored build directory and regenerated on demand.

**Reason.** One traced DiT-XL/2 block is roughly 120 MB of fp32, dominated by the
weights (`w_adaln` 31.8 MB, `w_fc1` and `w_fc2` 21 MB each, `w_qkv` 15.9 MB). Putting
that in git history is permanent and it buys nothing that the manifest does not: the
tensors are a pure function of (checkpoint, pinned input, pinned code), and D-105
exists precisely to make that function deterministic. Git LFS would work but adds a
dependency to a fork whose upstream does not use it, for an artifact we can rebuild in
seconds.

The manifest is the frozen artifact and the checksums are the contract. This matches
how `docs/GOLDEN_MODEL_SPEC.md` §10 frames its corpus — hashes, with the vectors
regenerated — and it means a reader can verify our golden without trusting our bytes:
regenerate, compare sha256s.

**Expected effect.** `workloads/dit/check_determinism.sh` already asserts two
independent traces agree; committing the manifest extends that to "agrees with the run
that produced the committed manifest", across machines and across time.

**How to apply.** Regenerating and getting different checksums is a **failure**, not a
refresh. Investigate before updating the manifest, and if the change is legitimate
(a torch bump, a deliberate pin change), record it here with the reason — a manifest
that gets silently rewritten whenever it disagrees is not a contract.

**Keep / revert.** Keep. Revisit only if a downstream phase needs the exact bytes
available without a torch install, which would be an argument for LFS, not for git.
