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

---

## D-108 — Gate A closed on its simulation legs; what "reproduces upstream" can honestly mean here

**Date:** 2026-08-28 · **Roadmap phase:** 0 · **Status:** adopted

**Result.** `rpu/scripts/gate-a.sh` at `FSA4X4Fp16Config`: 8 PASS, 1 SKIP.
`build-setup.sh --skip-ctags --skip-firesim --skip-marshal` completes clean, the
Verilator simulator builds in 53 s, and `main.py --seq_q 4 --seq_kv 4 --diff` runs to
`*** PASSED ***` in 4587 simulation cycles.

```
Error of FSA vs torch:         MAE 2.9919e-04  RMSE 3.7714e-04  MaxErr 7.5042e-04
                               RelErr 1.6433e-03  MaxRelErr 2.1891e-02
Error of PyEasyFloat vs torch: MAE 2.9919e-04  RMSE 3.7714e-04  MaxErr 7.5042e-04
                               RelErr 1.6433e-03  MaxRelErr 2.1891e-02
```

Performance counters: `execTime=353`, `mxActive=67`, `mxBubble=209`, `dmaActive=22`,
`mxInst=5`, `dmaInst=4`, `rawInst=32`.

**What the numbers actually establish.** The two rows are *identical to every printed
digit*. That is the result worth having: the RTL under Verilator agrees with the
PyEasyFloat software golden exactly, and both diverge from torch by the same amount,
which is the fp16 arithmetic difference and not a hardware bug. The RTL ↔ golden leg is
bit-exact; the golden ↔ torch leg is a tolerance comparison. Those are the two arrow
types the roadmap's verification chain distinguishes, visible in one run.

**What they do not establish, and a discrepancy not to paper over.** FSA's README
publishes `MAE 9.6587464e-05` for what looks like the same invocation. We get
`2.9919e-04`. The README also prints an `MSE` key where the pinned code prints `RMSE`,
so the README demonstrably predates the pinned commit and its figure is not a
reproduction target. **We therefore cannot claim to have reproduced upstream's published
number**, only that the flow runs and that RTL and golden agree exactly. That is enough
to pass Gate A as the roadmap words it ("reproduce upstream behaviour"), and it is not
enough to claim more.

**The FPGA leg is SKIP, so Gate A is not fully closed.** Per D-102, the gate asks for
three-way agreement and we have demonstrated two-way. The script prints
`gate is NOT fully closed`, and no summary of this work should round that up.

**Keep / revert.** Keep. Re-run from `msaga-main` at any time to confirm the backbone
still reproduces without our changes present (D-106).

---

## D-109 — Gate B needs a `GemmExecPlan`; settled by experiment, not by reading

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** adopted

**The question.** Could `C = A @ B` be had for free? `AttentionValueExecPlan` contains
no `load_reg_*` and no `update_reg`, so it never writes the stationary register — it
multiplies whatever `reg` already holds by the streamed operand. In the attention kernel
`reg` holds `P` only because `ATTN_SCORE` overwrote it. Issue `LOAD_STATIONARY(A)` then
`ATTN_VALUE(B)` with no score step between, and `reg` should still hold `A`. If so, Gate
B would collapse to Python tiling with zero RTL work.

**The experiment.** `rpu/experiments/gate_b_probe.py`, run against the real Verilator
RTL at `FSA4X4Fp16Config`. It issues exactly that two-instruction sequence on random
fp16 matrices and compares the drained accumulator with numpy.

**Result: no.** `max rel err 1.28`. The simulation completes cleanly
(`*** PASSED ***`, `execTime=210`, `mxInst=2`) and the outputs are plausibly-scaled
finite numbers — it computes *something*, just not `A @ B`.

To rule out the boring explanation, the probe then checks all eight operand orders and
transposes against both the raw accumulator tile and its transpose. **All sixteen
combinations land at rel err 1.13 to 1.53**; the closest, `A.T @ B` against `C_t.T`, is
still 1.126. So this is not a layout bug in the probe. The composite genuinely does not
compute a general matrix product.

**Leading hypothesis, not yet confirmed.** `AttentionValueExecPlan` asserts `acc_ui`,
so `macUnit.io.in_c := io.u_input.bits` — each PE accumulates the value arriving from
*above*. For the top row, `SystolicArray.scala` wires that input to the `CMP` unit's
`d_output`. The comparator array is stateful and is primed by `ATTN_SCORE`'s
`UPDATE` / `PROP_MAX` / `PROP_ZERO` command sequence. Run `ATTN_VALUE` without the score
step and the addend streaming in from the top is stale comparator state rather than
zero. That would explain finite-but-wrong output exactly. Confirming it means either a
waveform or a probe that drives the comparators to a known state first; neither is
needed to make the phase-2 decision.

**Consequence.** `GemmExecPlan` is required, as `GATE_B_FEASIBILITY.md` predicted after
its correction. Its shape is unchanged by this result: `AttentionScoreExecPlan`'s first
four declarations for the streaming multiply, `AttentionValueExecPlan`'s
`readAccRAM` + `ACC_SA` drain for the writeback, and a deliberate zero seed for the
accumulate input rather than whatever the comparators hold.

**Why this is recorded as a decision rather than a note.** It cost one experiment to
convert a contested source reading into a fact, and the contest was real — this file's
own feasibility note argued both sides before the probe settled it. The probe is kept
and is cheap to re-run; if a future change makes the two-instruction sequence work, it
will say so.

---

## D-110 — **D-109 is retracted.** The existing instructions do compute a general GEMM

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** supersedes D-109

**Retraction.** D-109 concluded, from an experiment on real RTL, that
`LOAD_STATIONARY` + `ATTN_VALUE` does not compute `A @ B` and that a `GemmExecPlan` was
therefore required. **That conclusion is wrong.** It was committed, and it was reported
as settled fact. Both instructions compute the product correctly:

```
func=2 (ATTN_VALUE)    max rel err vs A@B : 2.497088e-08
func=5 (GemmExecPlan)  max rel err vs A@B : 2.497088e-08
```

Identical to the last digit, on `[4x4] @ [4x4]` fp16 into fp32.

**What was actually wrong: the stationary operand layout.** The array computes

```
C = rev_both(S) @ B
```

where `S` is the stationary tile as loaded and `rev_both` reverses **both** rows and
columns. To get `A @ B`, load `S = A[::-1, ::-1]`. `gate_b_probe.py` instead copied
`A_tile.reverse(dim=0)` from the attention kernel — a single row reversal, which is the
convention `AttentionScoreExecPlan`'s upward flow needs, not this one.

**Why the sweep did not catch it.** D-109 leaned on a sixteen-way check of operand
orders and transposes and reported that none matched, treating that as proof the
mechanism was broken. The sweep permuted the **product**; the error was in the
**stationary operand**, and no transpose of `A @ B` equals `rev_both(A) @ B` for general
`A`. Sixteen negative results looked like strong evidence and were simply the wrong
sixteen.

**What found it.** Identity operands. `B = I` makes the output *be* the transform of
`A`, and it named `rev_both` in one run. `A = I` then confirmed the same model
predicts the other case. That diagnostic should have come before any conclusion was
recorded — it localizes, where pass/fail only judges.

**Also unsupported: the stale-comparator hypothesis.** D-109 proposed that `ATTN_VALUE`
without a preceding `ATTN_SCORE` accumulates stale `CMP` state through `acc_ui`. Since
`ATTN_VALUE` is now shown to be exactly correct in that situation, there is no such
effect to explain. Recorded as withdrawn, not merely unconfirmed.

**Consequence for Gate B.** Single-tile `C = A @ B` needs **no RTL change**. The
original feasibility claim — the one this file talked itself out of — was right.
`GemmExecPlan` and `RpuGemm*Config` are kept because they build clean, elaborate, and
produce identical results, and because dropping the online-softmax declarations is
plausibly cheaper in cycles; but **it must now justify itself on measurements**, not on
being necessary. If it cannot, it should be deleted rather than kept out of sympathy.

**Open, and not to be confused with the above.** Two-k-tile accumulation currently
fails for `func=2` and `func=5` *identically* (rel err ~9e36, garbage in the
accumulator). Because both fail the same way, this is a defect in the probe kernel's
double-buffering and semaphore handling, not a property of either plan. Gate B is not
closed until tiling works and the phase-1 shapes run.

**Process note, which is the durable part.** One negative experiment was treated as
decisive while an untested assumption — the operand layout — sat underneath it. The
lesson is not "run more experiments" but "when an experiment says a mechanism is
broken, first make it produce a known answer." An identity matrix costs one run and
distinguishes *wrong plumbing* from *wrong mechanism*; a sweep over plausible outputs
does not.

---

## D-111 — GEMM accumulation needs `scale = 1`; the accumulator multiplies by a stale register

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** adopted

**The bug.** Tiled GEMM passed every non-accumulating case at ~3e-8 and failed every
accumulating one, always by the same amount (rel ~9.13e36, and exact `0.0` in some
elements). Fences, `waitPrevAcc` (which maps to full serialization, `!io.busy`) and
double-buffer fixes changed nothing — the constancy was the clue: nothing was racing.

**The cause.** `Accumulator.scala` implements the accumulate command as

```
acc sa: out <- scale * sram_in + sa_in
```

where `scale` is `Seq.fill(cols) { Reg(accType) }` — **per-column, and never reset**. In
FlashAttention it carries the online rescale factor `exp(m_old - m_new)`, written by
`ATTN_SCORE` through `EXP_S1`/`EXP_S2` before every `ATTN_VALUE`. A plain GEMM has no
such factor and requires `scale = 1`, but nothing in a GEMM sequence ever writes it.

A single-tile GEMM cannot see this. The first k-tile sets `MatrixInstructionAcc.zero`,
which makes `sram_in` the ZERO constant, so `scale * 0` vanishes whatever `scale` holds.
The register only becomes visible from the second k-tile onward — which is exactly the
boundary the test results drew.

**Why upstream never hit it.** FSA's own `main.py --seq_q 4 --seq_kv 4` produces a
single K block, so it issues one `ATTN_VALUE` per output tile and never accumulates. At
`--seq_kv 8` it does accumulate, and passes — because `ATTN_SCORE` sets `scale` for it.
Gate A therefore passed without exercising the path at all. Worth remembering when
reading any conformance gate: passing says what ran, not what works.

**The fix.** `AccConstIdx` offers only `ZERO`, so there is no constant 1.0 to read.
Added `SetAccScale` (func 6) — `readAccRAM` + `SET_SCALE`, structurally
`AttentionLseNormScale` without the reciprocal.

The first attempt had the host DMA a row of 1.0 into accumulator SRAM. **That hangs the
simulator.** `DMAInstructionSRAM.isAccum` is declared in the instruction bundle and read
*nowhere* in the RTL — `grep -rn '\.isAccum' generators/fsa/src` returns nothing — so a
DMA targeting the accumulator never completes and its semaphore is never released. The
attention kernel only ever stores *out* of the accumulator, so upstream never exercises
that direction. This is also why the harness now passes a bounded `max_cycles` instead
of `0`: with unlimited cycles a deadlock hangs forever rather than failing.

The working approach computes the 1.0 row **on the array**. A GEMM with `A[:, 0] = 1`
and `B[0, :] = 1` produces exactly all-ones, every value exactly representable in fp16,
so the primed scale is 1.0 and not 1.0 ± an ulp. `SET_SCALE` then reads one row of it.
It writes into the same `c_acc` tile the real GEMM uses — `accRows = 1 + rows` does not
fit two `(rows, cols)` tiles — and the first real k-tile sets `zero=True` and overwrites
it.

**What this does and does not say about `GemmExecPlan`.** D-110 required the new plan to
justify itself on measurements rather than necessity. This does **not** rescue it: the
instruction that is genuinely required is `SetAccScale`, and `ATTN_VALUE` plus
`SetAccScale` may well be sufficient for the whole thing. `GemmExecPlan` still has to
earn its place on cycle counts, and the test harness runs both func codes so that
comparison is one flag away.

**Generalises beyond this bug.** Any RPU operation reusing the accumulator inherits an
unreset register whose meaning is attention-specific. Phase 8's weight-streaming and
FP4/FP8 work will touch the same path, and the same question — *what is `scale` when I
arrive?* — should be asked there rather than rediscovered.

---

## D-112 — `GemmExecPlan` is cycle-identical to `ATTN_VALUE` ~~and should be deleted~~

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** **REVERSED by D-113's root cause**

**What was measured.** `rpu/experiments/gemm_cycles.py`, `RpuGemm4X4Fp16Config`:

| shape | ATTN_VALUE | GemmExecPlan | delta |
|---|---|---|---|
| single tile | 397 | 397 | 0 |
| k x4 | 859 | 859 | 0 |
| m x n x k | 1607 | 1607 | 0 |

`mxActive`, `mxBubble` and `mxInst` identical too. On that basis this entry concluded the
plan earned nothing and should be deleted.

**Why that was wrong.** Both measurements were taken at **4x4**, the one array size where
`ATTN_VALUE` happens to be correct for a bare GEMM. At 8x8 and 16x16 `ATTN_VALUE`
corrupts whole drain steps and `GemmExecPlan` does not — see D-113. The plan's
`setComparator(0, rows, PROP_ZERO)` is the difference, and it is load-bearing.

Cycle-identity is therefore the *good* outcome, not evidence of uselessness:
`GemmExecPlan` buys correctness at every array size for **zero cycles**.

**Decision reversed.** Keep `GemmExecPlan` and `RpuMxFunc.GEMM`. It is the default
function code in `rpu_gemm.py`. `ATTN_VALUE` is kept selectable via `--func 2` precisely
because the contrast is the regression test for D-113.

**Lesson, and it is the same one as D-110.** A measurement taken at one configuration was
generalised to the design. Both times the error was not in the measurement but in the
scope silently attached to it. Benchmarks and correctness tests need their configuration
in the claim, not just in the log.

---

## D-113 — **RESOLVED.** `ATTN_VALUE` accumulates from an undriven, unreset comparator path

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** root-caused and fixed

**Symptom.** Tiled GEMM built on `ATTN_VALUE` was exact at 4x4 and corrupted whole drain
steps at 8x8 and 16x16 — entire rows of ~1e14, deterministic across seeds and runs, with
every surviving element arithmetically correct.

**Root cause.** `AttentionValueExecPlan` asserts `acc_ui`, so each PE takes its addend
from the neighbour above, and `SystolicArray` wires the top row's input to the `CMP`
unit's `d_output`. **`ATTN_VALUE` issues no comparator command**, so that path is
undriven, and every inter-PE pipeline register is built by
`pipe_no_reset = withReset(false.B){ Pipe(in) }` — **data and valid both power up
unreset**. The top row therefore accumulates power-on garbage, which lands in whichever
drain steps it happens to reach.

**Proof, three independent ways.**

1. *Issue `ATTN_SCORE` first and the drain is clean.* Zero garbage at 16x16, versus rows
   3 and 11 without it. `ATTN_SCORE` drives the comparators (`UPDATE`, `PROP_MAX`,
   `PROP_ZERO`) and flushes the array.
2. *The corrupted rows track the Verilator `$random` seed*, which is definitive for
   uninitialised state:

   | seed | garbage rows |
   |---|---|
   | `e23cbb39` (default) | 3, 11 |
   | 1 | 4, 7, 8, 14, 15 |
   | 7 | 1, 3, 4, 5, 13 |
   | 12345 | 1, 3, 5, 6, 7, 10, 11, 12, 13 |

3. *Priming with repeated `ATTN_VALUE` does **not** help*, which rules out a generic
   pipeline warm-up and points specifically at the comparator path that `ATTN_VALUE`
   never drives.

This also explains every earlier confusion: deterministic per build (fixed seed), no
clean scaling law (it depends on random values and pipeline depth), and 4x4 clean **by
luck of the seed** rather than by construction.

**The fix was already written.** `GemmExecPlan`'s `setComparator(0, rows, PROP_ZERO)` —
added in the first place to force a defined zero addend — is exactly the missing drive.
It was never tested above 4x4 until now.

| config | `ATTN_VALUE` (func 2) | `GemmExecPlan` (func 5) |
|---|---|---|
| 4x4 | clean | clean |
| 8x8 | 32/64 elements garbage | **0** |
| 16x16 | 32/256 elements garbage | **0** |

Full suite with func 5: **21/21 PASS across 4x4, 8x8 and 16x16**, rel err 2.5e-08 to
1.75e-07, scaling with contraction length as fp32 accumulation should.

**Consequences.**

- D-112 is reversed. `GemmExecPlan` earns its place: correctness at every array size for
  zero extra cycles. It is now the default in `rpu_gemm.py`.
- D-110's retraction was itself over-broad. "The existing instructions do compute a
  general GEMM" holds only at 4x4, and only because of the seed.
- This is a latent **upstream** hazard, not merely ours. Any future instruction that
  asserts `acc_ui` without driving the comparators inherits it, and on silicon the
  power-on state is arbitrary but fixed — it would present as persistently wrong results
  rather than as noise. Worth reporting to VCA-EPFL.
- `--func 2` is kept selectable so the contrast remains the regression test.

**Method note.** Five hypotheses were refuted before this one landed, and the probe that
cracked it was the cheapest: change one instruction in the sequence and see whether the
symptom moves. The seed sweep then converted a plausible story into proof. When a
symptom is deterministic but has no structural explanation, vary the *randomness source*
early — it separates "uninitialised" from "miscomputed" in one run.

---

## D-114 — Standing paranoia protocol, and an audit of unreset state

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** adopted

**Decision.** `rpu/PARANOIA.md` is the standing protocol, and its rules are enforced by
`rpu/scripts/gate-b.sh` rather than left to discipline: every gate runs **three array
sizes** and **three RTL random seeds**, and a firing Verilator assertion fails the case.

**Reason.** Four defects in one phase shared one shape — *silent, deterministic,
plausible wrong answers* that read as our own bug. Two of them (D-110, D-112) were
misdiagnosed as our error before the real cause surfaced. Rules that live only in a
report get skipped; rules wired into the gate do not.

**The audit.** `grep -n "= Reg(" --include=*.scala` over the FSA RTL found five unreset
registers and two comments openly stating a reset precondition. The one that matters
next:

```scala
// as long as exp2 is not the first operation, exp2Done does not need to be reset
val exp2Done = Reg(Bool())          // sa/PE.scala:55
```

That is **D-113's exact shape, unfired**: unreset state guarded by an undocumented
ordering assumption. Roadmap phase 4 adds GELU and softmax, which drive `exp2`. Check it
before debugging any wrong answer there.

The full table is in `PARANOIA.md` §7 and covers `pipe_no_reset` (D-113), `PE.reg`,
`Accumulator.scale` (D-111), and three unaudited DMA/decoder buffers.

**Measured effect.** Gate B at 16x16 across `$random` seeds 1, 7 and 12345 gives
**identical** rel err per case, which is the positive statement that the result does not
depend on power-on state. Before D-113's fix the same sweep moved the corrupted rows
every time.

**Cost.** 3 configs x 7 cases x 3 seeds = 63 simulator runs per gate invocation instead
of 7. Minutes, not hours, and cheap against the several hours the first misdiagnosis
cost.

**Keep / revert.** Keep. Drop a rule only by deleting the incident that justifies it,
which is not something that happens.

---

## D-115 — Real DiT GEMMs run on the array; three of seven shapes do not tile

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** adopted

**Result.** `rpu/experiments/dit_gemm_test.py` runs the phase-1 GEMM cases — operands
taken from the frozen DiT-XL/2 trace, not from a random generator — on the array. The
tileable cases pass at their **full contraction depth**:

| case | slice run | k-tiles | rel err | tol |
|---|---|---|---|---|
| `qkv_proj` | `[48x1152]@[1152x16]` | 72 | 4.07e-05 | 2.5e-03 |
| `attn_out_proj` | `[48x1152]@[1152x16]` | 72 | 1.03e-05 | 2.5e-03 |
| `mlp_fc1` | `[48x1152]@[1152x16]` | 72 | 6.12e-06 | 2.5e-03 |
| `mlp_fc2` | `[16x4096]@[4096x16]` | 256 | 1.18e-05 | 4.8e-03 |

`K` is the **complete** dimension of the real GEMM in the first three; only `M` and `N`
are sliced, because full size is 83k-332k stationary loads and that is not a Verilator
workload. Every reported number carries its slice (D-114 rule 1).

**Three shapes do not tile, and the reasons are structural, not harness limits.**

| case | shape | why |
|---|---|---|
| `attn_scores_h0` | `[256x72]@[72x256]` | `K = 72` is not a multiple of `rows = 16` |
| `attn_ctx_h0` | `[256x256]@[256x72]` | `N = 72` is not a multiple of `rows = 16` |
| `adaln` | `[1x1152]@[1152x6912]` | `M = 1` is not a multiple of `cols = 16` |

The first two are **D-104 arriving on schedule**. DiT-XL/2 has `d_head = 1152/16 = 72`,
FSA binds the head dimension to the array's row count, and 72 does not tile onto 16.
D-104 deferred this as a phase-2/4 mapping decision and flagged that it must not go
quiet; it has not.

**And the sweep sharpened it into an arithmetic fact.** At `RpuGemm8X8Fp16Config`,
**six of seven cases tile**, including both per-head attention GEMMs — because
`72 = 9 x 8`. `attn_scores_h0` runs `[224x72]@[72x8]`, 9 k-tiles, rel 1.44e-05;
`attn_ctx_h0` runs `[64x256]@[256x8]`, rel 8.93e-07.

`72 = 2^3 x 3^2`, so its divisors are {1,2,3,4,6,8,9,12,18,24,36,72}. **No power of two
above 8 divides it.** That is the whole difficulty in one line:

| array rows | `d_head = 72` |
|---|---|
| 4, 8 | tiles exactly |
| 16, 32, 64, **128** | does not tile |

The RPU targets `d_head = 128` on a 128x128 array (`GOLDEN_MODEL_SPEC` §2), so the
bring-up workload and the target machine disagree in a way no array size resolves for
both. The options are now concrete rather than vague: run bring-up attention on an
8-row array (exact, but a small array whose utilisation numbers do not transfer); pad
72 -> 128 and quote the masked fraction with every derived number; or tile 72 as 64 + 8.
The non-attention GEMMs are unaffected — they carry no `d_head` — so this constrains
phases 4 and 5, not phase 2.

**`adaln` fails on every array size** (`M = 1` vs `cols`), which is a scheduling problem
rather than a geometry one: batch the conditioning across the CFG pair, or accept a
`cols`-times-underutilised pass. The third is different in kind: `adaln`'s `M = 1` is a *batch* of one
conditioning vector, and a systolic array wants `M = cols` rows of work. That is a
scheduling question (batch the conditioning across the CFG pair, or accept a
`cols`-times-underutilised pass), not an arithmetic one.

**A defect in the test, caught and fixed before it flattered us.** The first version grew
the slice along M then N and always ended at `kt == 1`, so it never exercised
k-accumulation — the exact path D-111's stale accumulator scale lived on. A slice that
skips the interesting axis is not a test of that axis. It now grows K first and asserts
`kt > 1`.

**Seed independence.** Every case was run at `$random` seeds 1, 7 and 12345 (D-114
rule 2) at both 8x8 and 16x16. Rel err is **identical to every digit** across seeds, so
these results do not depend on power-on state.

**Tolerance caveat, stated rather than buried.** The reference is a float32 numpy
matmul, and the tolerance `max(1e-3, 3e-4 * sqrt(k/rows))` is a plausible envelope, not
a derived bound — it is loose by roughly two orders of magnitude against the observed
errors. The rigorous comparison is against **PyEasyFloat**, FSA's own bit-accurate model,
which is what upstream checks against and what makes the RTL ↔ golden leg bit-exact
rather than approximate. Wiring that in is the right next hardening step and is not yet
done.

---

## D-116 — The RTL ↔ golden leg is now bit-exact, and building the golden found a real modelling error

**Date:** 2026-08-28 · **Roadmap phase:** 2 · **Status:** adopted

**What changed.** Gate B compared the array against a float32 numpy matmul under a
tolerance `max(1e-3, 3e-4*sqrt(k/rows))`. D-115 flagged that as an envelope somebody
guessed, loose by two orders of magnitude, and noted the honest comparison was
PyEasyFloat — FSA's own bit-accurate model, the one `main.py --diff` checks attention
against. `rpu/experiments/gemm_golden.py` now provides it, and
`gate_b_test.py --bitexact` requires **exact equality**, not a tolerance.

This matters because the roadmap marks RTL ↔ golden as a `◄──►` arrow. A tolerance
comparison cannot support a bit-exactness claim, and until now Gate B was quietly making
the weaker one.

**Result: 42/42 at `rel 0.000e+00`** — 3 array sizes x 7 cases x 2 RTL seeds, zero
mismatches. The array reproduces the golden exactly at every configuration.

**Building it immediately caught a modelling error, which is the point.** The first
golden carried one continuous fma chain across all of K. Single-tile, m-tiling and
n-tiling matched bit-exactly; **every k-accumulating case missed by ~1 ulp**. The
hardware does not work that way:

- each k-tile contracts *inside the array from zero*, the partial sum flowing down a
  column through `rows` PEs, in reversed order (matching `fa_ref.py`'s `__mul_pv`);
- only then is the tile merged into accumulator SRAM by `ACC_SA`, which is
  `out = scale * sram_in + sa_in` — a single fused rounding, with `scale` primed to 1.0
  by `SetAccScale` (D-111).

Modelling it as per-tile contraction plus an `fma(1.0, acc, tile)` merge makes all seven
cases exact. Note which side was wrong: the RTL was right the whole time, and a
tolerance comparison would never have revealed the golden's error at all. **A tolerance
hides modelling mistakes in both directions.**

**Where the old numbers went.** The previously reported rel errors against numpy
(2.5e-08 to 1.75e-07) were **entirely reduction-order difference, not hardware error** —
the bit-exact golden itself differs from numpy fp32 by 5.44e-08 on the same shapes.

**Phase 3 groundwork, for free.** `gemm_golden.py` is a small, working instance of what
phase 3 must build at scale: exact arithmetic semantics — operand widths, reduction
order, rounding points, accumulator merge — expressed in software and checked against
RTL. `GOLDEN_MODEL_SPEC` §3 and §4 specify precisely these properties for the RPU.

**Kept, not replaced.** The tolerance path remains the default because the bit-exact
golden is a scalar FMA per MAC and too slow for large shapes. `--bitexact` is the
stronger claim; the tolerance path is the fast screen.

---

## D-117 — Phase 3 started: §4's reduction backbone, with open DECIDEs made unfakeable

**Date:** 2026-08-28 · **Roadmap phase:** 3 · **Status:** in progress

**What exists.** `rpu/golden/` now implements the part of `GOLDEN_MODEL_SPEC` the spec
itself calls "the bit-exactness backbone":

- `config.py` — one configuration object. Every §11 open decision (**DECIDE-1**
  activation FP8 flavour, **DECIDE-3** exact vs per-node-rounded tree, **DECIDE-4** tree
  width) has **no default**; constructing a config without naming it raises. The spec
  says open parameters are "marked DECIDE with its owning gate, never silently
  invented", and this makes that mechanical rather than aspirational. A separate
  `working_assumption()` returns the spec's own stated placeholders, labelled as such.
- `reduce.py` — §4 in full: k-blocks of 128, a fixed `tree_width`-input adder tree in
  ascending k evaluated exactly, tree outputs accumulated into FP32 in ascending order
  with one RNE add each, independent per-output accumulators. Plus `dot_linear_order`,
  the §10 must-fail mutant, kept deliberately *in the same file* as the thing it mutates.
- `test_reduce.py` — 15 checks, all passing.

**A real correctness problem found and fixed in the process.** The first `_rne_f32` took
the obvious shortcut, `np.float32(float(fraction))`, and asserted the exact value was
representable in float64 first. **That assert fires on ordinary random vectors**: an
8-product tree with a realistic exponent spread routinely needs more than 53 bits, and
the shortcut double-rounds. It is now a direct Fraction → float32 round-to-nearest-even
— binade search, quantum exponent with the denormal floor at 2⁻¹⁴⁹, ties-to-even,
overflow to infinity — verified against numpy on 500 float64-exact values, on 1/3, on
both exact ties at 2²⁴, on the smallest denormal, and on overflow.

§3 asks for "round-to-nearest-even at every format boundary". This is that boundary, and
approximating it would have quietly contaminated every downstream comparison.

**Two of my own tests were wrong, and the model was right both times.**

1. The tree test expected `2²⁴ + 7 = 16777223`. That is not representable in float32
   (the ulp at 2²⁴ is 2), so RNE gives 16777224 — which is what the model produced.
2. The accumulation-order test used random vectors, failed to distinguish forward from
   reversed order, and therefore asserted nothing. Replaced with a known-answer
   construction (PARANOIA rule 3): one group of 2²⁴ and fifteen groups of 1. Forward,
   each `2²⁴ + 1` is an exact tie and RNE swallows it → 2²⁴. Reversed, the ones
   accumulate to 15 first → 2²⁴ + 16. Order is now *demonstrated* load-bearing rather
   than asserted.

**Explicitly not done.** MXFP4/NVFP4 dequant and the FP8 formats (§3), the datapath
blocks (§5), memory semantics (§6), L2 trace and L3 state conformance (§1), and the
remaining §10 mutants. This is the reduction backbone and the configuration discipline
only.

---

## D-118 — §5 datapath blocks; DECIDE-5/6 are flags with spec-given defaults, not blockers

**Date:** 2026-08-28 · **Roadmap phase:** 3 · **Status:** adopted

**Correcting my own reading.** I reported §5.3 as blocked on DECIDE-5 (hardware `exp`)
and DECIDE-6 (probability precision). Re-reading it, the spec already settles the golden
model's behaviour: *"the golden default is correctly-rounded FP32 `exp`; the hardware
approximation (LUT/poly, ExpMul-class) must be specified to the bit at gate 1 and the
golden model then switches to it."* DECIDE-5 and DECIDE-6 are flags with stated
defaults, exactly like DECIDE-3/4 — not holes. §5.3 was never blocked.

Selecting `ExpImpl.HARDWARE_APPROX` **raises**, deliberately: inventing an
approximation to fill the gap would silently create the architectural commitment gate 1
is supposed to make.

**Implemented.** 5.1 dequant row, 5.2 matmul (FP8 operands, dequantized weights, FP32
per §4, FP8 re-quantization at the tensor boundary, with the raw accumulator exposed for
§9(c)), the 5.3 online-softmax streamer, 5.4 LayerNorm/GELU, and the 5.5 FLOP-share
checksum. All arithmetic routes through `reduce.dot`, so reduction order lives in one
place.

**Two real bugs, caught by tests derived from the spec's own words.**

1. *The online softmax rescaled the running sum but not the values already emitted.*
   Probabilities summed to **2.368** instead of 1. §5.3 makes the recurrence order part
   of the contract, so the sum-to-1 check is not decoration — it is the thing that
   detects a wrong recurrence. Fixed by rescaling prior tiles by `exp(m_old - m_new)`
   alongside the sum.
2. §10's mutant — *"running max updated in descending tile order"* — is implemented
   beside the real streamer and is confirmed detectable.

**Fourth test-expectation bug of the session; the model was right again.** I asserted
GELU is monotone. It is not: it dips below zero with a minimum near `x = -0.75`. The
test now checks the actual shape — minimum location, monotone above it, `→ x` for large
positive, `→ 0` for large negative. The running tally is worth stating plainly: four
wrong hand-written expectations, zero wrong model behaviours, and the two suites that
check against an **independent** reference (ml_dtypes for FP8, PyEasyFloat for the GEMM)
produced none of them.

**FLOP-share checksum.** §5.5 calls its shares "checksum for the model". Recomputing
them from §2's contract shapes gives agreement within 0.82 points (worst:
cross-attention 0.062 vs 0.070). The spec tags those shares `[S]` — simulated/derived —
so this is a consistency check on the model's geometry, not a bit-exact claim, and it is
recorded as such.

**Still open in phase 3:** §5.3's two µcode variants (static max-bound, FLASH-D), §5.6
update engine (blocked on DECIDE-10, the update-engine ISA doc, which does not exist),
§6 memory semantics, §7 named state, §8 chunk execution, and L2/L3 conformance.

---

## D-119 — Phase 4: elementwise ops have no direct expression on the array; three options costed

**Date:** 2026-08-29 · **Roadmap phase:** 4 · **Status:** analysis adopted, RTL option open

**Finding.** FSA's PE computes `reg * l_input + c` with partial sums flowing down a
column, so a column performs a length-`rows` *contraction*. Per-channel elementwise
multiply — needed by adaLN modulation and the gated residual, both twice per DiT block —
has no direct expression. The obvious mapping, "row 0 multiplies and the rest pass
through", is **not expressible**: `ControlGen.FlowRange.update` loops
`for row <- 0 until rows` on every primitive, so there is no per-row control.

**Options, in `rpu/PHASE4_MODULATION.md`.** (A) diagonal matmul — no RTL change, but
**99.2% of MACs wasted** at 128x128, and it inflates exactly the utilisation and J/block
numbers phase 6 exists to measure; (B) fold the scale into adjacent weights — exact and
free, implemented and tested as `fold_scale_into_weights`, but the scales are per-step
and the RPU streams 4-bit weights from DRAM, so folding rewrites the weight stream every
step and defeats weight streaming; (C) add a row-restricted `FlowRange` plus an
elementwise plan — a bounded control change, with `ControlGen.optimize()` as the risk and
its `verify()` as the safety net.

**Recommendation: A — and this reverses the recommendation this entry first carried.**
The "99.2% waste" figure is a ratio *within the op*, and quoting it alone implies the op
matters. Measured against a whole block
(`rpu/experiments/modulation_cost.py`), option A costs:

| workload | array rows | elementwise share of block | option A overhead |
|---|---|---|---|
| DiT-XL/2 bring-up | 128 | 0.026% | +3.31% |
| DiT-XL/2 bring-up | 16 | 0.026% | +0.41% |
| RPU contract (§2) | 128 | 0.007% | **+0.84%** |

Under one percent at the shapes the RPU is designed for. Spending a `ControlGen` change
plus an `optimize()` rewrite — the least testable part of the control path — to recover
0.84% is a bad trade. Take A; revisit C only if a *measurement* shows the overhead
matters, or if a later op needs genuine elementwise capability for a non-cost reason.

**Recorded because the error is instructive.** This is the same failure as D-110 and
D-112: a true number generalised past what it supports. There the scope was a
configuration; here it was a denominator. PARANOIA rule 1 now reads as covering both —
a ratio without its base is as unscoped as a benchmark without its config.

**Measured, not assumed: the `exp2Done` hazard does not fire today.** PARANOIA rule 8
flagged `PE.scala:55` as unreset with an undocumented ordering precondition, and phase 4
drives `exp2` through GELU and softmax. Checked: `ATTN_SCORE` fires `mac` at cycle 1,
which clears `exp2Done` long before `exp2` at cycle `2*rows+4`, and attention results are
**identical across RTL seeds 1, 7 and 12345**. So the precondition is self-satisfied by
that plan. **The constraint transfers**: any new plan driving `exp2` must fire a
non-`exp2` PE control first.

**A trap noted for LayerNorm.** Its mean is a contraction against a ones-vector, which
the array does natively — but its variance needs a sum of *squares*, so it hits the same
elementwise problem. LayerNorm is not "free because the array reduces".

**Golden side is done regardless of the RTL choice.** `modulate`, `gated_residual`,
`layernorm_fp32`, `gelu_tanh_fp32` and both folding identities are implemented and
tested, matching the pinned DiT-XL/2 forms exactly. Whichever mapping wins has something
exact to be verified against.

---

## D-120 — Phase 4 ops run on the array; LayerNorm turns out to be free

**Date:** 2026-08-29 · **Roadmap phase:** 4 · **Status:** adopted

**Measured on real RTL** (`rpu/experiments/phase4_modulate.py`, `RpuGemm16X16Fp16Config`,
checked against `rpu/golden/datapath.py`):

| op | mapping | result |
|---|---|---|
| `x * (1 + scale)` | `X @ diag(s)`, non-zero k-tile only | rel 3.10e-04 vs numpy |
| full adaLN `x*(1+s)+shift` | above, plus a vector add | rel 2.98e-04 vs the golden |
| row mean | `x · ones` | **exact**, rel 0.0 |
| row sum-of-squares | `x · x` | rel 9.34e-08 |
| variance | `sum_sq/C − mean²` | correct |

No RTL assertions fired. The rel ~3e-04 on the scaled paths is fp16 rounding of the
scale vector, not error in the mapping.

**A second correction to my own analysis, in the same direction as D-119.**
`PHASE4_MODULATION.md` claimed LayerNorm's variance "hits the same elementwise problem"
because it needs a sum of squares. **That is wrong.** `sum(x_i²)` is the dot product of
a row *with itself* — precisely what a systolic column computes. No elementwise square
is required, so LayerNorm pays **nothing**: mean and sum-of-squares are both native
contractions at full efficiency.

So the phase-4 cost picture is smaller than either version of this analysis first said.
Only the per-channel scalings — two modulations and two gates per block — pay option A's
`rows`× tax, and D-119 measured that at **+0.84%** of a block at RPU contract shapes.
LayerNorm, the op that looked most awkward, is free.

**Pattern, now three deep.** D-119 was a ratio quoted without its base; this was an
op assumed to need a capability it does not. Both were resolved by computing or running
the thing rather than reasoning about it further. The cheap move each time was the one
I skipped first.

**Remaining for phase 4:** GELU on the array. Unlike the others it is genuinely
elementwise-nonlinear, and FSA's `exp2` path in the PE is the natural vehicle — subject
to the `exp2Done` ordering constraint measured in D-119.

---

## D-121 — GELU: costed, bounded, and deliberately not decided

**Date:** 2026-08-29 · **Roadmap phase:** 4 · **Status:** study delivered; gate-1 decision left open

**Cost, first.** Applying D-119's lesson before choosing a mechanism: GELU is
`M x C_ff` elementwise operations, which is **0.0258% of block MACs at DiT-XL/2 and
0.0030% at RPU contract shapes**. Whatever mapping wins, GELU's compute is negligible.
The question is feasibility and precision, not throughput.

**Mechanism.** FSA already evaluates `exp2` as a piecewise-linear function *inside the
MAC* — `exp2PwlIntercepts` / `exp2PwlSlopes`, selected by the comparator's
`PROP_EXP2_INTERCEPTS` and latched by `PE.exp2Done`. tanh, and therefore tanh-GELU, is
reachable from that path plus the accumulator's `RECIPROCAL`. So the capability exists;
what does not exist is a specified piece count.

**Why I did not pick one.** `GOLDEN_MODEL_SPEC` DECIDE-5 requires the hardware
approximation be "specified to the bit at gate 1". Choosing a piece count here would be
precisely the silent architectural commitment `config.py` refuses everywhere else.
`datapath.py` already raises on `ExpImpl.HARDWARE_APPROX` for the same reason.

**What I produced instead: the study the decision should be made against.**
`rpu/golden/gelu_study.py`, measured on the **real activation distribution from the
phase-1 trace** (1,179,648 values from `mlp_fc1_out`, range [-10.528, 5.316],
p99.9 = 2.121) rather than a uniform grid:

| PWL pieces | max rel err | vs FP8 E4M3 half-ulp (3.125e-02) |
|---|---|---|
| 4 | 3.198e-02 | above |
| 8 | 2.888e-02 | below |
| 16 | 1.419e-02 | below |
| 32 | 4.363e-03 | below |
| 64 | 1.151e-03 | far below |
| 128 | 2.918e-04 | far below |

**The finding that bounds the choice.** §3 re-quantizes vector-op outputs to FP8 at the
tensor boundary, and E4M3 carries three explicit mantissa bits, so a half-ulp is
`2⁻⁵ = 3.1e-02` relative. **Piece counts past ~16 buy accuracy the FP8 boundary
immediately discards.** 8 pieces already sits under FP8 noise; 16 gives roughly 2x
margin. Anything beyond that is paying area and latency for bits that are rounded away
one operation later.

That bounds the decision to a narrow range without making it. The golden's own fp32
tanh-GELU is accurate to 2.365e-07 against a float64 reference, so it remains the
reference the hardware form will be checked against.

**Phase 4 status.** Modulation and the gated residual: mapped and measured (D-120).
LayerNorm: free, native contractions (D-120). GELU: costed, mechanism identified,
precision bounded, awaiting gate 1. The RTL for GELU is the one piece of phase 4 that
should not be written yet.

---

## D-122 — FP8 needs no datapath change; MXFP4 does. Phase 8 is smaller than it looks

**Date:** 2026-08-29 · **Roadmap phase:** 5/8 · **Status:** demonstrated

**Finding.** `FPArithmeticImpl(mulEW, mulMW, addEW, addMW)` is **width-generic**. Its
only constraints are `require(mulEW <= addEW && mulMW <= addMW)` and
`require(addEW - 1 >= log2Up(pwlPieces))`. The OCP FP8 formats therefore fall out as
parameters:

```scala
E4M3 = FPArithmeticImpl(4, 3, 8, 23)
E5M2 = FPArithmeticImpl(5, 2, 8, 23)
```

**Demonstrated, not inferred.** `RpuGemm16X16E4M3Config` elaborates, builds, and runs.
Its generated `FSAConfig.json` reports `"e_type": "e4m3", "a_type": "fp32"`, and a GEMM
on real E4M3 operands gives **max rel err 3.20e-03** against an fp32 reference, with the
leading values matching exactly. Roadmap phase 8 lists "FP8 attention" as if it were a
datapath addition. **It is a config line.**

**What is *not* free, and this is the part that matters.** §3 specifies weights as
**MXFP4** — E2M1 elements with a shared E8M0 scale per 32-element block, so that
"dequant = exponent add, no multiplier". FSA has **no block-scale mechanism at all**.
`FPArithmeticImpl(2, 1, 8, 23)` satisfies the width constraints and would give FP4
*elements*, but without a shared block exponent the dynamic range collapses — E2M1 alone
spans 0.5 to 6, which is useless for real weights. So:

| phase-8 item | actual cost |
|---|---|
| FP8 activations / attention | **a config parameter** — demonstrated working |
| FP4 elements | a config parameter (untested, constraint-legal) |
| **MXFP4 microscaling** | **a genuine datapath and weight-path addition** — block scales, per-block exponent add, and the scale plumbing to reach the PE |

The golden model already implements the MXFP4 semantics exactly
(`rpu/golden/formats.py`, validated against `ml_dtypes` on all 512 FP8 codes), so
whatever hardware is built has an exact reference waiting.

**Integration shims recorded.** FSA's Python side had never seen an FP8 config:
`config.py` evals the format name from the JSON but `dtype.py` defines only `fp8`, and
`from_numpy` calls `np.finfo` directly, which rejects `ml_dtypes`. Both are shimmed in
`rpu/experiments/rpu_gemm.py` rather than in the submodule (D-106). These are upstream
gaps, not bugs — nothing upstream generates an FP8 config.

**Why this belongs in phase 5's record.** Phase 5 runs a DiT block through
functional golden → numerical golden → RTL. The numerical golden targets FP4/FP8 while
the RTL was fp16-only, so the chain's bit-exact arrow had no common format. FP8 closes
half that gap today; MXFP4 remains the open half.

---

## D-123 — Phase 5: the verification chain runs end to end, with both arrow types honoured

**Date:** 2026-08-29 · **Roadmap phase:** 5 · **Status:** simulation legs closed

**Result.** `rpu/experiments/phase5_chain.py` runs one real DiT-XL/2 block through
`functional golden → numerical golden ↔ RTL`, stage by stage, on operands from the
phase-1 trace. `RpuGemm16X16Fp16Config`, slice of 16 tokens x 16 output channels at
**full K = 1152**:

| stage | vs functional golden (tolerance arrow) | vs numerical golden (exact arrow) |
|---|---|---|
| adaLN modulate | rel 1.342e-03 | — (vector op) |
| qkv accumulator (§9c) | rel 3.311e-04 | **BIT-EXACT** |
| qkv + bias | rel 4.097e-04 | — |
| mlp_fc1 accumulator (§9c) | rel 1.816e-04 | **BIT-EXACT** |
| layernorm mean | rel 4.788e-05 | native contraction |

**The point is the two columns, not the pass.** The roadmap draws `──►` from the
PyTorch golden and `◄──►` between the numerical golden and RTL, and this is the first
artifact that checks each with its own semantics: a *bound* on the first, *equality* on
the second. The fp16 quantization loss is reported rather than absorbed into a
tolerance, and the bit-exactness is asserted rather than approximated. Conflating them
is the failure D-116 caught, and this gate is built so it cannot recur.

**Honest scope.** Not the whole block. Attention is absent — DiT-XL/2's `d_head = 72`
does not tile on a 16-row array (D-115), and it needs the 8-row config. GELU is absent
by design (D-121, gate 1). Residual adds and the bias are vector operations, not array
work. The FPGA leg is SKIP, so **phase 5 as the roadmap words it is not closed** —
`gate-phase5.sh` says so in its own output rather than leaving the caveat to a summary.

**What it does establish.** Every linear stage of a real DiT block, on real checkpoint
operands, at full contraction depth, is bit-exact against a model of the hardware's own
arithmetic — and the divergence from fp32 PyTorch is quantization, quantified, at
1e-4 to 1e-3.

---

## D-124 — Attention joins the chain, bit-exact; FSA's *fused* attention cannot take this workload

**Date:** 2026-08-29 · **Roadmap phase:** 5 · **Status:** adopted

**Result.** Both attention GEMMs now run in the phase-5 chain at
`RpuGemm8X8Fp16Config`, 16 tokens x 16 channels, head 0 of the real DiT-XL/2 trace:

| stage | vs functional golden | vs numerical golden |
|---|---|---|
| attn scores `Q@K^T` | rel 3.962e-04 | **BIT-EXACT** |
| attn context `P@V` | rel 3.745e-04 | **BIT-EXACT** |

Seven of seven chain stages pass. Four are bit-exact against a model of the hardware's
own arithmetic.

**Why as GEMMs and not as FSA attention — a hard constraint, not a convenience.**
`VCA-EPFL/FSA` issue #5's maintainer answer states the mapping is `R == d == Bc`,
`C == Br`: the head dimension must *equal* the array's row count, with no tiling over
`d`. DiT-XL/2 has `d_head = 72`, so **FSA's fused FlashAttention would require a 72-row
array** — legal (`defaultFSAParams` takes any `Int`) but a bring-up-only geometry whose
utilisation figures transfer nowhere.

As separate GEMMs the contraction tiles freely, and `72 = 9 x 8` fits an 8-row array. So
the chain gets real attention today at the cost of not using the accelerator's defining
feature. That is the honest trade and it is worth stating plainly: **this workload does
not fit FSA's fused attention at any power-of-two array size** (D-115: 72's divisors
stop at 8).

Softmax runs in the golden model. FSA's fusion is an optimisation, not a correctness
requirement, and keeping it off-array isolates the GEMM claim from the `exp2` path.

**Consequence for the roadmap.** Phase 5's block coverage is now as complete as the
hardware allows: every linear stage bit-exact, attention included. What remains is
blocked or deferred rather than unattempted — GELU (D-121, gate 1), the FPGA leg
(D-102), and fused attention (needs either a 72-row array or a `d_head` the array
divides).

The `d_head = 72` decision flagged in D-104 has now cost something concrete twice: it
excludes attention from 16-row configs, and it excludes fused attention entirely. Worth
revisiting before phase 7 commits to a full one-step DiT.

---

## D-125 — Phase 7 uses a tile-aligned synthetic DiT, because it is testing a different thing than phase 5

**Date:** 2026-08-29 · **Roadmap phase:** 7 · **Status:** adopted

**The question D-124 forced.** Phase 7 runs "a complete small one-step DiT through the
same reusable array". DiT-XL/2's `d_head = 72` excludes fused attention at every
power-of-two array size, so a phase-7 built on the phase-1 workload would again run
attention as separate GEMMs and never exercise the accelerator's defining feature.

**Decision.** Phase 7 uses a **small synthetic DiT with `d_head == array rows`**, so
FSA's fused FlashAttention runs. Phase 5 keeps the real DiT-XL/2 checkpoint.

**Why that is not a dodge.** The two phases establish different properties, and using
one workload for both would weaken each:

| phase | question | needs |
|---|---|---|
| 5 | do the *values* match a real model? | real pretrained weights, real activations — **fidelity** |
| 7 | does the *whole pipeline* run on the array? | every op exercised, fused attention included — **completeness** |

Phase 5 already answered fidelity with the real checkpoint: seven stages, four bit-exact
against the hardware's own arithmetic (D-123, D-124). Repeating that at phase 7 adds
nothing, while a tile-aligned config buys the one thing phase 5 could not have —
attention running as attention rather than as two GEMMs.

**Explicitly labelled.** The phase-7 model is synthetic and randomly initialised. It is
a **pipeline test, not a fidelity claim**, and no number from it may be quoted as a
statement about DiT-XL/2 or about model quality. D-104 rejected random weights for the
phase-1 *workload freeze* for exactly the right reason — a golden model needs a real
checkpoint. That reasoning does not transfer here, because phase 7 is not producing a
golden.

**Configuration.** `d_head = rows`, `heads` free, `hidden = rows * heads`, so
`R == d == Bc` and `C == Br` hold (FSA issue #5). At `RpuGemm16X16Fp16Config` that is
`d_head = 16`.

**What this does not resolve.** The RPU's own target is `d_head = 128`
(`GOLDEN_MODEL_SPEC` §2), which *is* tile-aligned to a 128-row array. So the mismatch is
a property of the **bring-up workload**, not of the RPU — DiT-XL/2 is awkward, the RPU
is not. That reframing is worth carrying into phase 8: the architecture does not inherit
this problem.

---

## D-126 — Phase 7: a complete DiT block runs on the array, fused attention included

**Date:** 2026-08-29 · **Roadmap phase:** 7 · **Status:** simulation legs closed

**Result.** `rpu/experiments/phase7_dit.py` at `RpuGemm16X16Fp16Config`, synthetic
tile-aligned DiT (`d_head = 16 == rows`, 2 heads, hidden 32, 16 tokens):

| stage | on the array | vs float reference |
|---|---|---|
| adaLN modulation | diagonal matmul (option A) | rel 2.744e-04 |
| QKV projection | GEMM | rel 5.009e-05 |
| **attention** | **FSA fused: LOAD_STATIONARY / ATTN_SCORE / ATTN_VALUE / reciprocal / LSE-norm** | rel 4.866e-04 |
| output projection | GEMM | rel 1.049e-04 |
| gated residual | vector op | — |
| GELU | golden only (D-121, gate 1) | — |

**The result that matters is the attention row.** Every earlier phase ran attention as
two separate GEMMs, because DiT-XL/2's `d_head = 72` excludes fusion at any
power-of-two array size (D-124). This is the first time FSA's **fused FlashAttention** —
the accelerator's entire reason to exist — has carried a DiT-shaped workload end to end
in this program, and it did so at rel 4.87e-04 against a float64 reference.

**Reused, not reimplemented.** The attention call is upstream's own
`scaled_dot_product_attention` kernel from `generators/fsa/python/main.py`, driven with
our synthetic model's Q/K/V. Writing a second attention kernel would have tested our
kernel, not theirs.

**Scope, restated because it is easy to overclaim.** The model is synthetic and randomly
initialised (D-125). This is **pipeline completeness**, not fidelity. Phase 5 owns
fidelity, on the real DiT-XL/2 checkpoint, and its numbers are the ones that describe
the workload. `gate-phase7.sh` carries that caveat in its own header and output.

**Roadmap position.** Phases 0, 1, 2, 3, 4, 5 and 7 now have their simulation legs
closed. Phase 6 and phases 10-12 are hardware-blocked (D-102). Phase 8 is partially
done — FP8 is a config parameter and demonstrated (D-122); MXFP4 microscaling is the one
genuine datapath addition remaining. Phase 9's simulator ↔ RTL correlation is reachable;
its FPGA leg is not.

---

## D-127 — Phase 8 pre-RTL study: MXFP4 on real DiT weights

**Date:** 2026-08-29 · **Roadmap phase:** 8 · **Status:** study delivered; no decision made

**Context.** D-122 split phase 8: FP8 is a config parameter and demonstrated working;
**MXFP4 microscaling is the one genuine datapath addition** FSA lacks. Before building
it, measure it — on the real DiT-XL/2 weights from the phase-1 trace, not on synthetic
tensors. The insertion point is confirmed: between `spRAM.fullRead` and `inputDelayer`
in `FSA.scala`.

**Quantization error** (E2M1 elements, one E8M0 scale per 32, round-to-nearest):

| tensor | MXFP4 rms rel | FP8 E4M3 rms rel | fp16 rms rel |
|---|---|---|---|
| `w_qkv` | 0.1308 | 0.0280 | 0.000207 |
| `w_proj` | 0.1209 | 0.0268 | 0.000208 |
| `w_fc1` | 0.1247 | 0.0271 | 0.000207 |
| `w_fc2` | 0.1492 | 0.0277 | 0.000208 |

**Memory, which is the point.** 4 bits per element plus 8 bits of scale per 32 elements
= **4.25 bits/weight**. One block's weights: fp16 30.4 MiB → MXFP4 8.1 MiB, a **3.76x**
reduction. §2 puts 14B weights at 7.0 GB in 4-bit under weight streaming with no
resident weights, so DRAM traffic scales directly with bits/weight — at 16-bit the same
shapes would be 28 GB per step, which is not a bandwidth budget that closes. **This is
the lever the architecture rests on.**

**The finding worth carrying forward: MXFP4 error does not average out.** Measured
end-to-end on the qkv projection with real activations at K = 1152:

| weights | output rms rel |
|---|---|
| fp16 | 0.00017 |
| MXFP4 | 0.07826 |

Per-weight error 0.1336 → output error 0.0783 is only a **1.7x** reduction, where
independent noise over K = 1152 would give roughly √K ≈ 34x. MXFP4's error is
*correlated with weight magnitude* — relative error is roughly uniform across elements —
so it behaves like a systematic scaling rather than noise, and the contraction absorbs
almost none of it. Any error budget that assumed √K averaging is wrong by more than an
order of magnitude.

**Read as an upper bound, not a verdict.** These weights were trained in fp32 and
quantized round-to-nearest with **no calibration**. §3 specifies a weight image
"quantized offline", which in practice means calibration or quantization-aware training,
and §2 assumes 4-bit weights as a *premise* rather than something to be discovered. The
honest statement is: naive MXFP4 of an fp32 DiT checkpoint costs ~7.8% output error, and
the gap between that and an acceptable figure is the work a quantization pipeline does.

**Not a decision.** DECIDE-1 and DECIDE-2 belong to the pre-RTL numerics study's owner.
This is input to it, measured on real weights, together with the GELU piece-count study
(D-121).

---

## D-128 — Phase 9: a descriptor-driven cycle model; and the GEMM is latency-bound

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** simulator ↔ RTL leg partially closed

**Result.** `rpu/experiments/phase9_cycle_model.py` predicts `execTime` from the
instruction schedule, calibrated on **one** shape and validated on six held out
(`RpuGemm4X4Fp16Config`):

| shape | tiles | predicted | measured | error |
|---|---|---|---|---|
| single tile *(calibration)* | 1x1x1 | 220 | 221 | −0.5% |
| k x2 | 1x1x2 | 512 | 562 | −8.9% |
| k x4 | 1x1x4 | 808 | 870 | −7.1% |
| n x2 | 1x2x1 | 441 | 430 | +2.6% |
| m x2 | 2x1x1 | 441 | 430 | +2.6% |
| m x n | 2x2x1 | 882 | 848 | +4.0% |
| m x n x k | 2x2x2 | 1617 | 1651 | −2.1% |

**Against the roadmap's <5% target: mean 4.5% meets it, worst-case 8.9% does not.**
Both are reported because the roadmap does not say which it means, and quoting only the
favourable reading is the D-119 error.

**The model is derived, not fitted.** Every term but one comes from the execution plans
and the controller: `LoadStationary.setConflictFree(cols-1)`,
`GemmExecPlan.setConflictFree(2*rows-2)`, `accumulateMaxCycle = 2*rows+cols`, and the
`waitPrevAcc` serialisation that `MatrixEngineController` implements as
`canEnq = !io.busy`. The single calibrated constant is DMA latency — a property of the
TileLink/DRAM path, not of the array, and not derivable from the plans. Fitting all
seven shapes would have scored well and demonstrated nothing.

**The architectural finding is bigger than the model.** The counter decomposition says
this GEMM is **latency-bound**:

| shape | execTime | mxActive | mxBubble | dmaActive |
|---|---|---|---|---|
| single tile | 221 | 17 | 123 | 16 |
| k x4 | 870 | 87 | 712 | 64 |
| m x n x k | 1651 | 155 | 1445 | 124 |

`mxBubble` is **56-88%** of runtime while both `mxActive` and `dmaActive` stay small.
The array is not compute-limited and not bandwidth-limited — it is **waiting**, at
~66 cycles per DMA instruction. This matters well beyond phase 9: D-115 counted
83k-332k stationary loads for a full DiT GEMM, and at this issue rate that is the
binding cost, not the arithmetic. Any energy or utilisation figure taken from this
configuration would be measuring stalls.

**The residual has a named mechanism.** k-heavy shapes are under-predicted by 7-9%
because `waitPrevAcc` blocks the *next* tile's DMA from prefetching across the
serialisation point, so its latency is fully exposed rather than overlapped. Closing
that needs either a second calibrated constant — which starts being a fit — or a model
that tracks DMA/compute overlap explicitly. The honest position is that the mechanism is
identified and the model is not yet accurate enough to claim <5% worst-case.

**The FPGA correlation leg is SKIP** (D-102), so phase 9 as the roadmap words it —
simulator ↔ RTL ↔ FPGA — is not closed.

---

## D-129 — Two optimisations tested and refuted; the stall is scratchpad depth, and Gate B predicted it

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** measured; the fix is a capacity change

**Context.** D-128 found the tiled GEMM is latency-bound — `mxBubble` is 56-88% of
runtime at ~66 cycles per DMA — and hypothesised `waitPrevAcc` serialisation as the
cause. Two candidate fixes were tested.

**1. Remove `waitPrevAcc`. No effect.** It was set while chasing D-111, *before* the
real cause (the unreset accumulator scale) was found, and I noted at the time it did not
fix anything. It also does not cost anything:

| shape | with | without |
|---|---|---|
| k x4 | execTime 870, bubble 712 | 870, 712 |
| m x n x k | 1651, bubble 1445 | 1651, 1449 |

Correctness identical. **D-128's hypothesis is withdrawn.**

**2. Software-pipeline the k loop (prefetch distance 1). No effect.** Issuing the loads
for tile k+1 before the MX ops for tile k, into the second buffer: execTime 562 / 870 /
1651 and bubbles 428 / 712 / 1445 — **identical to the unpipelined kernel, to the
cycle**.

**Why, quantitatively.** The k loop is a serial dependency chain: every k-tile
accumulates into the same accumulator tile, so the MX engine cannot run ahead. Prefetch
distance 1 buys one iteration of overlap, which at ~12 cycles per k-iteration hides
about 12 of the ~66-cycle DMA latency. Hiding it fully needs a prefetch distance of
roughly `66/12 ≈ 6` iterations — that is **6+ buffers, not 2**.

And that does not fit. `Configs.defaultFSAParams` sizes the scratchpad as
`spadRows = 2*cols + 4*rows`, which is 24 rows at 4x4 — exactly six 4-row tiles, for
*both* operands together. The scratchpad cannot hold enough tiles in flight to cover its
own memory latency.

**This is Gate B's claim 5, arriving.** `GATE_B_FEASIBILITY.md` predicted before any of
this ran: *"the scratchpad and accumulator sizing is the part that actually costs
time"*, flagged as the claim most likely to be wrong in the expensive direction. It was
right, and for a reason the note did not anticipate: not capacity for correctness, but
**capacity for latency hiding**.

**Consequence.** The GEMM kernel is not the problem and no kernel-level scheduling fixes
it. The lever is `spadRows` — a `FSAParams` change, cheap to try — and the experiment is
well defined: raise the scratchpad allocation, deepen the prefetch, and watch `mxBubble`.
That is the next phase-9 step, and it is also the one that decides whether any
utilisation or J/block figure from this design is worth quoting (D-128).

**Kept, not reverted.** The pipelined kernel is neutral at distance 1 and is the correct
structure for a deeper prefetch, so it stays with this measurement recorded beside it.

---

## D-130 — Four stall hypotheses refuted; the evidence points at accumulator capacity

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** open, with a named experiment

**The measurement that will not move.** D-128 found `mxBubble` at 56-88% of runtime.
Four independent interventions, each targeting a plausible cause, changed **nothing**:

| # | intervention | result |
|---|---|---|
| 1 | remove `waitPrevAcc` (full serialisation) | execTime 870 → 870, bubble 712 → 712 |
| 2 | software-pipeline the k loop, prefetch distance 1 | 562/870/1651 → identical, to the cycle |
| 3 | **4x the scratchpad** (`spadRows` 24 → 96), prefetch depths 2, 3, 4, 6, 8 | 1486 at *every* depth, bubble 1289 at every depth |
| 4 | **deepen the DMA instruction queue** 2 → 16 (`rpu/patches/01`) | 1486 at every depth, unchanged |

Intervention 4 was worth doing on its own evidence: `dmaInst = Queue(decoder.io.outDMA,
pipe = true)` takes Chisel's **default of 2 entries** while `mxInst` gets
`mxInflight = 8`, and the decoder is a single in-order splitter, so a full DMA queue
stalls everything behind it. That asymmetry is real. It is simply not the bottleneck.

**What perfect insensitivity means.** A stall that ignores buffering, queue depth, issue
order and serialisation flags is not a *resource* stall. It is a **dependency** stall.

**The hypothesis the evidence now supports.** The k loop is a strict serial chain
through a single accumulator tile: each GEMM reads `c_acc`, accumulates, and writes it
back, so the next k-tile has a read-after-write dependency on the previous one. No
amount of prefetched *data* helps, because the *compute* cannot start early. The
arithmetic confirms the shape: `mxActive = 155` over 8 k-tiles is ~19 cycles of work per
tile against ~186 cycles elapsed — ~167 cycles per tile spent waiting on the accumulator
round trip.

`accRows = 1 + rows` holds exactly **one** output tile plus the log-exp-sum row. There is
no second accumulator to put independent work in.

**This is Gate B claim 5, third time and now precisely.** The feasibility note predicted
"the scratchpad and accumulator sizing is the part that actually costs time". D-129
attributed it to scratchpad depth and was wrong — intervention 3 refutes that directly.
The cost is in the **accumulator**, and it is not capacity for correctness or for
prefetch, but capacity for **independent work**.

**The experiment that would settle it.** Raise `accRows` to hold `A` output tiles, and
restructure the kernel to interleave `A` independent (m,n) output tiles through the k
loop, so tile *j*'s accumulate overlaps tile *j+1*'s compute. If `mxBubble` falls, the
dependency hypothesis is confirmed and the design lever is accumulator capacity. If it
does not, the stall is in the accumulator's own latency and no scheduling fixes it.

Until that runs, **no utilisation or J/block figure from this design should be quoted**
(D-128): at 87% bubble they would be measuring a dependency stall, not the machine.

**Patch handling.** `rpu/patches/` now exists with apply/revert scripts, so a submodule
change can be tested without the untracked-edit problem D-106 forbids. Patch 01 is
**kept as a file and reverted in the tree**, since carrying an applied change that
demonstrably buys nothing is exactly the kind of unjustified complexity that later reads
as intentional.

---

## D-131 — The stall explained: DMA cost is fixed per transfer, so it amortises with array size

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** root-caused; not an architectural flaw

**Fifth refutation, then the answer.** D-130's named experiment — raise `accRows` to
hold 4 output tiles and interleave independent (m,n) tiles through the k loop — was run
on `RpuGemm4X4DeepAccConfig` (`acc_size` 272 B = 17 rows = 1 + 4x4):

| mode | execTime | mxActive | mxBubble | bubble% |
|---|---|---|---|---|
| serial (today) | 2883 | 291 | 2581 | 89.5% |
| interleaved x2 | 2861 | 291 | 2561 | 89.5% |
| interleaved x4 | 2850 | 291 | 2550 | 89.5% |

**1.1%.** The accumulator-dependency hypothesis is refuted too — five for five.

**What `mxActive` being identical every time was telling me.** The work never changed
and the stall never changed, across every scheduling intervention. So the stall is not
schedulable at all. Holding the *tile count* fixed at 2x2x4 and varying only the array
size:

| config | tile bytes | execTime | mxActive | bubble% | **cycles/DMA** |
|---|---|---|---|---|---|
| 4x4 | 32 | 2883 | 291 | 89.3% | **75.9** |
| 8x8 | 128 | 2891 | 563 | 82.5% | **76.1** |
| 16x16 | 512 | 2910 | 1047 | 67.7% | **76.6** |

**Cycles per DMA is constant at ~76 across a 16x change in transfer size**, and total
runtime barely moves (2883 → 2910) while the useful work quadruples twice
(291 → 563 → 1047). The DMA cost is **fixed per transfer, independent of size**: pure
latency, serialised, and unaffected by anything the kernel does.

**Why the bubble falls anyway.** It is not that the stall shrinks — it is that the
compute grows into it. 89.3% → 82.5% → 67.7% is `mxActive` rising against a constant
DMA cost. Extrapolating the same trend, a 128x128 array moves 32 KB per tile and does
~64x the work of 16x16 per transfer, so the transfer stops being the limiter entirely.

**The conclusion that matters. This is not an architectural flaw; it is an artifact of
tiny tiles on a small array.** The RPU targets 128x128 (`GOLDEN_MODEL_SPEC` §4/§5.2),
where the same fixed latency is amortised over ~4000x more MACs per transfer than at
4x4. Five scheduling interventions failed because there was nothing to schedule around —
the fix is tile size, and the design already has it.

**Consequence for D-128's warning, which stands but narrows.** Utilisation and J/block
figures must not be quoted *from the small configs* — at 4x4 they measure a fixed
transfer latency, not the machine. Figures from 128x128 would not have this problem, and
the cycle model should carry `~76 cycles/transfer` as a measured constant rather than
treating it as a stall to be optimised away.

---

## D-132 — **OPEN DEFECT** at 32x32, and a correction to D-131's evidence

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** open

**Correction first.** D-131 reported a bubble trend of 89.3% → 82.5% → 67.7% → **40.7%**
across 4x4, 8x8, 16x16 and 32x32. **The 32x32 run produced wrong values** (rel err
4.34e-01) and I did not check correctness before quoting its cycle count. The trend
itself still stands on the three configurations that were verified correct — 89.3% →
82.5% → 67.7%, with cycles/DMA constant at 75.9 / 76.1 / 76.6 — and the 32x32 point
should be treated as indicative only until the defect below is fixed. Quoting a
performance number from a run I had not checked for correctness is the same class of
error as D-119 and D-129: a real measurement, reported past what it supports.

**The defect.** `RpuGemm32X32Fp16Config` fails on a **single tile**:

| RTL seed | rel err |
|---|---|
| default | `inf` |
| 1 | `nan` |
| 7 | 2.633e+36 |

Seed-dependent inf/NaN on one tile is D-113's signature: **uninitialised state reaching
the accumulator**. 4x4, 8x8 and 16x16 are all correct, so it is size-dependent.

**Ruled out.**

- *ISA address width.* `SPAD_MAX_ADDR_BITS = ACC_MAX_ADDR_BITS = 20`; 32x32 needs 8 bits
  for `spadRows = 192` and 6 for `accRows = 33`.
- *`PROP_ZERO` window too short.* `GemmExecPlan` drove the comparator for `rows` cycles
  while `mac.flow_down(1, rows)` consumes it until `2*rows - 1`. Extending it to
  `2*rows` is correct on its own terms and is **kept** — but it does not fix 32x32, and
  4x4 still passes at rel 2.497e-08 after the change, so it is not a regression either.

**The ladder was run. Results, and ownership.**

1. *`B = I` at 32x32:* 257 of 1024 elements garbage, **all 32 rows** affected but only
   **16 of 32 columns**, at `[0, 1, 4, 8, 12, 13, 16, 17, 18, 20, 21, 22, 23, 24, 28,
   29]`. Column-structured, which is the *opposite* of D-113's row/drain-step signature.
2. *Data dependence:* the bad column set is **identical** across data seeds. Structural,
   not numeric. 16x16 under the same probe is exact (rel 0.0).
3. *Capacity:* not the cause. At 32x32 the kernel uses 128 of 192 scratchpad rows and
   32 of 33 accumulator rows. ISA address fields are 20 bits against 8 and 6 needed.
4. **Ownership: ours.** `main.py --seq_q 32 --seq_kv 32` — upstream's own kernel and
   plans on this exact config — **passes**, with FSA matching PyEasyFloat to every
   printed digit. The 32x32 configuration is sound. The defect is in our GEMM path.
5. *Is `PROP_ZERO` simply an insufficient substitute for `ATTN_SCORE`'s priming?*
   Partly. Issuing a real `ATTN_SCORE` before `ATTN_VALUE` at 32x32 cuts the corruption
   from 467 to 118 elements of 1024 — **but does not eliminate it**, so priming is not
   the whole story either.

**Where that leaves it.** A defect that is ours, structural, column-indexed,
size-dependent, and survives both `PROP_ZERO` and real `ATTN_SCORE` priming. Every cheap
discriminator is now spent. The next step is the one this program has repeatedly avoided
and has now clearly earned: **a waveform**. `make debug CONFIG=RpuGemm32X32Fp16Config`
then `main.py --vcdfile`, per FSA issue #9, watching `accumulator.io.sa_in` and
`accRAM.fullWrite` for the bad columns during the drain window. That distinguishes
"garbage arrives from the array" from "the write lands wrong" in one look, which no
amount of black-box probing has managed.

**Not blocking.** 4x4, 8x8 and 16x16 are verified bit-exact and carry every result the
program depends on, including D-131's trend. 32x32 is a capability gap, not a
regression.

**Consequence.** Gate B's multi-config claim now covers 4x4, 8x8 and 16x16 only.
`gate-b.sh` should not be extended to 32x32 until this is resolved, and no number from
that configuration should be quoted meanwhile.

---

## D-133 — D-132 was two defects. One is fixed: `NaN * 0 = NaN` through an unreset per-column register

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** half resolved

**Found without the waveform, from the column-structure clue.** D-132's corruption was
*column*-indexed, and the design has exactly one piece of per-column state:

```scala
val scale = Seq.fill(cols) { Reg(accType) }        // Accumulator.scala:46, UNRESET
// acc sa: out <- scale * sram_in + sa_in
```

A k=0 tile sets `zero=True`, so `sram_in` is the ZERO constant and `scale * 0` is 0 —
**for any finite scale**. But `NaN * 0 = NaN`, and `Inf * 0 = NaN`. Any column whose
power-on `scale` holds a NaN or Inf pattern poisons its own output *even on a single
tile with a zeroed accumulator read*.

**And the guard was mine.** `rpu_gemm.py` primed the scale only `if s.kt > 1`. I added
that condition during the 16x16 investigation as an isolation step — "skip priming when
there is no accumulation, to see whether priming is the culprit" — and never removed it.
It looked safe precisely because `scale * 0 = 0` is true for every finite value, so it
survived 4x4, 8x8 and 16x16, where no column happened to power up NaN.

**Fixed.** Priming is now unconditional. The result:

| config | before | after | seed-dependent? |
|---|---|---|---|
| 32x32 | `inf` / `nan` / 2.633e+36 | **9.366e-01** | **no** — identical at both seeds |
| 16x16 | 1.586e-07 | 1.586e-07 | no |
| 4x4 | 2.497e-08 | 2.497e-08 | no |

Non-finite values are gone and the result is now **seed-independent**, which is the
signature that uninitialised state is no longer reaching the output. That half of D-132
is closed.

**The other half is still open.** 32x32 now returns a *deterministic, finite* rel err of
9.366e-01 while 16x16 and 4x4 stay exact. Deterministic and seed-independent means logic
or layout, not power-on state — a different defect that the NaN was masking.

**Leading hypothesis for the remainder, and a cleaner fix for both.** The priming
sequence is self-referential: it computes the all-ones tile with a GEMM whose own
`ACC_SA` multiplies by the very register it is trying to initialise. If any column's
scale is NaN when the priming GEMM runs, that column's "ones" is NaN, and `SET_SCALE`
then loads NaN back into scale. D-111 identified a route that avoids the dependency
entirely: `EXP_S1` sets `scale <- sa_in * attentionScale + 0` — **no dependence on the
old scale** — and `EXP_S2` then applies `pow2`, so with `sa_in = 0` the pair yields
`scale = 2^0 = 1` deterministically. Adding that pair to `GemmExecPlan` is a small,
derived change and is the next step.

**Method note.** The waveform D-132 called for was not needed. The column-structure
observation plus one grep for per-column state was enough, and it was available two
cycles earlier — the same "look at what the symptom's *shape* implicates" move that
cracked D-113 via `B = I`. Reaching for the heavy tool is not the same as reaching for
the right one.

---

## D-134 — **STUCK** on 32x32. The scale fix works; something else is size-dependent

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** stuck, isolated, not on the critical path

**What was built.** `SetAccScaleOne` (func 7) sets `scale <- 1.0` with **no dependence
on the register's previous value**, closing D-133's self-referential loop. It holds the
comparators at `PROP_ZERO`, walks the zeros down with `flow_ud`, and derives the scale
from them: `EXP_S1` gives `scale <- 0 * k + 0 = 0`, `EXP_S2` gives `scale <- 2^0 = 1`.
No operand, no scratchpad read, no accumulator read.

**It works — at 4x4.** All three shapes pass with the new plan and no regression:
single 2.497e-08, k x2 5.439e-08, m x n x k 5.700e-08. The mechanism is sound.

**32x32 still fails**: single `inf`, k x2 7.475e-01, m x n x k 6.908e-01.

**Declaring stuck, and why.** Three review cycles on one defect. The pattern is now
"same defect unresolved, repeated failed approach", which is the criterion. What has been
tried and eliminated:

| tried | result |
|---|---|
| ISA address widths | adequate (20 bits vs 8 and 6) |
| scratchpad / accumulator capacity | 128/192 and 32/33 rows, fits |
| `PROP_ZERO` window `rows` → `2*rows` | correct on its own terms, kept, no fix |
| unconditional scale priming | **fixed the NaN half** (D-133), residual remains |
| `SetAccScaleOne`, no back-dependence | works at 4x4, no effect at 32x32 |
| upstream `main.py` at 32x32 | **passes** — the config is sound, the defect is ours |

**Root-cause reasoning about the remainder.** Everything that fixed a 32x32 symptom also
changed 4x4, and everything that works at 4x4 fails at 32x32, so the fault scales with
`rows`/`cols` rather than with any capacity or initialisation state. The plans' cycle
arithmetic is the prime suspect: `SetAccScaleOne` asserts `EXP_S1` at `rows + cols`,
which is where the *first* de-skewed column arrives, but `flow_ud.flow_down(1, rows)` has
`effEnd = 2*rows`, so at 32x32 the zero wave is still in flight when the scale is
sampled, where at 4x4 it has essentially landed. Several plans share this shape
(`readAccRAM(rows + cols - 1, rows)`, `setAccumulator(rows + cols, rows)`), and a
half-cycle assumption that holds when `rows` is small need not hold when it is 32.

**The unblocking action, and this time it really is the waveform.** D-132 named it, D-133
found a cheaper win instead, and that seam is now exhausted. Concretely:

```
make -C sims/verilator debug CONFIG=RpuGemm32X32Fp16Config
cd generators/fsa/python && uv run main.py --config RpuGemm32X32Fp16Config \
    --vcdfile /tmp/32x32.vcd    # per FSA issue #9
```

then, on a single-tile GEMM, inspect across cycles `rows+cols-4 .. rows+cols+4`:
`accumulator.io.sa_in`, `accumulator.io.ctrl_in.cmd`, the `scale` registers, and
`accRAM.fullWrite.valid/data`. The one question to answer: **at the cycle `EXP_S1`
fires, is `sa_in` actually zero?** If not, the plan's timing constant is wrong and the
fix is arithmetic on the plan, not more experiments.

**Not on the critical path.** 4x4, 8x8 and 16x16 are verified bit-exact and carry every
result this program has claimed — Gate B, phases 5 and 7, D-131's amortisation trend.
32x32 is a capability gap. It should not block phase 8's MXFP4 work, which is the larger
remaining item.

---

## D-135 — PARANOIA rule 5 caught a stale-binary claim one cycle after it was written

**Date:** 2026-08-29 · **Roadmap phase:** 9 · **Status:** resolved

**What the rule caught.** D-134 added rule 5 — *test the artifact you just built* — after
I twice waited on a log line instead of a timestamp. The very next review applied it and
found something worse than the original incident:

| simulator | built | vs the func-7 change (~08:30) |
|---|---|---|
| `RpuGemm8X8Fp16Config` | Aug 28 19:02 | **stale** |
| `RpuGemm16X16Fp16Config` | Aug 28 18:42 | **stale** |
| `RpuGemm16X16E4M3Config` | Aug 29 01:35 | **stale** |

`rpu_gemm.py` had been switched to emit `SET_ACC_SCALE_ONE` (func 7, D-134). Those three
binaries do not implement it, so the instruction would have been a no-op and the scale
never primed. **Gate B's multi-config claim was resting on simulators that predated the
kernel they were supposedly testing** — and nothing in the last report said so, because
nothing checked.

**Rebuilt and re-verified.** All three rebuilt (09:04, 09:05, 09:06), then Gate B re-run
across 4x4, 8x8 and 16x16: **21/21 passed, zero failures**. `SetAccScaleOne` is correct
at every size that works, and the switch caused no regression.

**Why this is worth its own entry.** The failure mode is not "a test broke" — it is
"a test kept passing while measuring the wrong thing", which is the same shape as
D-113's seed dependence, D-115's `kt == 1` slice, and D-118's random vector that
distinguished nothing. Four instances now, all of the form *the check ran and proved
less than it appeared to*. Rule 5 turned a silent one into a caught one within a single
cycle, which is the strongest argument yet for writing the protocol down rather than
carrying it in working memory.

**Standing consequence.** Any claim about a configuration must be made against a binary
newer than the last kernel or plan change. `stat -c %Y` on the simulator, compared to the
edit time — not a log line, not a memory of having rebuilt.

---

## D-136 — FP8 runs on real DiT operands; and the tolerance is hiding something

**Date:** 2026-08-29 · **Roadmap phase:** 8 · **Status:** delivered, with an open question

**Result.** `RpuGemm16X16E4M3Config`, phase-1 DiT-XL/2 operands cast to E4M3, run on the
array at **full contraction depth**:

| case | slice | k-tiles | rel err | tolerance |
|---|---|---|---|---|
| `qkv_proj` | `[48x1152]@[1152x16]` | 72 | 8.794e-02 | 5.1e-01 |
| `attn_out_proj` | `[48x1152]@[1152x16]` | 72 | 3.241e-02 | 5.1e-01 |
| `mlp_fc1` | `[48x1152]@[1152x16]` | 72 | 1.412e-02 | 5.1e-01 |
| `mlp_fc2` | `[16x4096]@[4096x16]` | 256 | 1.056e-02 | 9.6e-01 |

That is roadmap phase 8's "FP8 attention, FP4 linear compute" half-delivered: **FP8
works on real workload operands, with no datapath change** (D-122), only a config line
and Python shims.

**The part I am not going to let pass.** The reference is
`As.astype(float32) @ Bs.astype(float32)` where `As`/`Bs` are **already E4M3-quantized**.
So input quantization is *already accounted for*, and both sides then accumulate in
fp32. On that comparison the residual should be reduction-order noise — the ~1e-7 seen
in the fp16 path — not **8.8e-02**.

D-122 measured a single 16x16 E4M3 tile at rel 3.2e-03. At K = 1152 (72 k-tiles) it is
8.8e-02, a factor of ~27 where `sqrt(72) ≈ 8.5` would be the independent-noise
expectation. Either E4M3 rounding error compounds non-independently through the
accumulator — plausible, and the same correlated-error effect D-127 found for MXFP4
weights — **or there is a defect in the FP8 path that the tolerance is masking**.

**And the tolerance is far too loose to tell.** I set `6e-2 * sqrt(k/rows)`, giving
5.1e-01 and 9.6e-01 — wide enough to pass almost anything. That is exactly the failure
D-116 named: *a tolerance hides modelling errors in both directions*. These cases are
marked PASS by a guard rail, not by a claim.

**Next step, concretely.** Parameterise `rpu/golden/gemm_golden.py` on the multiply
format — it currently hardcodes `MUL_EW, MUL_MW = 5, 10` — and run the E4M3 path against
a **bit-exact** reference the way D-116 did for fp16. That answers the question in one
run: if the array matches exactly, the 8.8e-02 is genuine E4M3 arithmetic and the number
is a real datapoint for the numerics study; if it does not, there is an FP8 defect and
the loose tolerance was concealing it.

Until that runs, **the FP8 error figures above should not be quoted as E4M3's accuracy**
— only as "the array produced these, and they have not yet been checked against exact
E4M3 arithmetic".

---

## D-137 — The FP8 discrepancy is subnormal handling, and it violates §3

**Date:** 2026-08-29 · **Roadmap phase:** 8 · **Status:** root-caused; a real spec conflict

**D-136's open question, answered.** The array's E4M3 output differed from a *bit-exact*
E4M3 reference by the same amount it differed from numpy (3.198e-03), and the reference
and numpy agreed with each other — so the array was not doing exact E4M3 arithmetic, and
the loose tolerance had been hiding it.

**Isolated in one run.** Same config, same shapes, only the operand *range* varied:

| operands | E4M3 subnormals present | vs bit-exact golden |
|---|---|---|
| `normal(0, 1)` | 4 | differs, rel 3.198e-03 |
| `uniform [0.5, 4]` | 0 | **BIT-EXACT** |
| `uniform [1, 2]` | 0 | **BIT-EXACT** |

The discrepancy is **entirely subnormal operands**. With none present the FP8 path is
bit-exact, which also confirms there is no other defect in it.

**Why fp16 never showed this.** E4M3 has bias 7, so its smallest normal is `2^-6 =
0.0156` and its subnormals occupy `0.002 … 0.0137` — a range `normal(0,1)` populates
constantly. fp16's smallest normal is `6.1e-5`, which random data essentially never
reaches. The fp16 path is bit-exact (D-116) because it never exercises the case.

**And upstream documents it.** FSA's README: *"FSA uses hardware floating-point
arithmetic from EasyFloat, which **simplifies subnormal handling** compared to
HardFloat."* This is a known, deliberate simplification — not a bug in FSA. It only
becomes material when the element format is narrow enough for subnormals to be common.

**The conflict, which is the point for phase 8.** `GOLDEN_MODEL_SPEC` §3 requires:

> Rounding: round-to-nearest-even at every format boundary; saturating casts …;
> **FP8/FP4 denormals supported as per OCP MX spec**

FSA's arithmetic does not meet that for narrow formats. So phase 8's "FP8 attention,
FP4 linear compute" is **not** simply a config parameter after all — D-122's finding
holds for the datapath *width*, but the RPU's denormal requirement needs arithmetic FSA
does not implement. MXFP4 is worse: E2M1's only subnormal is 0.5, which is 1 of its 8
magnitudes, so a large fraction of every weight block lands there.

**Consequences.**

- D-122 is narrowed, not retracted: FP8 works as a config parameter, and is bit-exact
  **on normal-range operands**. The claim now carries that qualifier.
- D-136's DiT figures (rel 1.06e-02 … 8.79e-02) are explained: real activations and
  weights cast to E4M3 contain many subnormals. They are **not** E4M3's intrinsic
  accuracy; they are EasyFloat's subnormal behaviour plus quantization, and should not
  be quoted as either alone.
- Phase 8's real cost list gains an item: **OCP-conformant subnormal handling in the
  MAC**, alongside MXFP4 block scaling (D-127).
- `rpu/golden/formats.py` already implements OCP denormals exactly and is validated
  against `ml_dtypes` on all 512 codes, so the reference for whatever gets built exists.

**Method note.** This is the third time a loose tolerance concealed a real effect
(D-116, D-136, here) and the second time the fix was to demand equality and then vary
one input property until the difference moved. Ranges are a diagnostic axis, like seeds
and configurations.

---

## D-138 — L2 and L3 conformance implemented; all four §10 mutants now exist

**Date:** 2026-08-29 · **Roadmap phase:** 3 · **Status:** delivered

**The gap this closes.** §1 defines three conformance levels. `reduce.py` and
`datapath.py` served **L1** only; **L2** (the memory access trace) and **L3** (state
inspectable between chunks) had no implementation at all, and were the largest unblocked
item left in phase 3.

**`rpu/golden/state.py`** implements §7's named objects — `WEIGHT_IMAGE`, `KV_RING[L]`
with pointers, `TEXT_KV[L]`, `DIFFUSION_LATENT`, `SCHEDULE_IMAGE`, `MODE_REG`,
`ACTION_BUFFER` — plus §6's address map, conveyor trace and KV ring. 17 checks, all
passing; `check-golden.sh` now runs 87.

**Two places the spec dictates the *implementation*, not just the behaviour, and why.**

- §6: *"No copies, no remapping — pointer arithmetic only, and the golden model must
  implement it as such so wraparound addressing is exercised."* A ring that shifts data
  returns identical values and is therefore invisible to any value-level test.
- §6 F2: *"one read per weight block per step regardless of branch count."* Double-
  fetching for the CFG pair changes nothing about the values — only the trace.

Both are cases where the golden model's *construction* is the thing under test, which is
exactly why §10 lists their negations as mutants.

**All four §10 mutants are now implemented and confirmed detectable:**

| mutant | detector | status |
|---|---|---|
| linear-order accumulation instead of the tree | §4 order vectors | D-117 |
| running max in descending tile order | softmax sum-to-1 | D-118 |
| double-fetched CFG weights | **L2 trace only** | here |
| ring as memcpy | **wraparound addressing only** | here |

The two added here are the ones no value comparison can catch, which is the point of
having L2 and L3 at all. Each mutant lives in the same file as the thing it mutates, so
it cannot rot into a test of nothing.

**Kept honest about open decisions.** `weight_trace` orders blocks layer-major then
QKV / attn-out / FFN-in / FFN-out / cross-attn, and says in its docstring that
**DECIDE-11** (freeze the intra-layer order) is the parameter this will fix. `BRANCHES`
deliberately has no entry for Deadline mode, because **DECIDE-9** (guidance on/off there)
is open — a test asserts that absence, so a later default cannot be added silently.

**Phase 3 remaining:** §8 chunk execution (the superloop), §5.6 update engine (blocked on
DECIDE-10, whose ISA document does not exist), and the §5.3 softmax variants.

---

## D-139 — §8 chunk execution: the golden model's last unblocked section

**Date:** 2026-08-29 · **Roadmap phase:** 3 · **Status:** delivered

**`rpu/golden/chunk.py`** implements §8's superloop in the normative order: ingest and
encode, fresh-token K/V against the ring window, `S(mode)` DiT steps with an F2-shared
weight stream, action head, commit. It owns *ordering and trace*; `datapath.py` supplies
values and `state.py` supplies the L2/L3 objects. 17 checks, all passing.
`check-golden.sh` now runs **108**.

**Chunk purity is enforced structurally, not asserted.** §7 says a chunk is a pure
function of `(state, fresh tokens, mode)` and that *"that property is itself a
conformance test"*. `run_chunk` copies the state and returns a new one rather than
mutating, so the test can run the same chunk twice from the same state and compare —
which it does, on both the resulting state and the emitted trace. A version that mutated
in place would make the property untestable rather than false, which is the worse
failure.

**F2 is checked where it can actually be violated.** For Quality mode: 36 weight reads
for 3 steps x 12 blocks, with `branches = 2`. The branch loop must not appear in that
count, and the test asserts `steps x blocks` rather than `steps x blocks x branches` —
so a future implementation that fetches per branch fails here as well as against the
D-138 mutant.

**Three open decisions raise instead of guessing**, extending `config.py`'s discipline
into the superloop:

| decision | what raises |
|---|---|
| DECIDE-12 (VAE encoder scope) | `encode_tokens` — §8.1's working assumption is upstream, so `run_chunk` takes encoded tokens |
| DECIDE-10 (update engine ISA) | `update_engine` — the document does not exist |
| DECIDE-9 (Deadline guidance) | `run_chunk` in Deadline mode — the branch count is unknown |

Each is a place where a plausible default would have been easy and would have quietly
become an architectural commitment.

**Phase 3 status.** L1 value conformance, L2 trace and L3 state are implemented for the
sections that are not blocked. What remains is blocked, not unattempted: §5.6 (DECIDE-10),
§5.3's two µcode softmax variants, and the DECIDE-1/2/3/5/6 numerics choices that gate
full L1. The golden model is now as complete as the open decisions permit.
