# RPU FPGA Program — execution roadmap

The plan of record. Revised 2026-08-28 to name the golden functional model as its own
phase: it is a foundational verification artifact, not a step inside workload bring-up.
Deviations go in `DECISIONS.md`, not here.

## The verification chain

Two chains, kept separate on purpose. Values first:

```
   PyTorch functional golden  ──►  RPU numerical golden  ◄──►  RTL  ◄──►  FPGA
        (phase 1)                      (phase 3)            (phase 4+)   (phase 5+)
```

Then, independently, time:

```
   static schedule  ──►  cycle model  ◄──►  RTL / FPGA timing
                          (phase 9)
```

**The arrows differ, and conflating them is the failure mode this structure exists to
prevent.** `──►` is a *derivation under stated tolerance*: the PyTorch golden is fp32
and defines what is computed; the RPU numerical golden narrows it to exact FP4/FP8
arithmetic and is expected to diverge from PyTorch within a bound that is derived, not
tuned. `◄──►` is *bit-exact agreement*: numerical golden, RTL and FPGA must produce
identical bits at every observation point, and any difference is a bug in one of them.

Never quote a value-chain result as a timing-chain result, or a tolerance comparison as
a bit-exactness claim.

## Phases

| # | Phase | Chain position |
|---|---|---|
| 0 | Reproduce Chipyard-FSA unchanged: Verilator + U55C FPGA + existing PyTorch reference | backbone |
| 1 | **Freeze the DiT workload and golden functional model** — pin model/checkpoint/input, dump deterministic intermediate tensors from one real transformer block | functional golden |
| 2 | Prove general GEMM: add tiled GEMM mode to the FSA systolic array, validate against the golden reference | RTL ↔ golden |
| 3 | **Build the RPU numerical golden model** — encode the exact FP4/FP8/accumulation/rounding behaviour RTL must reproduce | numerical golden |
| 4 | Add missing transformer ops: AdaLN/LayerNorm, GELU, residual/modulation | RTL |
| 5 | Run one complete DiT block: functional golden → numerical golden → Verilator → FPGA | full chain |
| 6 | Measure FPGA: cycles, utilization, HBM traffic, stalls, power, J/block | measurement |
| 7 | Run a complete small one-step DiT | full chain |
| 8 | Evolve into the first-silicon RPU architecture | architecture |
| 9 | Build and correlate the cycle model: simulator ↔ RTL ↔ FPGA to <5% cycle error | timing chain |
| 10 | Benchmark the identical workload on Jetson Thor | measurement |
| 11 | ASIC synthesis/P&R + workload-driven power | ASIC |
| 12 | ASIC-vs-Thor tapeout decision | decision |

### Why phases 1 and 3 are separate

They answer different questions and can fail independently.

Phase 1 asks *what does this model compute* — it is a semantic reference, produced by
running a real pretrained checkpoint on pinned inputs and dumping intermediates. It has
no opinion about number formats. Its correctness criterion is determinism and
provenance: same checkpoint, same input, same tensors, every time.

Phase 3 asks *what bits must the hardware produce* — reduction order, rounding mode,
requantisation points, accumulator width, the `exp` approximation. Its correctness
criterion is that RTL reproduces it exactly. `rpu/docs/GOLDEN_MODEL_SPEC.md` is the
specification for this artifact: §1 conformance levels (L1 value / L2 trace / L3 state),
§3 number formats and rounding, §4 deterministic reduction, §9 observation points, §10
test plan. Its §11 lists thirteen open `DECIDE` items — those are phase 3's blocking
list, and six of them (FP8 flavour, requantisation points, adder-tree rounding,
adder-tree width, hardware `exp`, probability precision) are exactly this phase's
content.

Building phase 3 without phase 1 gives you bits with nothing to check their meaning
against. Building phase 1 without phase 3 gives you meaning with no bit-exactness
target. Neither substitutes for the other, and neither is a step inside the other.

## Repository hierarchy

**1. Main repository — this fork.** `VCA-EPFL/chipyard-fsa` is the only top-level
hardware repository. All RPU development happens inside it. Do not create a separate
project that later tries to combine Chipyard, FSA and FPGA infrastructure.

```
chipyard-fsa/
├── generators/fsa/     # FSA accelerator source -- the thing that becomes the RPU
├── sims/verilator/     # RTL simulation
├── fpga/               # U55C FPGA flow
├── vlsi/               # Chipyard/Hammer ASIC flow
├── rpu/                # ours: roadmap, decisions, spec snapshots
│   └── golden/         # ours: RPU numerical golden model (phase 3)
└── workloads/dit/      # ours: DiT exporter + functional golden (phase 1)
```

Setup, run before any modification:

```bash
./build-setup.sh --skip-ctags --skip-firesim --skip-marshal
source env.sh
```

Pin the exact upstream commit first. **Pinned: `fa8665b7`.**

**2. Main accelerator source — modify this.** `generators/fsa/` (from `VCA-EPFL/FSA`)
provides the systolic array, execution plan, DMA, memory interfaces, floating-point
datapath, attention, performance counters, and a Python API/golden model. The top-level
integration is `AXI4FSA.scala`; operation and control behaviour is generated from
`ExecutionPlan.scala`. Early RPU changes belong under `generators/fsa/src/` and
`generators/fsa/python/`. Do not fork FSA into an independent build system.

**3. Chipyard — inherited infrastructure.** Use it for Chisel elaboration, Verilator,
AXI/memory integration, U55C integration, Vivado bitstream generation and the later
Hammer VLSI flow. Do not replace these with Allo, FINN, Brainsmith, TACCEL
infrastructure, custom FPGA shells or custom simulation infrastructure unless the
Chipyard path provably cannot meet a specific requirement. For the first prototype,
prefer FSA's Direct AXI4 integration (`AXI4FSAMxNConfig`), which connects the
accelerator straight to backing DRAM/HBM without a larger processor/TileLink path.

**4. Gemmini — reference donor only.** Consult `ucb-bar/gemmini` for tiled GEMM,
weight- and output-stationary execution, scratchpad organisation, accumulator SRAM, DMA
scheduling, tiling, I-GELU, LayerNorm and Softmax. Port small ideas into FSA. Do not
instantiate both accelerators, and do not import the Gemmini runtime unless the
arbitrary-GEMM feasibility gate fails. **Decision gate:** if clean general GEMM support
in FSA is straightforward, FSA becomes the RPU array and the search stops. If FSA is
structurally too specialised, stop and evaluate Gemmini as the compute generator with
FSA's attention ideas ported into it. Do not maintain both paths.

**5. DiT/PixArt — workload only.** An official PyTorch diffusion-transformer
implementation, used to load the pretrained model, run it, trace one transformer block,
and dump weights, activations and golden intermediates. The model repo must not become
part of the FPGA build system. The RPU-owned exporter lives in `workloads/dit/` and
converts PyTorch tensors into the formats `generators/fsa/python/` consumes:

```
PyTorch DiT → RPU exporter → static execution plan / descriptors → FSA-derived RPU
```

## Explicitly excluded from the dependency graph

TACCEL, APEX, Brainsmith, Allo, FINN, FINN-T, Diff-DiT, HG-PIPE, LUT-LLM — reading and
reference material only (Diff-DiT for low-bit DiT datapath ideas, TACCEL for
scheduling/compiler ideas, LUT-LLM for V80/HBM/power methodology, APEX for verification
ideas). No production dependency without a documented reason why Chipyard-FSA cannot
provide the functionality.

## Gate A — prove the backbone (phase 0)

From a clean checkout, FSA Python reference ↔ FSA Verilator ↔ FSA U55C FPGA must
reproduce upstream behaviour. The upstream flow builds a U55C bitstream from `fpga/`
and executes from `generators/fsa/python/`, comparing FPGA output against Torch.
**If this does not reproduce, stop before RPU development.**

## Gate B — prove FSA can become the RPU (phase 2)

Add `C = AB` as a standalone general tiled-GEMM mode on the existing FSA systolic
array. Run PyTorch GEMM ↔ Verilator ↔ U55C FPGA, including a real transformer-shaped
matrix, validated against the phase-1 golden reference. If it works cleanly, freeze the
backbone choice and stop searching for other accelerator repositories. If it needs a
fundamental rewrite of the array/control architecture, stop and evaluate Gemmini as the
base array with FSA attention mechanisms on top, inside the same Chipyard ecosystem.
That is the only planned fallback.

**Phase 1 is not gated behind Gate B.** It needs a checkpoint, a GPU and determinism —
none of which depend on the array question. It should run in parallel with, or ahead
of, the GEMM work.
