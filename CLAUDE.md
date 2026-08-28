# chipyard-fsa — RPU fork

Fork of `VCA-EPFL/chipyard-fsa`, pinned at **`fa8665b7`**. This is the *only*
top-level hardware repository for the RPU program; `rpu/EXECUTION_ROADMAP.md` is the
plan of record and `rpu/README.md` gives the source-of-truth hierarchy.

Upstream remote is `upstream`; ours is `origin` (`Zarand3r/chipyard-fsa`).

## What is ours vs upstream

Ours: `rpu/`, `workloads/`, `.claude/`, this file. Everything else is upstream and
should stay mergeable — prefer adding a file over editing one, and when an upstream
file must change, record why in `rpu/DECISIONS.md`.

Accelerator changes belong in `generators/fsa/src/` and `generators/fsa/python/`.
A new array operation is a new `ExecutionPlan` subclass plus a `func` code in
`MatrixInstructionHeader` — the control logic is generated from the plan, so the
datapath usually does not need touching. See `rpu/GATE_B_FEASIBILITY.md`.

## Working agreements

- **The gates are real.** Gate A (backbone reproduces) blocks all RPU development;
  Gate B (general GEMM on the FSA array) blocks the backbone decision. Do not start
  phase 3 work before they close.
- **A skipped leg is not a passing gate.** The FPGA legs cannot run on this machine
  (`rpu/DECISIONS.md` D-102). Report them SKIP, and never summarise a two-way agreement
  as if it were the three-way one the gate asks for.
- **Label predicted vs measured.** Static source reading and analytical models are
  predictions; Verilator and FPGA numbers are measurements. Never quote one as the
  other.
- **Record deviations in `rpu/DECISIONS.md`** as they are made. A decision that exists
  only in a commit message is lost.
- **One source of truth per layer.** No second implementation of a row in the
  `rpu/README.md` table.

## Toolchain

```bash
source env.sh                     # chipyard: conda env, sbt, riscv tools, CIRCT
source ~/.local/bin/eda-env.sh    # oss-cad-suite: verilator 5.051, yosys, iverilog, verible
```

Conda is Miniforge, not Miniconda — see `rpu/DECISIONS.md` D-101. `uv` drives the FSA
Python API (`generators/fsa/python`).

No Vivado and no U55C board here, so `fpga/ make bitstream` and every FPGA run are
unavailable. Verilator, the FSA Python golden model, and an RTX PRO 6000 GPU for
workload tracing are available.

## Skills

Project-scoped, in `.claude/skills/`. Start from `chip-flow`.
