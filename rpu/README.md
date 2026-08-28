# RPU

Everything in this directory is **ours**, not upstream Chipyard or FSA. It exists so
that RPU material has somewhere to live that will never collide with a `git pull
upstream`. Nothing here is on the Chipyard build path.

| Path | Holds |
|---|---|
| `EXECUTION_ROADMAP.md` | The plan of record: phases 0–12, the repository hierarchy, both go/no-go gates |
| `DECISIONS.md` | Every deviation from the roadmap, and every choice a reader would otherwise reverse-engineer |
| `GATE_B_FEASIBILITY.md` | Source-level prediction, written before the array ran, of whether FSA can take general GEMM |
| `docs/` | Vendored snapshots of the RPU spec — see `docs/PROVENANCE.md` |
| `golden/` | RPU numerical golden model (phase 3): exact FP4/FP8/accumulation/rounding |
| `scripts/` | The gates — `gate-a.sh` and, later, `gate-b.sh` |
| `../workloads/dit/` | Workload freeze + functional golden (phase 1); the PyTorch model repo stays outside the build |

## Source-of-truth hierarchy

Per the roadmap, one source of truth per layer. Do not create a second implementation
of any row.

| Layer | Lives in |
|---|---|
| Workload semantics | official PyTorch DiT (reference clone, outside this repo) |
| Workload functional golden | `workloads/dit/` (phase 1) — pinned checkpoint, pinned input, dumped intermediates |
| RPU numerical semantics | `rpu/golden/` (phase 3), specified by `rpu/docs/GOLDEN_MODEL_SPEC.md` |
| FSA hardware numerics | `generators/fsa/python/` + RPU extensions |
| Hardware implementation | `generators/fsa/src/` |
| RTL simulation | `sims/verilator/` |
| Physical FPGA | `fpga/` |
| ASIC implementation | `vlsi/` (Hammer) |
| Performance model | our simulator, consuming the same RPU schedule |
| RPU target spec | `rpu/docs/` here, snapshotted from `~/substrate/docs` |

## Excluded from the dependency graph

TACCEL, APEX, Brainsmith, Allo, FINN, FINN-T, Diff-DiT, HG-PIPE, LUT-LLM. Reading and
reference material only. No production dependency on any of them without a written
decision in `DECISIONS.md` saying why the Chipyard–FSA stack could not provide it.
