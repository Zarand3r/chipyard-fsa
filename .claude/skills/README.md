# Project skills

Skills scoped to this repository. Claude Code loads them automatically when working
here; they are not installed user-wide, so they travel with a clone and do not leak
into other projects.

`chip-flow` is the entry point — it names the six stages, the gate between them, and
routes to the rest.

## Authored for this project

| Skill | Stage |
|---|---|
| `chip-flow` | Router for the whole flow |
| `execution-model-from-spec` | 1 — spec to sweepable perf/resource model |
| `golden-model-first` | 2 — bit-accurate reference and frozen corpus |
| `accelerator-scheduling` | 3 — tiling, loop order, banking, instruction stream |
| `hls-dataflow-tapa` | 4 — Vitis HLS / TAPA dataflow kernels |
| `sv-verification-stack` | 5 — Verible/slang, Verilator, cocotb, SymbiYosys, Yosys |

## Vendored from claude-for-hardware

The following nine are third-party, copied unmodified from
[Midstall/claude-for-hardware](https://github.com/Midstall/claude-for-hardware)
at commit `a4c4a00` (2026-08-01), licensed **Apache-2.0**. The upstream license text
is in `LICENSE.claude-for-hardware`; this notice satisfies its attribution
requirement. They are not covered by this repository's own license.

`differential-verification`, `silicon-grade-discipline`, `rtl-area-timing`,
`fpga-synthesis-fit`, `fpga-bringup`, `hdl-module-design`, `soc-integration`,
`codegen-validation`, `tapeout-precheck`

**Caveat:** each carries a "Midstall House Style" section referencing that
organization's internal projects (Heimdall, Aegis, River). Those paragraphs are
someone else's conventions, not this project's — ignore them. Upstream's ROHD/Dart,
Nix, and firmware-boot skills were deliberately not taken.

To update, re-copy from upstream rather than editing in place, and re-check this
notice against the new commit.
