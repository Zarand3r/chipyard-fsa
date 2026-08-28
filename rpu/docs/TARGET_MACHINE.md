# Target machine — first-silicon specs (derived index)

**Not authoritative.** Every number here is defined in `~/substrate/docs/`; this file is
a convenience index snapshotted from `substrate@3ff44a9` (2026-08-27). When a value
matters, read the source column, not this table. Provenance tags are carried over
unchanged: `[T]` theoretical/target (pre-gate-1), `[S]` simulated/derived,
`[X]` external published result.

The column that earns this file is the last one: **what the golden model can actually
check.** `GOLDEN_MODEL_SPEC.md` specifies *values*; these bullets are mostly *physical*.
Confusing the two is how a tile-width change in `CHIP_SPEC` silently invalidates
`GOLDEN_MODEL_SPEC` §4.

## The ten

| # | Claim | Tag | Authoritative in | Golden-model coverage |
|---|---|---|---|---|
| 1 | 1.9 M FP4 MACs @ ~1.05 GHz (4 PF dense), 128×128 weight-streaming systolic tiles; `5120 = 8×640`, `13824 = 8×1728` divide the array exactly | [T] | `CHIP_SPEC.md`, `CHIP_LAYOUT.md`, `PERF_LEVERS.md` | **Partial** — tile geometry is §4/§5.2 (128×128 tiles, k-blocks of 128). MAC count, clock, PF are timing-domain, out of scope by §Scope |
| 2 | FP8 multiply into FP32 accumulators in dedicated SRAM beneath the array (TPUv1 narrow-multiply/wide-accumulate) | [T]/[X] | `CHIP_SPEC.md` | **Partial** — the arithmetic is §3/§4 and is bit-checkable. Physical placement is not modeled |
| 3 | 256-bit LPDDR5X @ 307 GB/s, ≥16 GB; prefetcher is a counter because the chip walks the same ~44 GB sequence every chunk | [T] | `CHIP_LAYOUT.md`, `MEMORY_BANDWIDTH.md`, `CHIP_ROADMAP.md` (44 GB) | **The *reason* is L2** — §6's conveyor invariant is exactly what makes a counter sufficient, and it is testable. The interface numbers are not |
| 4 | ~90 MB SRAM for stream buffers and activation spine — no resident weights | [T] | `CHIP_LAYOUT.md`, `CROSS_MODEL_DESIGN.md`, `PERF_LEVERS.md` | **Property yes, capacity no** — "no resident weights" follows from weight-streaming + §6. The 90 MB that *makes it true* is invisible (see Gap 2) |
| 5 | KV ring in DRAM, pointer-advanced at chunk end; both guidance branches share each fetched weight and context block | [T] | `CHIP_SPEC.md`, `system-design.md` | **Full** — §6 ring ("pointer arithmetic only"), §5.2 F2 rule, §8.5 commit. Both have must-fail mutants in §10 |
| 6 | Per-bank DRAM refresh in known idle windows; unmanaged costs 14–28 % of effective bandwidth | [S] | `PERF_LEVERS.md`, `WHITEPAPER.md` | **Excluded by definition** — §6: "Refresh, ECC: timing-domain; invisible to the golden model" |
| 7 | No OS, zero instruction fetch, no kernels; ~1,200 GPU kernel launches per chunk become one fixed schedule. TPUv1: 80 % of peak @ 7 ms p99 vs 37 % for K80 | [T]/[X] | `WHITEPAPER.md`, `PROGRAM.md`; TPUv1 = ISCA 2017 | **Mechanism yes** (`SCHEDULE_IMAGE`, §7). The comparatives are evidence, not spec |
| 8 | Schedule ROM with 1/2/3-step modes, generated offline by the calibrated simulator with a worst-case deadline certificate; weights/scales/counts/schedules load from an image; tile geometry and formats fixed in mask | [T] | `CHIP_SPEC.md`, `SIMULATORS.md` | **Modes and images yes** (§2 `S`, §7 `SCHEDULE_IMAGE`/`MODE_REG`). Certificate and offline generation are `sim/`'s output, correctly not here |
| 9 | ~2 % of datapath programmable for flow-ODE / CEM planning; JEPA costs 1.9 % of DreamZero-path perf/W, both stream ~7 GB, one memory tier serves both | [T]/[S] | `CROSS_MODEL_DESIGN.md`, `papers/MEMO_V2_NOTES.md` | **Block yes** — §5.6 "the programmable 2%", flow-ODE/CEM, DECIDE-10. The perf/W and GB figures are analytical results |
| 10 | No further conventional-architecture multiplier; grids of small cores, reconfigurable fabrics, exposed datapaths pay for flexibility in measured efficiency | [S] | `PERF_LEVERS.md`, `WHITEPAPER.md` | **Not a spec item** — design rationale |

Roughly: **1 fully covered, 7 partially, 2 not at all** — and for eight of those the
omission is correct scoping, not a defect.

## Three real gaps this comparison exposes

These are value-relevant and belong in `GOLDEN_MODEL_SPEC`, not here.

1. **Divisibility is assumed, never stated.** `5120/128 = 40` and `13824/128 = 108`,
   both exact, so padding is never needed at contract shapes. But the spec never says
   *"tensor dims always divide the tile; padding never occurs."* A future config that
   breaks divisibility needs padding plus masking, and masked lanes summed into an
   accumulator change values. Wants one normative sentence and a §10 mutant that pads
   without masking.

2. **90 MB is what makes the conveyor invariant true, and it is invisible.** §6 says
   SRAM residency shows up "at L2 only as the absence of re-reads within a tile," but
   the capacity guaranteeing that is nowhere in the spec. As written, a 9 MB
   implementation is still L1-conformant while emitting a completely different trace.
   Fix by stating the invariant as a hard rule — *no weight block is read twice within
   a step* — rather than by importing the capacity number.

3. **44 GB/chunk is a free cross-check going unused.** It is derivable from the L2
   trace the golden model already emits. If trace volume disagrees with
   `CHIP_ROADMAP`'s 44 GB, one of the two is wrong — and that catches address-map
   errors before RTL exists. Same trick §5.5 already uses with the FLOP-share checksum.

## Verifying this snapshot

```bash
grep -rn "1.05 GHz\|307\|90 MB" ~/substrate/docs/CHIP_SPEC.md ~/substrate/docs/CHIP_LAYOUT.md
git -C ~/substrate log -1 --format=%h   # compare against 3ff44a9
```
