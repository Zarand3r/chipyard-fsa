# Spec provenance

Documents in this directory are copied from other repositories. They are **vendored
snapshots, not the source of truth** — the upstream copy is authoritative and will
drift. Re-copy rather than editing in place, and update the commit hash below.

| File | Source | Upstream commit | Copied |
|---|---|---|---|
| `GOLDEN_MODEL_SPEC.md` | `~/substrate/docs/GOLDEN_MODEL_SPEC.md` | `3ff44a9` (2026-08-27) | 2026-08-28 |
| `TARGET_MACHINE.md` | derived index over `~/substrate/docs/` (many files) | `3ff44a9` (2026-08-27) | 2026-08-28 |

These files reached this repository second-hand: copied into `rpu_simulation` on
2026-08-28, then into `rpu_simulation_2`, then here unchanged the same day. The
upstream in `~/substrate/docs/` is the authoritative copy for all of them. See
`../DECISIONS.md` D-103 for why this fork is now the only hardware repository.

`TARGET_MACHINE.md` is a *derived* summary, not a copy of any one upstream file — it
indexes the first-silicon claims to their authoritative source and records whether the
golden model can check each. Its numbers are convenience restatements and will drift;
the source column is the truth.

## Not copied (cited by GOLDEN_MODEL_SPEC but left upstream)

`GOLDEN_MODEL_SPEC.md` derives from and references these, which live in
`~/substrate/docs/` and were not pulled in:

`CHIP_SPEC.md` · `CHIP_LAYOUT.md` · `system-design.md` · `CROSS_MODEL_DESIGN.md` ·
`PERF_LEVERS.md` · `MEMORY_BANDWIDTH.md`

Pull them in only if this repo needs to resolve a reference on its own; each copy is
another thing that can go stale.

## Verifying the snapshot is current

```bash
diff rpu/docs/GOLDEN_MODEL_SPEC.md ~/substrate/docs/GOLDEN_MODEL_SPEC.md
```
