"""How much does option A (diagonal matmul) actually cost?

rpu/PHASE4_MODULATION.md quoted "99.2% of MACs wasted" for the diagonal-matmul mapping.
That number is real but it is a ratio *within the op*, and quoting it alone implies the
op dominates. It does not: an elementwise op is O(M*C) where a projection is O(M*C*C).
This computes the overhead against the whole block, which is the number that matters.

Key tiling point: diag(s) is block-diagonal, so for output n-tile j only k-tile j is
non-zero. A scheduler that skips the zero k-tiles pays `rows` times the useful work, not
`rows * C/rows`.
"""
from __future__ import annotations

def block_macs(M: int, C: int, C_ff: int, N_ctx: int, N_text: int) -> dict[str, int]:
    return {
        "qkv":        M * C * 3 * C,
        "attn_qk":    M * N_ctx * C,
        "attn_pv":    M * N_ctx * C,
        "out_proj":   M * C * C,
        "cross_attn": 2 * M * N_text * C + M * C * C,
        "ffn_fc1":    M * C * C_ff,
        "ffn_fc2":    M * C_ff * C,
    }


def report(name: str, M: int, C: int, C_ff: int, N_ctx: int, N_text: int,
           rows: int, n_elementwise: int) -> None:
    mm = block_macs(M, C, C_ff, N_ctx, N_text)
    total = sum(mm.values())
    useful = n_elementwise * M * C          # 2 modulations + 2 gates, per channel
    diag = useful * rows                    # option A: rows-deep contraction per element
    print(f"\n{name}  (M={M}, C={C}, C_ff={C_ff}, N_ctx={N_ctx}, array rows={rows})")
    print(f"  block matmul MACs          {total:>18,}")
    print(f"  elementwise useful MACs    {useful:>18,}   "
          f"({100*useful/total:.3f}% of the block)")
    print(f"  option A cost (x{rows})       {diag:>18,}   "
          f"(+{100*diag/total:.2f}% on top of the block)")
    print(f"  waste WITHIN the op        {100*(1-1/rows):.1f}%   "
          f"<- the number the doc quoted")


# DiT-XL/2, the phase-1 pinned bring-up workload: 4 elementwise ops per block
# (2 adaLN modulations + 2 gated residuals).
report("DiT-XL/2 bring-up", M=256, C=1152, C_ff=4608, N_ctx=256, N_text=0,
       rows=128, n_elementwise=4)
report("DiT-XL/2 on a 16-row array", M=256, C=1152, C_ff=4608, N_ctx=256, N_text=0,
       rows=16, n_elementwise=4)

# RPU contract shapes, GOLDEN_MODEL_SPEC §2. DiT blocks there carry cross-attention too,
# so 6 elementwise ops per block is the conservative count.
report("RPU contract (DreamZero-class)", M=3120, C=5120, C_ff=13824, N_ctx=18720,
       N_text=256, rows=128, n_elementwise=6)
