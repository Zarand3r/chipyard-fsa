# Phase 4: where do the elementwise ops run?

Roadmap phase 4 adds AdaLN/LayerNorm, GELU, and residual/modulation. Numerically these
are trivial and `rpu/golden/datapath.py` already implements them exactly. The whole
difficulty is architectural, and it is worth settling before writing RTL.

## The constraint

FSA's PE computes `reg * l_input + c`, and partial sums flow **down a column**, so a
column of `rows` PEs performs a length-`rows` contraction. That is a dot product, not an
elementwise product.

The natural mapping — "row 0 multiplies, rows 1..n-1 pass the result through" — is **not
expressible**. `ControlGen.FlowRange.update` loops `for row <- 0 until rows` on every
primitive, so `parallel`, `flow_up` and `flow_down` all drive *every* row. There is no
per-row control primitive.

So a per-channel elementwise multiply — which is what adaLN modulation
`x*(1+scale)+shift` and the gated residual `x + gate*branch` both need — has no direct
expression on this array.

## Three options, costed

### A. Diagonal matmul  — **recommended**

`X * s` becomes `X @ diag(s)`. Correct, needs **no RTL change**, works today.
`diag(s)` is block-diagonal, so for output n-tile *j* only k-tile *j* is non-zero; a
scheduler that skips the zero k-tiles pays `rows` times the useful work, not
`rows * C/rows`.

**Cost, measured rather than asserted** (`rpu/experiments/modulation_cost.py`):

| workload | array rows | elementwise share of block | option A overhead |
|---|---|---|---|
| DiT-XL/2 bring-up | 128 | 0.026% | **+3.31%** |
| DiT-XL/2 bring-up | 16 | 0.026% | +0.41% |
| RPU contract (§2) | 128 | 0.007% | **+0.84%** |

An earlier version of this document quoted "99.2% of MACs wasted" and stopped there.
That ratio is true *within the op* and it is misleading on its own: an elementwise op is
`O(M*C)` where a projection is `O(M*C*C)`, so at RPU contract shapes the whole thing
costs **under one percent** of the block. The waste is enormous and the op is tiny.

At `C = 5120` the overhead falls as `rows/C` grows less significant, which is why the
RPU's own shapes are cheaper than the bring-up model's.

### B. Fold the scale into adjacent weights

Exact, and costs **zero** array operations:

```
(x * s) @ W  ==  x @ (diag(s) @ W)      # input-side, row-scaling
g * (c @ W)  ==  c @ (W @ diag(g))      # output-side, column-scaling
```

Both identities are implemented and tested in `golden/datapath.py`
(`fold_scale_into_weights`). Every adaLN scale and gate in a DiT block sits adjacent to
a linear layer, so all four are foldable in principle.

**Why it does not work for the RPU as specified.** The scales are recomputed *every
step* from the conditioning vector, and the RPU streams 4-bit weights from DRAM
(`GOLDEN_MODEL_SPEC` §2: 14B params, 7.0 GB, weight-streaming, no resident weights).
Folding a per-step scale means rewriting the weight stream every step, which defeats
weight streaming entirely and moves the cost from compute to the memory system — the one
place the design has least headroom.

Folding **is** viable where the conditioning is constant across a chunk. Whether that
holds is a scheduling question, not an arithmetic one, and it is worth checking against
the mode definitions before dismissing B.

### C. Add a per-row control primitive

Add a row-restricted `FlowRange` to `ControlGen` and an elementwise execution plan that
drives `mac` on one row and `flow_ud` on the rest. This is the honest fix: the array
gains a genuine elementwise capability at the cost of a control change.

Scope: `ControlGen` (a row range on `FlowRange`, and `optimize()` must keep working —
it currently reconstructs flows assuming full-height ranges), one new `ExecutionPlan`,
one function code. The optimizer is the risk; `verify()` will catch a mistake loudly,
which is the good case.

## Recommendation

**A.** It needs no RTL change, no control-plane risk, and costs **+0.84% of block MACs
at RPU contract shapes**. Spending a `ControlGen` change and an optimizer rewrite to
recover under a percent is a bad trade, and option C's risk sits in `optimize()`, which
is the least testable part of the control path.

Revisit C only if a measurement — not an estimate — shows the overhead matters, or if
some later op needs genuine elementwise capability for a reason other than cost.

B stays rejected for the streaming case but should be re-examined if conditioning turns
out to be chunk-static, because zero-cost is hard to beat.

**The recommendation was reversed by doing the arithmetic.** It was C until the overhead
was computed against a whole block instead of against the op. Both numbers were correct;
only one of them answered the question.

None of this is blocked on `d_head = 72` (D-115): modulation, gates, GELU and LayerNorm
are per-channel and carry no head dimension.

## Two things to check before writing the RTL

1. **`exp2Done` (PARANOIA rule 8).** `PE.scala:55` is unreset, guarded by the comment
   "as long as exp2 is not the first operation". Measured 2026-08-29: the existing path
   is safe because `ATTN_SCORE` fires `mac` at cycle 1, which clears the register long
   before `exp2` at cycle `2*rows+4`, and attention results are identical across RTL
   seeds 1, 7 and 12345. **The constraint transfers to any new plan**: a plan that
   drives `exp2` must fire a non-`exp2` PE control first, in the same plan.
2. **LayerNorm needs a row reduction, and one of its terms has the same problem.** The
   mean is a contraction against a ones-vector, which the array does natively. The
   variance needs a sum of *squares*, i.e. an elementwise square first — option A, B or
   C again. Do not plan LayerNorm as "free because the array reduces".
