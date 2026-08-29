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

### A. Diagonal matmul

`X * s` becomes `X @ diag(s)`. Correct, needs **no RTL change**, works today.

Cost: the array performs a length-`rows` contraction where one term is wanted. At
`fsa128x128` that is **127/128 = 99.2% of the MACs wasted** on every modulation and
every gate. DiT-XL/2 has two modulations and two gates per block.

This is the trap D-104 and the sibling repo's D-005 both name: it inflates exactly the
utilisation and J/block figures phase 6 exists to measure. Usable as a bring-up crutch
**only if every derived number is quoted with the waste**, and a "temporary" crutch in an
energy comparison has a way of surviving into the results table.

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

**C for the RPU, A only for bring-up and only with the waste quoted.** B is rejected for
the streaming case but should be re-examined if conditioning turns out to be
chunk-static, because zero-cost is hard to beat.

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
