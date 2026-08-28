# Gate B pre-read — can FSA become the RPU array?

**Status:** source analysis only. Written before the array was built or simulated, so
every claim here is a *prediction* to be checked against Gate B's actual run, not a
result. Read against FSA @ the revision pinned by `chipyard-fsa@fa8665b7`.

Gate B asks: can `C = AB` be added as a general tiled-GEMM mode on the existing FSA
systolic array, or does it require a fundamental rewrite of the array/control
architecture? The four files that answer it are `sa/PE.scala`, `sa/SystolicArray.scala`,
`ExecutionPlan.scala`, and `Configs.scala`.

## Prediction: yes, and the array is not the hard part

### The PE is already a weight-stationary MAC

`PE.scala` holds one stationary `reg` of `elemType`, loadable from the left
(`load_reg_li`) or from above (`load_reg_ui`), and computes
`reg * l_input + c` where `c` is selected from the up or down neighbour. `PECtrl`
carries exactly nine control bits, and general GEMM needs only four of them —
`load_reg_li`, `flow_lr`, `mac`, `acc_ui`. All four already exist and are already
exercised.

The FlashAttention-specific machinery is `exp2` (the PE computes `2^x` in place using
its own MAC) and the `CMP` row along the top (running row-max for online softmax).
GEMM simply does not assert them. Nothing has to be removed.

### An existing execution plan is already a GEMM

`AttentionValueExecPlan` computes `O = P @ V`:

```
readScratchPad(0, rows, None)     // stream V from the scratchpad
mac.flow_down(1, rows)            // multiply against the stationary tile
acc_ui.flow_down(1, rows)         // accumulate downward
flow_lr.flow_down(1, rows)
readAccRAM(rows + cols - 1, rows, None)
setAccumulator(rows + cols, rows, AccumulatorCmd.ACC_SA)
```

That is `C += A·B` with `A` stationary — structurally the inner step of a tiled GEMM
already. Preceded by `LoadStationary`, it is the whole k-tile loop body.

### Accumulate-vs-seed across k-tiles already has an encoding

`MatrixInstructionAcc` carries a `zero` bit, and `AccReadDesc.toHardware` maps it to
`is_constant := ... || rs2.zero` with `const_idx = AccConstIdx.ZERO`. So "first k-tile
seeds the accumulator, later k-tiles accumulate" is expressible in the ISA as it
stands. `MatrixInstructionSpad` and `MatrixInstructionAcc` both carry signed `stride`
fields, which is what arbitrary tiling needs.

### Adding an operation is adding a subclass, not touching the datapath

`ExecutionPlan` is a cycle-indexed *declarative schedule*: a plan declares which
control signals fire on which cycles (`mac.flow_down(1, rows)`), and `ControlGen`
synthesises the control hardware from that declaration.

The dispatch is better than "an extension point" — it is a **constructor parameter with
a default**. `FSAParams.supportedExecutionPlans` (`FSA.scala:29`) is a
`(Int, Int, HasArithmeticParams) => Seq[(UInt, ExecutionPlan)]`, defaulting to the five
attention plans:

```scala
supportedExecutionPlans = { (rows, cols, ap) => Seq(
  ISA.MxFunc.LOAD_STATIONARY          -> new LoadStationary(rows, cols),
  ISA.MxFunc.ATTENTION_SCORE_COMPUTE  -> new AttentionScoreExecPlan(rows, cols, ap),
  ISA.MxFunc.ATTENTION_VALUE_COMPUTE  -> new AttentionValueExecPlan(rows, cols),
  ISA.MxFunc.ATTENTION_LSE_NORM_SCALE -> new AttentionLseNormScale(rows, cols, ap),
  ISA.MxFunc.ATTENTION_LSE_NORM       -> new AttentionLseNorm(rows, cols)) }
```

So a GEMM-capable design is a *new config that passes a different sequence*, not an
edit to the accelerator. And the encoding has room: `ISA.MX_FUNC_BITS = 5` gives 32
function codes, of which 5 are used. `MxFunc.GEMM = 5.U` costs nothing.

Concretely, phase 2 is:

| Change | File | Ours or upstream |
|---|---|---|
| `GemmExecPlan` | new file under `generators/fsa/src/main/scala/fsa/` | ours |
| `GEMM = 5.U` function code | `isa/ISA.scala`, or our own object to avoid the edit | avoidable |
| Config passing the extended plan sequence | new file | ours |
| `MxFunc.GEMM = 5` | `python/fsa/instructions.py` | small upstream edit |
| Tiling for arbitrary M/N/K | `python/fsa/kernel.py` or ours alongside | ours |

That is one small upstream edit, and even it is avoidable. It fits D-106's rule of
adding files rather than changing them.

## Where the friction actually is: memory sizing, not compute

`Configs.defaultFSAParams` sizes the scratchpad and accumulator to FlashAttention's
working set and nothing else:

```scala
spadRows = 2 * cols + 4 * rows,   // 2 tiles for Q, 2x2 tiles for K/V double-buffering
accRows  = 1 + rows,              // 1 row for log-exp-sum, 1 tile for output O
```

A general GEMM over transformer-shaped matrices wants a different partitioning and,
for useful M-blocking, more accumulator rows than `1 + rows`. `FSAParams` is a plain
case class, so this is a parameter change rather than an architectural one — but it is
the part that will actually cost time, and it is where a "straightforward" Gate B could
turn into a real one. **Size the accumulator deliberately and record the choice**;
do not let it default to the attention sizing and then report the utilisation number
that results.

Two smaller notes:

- `SystolicArray.scala` hardwires the bottom-row accumulator input to zero with a
  `// TODO: control the bottom input`. For GEMM a zero seed is exactly right, so the
  TODO is not in our way — but it is also the hook Phase 8 will want.
- `Configs` parameterises the array to `fsa128x128`, which is the same tile geometry
  the RPU spec fixes (`GOLDEN_MODEL_SPEC` §4/§5.2, 128×128 weight-streaming tiles).
  That alignment is convenient and worth not breaking.

## One binding that GEMM escapes and attention does not

`generators/fsa/python/main.py` drives the kernel with `d=cfg.sa_rows, br=cfg.sa_cols,
bc=cfg.sa_rows` — the **head dimension is the array's row count**. That is a hard
coupling for the attention path, and it is what makes the phase-1 workload's
`d_head = 72` awkward against a 128-row array (see `DECISIONS.md` D-104).

General GEMM does not inherit the coupling: `C = AB` has no head dimension, and M, N, K
are all free once tiling exists. So the `d_head` question does not sit inside Gate B,
and Gate B passing must not be read as having settled it. It is a phase-2/4 mapping
decision with its own entry to write.

## What this does not cover

`ArithmeticImpl` is instantiated only at fp16/bf16/fp32 widths
(`FPArithmeticImpl(expWidth, sigWidth, accExp, accSig)`). There is **no FP8 and no FP4**
today. Phase 8 needs new `ArithmeticImpl` instances, and the `Arithmetic` typeclass is
the right seam for them — but nothing in Gate B tests that seam, so Gate B passing says
nothing about whether the narrow-format work is cheap.

## How this prediction should be scored

Written before the array was built. When Gate B actually runs, come back and mark each
claim kept or broken — a feasibility note that is never scored is just optimism with a
date on it. The claims are:

1. GEMM needs no new PE control bits. *(from `PECtrl`'s nine signals)*
2. `AttentionValueExecPlan` is structurally the k-tile loop body.
3. Accumulate-vs-seed across k-tiles is already encodable (`MatrixInstructionAcc.zero`).
4. Adding the op requires at most one small upstream edit.
5. **The scratchpad and accumulator sizing is the part that actually costs time.**

Claim 5 is the one most likely to be wrong in the expensive direction. The others are
structural readings that are hard to be wrong about; claim 5 is a judgement about
effort.

## The call this feeds

The roadmap's decision gate is "if adding clean general GEMM support to FSA is
straightforward, FSA becomes the RPU array; otherwise evaluate Gemmini." On the source
alone the answer looks like **FSA**, with the caveat that the evidence is static
reading, and the scratchpad/accumulator resizing is the piece most likely to be worse
than it looks. Consult Gemmini for scratchpad organisation, accumulator SRAM and
tiling — the roadmap's stated use for it — rather than as a fallback array.
