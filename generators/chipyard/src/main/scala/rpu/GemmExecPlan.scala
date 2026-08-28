// RPU-owned. Lives in the chipyard generator rather than under generators/fsa/ because
// generators/fsa is a git submodule pointing at VCA-EPFL/FSA -- putting our source
// there would leave it untracked by this repository (rpu/DECISIONS.md D-106). Scala
// does not require the directory to match the package, and the chipyard generator
// already depends on the fsa classpath (see config/FSAConfig.scala), so declaring
// `package fsa` here is legal and keeps our code in our fork.
package fsa

import chisel3._
import fsa.isa._
import fsa.sa._

/** Function codes for RPU-added matrix operations.
  *
  * `ISA.MX_FUNC_BITS` is 5, giving 32 codes, of which upstream uses 0-4. Defining ours
  * here rather than editing `isa/ISA.scala` keeps the submodule untouched.
  */
object RpuMxFunc {
  def GEMM = 5.U
}

/** `C = A * B` for one tile: the general matrix product FSA does not otherwise have.
  *
  * Derived from the two attention plans, which between them contain every piece:
  *
  *   - the streaming multiply against the stationary register, from
  *     `AttentionScoreExecPlan`'s first declarations;
  *   - the drain into accumulator SRAM, from `AttentionValueExecPlan`'s tail;
  *   - none of the online-softmax machinery in between (no comparator max tracking,
  *     no `exp2` chain, no exp-sum).
  *
  * The one piece that is neither plan's is `PROP_ZERO`. `acc_ui` makes each PE take its
  * addend from the neighbour above, and `SystolicArray` wires the top row's input to the
  * `CMP` unit's `d_output`. The comparator array is stateful. `rpu/experiments/
  * gate_b_probe.py` showed empirically that issuing `LOAD_STATIONARY` then `ATTN_VALUE`
  * with no score step produces finite-but-wrong results (rel err 1.28, and no operand
  * layout reproduces it) -- consistent with a stale comparator addend entering the top
  * row. Holding the comparators at `PROP_ZERO` for the whole streaming window makes the
  * seed an explicit zero instead of whatever they happen to hold. See D-109.
  *
  * Accumulate-vs-seed across k-tiles is not handled here: it rides on
  * `MatrixInstructionAcc.zero`, which `AccReadDesc.toHardware` already maps to a
  * constant-zero accumulator read. Callers pass it per instruction.
  */
class GemmExecPlan(val rows: Int, val cols: Int) extends ExecutionPlan {
  // Stream the B tile out of the scratchpad, one row per cycle.
  readScratchPad(0, rows, None)
  // Release as soon as the last scratchpad read is issued, matching both attention
  // plans, so the next instruction's SRAM read can overlap this one's compute.
  releaseSemaphore(rows - 1)

  // Drive an explicit zero down from the comparator row for the whole window, one cycle
  // ahead of the MACs that consume it. This is the declaration that makes the plan a
  // GEMM rather than "ATTN_VALUE without a score step".
  setComparator(0, rows, CmpControlCmd.PROP_ZERO)

  // C = reg (the stationary A tile) * l_input (the streamed B tile), accumulating
  // downward so the partial sums drain out of the bottom row into `io.acc_out`.
  mac.flow_down(1, rows)
  acc_ui.flow_down(1, rows)
  flow_lr.flow_down(1, rows)

  // Same reasoning as AttentionValueExecPlan: the last compute of this instruction is
  // at 2*rows - 1, so release one cycle earlier to overlap the next SRAM read.
  setConflictFree(2 * rows - 1 - 1)

  // Drain into accumulator SRAM. ACC_SA adds the systolic array output to the value
  // read back, which combined with the `zero` bit gives seed-or-accumulate.
  readAccRAM(rows + cols - 1, rows, None)
  setAccumulator(rows + cols, rows, AccumulatorCmd.ACC_SA)
}

/** FSA parameter sets that additionally support `GEMM`.
  *
  * `FSAParams.supportedExecutionPlans` is a constructor parameter with a default, so a
  * GEMM-capable design is a different argument rather than an edit to the accelerator.
  */
object RpuConfigs {
  def withGemm(base: FSAParams): FSAParams = base.copy(
    supportedExecutionPlans = { (rows, cols, ap) =>
      Seq(
        ISA.MxFunc.LOAD_STATIONARY          -> new LoadStationary(rows, cols),
        ISA.MxFunc.ATTENTION_SCORE_COMPUTE  -> new AttentionScoreExecPlan(rows, cols, ap),
        ISA.MxFunc.ATTENTION_VALUE_COMPUTE  -> new AttentionValueExecPlan(rows, cols),
        ISA.MxFunc.ATTENTION_LSE_NORM_SCALE -> new AttentionLseNormScale(rows, cols, ap),
        ISA.MxFunc.ATTENTION_LSE_NORM       -> new AttentionLseNorm(rows, cols),
        RpuMxFunc.GEMM                      -> new GemmExecPlan(rows, cols)
      )
    }
  )

  lazy val gemm4x4     = withGemm(Configs.fsa4x4)
  lazy val gemm8x8     = withGemm(Configs.fsa8x8)
  lazy val gemm16x16   = withGemm(Configs.fsa16x16)
  lazy val gemm128x128 = withGemm(Configs.fsa128x128)
}
