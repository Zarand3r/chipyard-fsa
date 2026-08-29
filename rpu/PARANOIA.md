# Paranoia protocol

Every rule here was paid for by a specific defect. The citation is the point: a rule
without an incident behind it is superstition, and gets dropped the first time it is
inconvenient.

## The failure mode this program actually has

Not crashes. **Silent, deterministic, plausible-looking wrong answers** that read as
"my code is buggy". D-113 produced correct arithmetic in most elements and garbage in a
few rows, reproducibly, with no assertion firing. It was diagnosed as our bug twice
(D-109's retraction, D-112's deletion verdict) before the real cause surfaced.

Assume a wrong result is the hardware's contract, not your arithmetic, until a
known-answer probe says otherwise.

## Rules

### 1. A result is scoped to the configuration it was measured on — and a ratio to its base

**Incidents: D-110, D-112.** Both generalised a 4x4 measurement to "the design".
`ATTN_VALUE` computes a correct GEMM at 4x4 and corrupts rows at 8x8 and 16x16;
`GemmExecPlan` looked useless because the only size where its fix is unnecessary is the
size we benchmarked.

Gates run **>= 3 array sizes**. Any claim in a commit message, decision or report names
its configuration, or it is not a claim.

**Incident: D-119.** The same error with a denominator instead of a configuration.
"99.2% of MACs wasted" was true *within* an elementwise op and implied the op mattered;
against a whole block the overhead is **0.84%**, and the recommendation reversed once
the arithmetic was done. A ratio without its base is as unscoped as a benchmark without
its config. Compute the number that answers the question, not the one that is easy.

### 2. Sweep the RTL random seed

**Incident: D-113.** Uninitialised registers are invisible under a fixed `$random` seed
and look exactly like a logic bug. Varying the seed separates *uninitialised* from
*miscomputed* in one run — the corrupted rows moved with every seed.

`gate_b_test.py --vseeds a,b,c`. Identical results across seeds is a positive property,
not a formality. Do this **early** when a symptom is deterministic but has no structural
explanation.

### 3. Known-answer probes before random data

**Incident: D-110.** A sixteen-way sweep over operand orders and transposes "proved" the
mechanism broken. It permuted the *product* while the fault was in the *operand*.
`B = I` makes the output *be* the transform of the input and named the real convention
(`rev_both`) in one run.

Pass/fail judges. Identity localises. When something is wrong, first make the hardware
produce an answer you already know.

### 4. A tolerance hides modelling errors in both directions

**Incident: D-116.** Gate B compared the array against float32 numpy under a guessed
envelope, loose by two orders of magnitude. Replacing it with PyEasyFloat and demanding
*exact* equality immediately exposed an error — in the **golden**, not the RTL: it
carried one continuous fma chain across K, where the hardware contracts each k-tile from
zero and merges with a single fused `ACC_SA` rounding. Single-tile cases matched; every
k-accumulating case missed by ~1 ulp.

Under a tolerance that mistake is invisible forever, and so is any hardware error
smaller than the envelope. Where the roadmap draws a `<-->` arrow, demand equality and
model the arithmetic properly. Where a tolerance is genuinely required, it is *derived*
and its derivation is written down.

### 5. Test the artifact you just built, not the one that is lying around

**Incident: D-134.** Twice in one session I waited on a log line to decide a rebuild had
finished. The build scripts truncate their log but earlier runs leave matching `exit=`
text, so `grep` succeeded against a *stale* line and the tests ran new Python against old
simulators. Both rounds produced confident, wrong conclusions -- including an apparent
regression at 4x4 that did not exist.

Wait on the **artifact**: `stat -c %Y` on the simulator binary, compared against the time
the edit was made. A log line says something happened once; a timestamp says what is
actually on disk now.

### 6. Never scroll past an assertion

The Verilator build carries `--assert` and the RTL has `DelayedAssert`s on real
contracts (e.g. `PopCount(computeFlags) <= 1`). `rpu_gemm.make_engine` captures
simulator output and any assertion **fails the case**. Assertions must never be noise.

### 7. Bound the cycle limit

**Incident: DMA into accumulator SRAM (D-111).** `isAccum` is declared in the DMA
instruction bundle and read nowhere in the RTL, so the transfer never completes and the
semaphore never releases. With `max_cycles=0` that hung silently for six minutes. Probes
pass a bounded `max_cycles` so a deadlock fails loudly.

### 8. Upstream tests cover upstream's usage only

**Incidents: D-111, D-113.** FSA's `main.py --seq_kv 4` issues one K block, so it never
exercises accumulation at all — Gate A passed without touching the path. And
`ATTN_VALUE` is only ever issued downstream of `ATTN_SCORE`, so its precondition is
never tested.

A passing upstream suite says what ran, not what works. Before reusing an upstream
mechanism, ask what their tests actually drive.

### 9. Ask what the state is when you arrive

**Incident: D-113, and D-111 before it.** Both were state that someone else's
instruction normally initialises.

Known unreset state in FSA, from `grep -n "= Reg(" --include=*.scala`:

| Location | State | Precondition |
|---|---|---|
| `sa/SystolicArray.scala:38` | `pipe_no_reset = withReset(false.B){ Pipe(in) }` — every inter-PE pipe, **data and valid** | Something must flush the array. **This is D-113.** |
| `sa/PE.scala:55` | `exp2Done = Reg(Bool())` | Comment: *"as long as exp2 is not the first operation, exp2Done does not need to be reset"* — **a second undocumented ordering precondition, not yet hit.** Relevant when phase 4 adds softmax/GELU. |
| `sa/PE.scala:49` | `reg = Reg(elemType)` | Safe only because `LOAD_STATIONARY` always precedes use |
| `Accumulator.scala` | `scale = Seq.fill(cols){ Reg(accType) }` | **D-111.** Must be driven to 1.0 for a GEMM |
| `dma/LSQ.scala:29`, `dma/DMARequest.scala:31`, `frontend/Decoder.scala:17` | `Reg(Vec(...))` buffers | Presumed valid-bit guarded; unaudited |

Phase 8's weight-streaming and FP4/FP8 work touches these same paths. Re-read this table
before adding an instruction.

## Applying it to the next phases

- **Phase 3 (RPU numerical golden).** Software, so rules 1-2 do not apply, but rules 3-4
  do: build known-answer vectors before random ones. `GOLDEN_MODEL_SPEC` §10 already
  asks for must-fail mutants — those are rule 3 in the spec's own words.
- **Phase 4 (AdaLN/GELU/modulation).** GELU or softmax will drive `exp2`. Check
  `exp2Done` (rule 9) *before* debugging any wrong answer, and sweep seeds immediately.
- **Phase 6 (measurement).** Every number carries its configuration (rule 1), and any
  padding is quoted with it (D-104).
