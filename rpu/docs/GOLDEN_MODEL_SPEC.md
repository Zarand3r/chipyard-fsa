# GOLDEN_MODEL_SPEC.md — RPU functional specification for the golden model

**Written 2026-08-27.** This document specifies the RPU's observable behavior precisely
enough to implement a **golden functional model**: a bit-exact software reference that
RTL, FPGA, and silicon are verified against. Sources: CHIP_SPEC v0.2, CHIP_LAYOUT,
system-design.md; where those documents leave a parameter open it is marked **DECIDE**
with its owning gate, never silently invented. Provenance tags as usual; everything
architectural here is [T] until gate-1 measurements land.

**Scope rule:** the golden model defines *what* is computed (values, orders, state
transitions, memory traffic), not *how fast*. Timing, power, refresh scheduling, and
utilization belong to the cycle model (`sim/`) and the analytical instrument (`rpu/`).
One deliberate exception: **reduction order and rounding are architectural** — they are
specified exactly, because bit-exactness against RTL is impossible without them.

---

## 1. Conformance levels

A golden model implementation MUST provide:

- **L1 — value conformance:** given (weight image, initial state, input token streams,
  mode), produce bit-exact outputs for every observation point in §9.
- **L2 — trace conformance:** produce the external-memory access trace (address,
  direction, length order) for each chunk; the trace MUST be identical across chunks
  with the same mode (the conveyor invariant, §6).
- **L3 — state conformance:** expose every named persistent state object (§7) for
  inspection between chunks.

## 2. Workload contract (frozen; matches bench/contract.toml)

DreamZero-class 14B video-action diffusion transformer:

| Parameter | Value |
|---|---|
| Layers `L` | 40 |
| Hidden width `d` | 5120 |
| FFN width `d_ff` | 13824 |
| Attention heads `H` | 40 (head dim `d_h` = 128) |
| Context window `N_ctx` | 18,720 tokens (2 s) |
| Fresh tokens per chunk `N_new` | 1,560 (Deadline) / 3,120 (Quality, Balanced) |
| Text conditioning length | 256 tokens (cross-attention K/V source) |
| Weights | 14B params, 4-bit microscaled (7.0 GB) |
| Activations / KV | FP8 (KV window 7.7 GB) |
| Diffusion steps `S` | 1, 2, or 3 — mode-selected |
| Guidance | CFG pair (cond/uncond) in Quality and Balanced; **DECIDE-9** for Deadline |

The golden model is parameterized over this table (a config schema, §10) so the same
model serves the JEPA row and the cross-model proxies; the values above are the
conformance defaults.

## 3. Number formats and rounding

| Quantity | Format | Notes |
|---|---|---|
| Weights (storage) | **MXFP4 profile (default):** E2M1 elements, 32-element blocks, shared E8M0 (power-of-two) scale per block | dequant = exponent add, no multiplier |
| Weights (fallback profile) | **NVFP4:** E2M1 elements, 16-element blocks, FP8-E4M3 scale | one FP8 multiply per block; profile is a µcode field |
| Activations, KV | FP8 — **DECIDE-1:** E4M3 vs E5M2 split per tensor class (working assumption: E4M3 everywhere; E5M2 nowhere until a range analysis demands it) |
| MAC products | exact (formats are narrow enough that the product is representable before reduction) |
| Accumulation | **FP32**, default profile. Gated alternates (gate-1, task-checked): BF16, INT8 with block scale |
| Softmax arithmetic | FP32 internally (§5.3) |
| Vector ops (norms, GELU, residuals) | FP32 internally, FP8 at tensor boundaries — **DECIDE-2:** exact re-quantization points |
| Rounding | round-to-nearest-even at every format boundary; saturating casts (no inf/NaN propagation into FP8 tensors); FP8/FP4 denormals supported as per OCP MX spec |

**Weight quantization is out of scope**: the golden model consumes a prepared weight
image (quantized offline); it never quantizes weights itself.

## 4. Deterministic reduction (the bit-exactness backbone)

All matrix arithmetic MUST reduce in this exact order — this is the architectural
contract that makes golden-vs-RTL comparison exact rather than approximate:

1. **Dot products are tiled in k-blocks of 128** (one systolic tile traversal).
2. Within a k-block, dequantized products are summed by a **fixed 8-input combinational
   adder tree** (CHIP_SPEC §6: the 0.51× arithmetic-energy lever): products p[0..7] by
   ascending k reduce as `((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))`, evaluated exactly
   (no intermediate rounding — **DECIDE-3**, gate-1: exact tree vs FP32-rounded tree
   nodes; golden model implements both behind a flag, default exact-then-round).
3. Tree outputs accumulate into the FP32 accumulator **in ascending k-block order**
   (k = 0..K/128−1), one RNE-rounded FP32 add per tree output.
4. Accumulators for a given output tile are independent; no cross-output reduction
   exists anywhere in the datapath.
5. Adder-tree width 8 is itself a gate-1 decision (**DECIDE-4**: 4/8/16); the golden
   model parameterizes it. Bit-exactness claims are always *per configuration*.

## 5. Datapath blocks (functional semantics)

### 5.1 Dequant row
Input: MXFP4 (or NVFP4/INT4/INT2 — profile field) weight block + scale.
Output: the effective multiplicand consumed by the MAC.
MXFP4: value = E2M1 element with exponent increased by the block's E8M0 scale
(pure exponent add). NVFP4: element × E4M3 scale (one FP8 multiply, RNE).
INT4/INT2 paths exist in hardware (CHIP_SPEC §4 flag) — modeled, disabled by default,
and INT2 stays behind the no-aggressive-quantization constraint.

### 5.2 Matmul (weight-streaming systolic, functional view)
`Y[M,N] = X[M,K] · W[K,N]` with X in FP8, W dequantized per §5.1, Y in FP32 per §4,
then re-quantized to FP8 at the tensor boundary (§3). Tiling: output tiles of 128×128;
the golden model iterates tiles in row-major (M-tile, N-tile) order — order is
observable only through the memory trace (L2), not values (§4.4 independence).
The CFG pair shares every fetched weight block (F2 rule): one weight fetch feeds both
branch matmuls; the golden model must issue **one** trace read per weight block per
step regardless of branch count.

### 5.3 Attention (two fabric passes + online-softmax streamer)
Per layer, per head (d_h = 128), over the full window (N_ctx tokens of K/V from the
ring + the current step's fresh K/V):

- **Pass 1:** `S_raw = Q · K^T` on the main fabric (FP8 operands, FP32 accumulation
  per §4), scaled by 1/√d_h in FP32.
- **Streamer:** numerically stable softmax in FP32. Default: **online softmax**
  (running max m, running sum s over k-tiles in ascending tile order — the recurrence
  order is part of the spec). Two µcode-selectable variants, modeled behind flags:
  (a) **static max-bound** — per-layer precomputed max replaces the running max
  (candidate simplification, gate-1 task-accuracy check); (b) **FLASH-D** form —
  division folded into sigmoid evaluation (Tier-2 candidate). `exp` is **DECIDE-5**:
  the golden default is correctly-rounded FP32 `exp`; the hardware approximation
  (LUT/poly, ExpMul-class) must be specified to the bit at gate 1 and the golden model
  then switches to it. Probabilities re-quantize to FP8 for pass 2 (**DECIDE-6**:
  FP8 vs keeping FP16 through AV).
- **Pass 2:** `O = P · V` on the fabric, FP32 accumulation, output re-quantized FP8.

Cross-attention: identical machinery with K/V from the 256-token text encoding
(precomputed per chunk, resident in SRAM).

### 5.4 Vector unit
RMSNorm/LayerNorm (**DECIDE-7**: which, and where — DreamZero-class DiT block layout),
GELU, residual adds, RoPE/positional handling (**DECIDE-8**: the DiT's positional and
timestep/AdaLN conditioning scheme — must be pinned from the reference checkpoint
before the golden model can claim L1 over a full layer). All FP32 internal, FP8
boundaries, RNE.

### 5.5 DiT block (reference composition)
`x → [norm → QKV proj → attention → out proj → +residual] → [norm → cross-attn →
+residual] → [norm → FFN(GELU) → +residual]`, with conditioning applied per DECIDE-8.
FLOP shares at contract shapes (checksum for the model): self-attention 40.7%,
FFN 30.0%, QKV 16.7%, cross-attention 7.0%, output projection 5.6% [S].

### 5.6 Update engine (the programmable 2%)
Flow-ODE / CEM update between diffusion steps: µcoded, FP32. The golden model provides
a reference flow-ODE step (Euler; **DECIDE-10**: the shipped integrator and CEM
variant) and executes user µcode as plain FP32 software — the engine's ISA is a
separate document; until it exists this block's conformance level is L1 over the
reference integrator only.

## 6. Memory system (functional semantics)

- **Address map** (base addresses are config, layout is normative): weight region
  (blocks in exact conveyor stream order — layer-major, then per-layer operator order
  QKV, attn-out, FFN-in, FFN-out, cross-attn — **DECIDE-11**: freeze the intra-layer
  order), KV ring region, activation spill region, schedule/µcode image.
- **Conveyor invariant (L2):** each step issues the identical weight-region read trace;
  each chunk issues the identical full trace modulo ring-pointer offsets. One read per
  weight block per step (F2). The golden model emits the trace; the cycle model prices
  it; RTL must match it exactly.
- **KV ring:** ring buffer of per-layer K/V for N_ctx tokens. At chunk end: advance
  ring pointers by N_new (evicting the oldest N_new), append the fresh tokens' K/V
  (written during the final step's pass). No copies, no remapping — pointer arithmetic
  only, and the golden model must implement it as such so wraparound addressing is
  exercised. SRAM holds only in-flight window tiles (not modeled at L1; visible at L2
  only as the absence of re-reads within a tile).
- **Refresh, ECC:** timing-domain; invisible to the golden model by definition.

## 7. Named persistent state objects (L3)

`WEIGHT_IMAGE` (read-only after load) · `KV_RING[L]` + ring pointers ·
`TEXT_KV[L]` (per-chunk) · `DIFFUSION_LATENT` · `SCHEDULE_IMAGE` (µcode ROM: modes,
tile loops, stream order) · `MODE_REG` (Quality/Balanced/Deadline + profile fields:
weight profile, accumulator format, softmax variant, adder-tree width) ·
`ACTION_BUFFER` (output). State persists across chunks; a chunk is a pure function of
(state, fresh sensor tokens, mode) — that property is itself a conformance test.

## 8. Chunk execution (the superloop, one iteration)

1. Ingest fresh tokens; encode (VAE-encoder stage — **DECIDE-12**: in-scope for the
   golden model or upstream; working assumption upstream, tokens arrive encoded).
2. Compute fresh-token K/V; prefill attention primes against the ring window.
3. For step s = 1..S(mode): full DiT forward (40 layers per §5.5) — CFG pair with
   F2-shared weight stream where the mode has guidance; update engine (§5.6) advances
   the latent.
4. Action head (**DECIDE-13**: final projection shape and trajectory format) fills
   `ACTION_BUFFER`.
5. Commit: ring pointers advance, fresh K/V appended, latent state updated.

## 9. Observation points (what "bit-exact" is checked on)

Per configuration: (a) every layer's post-block activations, both branches;
(b) per-head attention outputs for designated probe layers {0, 19, 39};
(c) accumulator values for designated probe tiles before re-quantization;
(d) `ACTION_BUFFER`; (e) all L3 state after commit; (f) the L2 trace.
RTL comparison is equality, not tolerance — that is the point of §4.

## 10. Test plan (build these with the model, L15 discipline)

- **Unit vectors:** dequant (both profiles, denormals, scale extremes), adder tree
  (catastrophic-cancellation vectors that distinguish tree orders), online softmax
  (max at every tile position; static-bound divergence cases), RNE/saturation edges.
- **Mutants that must fail:** linear-order accumulation instead of the tree; running
  max updated in descending tile order; double-fetched CFG weights (breaks L2);
  ring implemented as memcpy (passes values, fails wraparound addressing vectors).
- **Layer and chunk vectors:** random-seeded, plus the FLOP-share checksum (§5.5) and
  chunk-purity test (§7). Golden outputs are committed as fixtures; the cycle model
  (`sim/`) consumes the same vectors for its independent FLOP/byte cross-check.

## 11. Open decisions (blocking full L1; each has an owner gate)

| # | Decision | Gate |
|---|---|---|
| 1 | FP8 flavor per tensor class | pre-RTL numerics study |
| 2 | Re-quantization points in vector path | pre-RTL numerics study |
| 3 | Adder-tree internal rounding | gate 1 |
| 4 | Adder-tree width (4/8/16) | gate 1 (with accumulator format) |
| 5 | Hardware `exp` approximation, bit-specified | gate 1 |
| 6 | Probability precision into pass 2 | gate 1 |
| 7 | Norm type and placement | reference-checkpoint pinning |
| 8 | Positional + conditioning scheme (RoPE/AdaLN) | reference-checkpoint pinning |
| 9 | Deadline-mode guidance on/off | mode definition freeze |
| 10 | Shipped integrator / CEM variant | update-engine ISA doc |
| 11 | Intra-layer conveyor order | pre-RTL, with SRAM banking |
| 12 | VAE-encoder scope | system boundary decision |
| 13 | Action-head format | reference-checkpoint pinning |

DECIDE-7/8/13 share one prerequisite: pinning the reference checkpoint's exact block
structure. Until then the golden model is L1-conformant per operator and per synthetic
layer, not against the real checkpoint.

## 12. Relationship to the rest of the program

Golden model (this spec, values) → cycle model `sim/` (time; already agrees with the
analytical model on FLOPs to 0.7%) → analytical instrument `rpu/` (energy/feasibility)
→ gate-1 synthesis (LibreLane/ASAP7-class flow) verifies RTL against the golden model
on the §10 vectors. The streaming/cross-model extensions (CROSS_MODEL_DESIGN) reuse
this spec unchanged except §2's parameterization and §8's loop cadence — that is the
point of specifying the machine, not the model.
