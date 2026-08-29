"""Phase 9: a descriptor-driven cycle model, correlated against RTL.

The roadmap asks for simulator <-> RTL <-> FPGA agreement to **<5% cycle error**. The
FPGA leg is blocked (D-102), so this closes the reachable half: predict `execTime` from
the instruction schedule alone, then measure it on Verilator.

The model is **derived from the execution plans, not fitted**. Each plan declares when
the next instruction may start via `setConflictFree`, and that is exactly the issue
interval:

    LoadStationary   setConflictFree(cols - 1)        -> cols - 1 cycles
    GemmExecPlan     setConflictFree(2*rows - 1 - 1)  -> 2*rows - 2 cycles
    SetAccScale      setConflictFree(1)               -> 1 cycle

plus a drain for the final instruction's accumulate phase, which `GemmExecPlan` puts at
`accumulateMaxCycle = 2*rows + cols`.

A fitted model would score well and predict nothing. Deriving it from the same
declarations the hardware is generated from means a disagreement is informative: either
the model misreads the schedule or the hardware is not doing what its plan says.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G                                                    # noqa: E402

COUNTERS = ("execTime", "mxActive", "mxBubble", "dmaActive", "mxInst", "dmaInst")


def counters() -> dict[str, int]:
    out = {}
    for c in COUNTERS:
        m = re.findall(rf"FSA:\s*{c}\s*=\s*(\d+)", G.sim_output())
        if m:
            out[c] = int(m[-1])
    return out


def measured_exec_time() -> int | None:
    """Read the counter out of rpu_gemm's own capture.

    An earlier version installed a second `subprocess.run` wrapper here, which
    `make_engine` promptly overwrote with its own -- so every measurement came back
    n/a. Use the one capture rather than racing it.
    """
    m = re.findall(r"FSA:\s*execTime\s*=\s*(\d+)", G.sim_output())
    return int(m[-1]) if m else None


def dma_count(mt: int, nt: int, kt: int) -> int:
    """DMA instructions the kernel issues: A and B per k-tile, one store per output
    tile, plus the two loads that prime the accumulator scale when k > 1."""
    return mt * nt * (2 * kt + 1) + (2 if kt > 1 else 0)


def predict(mt: int, nt: int, kt: int, rows: int, cols: int, dma_latency: float) -> int:
    """Cycles for a tiled GEMM.

    Two terms, and only the second carries a calibrated constant:

    * **compute**, derived from the plans' own `setConflictFree` declarations --
      `LoadStationary` at `cols-1`, `GemmExecPlan` at `2*rows-2`, plus
      `accumulateMaxCycle = 2*rows + cols` to drain the last accumulate;
    * **stall**, `dma_latency` cycles per DMA instruction. This one cannot be derived
      from the execution plans: it is a property of the TileLink/DRAM path, not of the
      array. It is calibrated on ONE shape and then used to predict the others.

    The measured decomposition is why the model has this shape at all: `mxBubble` is
    56-88% of `execTime` while `mxActive` and `dmaActive` are both small, so the tiled
    GEMM is **latency-bound**, not compute- or bandwidth-bound.
    """
    load_iv = cols - 1
    gemm_iv = 2 * rows - 2
    acc_max = 2 * rows + cols             # GemmExecPlan.accumulateMaxCycle
    prime = (load_iv + gemm_iv + 1) if kt > 1 else 0

    # k-tiles after the first set waitPrevAcc (rpu_gemm.mx_gemm), and
    # MatrixEngineController maps that to `canEnq = !io.busy` -- FULL serialisation,
    # not the conflict-free interval. So an accumulating tile costs its whole
    # accumulate phase and cannot overlap the next tile's load. Derived from the RTL,
    # not fitted: it is why k-heavy shapes were under-predicted by ~10%.
    serialised = max(kt - 1, 0) * (acc_max - gemm_iv)

    compute = mt * nt * (kt * (load_iv + gemm_iv) + acc_drain_term(rows, cols)
                         + serialised) + prime
    return int(round(compute + dma_latency * dma_count(mt, nt, kt)))


def acc_drain_term(rows: int, cols: int) -> int:
    return 2 * rows + cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
    args = ap.parse_args()
    rows, cols = G.load_config(args.config)
    R, C = rows, cols

    shapes = [
        ("single tile",   C,     R,     R),
        ("k x2",          C,     R, 2 * R),
        ("k x4",          C,     R, 4 * R),
        ("n x2",          C, 2 * R,     R),
        ("m x2",      2 * C,     R,     R),
        ("m x n",     2 * C, 2 * R,     R),
        ("m x n x k", 2 * C, 2 * R, 2 * R),
    ]
    print(f"{args.config}: {rows} rows x {cols} cols\n")
    print(f"{'shape':<12} {'tiles':<8} {'pred':>7} {'meas':>7} {'err':>8} "
          f"{'mxAct':>7} {'mxBub':>7} {'dmaAct':>7} {'dmaI':>5}")
    errs = []
    rng = np.random.default_rng(0)
    calib: float | None = None
    rowsdata = []
    for name, m, n, k in shapes:
        G.reset(); G.clear_assertions(); G.load_config(args.config)
        eng = G.make_engine(args.config)
        A = rng.normal(size=(m, k)).astype(np.float16)
        B = rng.normal(size=(k, n)).astype(np.float16)
        G.run(eng, A, B, rows, cols)
        meas = measured_exec_time()
        s = G.Shape(m, n, k, rows, cols)
        if calib is None and meas is not None:
            # Calibrate the one non-derivable constant on the FIRST shape only, then
            # predict every other shape with it. Fitting all seven would score well and
            # demonstrate nothing.
            lo, hi = 0.0, 500.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if predict(s.mt, s.nt, s.kt, rows, cols, mid) < meas:
                    lo = mid
                else:
                    hi = mid
            calib = (lo + hi) / 2
            print(f"  calibrated DMA latency on '{name}' only: "
                  f"{calib:.1f} cycles/DMA instruction\n")
        pred = predict(s.mt, s.nt, s.kt, rows, cols, calib or 0.0)
        if meas is None:
            print(f"{name:<12} {'--':<10} {pred:>10} {'n/a':>10} {'--':>9}")
            continue
        err = (pred - meas) / meas
        rowsdata.append((name, err))
        errs.append(abs(err))
        c = counters()
        print(f"{name:<12} {f'{s.mt}x{s.nt}x{s.kt}':<8} {pred:>7} {meas:>7} "
              f"{100*err:>7.1f}% {c.get('mxActive',-1):>7} {c.get('mxBubble',-1):>7} "
              f"{c.get('dmaActive',-1):>7} {c.get('dmaInst',-1):>5}")

    if errs:
        # The calibration shape is not a prediction; score the held-out six.
        held = [abs(e) for n_, e in rowsdata[1:]]
        worst, mean = 100 * max(held), 100 * float(np.mean(held))
        print(f"\n  (shape 1 calibrates; the {len(held)} below it are predictions)")
        print(f"\nworst {worst:.1f}%   mean {mean:.1f}%   "
              f"(roadmap target: <5%)")
        print("PHASE 9:", "MEETS the <5% target" if worst < 5 else
              f"MISSES the <5% target by {worst - 5:.1f} points")
        print("  FPGA correlation leg: SKIP (D-102).")
        return 0 if worst < 5 else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
