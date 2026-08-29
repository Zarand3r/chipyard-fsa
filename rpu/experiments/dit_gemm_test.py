"""Run the real phase-1 DiT GEMMs on the array, with real checkpoint data.

Gate B asks for "a real transformer-shaped matrix". These operands come from the
frozen DiT-XL/2 trace (workloads/dit), not from a random generator, so this exercises
the fp16 dynamic range an actual checkpoint produces.

Two things it reports honestly:

* **Tileability.** M must be a multiple of `cols`, N and K multiples of `rows`. Several
  real DiT shapes do not tile -- notably anything carrying `d_head = 72` (D-104) and
  `adaln`, whose M is 1. That is a real result about the workload/array pairing, not a
  harness limitation.
* **Cost.** Full-size shapes need up to ~250k stationary loads, which is not a Verilator
  workload. Each case is therefore run on a tileable *slice*, and the slice is printed
  with the result so no number is quoted as if it were the whole GEMM.

    uv run ../../rpu/experiments/dit_gemm_test.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G

GEMM_DIR = Path.home() / "rpu-simulation/chipyard-fsa/workloads/dit/build/gemm"


def elem_dtype(name: str):
    """Element dtype for the run. E4M3 is a config parameter, not a datapath change
    (D-122), so the same operands can be pushed through an FP8 array unchanged."""
    if name == "fp16":
        return np.float16
    import ml_dtypes
    return {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}[name]


def load_case(name: str, dt):
    a = np.load(GEMM_DIR / f"{name}.A.npy")
    b = np.load(GEMM_DIR / f"{name}.B.npy")
    return a.astype(dt), b.astype(dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
    ap.add_argument("--func", type=int, default=G.GEMM_FUNC)
    ap.add_argument("--max-loads", type=int, default=256,
                    help="cap on stationary loads per case, to keep Verilator sane")
    ap.add_argument("--vseeds", type=str, default="")
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "e4m3", "e5m2"),
                    help="element format; must match the config's e_type")
    args = ap.parse_args()
    dt = elem_dtype(args.dtype)

    rows, cols = G.load_config(args.config)
    manifest = json.loads((GEMM_DIR / "manifest.json").read_text())
    names = [n for n in manifest["cases"] if (GEMM_DIR / f"{n}.A.npy").exists()]

    print(f"{args.config}: {rows}x{cols}, elements {args.dtype}   "
          f"(M % {cols} == 0, N/K % {rows} == 0)\n")
    print(f"{'case':<16} {'full shape':<24} {'tiles':<14} {'full loads':>11}  status")
    runnable = []
    for name in names:
        A, B = load_case(name, dt)
        m, k = A.shape; _, n = B.shape
        why = []
        if m % cols: why.append(f"M={m} not x{cols}")
        if n % rows: why.append(f"N={n} not x{rows}")
        if k % rows: why.append(f"K={k} not x{rows}")
        shape = f"[{m}x{k}]@[{k}x{n}]"
        if why:
            print(f"{name:<16} {shape:<24} {'--':<14} {'--':>11}  SKIP: {'; '.join(why)}")
            continue
        s = G.Shape(m, n, k, rows, cols)
        print(f"{name:<16} {shape:<24} {f'{s.mt}x{s.nt}x{s.kt}':<14} "
              f"{s.stationary_loads:>11,}  tileable")
        runnable.append((name, A, B))

    if not runnable:
        print("\nno case tiles on this array")
        return 1

    print(f"\nrunning tileable cases on a slice capped at {args.max_loads} stationary loads:")
    vseeds = [int(x) for x in args.vseeds.split(",") if x.strip()] or [None]
    fail = 0
    for name, A, B in runnable:
        # Grow K FIRST. An earlier version grew M then N and always ended at kt == 1,
        # so it never exercised k-accumulation -- the exact path D-111's stale
        # accumulator scale lived on. A slice that skips the interesting axis is not a
        # test of the interesting axis.
        mt = nt = kt = 1
        while mt * nt * (kt + 1) <= args.max_loads and (kt + 1) * rows <= A.shape[1]:
            kt += 1
        while (mt + 1) * nt * kt <= args.max_loads and (mt + 1) * cols <= A.shape[0]:
            mt += 1
        while mt * (nt + 1) * kt <= args.max_loads and (nt + 1) * rows <= B.shape[1]:
            nt += 1
        assert kt > 1 or A.shape[1] < 2 * rows, "slice must exercise k-accumulation"
        m, n, k = mt * cols, nt * rows, kt * rows
        As, Bs = np.ascontiguousarray(A[:m, :k]), np.ascontiguousarray(B[:k, :n])
        ref = As.astype(np.float32) @ Bs.astype(np.float32)
        for vs in vseeds:
            G.reset(); G.clear_assertions()
            G.load_config(args.config)
            eng = G.make_engine(args.config, vseed=vs)
            got = G.run(eng, As, Bs, rows, cols, func=args.func)
            rel = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()
                        / max(np.abs(ref).max(), 1e-30))
            fired = G.assertions()
            # fp32 accumulation over k terms of real checkpoint fp16 values
            # E4M3 has 3 explicit mantissa bits (half-ulp 2**-5); fp16 has 10.
            # A shared tolerance would either pass everything or fail everything.
            base = 3e-4 if args.dtype == "fp16" else 6e-2
            tol = max(1e-3 if args.dtype == "fp16" else 2e-1,
                      base * (k / rows) ** 0.5)
            ok = rel < tol and not fired
            fail |= (not ok)
            tag = f" vseed={vs}" if vs is not None else ""
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<16}{tag:<12} "
                  f"slice [{m}x{k}]@[{k}x{n}]  {mt}x{nt}x{kt} tiles  "
                  f"{mt*nt*kt:>4} loads  rel {rel:.3e}  tol {tol:.1e}")
            for line in fired[:2]:
                print(f"        RTL ASSERTION: {line}")

    print("\nDIT GEMM:", "all slices PASSED" if not fail else "FAILED")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
