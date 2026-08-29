"""Which transform does the array apply to the stationary operand, at this array size?

`rev_both` was derived at 4x4, where rows == cols and many candidate conventions
coincide. Everything fails at 16x16, so re-derive rather than assume.

B = I makes the output *be* the transform of A.

    uv run ../../rpu/experiments/layout_probe.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G
import fsa as F
from fsa.tensor import MTile

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
ap.add_argument("--func", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prefence", action="store_true",
                help="drain the matrix engine before the GEMM")
ap.add_argument("--rev", action="store_true",
                help="read the stationary tile backwards, as the attention kernel does")
args = ap.parse_args()

REV = args.rev
PREFENCE = args.prefence
rows, cols = G.load_config(args.config)
eng = G.make_engine(args.config)
print(f"{args.config}: {rows} rows x {cols} cols")


@F.kernel
def one(A: MTile, B_t: MTile) -> MTile:
    M, K = A.shape
    N, _ = B_t.shape
    C_t = F.alloc_mem((N, M), F.fp32)
    At = F.alloc_spad((M, K)); Bt = F.alloc_spad((N, K))
    Ct = F.alloc_accumulator((N, M))
    sa, sb, sc = (F.Semaphore(id=i, n=2) for i in range(3))
    F.load_tile(A, At, sa)
    F.mx_load_stationary(At.reverse(dim=0) if REV else At, sa)
    F.load_tile(B_t, Bt, sb)
    if PREFENCE:
        # LoadStationary releases conflictFree at cols-1 but keeps writing `reg` via
        # load_reg_li.parallel(1, cols) until cycle cols. The GEMM's first MACs can
        # therefore read a partially loaded stationary register.
        F.fence(mx=True, dma=False, stop=False)
    G.mx_gemm(args.func, Bt, Ct, False, sb)
    F.store_tile(Ct, C_t, sc)
    F.fence(mx=True, dma=True, stop=True)
    return C_t


rng = np.random.default_rng(args.seed)
A = rng.normal(size=(cols, rows)).astype(np.float16)
B = np.eye(rows, dtype=np.float16)
# S is loaded verbatim; whatever comes back names the transform.
got = F.to_numpy(eng.execute(one(F.from_numpy(np.ascontiguousarray(A)),
                                 F.from_numpy(np.ascontiguousarray(B.T))))).T
Af = A.astype(np.float32)
cands = {
    "A":               Af,
    "rev_rows(A)":     Af[::-1, :],
    "rev_cols(A)":     Af[:, ::-1],
    "rev_both(A)":     Af[::-1, ::-1],
    "rev_cols(A) [REV]": Af[:, ::-1],
    "A.T":             Af.T if Af.shape[0] == Af.shape[1] else None,
    "rev_both(A).T":   Af[::-1, ::-1].T if Af.shape[0] == Af.shape[1] else None,
}
print(f"got shape {got.shape}, finite={np.isfinite(got).all()}, "
      f"max|got|={np.abs(got[np.isfinite(got)]).max() if np.isfinite(got).any() else float('nan'):.4g}")
hit = False
for name, c in cands.items():
    if c is None or c.shape != got.shape:
        continue
    r = np.abs(got.astype(np.float64) - c.astype(np.float64)).max() / max(np.abs(c).max(), 1e-30)
    mark = "  <== MATCH" if r < 1e-2 else ""
    print(f"  vs {name:<16} rel {r:.4e}{mark}")
    hit |= r < 1e-2
bad = np.abs(got) > 1e3
print(f"\n  garbage elements (|x| > 1e3): {bad.sum()} of {got.size}")
if bad.any():
    print("  garbage map (row-major, . = ok, X = garbage):")
    for r in range(got.shape[0]):
        print("   ", "".join("X" if bad[r, c] else "." for c in range(got.shape[1])))
    print("  garbage rows:", sorted(set(np.nonzero(bad)[0].tolist())))
    print("  garbage cols:", sorted(set(np.nonzero(bad)[1].tolist())))
ok = ~bad
if ok.any():
    for name, c in cands.items():
        if c is None or c.shape != got.shape:
            continue
        r = np.abs(got[ok].astype(np.float64) - c[ok].astype(np.float64)).max() / max(np.abs(c).max(), 1e-30)
        if r < 1e-2:
            print(f"  ignoring garbage, the good elements match {name} (rel {r:.3e})")

if not hit:
    print("\n  no candidate matches -- not a layout convention at this size")
    print("  got[0,:6] ", np.array2string(got[0, :6], precision=4))
    print("  A[0,:6]   ", np.array2string(Af[0, :6], precision=4))
    print("  got[:,0][:6]", np.array2string(got[:6, 0], precision=4))
