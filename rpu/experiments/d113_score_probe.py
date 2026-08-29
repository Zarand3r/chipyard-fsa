"""D-113: does ATTN_VALUE's drain need ATTN_SCORE in front of it?

Upstream only ever issues ATTN_VALUE downstream of ATTN_SCORE. Our GEMM issues it
directly, and corrupts whole drain steps at 8x8 and 16x16 while being exact at 4x4.

This runs the upstream three-instruction sequence -- LOAD_STATIONARY, ATTN_SCORE,
ATTN_VALUE -- through our own harness and asks only one question: are any output rows
garbage? The VALUES will not be A @ B (ATTN_SCORE overwrites the stationary register
with softmax output, which is the point of the fusion). Only the presence or absence of
garbage rows matters here.

    uv run ../../rpu/experiments/d113_score_probe.py --config RpuGemm16X16Fp16Config
"""
from __future__ import annotations
import argparse, os, sys
import subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G
import fsa as F
from fsa.tensor import MTile

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="RpuGemm16X16Fp16Config")
ap.add_argument("--score", action="store_true", help="issue ATTN_SCORE before ATTN_VALUE")
ap.add_argument("--prime", type=int, default=0,
                help="issue N throwaway GEMMs first to flush the unreset pipes")
ap.add_argument("--vseed", type=int, default=None,
                help="Verilator $random seed; changes the power-on contents of "
                     "registers built with pipe_no_reset")
args = ap.parse_args()

if args.vseed is not None:
    # The FSA engine builds its own simulator command; append the seed plusarg.
    _real = subprocess.run

    def _seeded(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd and "simulator-chipyard" in str(cmd[0]):
            cmd = list(cmd) + [f"+verilator+seed+{args.vseed}"]
        return _real(cmd, *a, **kw)

    subprocess.run = _seeded

rows, cols = G.load_config(args.config)
eng = G.make_engine(args.config)


@F.kernel
def k(Am: MTile, Bm: MTile, Km: MTile) -> MTile:
    M, K = Am.shape
    N, _ = Bm.shape
    out = F.alloc_mem((N, M), F.fp32)
    At = F.alloc_spad((M, K)); Bt = F.alloc_spad((N, K))
    Kt = F.alloc_spad((N, K))
    L = F.alloc_accumulator((1, cols))          # log-exp-sum row, as attention uses
    Ct = F.alloc_accumulator((N, M))
    sa, sb, sk = (F.Semaphore(id=i, n=2) for i in range(3))
    sc = F.Semaphore(id=3, n=2)
    F.load_tile(Am, At, sa)
    F.mx_load_stationary(At.reverse(dim=0), sa)
    if args.score:
        F.load_tile(Km, Kt, sk)
        F.mx_attn_score(Kt, L, False, sk, False)
    for _ in range(args.prime):
        # Flush: SystolicArray.pipe_no_reset builds every inter-PE pipeline register
        # with withReset(false.B), so both data and valid power up randomized. Pushing
        # a throwaway instruction through drives real values into every stage.
        F.load_tile(Bm, Bt, sb)
        F.mx_attn_value(Bt, Ct, False, sb)
    F.load_tile(Bm, Bt, sb)
    F.mx_attn_value(Bt, Ct, False, sb)
    F.fence(mx=True, dma=False, stop=False)
    F.store_tile(Ct, out, sc)
    F.fence(mx=True, dma=True, stop=True)
    return out


rng = np.random.default_rng(0)
A = rng.normal(size=(cols, rows)).astype(np.float16)
B = rng.normal(size=(rows, rows)).astype(np.float16)
Kk = rng.normal(size=(rows, rows)).astype(np.float16)
got = F.to_numpy(eng.execute(k(
    F.from_numpy(np.ascontiguousarray(A)),
    F.from_numpy(np.ascontiguousarray(B.T)),
    F.from_numpy(np.ascontiguousarray(Kk))))).T
bad = sorted(set(np.nonzero(np.abs(got) > 1e3)[0].tolist()))
print(f"\n{args.config}  ATTN_SCORE issued: {args.score}")
print(f"  garbage rows: {bad}   ({(np.abs(got) > 1e3).sum()} of {got.size} elements)")
print("  VERDICT:", "clean drain" if not bad else "drain still corrupted")
