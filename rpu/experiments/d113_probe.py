"""D-113: localize the 16x16 row corruption.

Three questions, each answered by comparing which output rows are garbage:

  1. B = I  -> output row r carries stationary row r'. Garbage tracks A.
  2. A = I  -> output row r carries B row r'.           Garbage tracks B.
     If the SAME rows are garbage in both, the fault is positional in the
     accumulator/readout and independent of either operand.
  3. store the same accumulator twice into two buffers. If both copies are
     garbage in the same places the accumulator content is bad; if they differ,
     the store path is bad.

    uv run ../../rpu/experiments/d113_probe.py --config RpuGemm16X16Fp16Config
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
args = ap.parse_args()

THRESH = 1e3


def bad_rows(x):
    return sorted(set(np.nonzero(np.abs(x) > THRESH)[0].tolist()))


def run_case(label, A, B, double_store=False, acc_offset_rows=0):
    G.reset()
    rows, cols = G.load_config(args.config)
    eng = G.make_engine(args.config)

    @F.kernel
    def k(Am: MTile, Bm: MTile) -> list[MTile]:
        M, K = Am.shape
        N, _ = Bm.shape
        outs = [F.alloc_mem((N, M), F.fp32) for _ in range(2 if double_store else 1)]
        At = F.alloc_spad((M, K)); Bt = F.alloc_spad((N, K))
        # Shift the accumulator tile's base address by allocating a filler first.
        # BankedSRAM selects banks from the LOW address bits, so if the corrupted rows
        # are a bank effect they move with the base; if they are drain-timing they stay.
        if acc_offset_rows:
            F.alloc_accumulator((acc_offset_rows, M))
        Ct = F.alloc_accumulator((N, M))
        sa, sb = F.Semaphore(id=0, n=2), F.Semaphore(id=1, n=2)
        sc = [F.Semaphore(id=2, n=2), F.Semaphore(id=3, n=2)]
        F.load_tile(Am, At, sa)
        F.mx_load_stationary(At, sa)
        F.load_tile(Bm, Bt, sb)
        G.mx_gemm(args.func, Bt, Ct, False, sb)
        F.fence(mx=True, dma=False, stop=False)
        for i, o in enumerate(outs):
            F.store_tile(Ct, o, sc[i])
        F.fence(mx=True, dma=True, stop=True)
        return outs

    res = eng.execute(k(F.from_numpy(np.ascontiguousarray(A)),
                        F.from_numpy(np.ascontiguousarray(B.T))))
    if not isinstance(res, list):
        res = [res]
    mats = [F.to_numpy(r).T for r in res]
    print(f"  {label:<28} garbage rows {bad_rows(mats[0])}")
    if double_store:
        b0, b1 = bad_rows(mats[0]), bad_rows(mats[1])
        same_bits = np.array_equal(np.nan_to_num(mats[0]), np.nan_to_num(mats[1]))
        print(f"  {'  second store':<28} garbage rows {b1}   "
              f"identical to first: {same_bits}")
    return mats[0]


rows, cols = G.load_config(args.config)
print(f"{args.config}: {rows} rows x {cols} cols, func={args.func}\n")
rng = np.random.default_rng(0)

A_rand = rng.normal(size=(cols, rows)).astype(np.float16)
B_rand = rng.normal(size=(rows, rows)).astype(np.float16)
I_a = np.eye(cols, rows, dtype=np.float16)
I_b = np.eye(rows, dtype=np.float16)

print("1/2. does garbage track the operands, or the output position?")
run_case("B = I  (output carries A)", A_rand, I_b)
run_case("A = I  (output carries B)", I_a, B_rand)
run_case("A = I, B = I", I_a, I_b)

print("\n3. is the accumulator content bad, or the store?")
run_case("random, stored twice", A_rand, B_rand, double_store=True)

print("\n4. do the bad rows move with the accumulator base address?")
run_case("acc offset 0 rows", A_rand, B_rand, acc_offset_rows=0)
run_case("acc offset 1 row",  A_rand, B_rand, acc_offset_rows=1)
