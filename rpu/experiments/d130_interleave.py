"""D-130's named experiment: does interleaving independent output tiles hide the stall?

Four interventions changed nothing (D-130), so the stall is a dependency, not a
resource. The k loop accumulates through ONE accumulator tile, giving each k-tile a
read-after-write dependency on the previous. This runs the same total work two ways:

  serial      one output tile at a time, k loop back to back      (today's kernel)
  interleaved N output tiles round-robin through their k loops    (needs accRows > 1 tile)

If the stall is the accumulator RAW dependency, interleaving gives the engine independent
work to run during each accumulate, and `mxBubble` should fall. If it does not fall, the
stall is the accumulator's own latency and no scheduling fixes it -- which is equally
worth knowing.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpu_gemm as G                                                    # noqa: E402
import fsa as F                                                          # noqa: E402
from fsa.tensor import MTile                                             # noqa: E402


def run_interleaved(config: str, A: np.ndarray, B: np.ndarray, rows: int, cols: int,
                    n_acc: int, func: int = G.GEMM_FUNC):
    """C = A @ B with `n_acc` output tiles accumulated concurrently."""
    m, k = A.shape
    _, n = B.shape
    s = G.Shape(m, n, k, rows, cols)
    s.check()
    tiles = [(mi, ni) for mi in range(s.mt) for ni in range(s.nt)]

    a_np = [G.stationary(A[mi * cols:(mi + 1) * cols, ki * rows:(ki + 1) * rows])
            for mi in range(s.mt) for ki in range(s.kt)]
    b_np = [np.ascontiguousarray(B[ki * rows:(ki + 1) * rows, ni * rows:(ni + 1) * rows].T)
            for ki in range(s.kt) for ni in range(s.nt)]
    oa, ob = G.ones_operands(rows, cols)

    @F.kernel
    def gemm(A_t: list[MTile], B_t: list[MTile], o_a: MTile, o_b: MTile) -> list[MTile]:
        out = [F.alloc_mem((rows, cols), F.fp32) for _ in tiles]
        ab = [F.alloc_spad((cols, rows)) for _ in range(2)]
        bb = [F.alloc_spad((rows, rows)) for _ in range(2)]
        accs = [F.alloc_accumulator((rows, cols)) for _ in range(n_acc)]
        sa = [F.Semaphore(id=0, n=2), F.Semaphore(id=1, n=2)]
        sb = [F.Semaphore(id=2, n=2), F.Semaphore(id=3, n=2)]
        sc = F.Semaphore(id=4, n=2)
        ss = F.Semaphore(id=5, n=2)

        if s.kt > 1:
            F.load_tile(o_a, ab[0], sa[0])
            F.mx_load_stationary(ab[0], sa[0])
            F.load_tile(o_b, bb[0], sb[0])
            G.mx_gemm(func, bb[0], accs[0], False, sb[0])
            G.mx_set_acc_scale(accs[0], ss)

        # Round-robin: for each k, step every tile in the group. Consecutive
        # instructions then touch DIFFERENT accumulators, so tile j's accumulate can
        # overlap tile j+1's compute.
        for g0 in range(0, len(tiles), n_acc):
            group = tiles[g0:g0 + n_acc]
            for ki in range(s.kt):
                for gi, (mi, ni) in enumerate(group):
                    buf = (ki * len(group) + gi) % 2
                    F.load_tile(A_t[mi * s.kt + ki], ab[buf], sa[buf])
                    F.mx_load_stationary(ab[buf], sa[buf])
                    F.load_tile(B_t[ki * s.nt + ni], bb[buf], sb[buf])
                    G.mx_gemm(func, bb[buf], accs[gi], ki > 0, sb[buf])
            F.fence(mx=True, dma=False, stop=False)
            for gi, (mi, ni) in enumerate(group):
                F.store_tile(accs[gi], out[g0 + gi], sc)
        F.fence(mx=True, dma=True, stop=True)
        return out

    eng = G.make_engine(config)
    outs = eng.execute(gemm([F.from_numpy(a) for a in a_np],
                            [F.from_numpy(b) for b in b_np],
                            F.from_numpy(G.stationary(oa)),
                            F.from_numpy(np.ascontiguousarray(ob.T))))
    if not isinstance(outs, list):
        outs = [outs]
    C = np.zeros((m, n), np.float32)
    for idx, (mi, ni) in enumerate(tiles):
        C[mi * cols:(mi + 1) * cols, ni * rows:(ni + 1) * rows] = F.to_numpy(outs[idx]).T
    return C


def counters() -> tuple[int, int, int]:
    o = G.sim_output()
    f = lambda c: int(re.findall(rf"{c}\s*=\s*(\d+)", o)[-1])
    return f("execTime"), f("mxActive"), f("mxBubble")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4DeepAccConfig")
    args = ap.parse_args()
    rows, cols = G.load_config(args.config)
    rng = np.random.default_rng(0)
    # 4 output tiles, 4 k-tiles each: enough independent work to interleave.
    m, n, k = 2 * cols, 2 * rows, 4 * rows
    A = rng.normal(size=(m, k)).astype(np.float16)
    B = rng.normal(size=(k, n)).astype(np.float16)
    ref = A.astype(np.float32) @ B.astype(np.float32)

    print(f"{args.config}: {rows}x{cols}   [{m}x{k}]@[{k}x{n}] "
          f"= {m//cols}x{n//rows} output tiles x {k//rows} k-tiles\n")
    print(f"{'mode':<16} {'execTime':>9} {'mxActive':>9} {'mxBubble':>9} "
          f"{'bubble%':>8} {'rel err':>11}")

    G.reset(); G.clear_assertions(); G.load_config(args.config)
    got = G.run(G.make_engine(args.config), A, B, rows, cols)
    et, ac, bu = counters()
    base = et
    print(f"{'serial (today)':<16} {et:>9} {ac:>9} {bu:>9} {100*bu/et:>7.1f}% "
          f"{float(np.abs(got-ref).max()/np.abs(ref).max()):>11.3e}")

    for n_acc in (2, 4):
        G.reset(); G.clear_assertions(); G.load_config(args.config)
        try:
            got = run_interleaved(args.config, A, B, rows, cols, n_acc)
            et, ac, bu = counters()
            rel = float(np.abs(got - ref).max() / np.abs(ref).max())
            print(f"{f'interleaved x{n_acc}':<16} {et:>9} {ac:>9} {bu:>9} "
                  f"{100*bu/et:>7.1f}% {rel:>11.3e}   {100*(et-base)/base:+.1f}%")
        except Exception as e:
            print(f"{f'interleaved x{n_acc}':<16} FAILED {type(e).__name__}: {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
