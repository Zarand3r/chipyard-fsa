"""Localize the GEMM dataflow fault with identity operands.

gemm_probe.py says GemmExecPlan does not compute A @ B, and gives output
bit-identical to the ATTN_VALUE probe. Pass/fail cannot distinguish "wrong plan
selected" from "right plan, wrong operand skew". Identity operands can:

    A = I  =>  C should equal B exactly. Any permutation of B's rows/columns in the
               output names the skew that is wrong.
    B = I  =>  C should equal A exactly. Same, for the stationary side.

Run from generators/fsa/python:
    uv run ../../rpu/experiments/gemm_identity.py --config RpuGemm4X4Fp16Config
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import fsa as F                                                       # noqa: E402
from gemm_probe_lib import gemm_one_tile, load_config, make_engine    # noqa: E402


def describe(name: str, got: np.ndarray, want: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print("got:\n", np.array2string(got, precision=3, suppress_small=True))
    print("want:\n", np.array2string(want, precision=3, suppress_small=True))
    if got.shape != want.shape:
        print(f"  shape mismatch: {got.shape} vs {want.shape}")
        return
    exact = np.allclose(got, want, atol=1e-2)
    print(f"  matches directly: {exact}")
    if exact:
        return
    # Name the permutation, if it is one.
    for label, cand in (
        ("row-reversed",        want[::-1, :]),
        ("col-reversed",        want[:, ::-1]),
        ("both-reversed",       want[::-1, ::-1]),
        ("transposed",          want.T if want.shape[0] == want.shape[1] else None),
    ):
        if cand is None or cand.shape != got.shape:
            continue
        if np.allclose(got, cand, atol=1e-2):
            print(f"  MATCHES {label} -- the skew convention is off by this")
            return
    print("  no simple permutation matches; not a pure ordering problem")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RpuGemm4X4Fp16Config")
    args = ap.parse_args()

    rows, cols = load_config(args.config)
    engine = make_engine(args.config)
    if engine is None:
        return 2
    print(f"config {args.config}: {rows} rows x {cols} cols")

    rng = np.random.default_rng(0)

    # A = I (cols x rows, so square only when cols == rows)
    if cols == rows:
        A_np = np.eye(cols, rows, dtype=np.float16)
        B_np = rng.normal(size=(rows, rows)).astype(np.float16)
        C = engine.execute(gemm_one_tile(
            F.from_numpy(A_np), F.from_numpy(np.ascontiguousarray(B_np.T))))
        describe("A = I, so C should be B", F.to_numpy(C).T, B_np.astype(np.float32))

    # B = I
    A_np = rng.normal(size=(cols, rows)).astype(np.float16)
    B_np = np.eye(rows, rows, dtype=np.float16)
    C = engine.execute(gemm_one_tile(
        F.from_numpy(A_np), F.from_numpy(np.ascontiguousarray(B_np.T))))
    describe("B = I, so C should be A", F.to_numpy(C).T, A_np.astype(np.float32))

    # A = I and B = I: output should be the identity. Anything else is pure dataflow.
    A_np = np.eye(cols, rows, dtype=np.float16)
    B_np = np.eye(rows, rows, dtype=np.float16)
    C = engine.execute(gemm_one_tile(
        F.from_numpy(A_np), F.from_numpy(np.ascontiguousarray(B_np.T))))
    describe("A = B = I, so C should be I", F.to_numpy(C).T,
             (A_np.astype(np.float32) @ B_np.astype(np.float32)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
