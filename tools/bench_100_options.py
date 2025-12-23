"""Benchmark: COS American pricing for 100 options (same T, many K).

This is designed to compare two code versions side-by-side by pointing at a
repo root (worktree) to import from.

Run examples:
    python tools/bench_100_options.py --repo-root .
    python tools/bench_100_options.py --repo-root _bench_baseline

Notes
-----
- Uses discrete cash dividends (converted internally) and an optional scheduled event.
- Uses mixed calls/puts (alternating) to mimic a realistic option chain.
- If the target version doesn't support vector `is_call`, it falls back to two calls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _insert_repo_root(repo_root: str) -> str:
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise SystemExit(f"repo root not found: {root}")
    # Ensure this import path wins.
    sys.path.insert(0, root)
    return root


def _time_once(fn):
    t0 = time.perf_counter()
    out = fn()
    return float(time.perf_counter() - t0), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--with-event", action="store_true", help="Include scheduled event jump")
    args = ap.parse_args()

    root = _insert_repo_root(args.repo_root)

    # Import after setting sys.path.
    from american_options import GBMCHF  # type: ignore
    from american_options.engine import COSPricer  # type: ignore

    DiscreteEventJump = None
    if args.with_event:
        from american_options import DiscreteEventJump as _DEJ  # type: ignore

        DiscreteEventJump = _DEJ

    S0 = 100.0
    r = 0.01
    q = 0.00
    vol = 0.20
    T = 1.00

    # Cash dividends
    divs = {
        0.25: (2.0, 0.0),
        0.75: (2.0, 0.0),
    }

    event = None
    if DiscreteEventJump is not None:
        # log-jumps (u>0, d<0)
        event = DiscreteEventJump(time=0.60, p=0.50, u=0.10, d=-0.08, ensure_martingale=True)

    # 100 options
    K = np.linspace(70.0, 130.0, 100, dtype=float)
    is_call = np.zeros_like(K, dtype=bool)
    is_call[::2] = True

    def price_batch():
        model = GBMCHF(S0, r, q, divs, {"vol": vol})
        pricer = COSPricer(model, N=int(args.N), L=float(args.L))

        try:
            # Newer versions: is_call can be a vector
            return np.asarray(
                pricer.american_price(K, T, steps=int(args.steps), is_call=is_call, use_softmax=False, event=event),
                dtype=float,
            )
        except Exception:
            # Fallback: two calls
            out = np.empty_like(K, dtype=float)
            if np.any(~is_call):
                out[~is_call] = np.asarray(
                    pricer.american_price(K[~is_call], T, steps=int(args.steps), is_call=False, use_softmax=False, event=event),
                    dtype=float,
                )
            if np.any(is_call):
                out[is_call] = np.asarray(
                    pricer.american_price(K[is_call], T, steps=int(args.steps), is_call=True, use_softmax=False, event=event),
                    dtype=float,
                )
            return out

    # Warm-up
    _ = price_batch()

    times = []
    for _ in range(int(args.reps)):
        dt, prices = _time_once(price_batch)
        times.append(dt)

    times = np.asarray(times, dtype=float)

    print(f"repo_root: {root}")
    print(f"with_event: {bool(args.with_event)}")
    print(f"N={args.N} L={args.L} steps={args.steps} n_opts={K.size}")
    print(f"timing_s: reps={args.reps}  avg={times.mean():.6f}  min={times.min():.6f}  max={times.max():.6f}")
    print(f"price_sanity: min={float(np.min(prices)):.6g} max={float(np.max(prices)):.6g}")


if __name__ == "__main__":
    main()
