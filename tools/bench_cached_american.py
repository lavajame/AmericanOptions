"""Micro-benchmark for cached COS American pricing.

Focus: same T, many K, with discrete dividends and optional scheduled event.

Run:
    python tools/bench_cached_american.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options import GBMCHF, DiscreteEventJump  # noqa: E402
from american_options.engine import COSPricer  # noqa: E402


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return float(time.perf_counter() - t0), out


def main() -> None:
    S0 = 100.0
    r = 0.01
    q = 0.0
    vol = 0.2
    T = 1.0

    # Cash divs
    divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}

    # Scheduled event (log-jumps)
    event = DiscreteEventJump(time=0.6, p=0.5, u=0.10, d=-0.08, ensure_martingale=True)

    K1 = np.array([100.0], dtype=float)
    K_multi = np.linspace(60.0, 140.0, 401, dtype=float)

    N = 512
    L = 10.0
    steps = 200

    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pricer = COSPricer(model, N=N, L=L)

    # Cold vs warm on the same pricer/context
    t_cold, _ = _time(lambda: pricer.american_price(K_multi, T, steps=steps, is_call=True, event=event))
    t_warm, _ = _time(lambda: pricer.american_price(K_multi, T, steps=steps, is_call=True, event=event))

    print(f"same pricer (cold): K=401 {t_cold:.4f}s")
    print(f"same pricer (warm): K=401 {t_warm:.4f}s")

    # Small call to ensure scalar-K still works
    t1, _ = _time(lambda: pricer.american_price(K1, T, steps=steps, is_call=True, event=event))
    print(f"same pricer: K=1 {t1:.4f}s")

    def run_once():
        m = GBMCHF(S0, r, q, divs, {"vol": vol})
        p = COSPricer(m, N=N, L=L)
        p.american_price(K1, T, steps=steps, is_call=True, event=event)
        p.american_price(K_multi, T, steps=steps, is_call=True, event=event)

    # Show shared cache benefit across fresh pricer instances (single run)
    dt_fresh, _ = _time(run_once)
    print(f"fresh pricer run (shared-cache warm): {dt_fresh:.4f}s")

    flags = np.zeros_like(K_multi, dtype=bool)
    flags[::2] = True
    mixed = pricer.american_price(K_multi, T, steps=100, is_call=flags, event=event)
    print(f"mixed is_call vector: shape={mixed.shape}, min={mixed.min():.6g}, max={mixed.max():.6g}")


if __name__ == "__main__":
    main()
