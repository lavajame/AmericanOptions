"""Benchmark analytic vs naive finite-difference sensitivities for a composite Merton+VG model.

Usage:
  C:/workspace/AmericanOptions/.venv/Scripts/python.exe tools/bench_sensitivities_merton_vg.py

This prints timings and max relative errors for dPrice/dParam computed via:
- Analytic pipeline: COSPricer(...).european_price(..., return_sensitivities=True, sens_method='analytic')
- Naive bump: central differences on the *full* European price for each parameter

Notes
-----
- This is a benchmark/diagnostic script (not a unit test). Timings are inherently noisy.
- We keep the option set small so it runs quickly.
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options.engine import COSPricer, CompositeLevyCHF  # noqa: E402


def build_model(*, S0: float, r: float, q: float, merton_params: dict, vg_params: dict) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        {},
        {
            "components": [
                {"type": "merton", "params": dict(merton_params)},
                {"type": "vg", "params": dict(vg_params)},
            ]
        },
    )


def bump_params(base_merton: dict, base_vg: dict, name: str, bump: float) -> tuple[dict, dict]:
    merton = copy.deepcopy(base_merton)
    vg = copy.deepcopy(base_vg)
    if name == "q":
        return merton, vg

    prefix, key = name.split(".", 1)
    if prefix.upper() == "MERTON":
        merton[key] = float(merton[key]) + float(bump)
    elif prefix.upper() == "VG":
        vg[key] = float(vg[key]) + float(bump)
    else:
        raise ValueError(f"Unexpected composite param prefix: {prefix}")
    return merton, vg


def _timeit(fn, *, repeat: int) -> tuple[float, object]:
    best = float("inf")
    out = None
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def main() -> int:
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.25
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    merton_params = {"sigma": 0.16, "lam": 0.8, "muJ": -0.08, "sigmaJ": 0.20}
    vg_params = {"theta": -0.20, "sigma": 0.14, "nu": 0.20}

    base = build_model(S0=S0, r=r, q=q, merton_params=merton_params, vg_params=vg_params)
    pricer = COSPricer(base, N=2048, L=12.0)

    sens_params = base.param_names() + ["q"]
    rel_step = 1e-4

    def run_analytic():
        return pricer.european_price(
            K,
            T,
            is_call=True,
            return_sensitivities=True,
            sens_method="analytic",
            sens_params=sens_params,
        )

    # Warm-up (import caches/JITs/BLAS)
    _ = run_analytic()

    t_ana, (price_ana, sens_ana) = _timeit(run_analytic, repeat=5)
    price_ana = np.asarray(price_ana, dtype=float)

    # Naive bumps: time the full loop as a block
    def run_bumps():
        out: dict[str, np.ndarray] = {}
        for name in sens_params:
            if name == "q":
                base_val = float(q)
            else:
                prefix, key = name.split(".", 1)
                base_val = float(merton_params[key] if prefix.upper() == "MERTON" else vg_params[key])
            h = rel_step * max(1.0, abs(base_val))
            if key in {"lam", "sigmaJ", "sigma", "nu"}:
                h = min(h, 0.25 * max(base_val, 1e-12))

            if name == "q":
                pr_p = COSPricer(build_model(S0=S0, r=r, q=q + h, merton_params=merton_params, vg_params=vg_params), N=2048, L=12.0)
                pr_m = COSPricer(build_model(S0=S0, r=r, q=q - h, merton_params=merton_params, vg_params=vg_params), N=2048, L=12.0)
            else:
                m_p, v_p = bump_params(merton_params, vg_params, name, +h)
                m_m, v_m = bump_params(merton_params, vg_params, name, -h)

                pr_p = COSPricer(build_model(S0=S0, r=r, q=q, merton_params=m_p, vg_params=v_p), N=2048, L=12.0)
                pr_m = COSPricer(build_model(S0=S0, r=r, q=q, merton_params=m_m, vg_params=v_m), N=2048, L=12.0)

            vp = np.asarray(pr_p.european_price(K, T, is_call=True), dtype=float)
            vm = np.asarray(pr_m.european_price(K, T, is_call=True), dtype=float)
            out[name] = (vp - vm) / (2.0 * h)
        return out

    # Warm-up bumps once
    _ = run_bumps()

    t_fd, sens_fd = _timeit(run_bumps, repeat=3)

    print("Composite sensitivities benchmark: Merton+VG")
    print(f"Strikes: {K.tolist()}  T={T}")
    print(f"Analytic (best of 5): {t_ana * 1e3:.2f} ms")
    print(f"Naive bumps (best of 3): {t_fd * 1e3:.2f} ms  (~{t_fd/t_ana:.1f}x slower)")
    print("-")

    header = f"{'param':<14} {'max|dP_ana|':>12} {'max|dP_fd|':>12} {'max rel err':>12}"
    print(header)
    print("-" * len(header))

    for name in sens_params:
        a = np.asarray(sens_ana[name], dtype=float)
        f = np.asarray(sens_fd[name], dtype=float)
        denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(f)))
        rel = float(np.max(np.abs(a - f) / denom))
        print(f"{name:<14} {np.max(np.abs(a)):12.4e} {np.max(np.abs(f)):12.4e} {rel:12.4e}")

    print("-")
    print(f"Price (analytic) min/max: {float(np.min(price_ana)):.6f} / {float(np.max(price_ana)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
