"""Compare classic vs Le Floc'h (2020) COS payoff coefficients.

This script focuses on European GBM where Black–Scholes is available as a reference.
It is meant to answer: does the Lefloch payoff-coeff tweak improve deep ITM/OTM stability
at fixed (N, L) vs our existing implementation?

Run:
  C:/workspace/AmericanOptions/.venv/Scripts/python.exe tools/compare_lefloch_payoff_coeffs.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.stats import norm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options import GBMCHF, VGCHF
from american_options.engine import COSPricer


def bs_call(*, S0: float, K: float | np.ndarray, r: float, q: float, vol: float, T: float) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)
    if vol <= 0:
        return np.maximum(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    sig_sqrt = vol * math.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrt
    d2 = d1 - sig_sqrt
    return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def main() -> None:
    S0 = 1.0
    r = 0.0
    q = 0.0
    divs = {}

    # Stress the short-maturity truncation sensitivity.
    T = 2.0 / 365.0

    # Wide strike sweep: deep ITM puts / deep OTM calls.
    K = np.concatenate([np.linspace(0.6, 1.4, 17), np.array([1.6, 1.8, 2.0, 2.2, 2.5])])

    N = 256
    L = 12.0

    # --- GBM: compare against Black–Scholes ---
    vol = 1.0
    model_gbm = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr_gbm = COSPricer(model_gbm, N=N, L=L)
    call_bs = bs_call(S0=S0, K=K, r=r, q=q, vol=vol, T=T)
    call_classic = pr_gbm.european_price(K, T, is_call=True, payoff_coeffs="classic")
    call_lefloch = pr_gbm.european_price(K, T, is_call=True, payoff_coeffs="lefloch")

    err_classic = np.abs(call_classic - call_bs)
    err_lefloch = np.abs(call_lefloch - call_bs)

    print("GBM COS payoff coeffs comparison (calls, BS reference)")
    print(f"  params: S0={S0} r={r} q={q} vol={vol} T={T:.8f}  N={N} L={L}")
    print(
        f"  classic: max_abs_err={float(np.max(err_classic)):.3e}  rmse={float(np.sqrt(np.mean(err_classic**2))):.3e}"
    )
    print(
        f"  lefloch: max_abs_err={float(np.max(err_lefloch)):.3e}  rmse={float(np.sqrt(np.mean(err_lefloch**2))):.3e}"
    )

    # --- VG: compare against a high-(N,L) COS reference ---
    # Reasonable parameters; goal is to detect any tail-stability improvement.
    theta, sigma, nu = -0.1, 0.2, 0.2
    model_vg = VGCHF(S0, r, q, divs, {"theta": theta, "sigma": sigma, "nu": nu})

    pr_vg = COSPricer(model_vg, N=N, L=L)
    call_vg_classic = pr_vg.european_price(K, T, is_call=True, payoff_coeffs="classic")
    call_vg_lefloch = pr_vg.european_price(K, T, is_call=True, payoff_coeffs="lefloch")

    pr_vg_ref = COSPricer(model_vg, N=8192, L=24.0)
    call_vg_ref = pr_vg_ref.european_price(K, T, is_call=True, payoff_coeffs="classic")

    vg_err_classic = np.abs(call_vg_classic - call_vg_ref)
    vg_err_lefloch = np.abs(call_vg_lefloch - call_vg_ref)

    print("\nVG COS payoff coeffs comparison (calls, high-(N,L) COS reference)")
    print(f"  params: theta={theta} sigma={sigma} nu={nu}  T={T:.8f}  N={N} L={L}")
    print(f"  classic: max_abs_diff={float(np.max(vg_err_classic)):.3e}  rmse={float(np.sqrt(np.mean(vg_err_classic**2))):.3e}")
    print(f"  lefloch: max_abs_diff={float(np.max(vg_err_lefloch)):.3e}  rmse={float(np.sqrt(np.mean(vg_err_lefloch**2))):.3e}")

    tail = np.argsort(K)[-6:]
    print("\nTail strikes (largest K):")
    for i in tail:
        print(
            f"  K={float(K[i]):.6g}  ref={float(call_vg_ref[i]):.6g}  classic={float(call_vg_classic[i]):.6g}  lefloch={float(call_vg_lefloch[i]):.6g}"
        )


if __name__ == "__main__":
    main()
