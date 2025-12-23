"""Plot implied-vol surfaces showing the impact of a discrete event jump.

Creates a 1x2 figure:
- Left: GBM (flat base vol) + event
- Right: VG (non-flat implied vol) + same event

Each panel shows:
- Filled contours: implied vol surface WITH event
- Black contour lines: implied vol surface WITHOUT event

No dividends are used.
"""

from __future__ import annotations

import os
import sys
from math import erf, exp, log, sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Allow running as: `python tools/plot_event_iv_surfaces.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from american_options import DiscreteEventJump, GBMCHF, VGCHF
from american_options.engine import COSPricer


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(*, S0: float, K: float, r: float, q: float, T: float, vol: float) -> float:
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if vol <= 0.0:
        fwd = S0 * exp((r - q) * T)
        return exp(-r * T) * max(fwd - K, 0.0)

    sig_sqrt = vol * sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrt
    d2 = d1 - sig_sqrt
    return S0 * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def implied_vol_call(*, target_price: float, S0: float, K: float, r: float, q: float, T: float) -> float:
    if not np.isfinite(target_price) or T <= 0.0:
        return float("nan")

    disc = exp(-r * T)
    fwd = S0 * exp((r - q) * T)

    intrinsic = disc * max(fwd - K, 0.0)
    upper = disc * fwd  # loose but safe bound

    if target_price < intrinsic - 1e-10 or target_price > upper + 1e-10:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S0=S0, K=K, r=r, q=q, T=T, vol=float(sig)) - float(target_price)

    lo, hi = 1e-6, 2.0
    flo, fhi = f(lo), f(hi)
    while np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) == np.sign(fhi) and hi < 8.0:
        hi = min(8.0, hi * 1.5)
        fhi = f(hi)

    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return float("nan")
    if np.sign(flo) == np.sign(fhi):
        return float("nan")

    sol = root_scalar(f, bracket=(lo, hi), method="brentq", xtol=1e-10)
    return float(sol.root) if sol.converged else float("nan")


def _iv_surface_from_cos(
    *,
    model,
    strikes: np.ndarray,
    maturities: np.ndarray,
    event: DiscreteEventJump | None,
    N: int,
    L: float,
) -> np.ndarray:
    pricer = COSPricer(model, N=N, L=L)

    iv = np.full((len(strikes), len(maturities)), np.nan, dtype=float)
    for j, T in enumerate(maturities):
        prices = pricer.european_price(strikes, float(T), is_call=True, event=event)
        for i, K in enumerate(strikes):
            iv[i, j] = implied_vol_call(
                target_price=float(prices[i]),
                S0=float(model.S0),
                K=float(K),
                r=float(model.r),
                q=float(model.q),
                T=float(T),
            )
    return iv


def main() -> None:
    os.makedirs("figs", exist_ok=True)

    # Common market inputs
    S0 = 100.0
    r = 0.02
    q = 0.0

    event_time = 0.05
    sigma = 0.15
    vg_theta = -0.14
    vg_nu = 0.20
    T = 0.5
    k_lim = np.exp(4.0 * sigma * np.sqrt(T))

    # Grid (strike x maturity)
    strikes = np.linspace(S0 / k_lim, S0 * k_lim, 41, dtype=float)
    maturities = np.linspace(0.01, T, 40, dtype=float)

    # Event definition (same for both models)
    event = DiscreteEventJump(
        time=event_time,
        p=0.5,
        u=float(np.log(1.04)),
        d=float(np.log(0.90)),
        ensure_martingale=True,
    )

    # Model 1: GBM (flat base vol)
    gbm = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": 0.20})

    # Model 2: VG (non-flat implied vol)
    vg = VGCHF(S0=S0, r=r, q=q, divs={}, params={"theta": vg_theta, "sigma": np.sqrt(sigma**2- vg_theta**2 * vg_nu), "nu": vg_nu})

    # COS settings
    N_gbm, L_gbm = 1024, 10.0
    N_vg, L_vg = 2048, 12.0

    # Compute surfaces with and without event
    iv_gbm_base = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=None, N=N_gbm, L=L_gbm)
    iv_gbm_evt = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=event, N=N_gbm, L=L_gbm)

    iv_vg_base = _iv_surface_from_cos(model=vg, strikes=strikes, maturities=maturities, event=None, N=N_vg, L=L_vg)
    iv_vg_evt = _iv_surface_from_cos(model=vg, strikes=strikes, maturities=maturities, event=event, N=N_vg, L=L_vg)

    # Plot: filled contours = with event, contour lines = no event
    T_grid, K_grid = np.meshgrid(maturities, strikes)

    all_evt = np.concatenate([iv_gbm_evt[np.isfinite(iv_gbm_evt)], iv_vg_evt[np.isfinite(iv_vg_evt)]])
    vmin = float(np.nanmin(all_evt))
    vmax = float(np.nanmax(all_evt))
    # Keep sane bounds
    vmin = max(0.01, vmin)
    vmax = min(3.0, vmax)

    levels = np.linspace(vmin, vmax, 14)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

    for ax, title, iv_evt, iv_base in [
        (axes[0], "GBM (flat vol) + event", iv_gbm_evt, iv_gbm_base),
        (axes[1], "VG (non-flat IV) + event", iv_vg_evt, iv_vg_base),
    ]:
        cf = ax.contourf(T_grid, K_grid, iv_evt, levels=levels, cmap="viridis")
        ax.contour(T_grid, K_grid, iv_base, levels=levels[::2], colors="k", linewidths=0.7, alpha=0.75)
        ax.axvline(float(event.time), color="w", lw=1.5, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Maturity T")

    axes[0].set_ylabel("Strike K")

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Implied vol (Black–Scholes)\nfill: with event, lines: no event")

    fig.suptitle(
        f"Implied-vol surfaces with a discrete event jump at t={event.time:.2f} (vertical line)\n"
        f"Event: p={event.p:.2f}, u={event.u:.3f}, d={event.d:.3f}, martingale_norm={event.ensure_martingale}",
        y=1.02,
    )

    fig.tight_layout()
    out = "figs/event_iv_surfaces_gbm_vs_vg.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
