"""Plot IV slices using calibrated Lévy model params.

Reads a calibration JSON produced by `tools/analyze_skew_persistence.py --calibrate-to-avgshape`
and plots implied-vol slices (European calls via COS) on a 1x3 grid.

Usage:
  python tools/plot_iv_slices_calibrated.py \
    --calib-json figs/skew_persistence_by_model_calib_T025_calib_params.json \
    --out figs/iv_slices_calib_T025.png \
    --Ts 0.05,0.25,1.0

Notes
-----
- x-axis is strike K; strikes are generated from a fixed log-moneyness window k=ln(K/F).
- Uses the same BS IV inversion logic as other tools scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import erf, exp, log, sqrt

import numpy as np
from scipy.optimize import root_scalar

# Allow running as: `python tools/plot_iv_slices_calibrated.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from american_options import CGMYCHF, KouCHF, MertonCHF, NIGCHF, VGCHF
from american_options.engine import COSPricer


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs_call_price(*, S0: float, K: float, r: float, q: float, T: float, vol: float) -> float:
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if vol <= 0.0:
        fwd = S0 * exp((r - q) * T)
        return exp(-r * T) * max(fwd - K, 0.0)

    sig_sqrt = vol * sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrt
    d2 = d1 - sig_sqrt
    return S0 * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def _implied_vol_call(*, target_price: float, S0: float, K: float, r: float, q: float, T: float) -> float:
    target_price = float(target_price)
    if not np.isfinite(target_price) or T <= 0.0:
        return float("nan")

    disc = exp(-r * T)
    fwd = S0 * exp((r - q) * T)
    intrinsic = disc * max(fwd - K, 0.0)
    upper = disc * fwd

    if target_price <= intrinsic + 1e-14:
        return 0.0
    if target_price < intrinsic - 1e-10 or target_price > upper + 1e-10:
        return float("nan")

    def f(sig: float) -> float:
        return _bs_call_price(S0=S0, K=K, r=r, q=q, T=T, vol=float(sig)) - target_price

    lo, hi = 1e-8, 2.0
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


def _parse_Ts(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]


def _make_grid(*, x_min: float, x_max: float, n: int) -> np.ndarray:
    n = int(n)
    if n < 5:
        raise ValueError("n must be >= 5")
    if n % 2 == 0:
        n += 1
    x = np.linspace(float(x_min), float(x_max), n, dtype=float)
    x[n // 2] = 0.0
    return x


def _build_model(name: str, *, S0: float, r: float, q: float, params: dict[str, float]):
    name_u = str(name).strip().upper()
    if name_u == "VG":
        return VGCHF(S0=S0, r=r, q=q, divs={}, params={"sigma": float(params["sigma"]), "theta": float(params["theta"]), "nu": float(params["nu"])})
    if name_u == "MERTON":
        return MertonCHF(
            S0=S0,
            r=r,
            q=q,
            divs={},
            params={"vol": float(params["vol"]), "lam": float(params["lam"]), "muJ": float(params["muJ"]), "sigmaJ": float(params["sigmaJ"])},
        )
    if name_u == "KOU":
        return KouCHF(
            S0=S0,
            r=r,
            q=q,
            divs={},
            params={"vol": float(params["vol"]), "lam": float(params["lam"]), "p": float(params["p"]), "eta1": float(params["eta1"]), "eta2": float(params["eta2"])},
        )
    if name_u == "CGMY":
        return CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(params["C"]), "G": float(params["G"]), "M": float(params["M"]), "Y": float(params["Y"])})
    if name_u == "NIG":
        return NIGCHF(S0=S0, r=r, q=q, divs={}, params={"alpha": float(params["alpha"]), "beta": float(params["beta"]), "delta": float(params["delta"]), "mu": float(params.get("mu", 0.0))})
    raise ValueError(f"Unsupported model: {name}")


def _iv_slice(*, model, T: float, k_grid: np.ndarray, N: int, L: float) -> np.ndarray:
    pricer = COSPricer(model, N=int(N), L=float(L))
    F = float(model.S0) * float(np.exp((float(model.r) - float(model.q)) * float(T)))
    strikes = F * np.exp(k_grid)
    prices = pricer.european_price(strikes, float(T), is_call=True)
    iv = np.full_like(strikes, np.nan, dtype=float)
    for i, K in enumerate(strikes):
        iv[i] = _implied_vol_call(target_price=float(prices[i]), S0=float(model.S0), K=float(K), r=float(model.r), q=float(model.q), T=float(T))
    return iv


def _sigma_from_avg_c2(models: dict[str, object], *, T: float) -> float:
    """Approx sigma via sigma(T) = sqrt( mean_i c2_i(T) / T )."""
    c2s = []
    for m in models.values():
        # All CF models in this repo expose _var2(T) (variance of log S_T).
        c2 = float(m._var2(float(T)))
        if np.isfinite(c2) and c2 >= 0.0:
            c2s.append(c2)
    if not c2s:
        return float("nan")
    c2_bar = float(np.mean(c2s))
    return float(np.sqrt(max(c2_bar / max(float(T), 1e-16), 0.0)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot 1x3 implied-vol slices using calibrated params.")
    p.add_argument("--calib-json", type=str, required=True)
    p.add_argument("--out", type=str, default="figs/iv_slices_calibrated.png")
    p.add_argument("--Ts", type=str, default="0.05,0.25,1.0")
    p.add_argument("--use-z", action="store_true", help="Plot vs z = ln(K/F)/(sigma*sqrt(T)) instead of k")
    p.add_argument("--z-min", type=float, default=-3.0)
    p.add_argument("--z-max", type=float, default=3.0)
    p.add_argument("--nz", type=int, default=61)
    p.add_argument("--k-min", type=float, default=-0.60, help="Used only when --use-z is not set")
    p.add_argument("--k-max", type=float, default=0.60, help="Used only when --use-z is not set")
    p.add_argument("--nk", type=int, default=61, help="Used only when --use-z is not set")
    p.add_argument("--N", type=int, default=2**12)
    p.add_argument("--L", type=float, default=12.0)
    args = p.parse_args(argv)

    with open(str(args.calib_json), "r", encoding="utf-8") as f:
        payload = json.load(f)

    S0 = float(payload.get("S0", 100.0))
    r = float(payload.get("r", 0.02))
    q = float(payload.get("q", 0.0))
    model_params: dict[str, dict[str, float]] = payload["models"]

    # Normalize keys to uppercase
    model_params_u = {str(k).upper(): {kk: float(vv) for kk, vv in v.items()} for k, v in model_params.items()}

    names = ["VG", "MERTON", "KOU", "CGMY", "NIG"]
    models = {nm: _build_model(nm, S0=S0, r=r, q=q, params=model_params_u[nm]) for nm in names if nm in model_params_u}

    Ts = _parse_Ts(str(args.Ts))
    if len(Ts) != 3:
        raise ValueError("--Ts must contain exactly 3 maturities")

    use_z = bool(args.use_z)
    if use_z:
        z_grid = _make_grid(x_min=float(args.z_min), x_max=float(args.z_max), n=int(args.nz))
    else:
        k_grid = _make_grid(x_min=float(args.k_min), x_max=float(args.k_max), n=int(args.nk))

    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)

    for ax, T in zip(axes, Ts):
        sigma_bar = _sigma_from_avg_c2(models, T=float(T)) if use_z else float("nan")
        if use_z and (not np.isfinite(sigma_bar) or sigma_bar <= 0.0):
            raise RuntimeError(f"Failed to compute sigma_bar for T={T}")

        if use_z:
            k_grid_eff = z_grid * float(sigma_bar) * float(np.sqrt(float(T)))
        else:
            k_grid_eff = k_grid

        for nm in names:
            iv = _iv_slice(model=models[nm], T=float(T), k_grid=k_grid_eff, N=int(args.N), L=float(args.L))
            x = z_grid if use_z else k_grid
            ax.plot(x, iv, marker="o", linewidth=1.6, markersize=3.5, label=nm)

        title = f"IV slice @ T={T:.2f}y"
        if use_z:
            title += f"\n$\\sigma_{{ref}}$={sigma_bar:.3f}"
        ax.set_title(title)
        ax.set_xlabel("z = ln(K/F) / (sigma_ref * sqrt(T))" if use_z else "k = ln(K/F)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Implied vol (BS, call)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)
    fig.suptitle("Calibrated Lévy models: IV slices", y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))

    fig.savefig(str(args.out), dpi=160)
    plt.close(fig)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
