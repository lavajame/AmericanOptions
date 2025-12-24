"""Calibration-stability diagnostic for composite Lévy models.

Goal
----
When calibrating, parameters that move the implied-vol surface in very similar ways
are hard to identify jointly (ill-conditioned Jacobian). This script compares
candidate composite models by:

- Computing an IV vector over a small (T, K) grid.
- Estimating the Jacobian dIV/dθ via finite differences.
- Reporting:
  * max/mean pairwise cosine similarity between parameter sensitivity vectors
  * Jacobian condition number (SVD)

This is a *local* diagnostic around a chosen baseline.

Run
---
python tools/diagnose_combo_calibration_stability.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Sequence

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt

# Allow running as a plain script without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options import CompositeLevyCHF
from american_options.engine import COSPricer


def bs_call(*, S0: float, K: float, r: float, q: float, vol: float, T: float) -> float:
    if T <= 0.0:
        return max(S0 - K, 0.0)
    sigma = max(float(vol), 1e-16)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_implied_vol_call(*, price: float, S0: float, K: float, r: float, q: float, T: float) -> float:
    price = float(price)
    if not np.isfinite(price) or T <= 0.0:
        return float("nan")

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    intrinsic = max(S0 * disc_q - K * disc_r, 0.0)
    upper = S0 * disc_q

    if price <= intrinsic + 1e-14:
        return 0.0
    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call(S0=S0, K=K, r=r, q=q, vol=sig, T=T) - price

    lo, hi = 1e-8, 2.0
    flo, fhi = f(lo), f(hi)
    while np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) == np.sign(fhi) and hi < 8.0:
        hi *= 1.5
        fhi = f(hi)

    if not (np.isfinite(flo) and np.isfinite(fhi)) or np.sign(flo) == np.sign(fhi):
        return float("nan")

    sol = root_scalar(f, bracket=(lo, hi), method="brentq")
    return float(sol.root) if sol.converged else float("nan")


def _cos_iv_vector(
    *,
    model_factory: Callable[[float], object],
    maturities: np.ndarray,
    strikes_by_T: Dict[float, np.ndarray],
    N: int = 256,
    L: float = 10.0,
    r: float,
    q: float,
    S0: float,
) -> np.ndarray:
    """Flatten IVs in maturity-major order."""
    out: List[float] = []
    for T in maturities:
        model = model_factory(float(T))
        pr = COSPricer(model, N=N, L=L)
        Ks = strikes_by_T[float(T)]
        prices = pr.european_price(Ks, float(T), is_call=True)
        for K, p in zip(Ks, prices):
            out.append(bs_implied_vol_call(price=float(p), S0=S0, K=float(K), r=float(r), q=float(q), T=float(T)))
    return np.asarray(out, dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _cosine_similarity_matrix(J: np.ndarray, param_index: Sequence[tuple[str, str]], *, abs_value: bool) -> pd.DataFrame:
    """Pairwise cosine similarity between Jacobian columns (parameter sensitivity directions).

    Parameters
    ----------
    J:
        Jacobian (n_obs, n_params)
    param_index:
        Sequence of (component, parameter_name) pairs, in the same order as columns of J.
    """
    n = len(param_index)
    mat = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = _cosine_similarity(J[:, i], J[:, j])
            if not np.isfinite(s):
                s = float("nan")
            mat[i, j] = abs(s) if abs_value and np.isfinite(s) else s
    idx = pd.MultiIndex.from_tuples(list(param_index), names=["component", "param"])
    return pd.DataFrame(mat, index=idx, columns=idx)


def _slug(s: str) -> str:
    s = s.strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "+", "-", "_"}:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _save_heatmap(df: pd.DataFrame, *, title: str, out_png: str, cmap: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(0.9 * df.shape[1] + 3.0, 0.8 * df.shape[0] + 2.5))
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))

    def _fmt_idx(x) -> str:
        # MultiIndex tuple -> label
        if isinstance(x, tuple) and len(x) == 2:
            return f"{x[0]}:{x[1]}"
        return str(x)

    ax.set_xticklabels([_fmt_idx(c) for c in df.columns], rotation=45, ha="right")
    ax.set_yticklabels([_fmt_idx(r) for r in df.index])
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@dataclass(frozen=True)
class StabilityReport:
    name: str
    param_names: List[str]
    max_abs_cos: float
    mean_abs_cos: float
    cond: float


def _report_from_jacobian(name: str, param_names: List[str], J: np.ndarray) -> StabilityReport:
    # Column-normalize for cosine comparisons
    cols = [J[:, j] for j in range(J.shape[1])]
    sims = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s = _cosine_similarity(cols[i], cols[j])
            if np.isfinite(s):
                sims.append(abs(s))

    svals = np.linalg.svd(J, compute_uv=False)
    svals = svals[np.isfinite(svals)]
    if svals.size == 0 or float(np.min(svals)) <= 0.0:
        cond = float("inf")
    else:
        cond = float(np.max(svals) / np.min(svals))

    return StabilityReport(
        name=name,
        param_names=param_names,
        max_abs_cos=float(np.max(sims)) if sims else float("nan"),
        mean_abs_cos=float(np.mean(sims)) if sims else float("nan"),
        cond=cond,
    )


def _make_strikes(*, S0: float, r: float, T: float, vol_ref: float, nK: int = 21, width_sigmas: float = 2.5) -> np.ndarray:
    F = S0 * math.exp(r * T)
    ks = np.linspace(-width_sigmas, width_sigmas, nK)
    # log-moneyness grid around forward
    K = F * np.exp(ks * vol_ref * math.sqrt(T))
    return np.asarray(K, dtype=float)


def main() -> None:
    # Baseline grid (kept small; this runs many COS pricings)
    S0, r, q = 100.0, 0.02, 0.0
    maturities = np.asarray([0.05, 0.25, 1.0], dtype=float)
    vol_ref = 0.20

    strikes_by_T = {float(T): _make_strikes(S0=S0, r=r, T=float(T), vol_ref=vol_ref) for T in maturities}

    # Use the same total variance target across models at each T.
    def target_c2(T: float) -> float:
        return float((vol_ref ** 2) * T)

    def merton_params_for_c2(T: float, c2: float, lam: float, muJ: float, sigmaJ: float) -> Dict[str, float]:
        jump_c2 = lam * T * (muJ * muJ + sigmaJ * sigmaJ)
        resid = c2 - jump_c2
        if resid < -1e-14:
            raise RuntimeError("Merton jump variance exceeds target")
        vol = math.sqrt(max(resid / T, 0.0))
        return {"vol": float(vol), "lam": float(lam), "muJ": float(muJ), "sigmaJ": float(sigmaJ)}

    def kou_params_for_c2(T: float, c2: float, lam: float, p: float, eta1: float, eta2: float) -> Dict[str, float]:
        jump_c2 = lam * T * (p * 2.0 / (eta1 * eta1) + (1.0 - p) * 2.0 / (eta2 * eta2))
        resid = c2 - jump_c2
        if resid < -1e-14:
            raise RuntimeError("Kou jump variance exceeds target")
        vol = math.sqrt(max(resid / T, 0.0))
        return {"vol": float(vol), "lam": float(lam), "p": float(p), "eta1": float(eta1), "eta2": float(eta2)}

    def vg_params_for_c2(T: float, c2: float, theta: float, nu: float) -> Dict[str, float]:
        per_t = c2 / T
        resid = per_t - (theta * theta) * nu
        if resid < -1e-14:
            raise RuntimeError("VG theta^2*nu exceeds target")
        sigma = math.sqrt(max(resid, 0.0))
        return {"theta": float(theta), "sigma": float(sigma), "nu": float(nu)}

    # Candidate composite models (both with 50/50 c2 split).
    def factory_merton_vg(shape: Dict[str, float]) -> Callable[[float], object]:
        def f(T: float):
            c2 = target_c2(T)
            c2h = 0.5 * c2
            m = merton_params_for_c2(T, c2h, lam=shape["lam"], muJ=shape["muJ"], sigmaJ=shape["sigmaJ"])
            v = vg_params_for_c2(T, c2h, theta=shape["theta"], nu=shape["nu"])
            return CompositeLevyCHF(S0, r, q, {}, {"components": [("merton", m), ("vg", v)]})
        return f

    def factory_kou_vg(shape: Dict[str, float]) -> Callable[[float], object]:
        def f(T: float):
            c2 = target_c2(T)
            c2h = 0.5 * c2
            k = kou_params_for_c2(T, c2h, lam=shape["lam"], p=shape["p"], eta1=shape["eta1"], eta2=shape["eta2"])
            v = vg_params_for_c2(T, c2h, theta=shape["theta"], nu=shape["nu"])
            return CompositeLevyCHF(S0, r, q, {}, {"components": [("kou", k), ("vg", v)]})
        return f

    # Baselines (shape params only; variance parameters are solved each T)
    base_mv = {"lam": 0.8, "muJ": -0.08, "sigmaJ": 0.08, "theta": -0.30, "nu": 0.15}
    base_kv = {"lam": 0.6, "p": 0.25, "eta1": 30.0, "eta2": 8.0, "theta": -0.30, "nu": 0.15}

    candidates: List[
        Tuple[
            str,
            Dict[str, float],
            Callable[[Dict[str, float]], Callable[[float], object]],
            List[str],
            List[tuple[str, str]],
        ]
    ] = [
        (
            "Merton+VG (shape-only, 50/50 c2)",
            base_mv,
            factory_merton_vg,
            ["lam", "muJ", "sigmaJ", "theta", "nu"],
            [("Merton", "lam"), ("Merton", "muJ"), ("Merton", "sigmaJ"), ("VG", "theta"), ("VG", "nu")],
        ),
        (
            "Kou+VG (shape-only, 50/50 c2)",
            base_kv,
            factory_kou_vg,
            ["lam", "p", "eta1", "eta2", "theta", "nu"],
            [("Kou", "lam"), ("Kou", "p"), ("Kou", "eta1"), ("Kou", "eta2"), ("VG", "theta"), ("VG", "nu")],
        ),
    ]

    reports: List[StabilityReport] = []
    jacobians: Dict[str, Tuple[List[str], List[tuple[str, str]], np.ndarray]] = {}

    for name, base, factory_builder, params, param_index in candidates:
        model_factory = factory_builder(base)
        iv0 = _cos_iv_vector(
            model_factory=model_factory,
            maturities=maturities,
            strikes_by_T=strikes_by_T,
            N=256,
            L=10.0,
            r=r,
            q=q,
            S0=S0,
        )

        # Finite differences in *shape* parameters, keeping the c2 split enforced.
        J = np.zeros((iv0.size, len(params)), dtype=float)
        for j, pname in enumerate(params):
            x0 = float(base[pname])
            # heuristic step sizes
            if pname in {"eta1", "eta2"}:
                h = 1e-2 * max(abs(x0), 1.0)
            elif pname in {"lam", "nu"}:
                h = 1e-3 * max(abs(x0), 1e-3)
            elif pname == "p":
                h = 5e-4
            else:
                h = 1e-3 * max(abs(x0), 1e-3)

            bumped = dict(base)
            bumped[pname] = x0 + h

            iv1 = _cos_iv_vector(
                model_factory=factory_builder(bumped),
                maturities=maturities,
                strikes_by_T=strikes_by_T,
                N=256,
                L=10.0,
                r=r,
                q=q,
                S0=S0,
            )

            dv = (iv1 - iv0) / h
            # Replace NaNs (from extreme moneyness) with zeros so they don't poison metrics.
            dv = np.where(np.isfinite(dv), dv, 0.0)
            J[:, j] = dv

        reports.append(_report_from_jacobian(name, params, J))
        jacobians[name] = (params, param_index, J)

    # Rank: lower max_abs_cos first, then lower condition number.
    reports.sort(key=lambda rep: (rep.max_abs_cos if np.isfinite(rep.max_abs_cos) else 1e9, rep.cond))

    print("\n=== Composite calibration-stability (local) ===")
    print(f"Grid: maturities={list(maturities)}, strikes per T={len(next(iter(strikes_by_T.values())))}")
    print("Metric interpretation: lower is better (less collinearity / better-conditioned).\n")

    for rep in reports:
        print(f"{rep.name}")
        print(f"  params: {rep.param_names}")
        print(f"  max |cos| between sensitivities: {rep.max_abs_cos:.4f}")
        print(f"  mean|cos| between sensitivities: {rep.mean_abs_cos:.4f}")
        print(f"  Jacobian cond (SVD):            {rep.cond:.2e}")
        print("")

        # Pairwise collinearity matrices (signed + absolute)
        params, param_index, J = jacobians[rep.name]
        df_signed = _cosine_similarity_matrix(J, param_index, abs_value=False)
        df_abs = _cosine_similarity_matrix(J, param_index, abs_value=True)

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_rows", 50)

        print("  Pairwise cosine similarity (signed):")
        print(df_signed.round(3).to_string())
        print("\n  Pairwise cosine similarity (absolute):")
        print(df_abs.round(3).to_string())
        print("")

        # Save CSVs + heatmaps
        os.makedirs(os.path.join(_ROOT, "figs"), exist_ok=True)
        slug = _slug(rep.name)
        out_csv_signed = os.path.join(_ROOT, "figs", f"calib_collinearity_signed_{slug}.csv")
        out_csv_abs = os.path.join(_ROOT, "figs", f"calib_collinearity_abs_{slug}.csv")
        df_signed.to_csv(out_csv_signed, index=True)
        df_abs.to_csv(out_csv_abs, index=True)

        out_png_signed = os.path.join(_ROOT, "figs", f"calib_collinearity_signed_{slug}.png")
        out_png_abs = os.path.join(_ROOT, "figs", f"calib_collinearity_abs_{slug}.png")
        _save_heatmap(
            df_signed,
            title=f"Pairwise cosine similarity (signed)\n{rep.name}",
            out_png=out_png_signed,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        _save_heatmap(
            df_abs,
            title=f"Pairwise cosine similarity (absolute)\n{rep.name}",
            out_png=out_png_abs,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )


if __name__ == "__main__":
    main()
