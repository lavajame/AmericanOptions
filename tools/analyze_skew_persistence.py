"""Compare skew/smile persistence vs maturity across non-flat Lévy models.

This script prices *European calls* via COS, inverts to Black–Scholes implied vols,
then computes simple smile metrics across maturities:

- ATM IV
- ATM skew: d(IV)/d(log-moneyness) at k=0
- ATM curvature: d^2(IV)/d(log-moneyness)^2 at k=0
- Smile RMS over |k|<=k_rms

Finally it ranks models by how persistent the skew is as maturity increases.

Usage:
  python tools/analyze_skew_persistence.py

Outputs:
  - figs/skew_persistence_by_model.csv

Notes
-----
- All models are normalized to have the *same variance rate* Var[log S_T]/T = target_vol^2.
  That makes the comparison about *shape persistence*, not overall level.
- Default "shape" parameters are taken from `plot_diagnostics.plot_levy_vs_equiv_gbm()`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
from scipy.optimize import least_squares, root_scalar
from scipy.special import gamma as sp_gamma

# Allow running as: `python tools/analyze_skew_persistence.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from american_options import CGMYCHF, GBMCHF, KouCHF, MertonCHF, NIGCHF, VGCHF
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: object


@dataclass(frozen=True)
class CalibResult:
    name: str
    params: dict[str, float]
    cost: float


def _variance_rate_target(target_vol: float) -> float:
    return float(target_vol) ** 2


def _build_models(*, S0: float, r: float, q: float, target_vol: float) -> list[ModelSpec]:
    """Build models with shared variance rate and equity-like left skew shapes."""

    var_rate = _variance_rate_target(target_vol)

    # Shape parameters (from plot_diagnostics.plot_levy_vs_equiv_gbm)
    merton_lam = 0.8
    merton_muJ = -0.08
    merton_sigmaJ = 0.08
    merton_jump_var_rate = float(merton_lam) * (float(merton_muJ) ** 2 + float(merton_sigmaJ) ** 2)
    merton_vol_sq = max(0.0, var_rate - merton_jump_var_rate)
    merton_vol = float(np.sqrt(merton_vol_sq))

    kou_lam = 0.6
    kou_p = 0.25
    kou_eta1 = 30.0
    kou_eta2 = 8.0
    kou_jump_var_rate = float(kou_lam) * (
        float(kou_p) * 2.0 / (float(kou_eta1) ** 2) + (1.0 - float(kou_p)) * 2.0 / (float(kou_eta2) ** 2)
    )
    kou_vol_sq = max(0.0, var_rate - kou_jump_var_rate)
    kou_vol = float(np.sqrt(kou_vol_sq))

    vg_theta = -0.30
    vg_nu = 0.25
    vg_sigma_sq = max(0.0, var_rate - (float(vg_theta) ** 2) * float(vg_nu))
    vg_sigma = float(np.sqrt(vg_sigma_sq))

    # CGMY: choose (G, M, Y) for asymmetry; tune C to match variance rate.
    cgmy_G = 1.5
    cgmy_M = 10.0
    cgmy_Y = 1.3
    cgmy_var_rate_per_C = float(sp_gamma(2.0 - cgmy_Y) * (cgmy_M ** (cgmy_Y - 2.0) + cgmy_G ** (cgmy_Y - 2.0)))
    cgmy_C = float(var_rate / max(cgmy_var_rate_per_C, 1e-300))

    # NIG: choose (alpha, beta) for skew; tune delta to match variance rate.
    nig_alpha = 15.0
    nig_beta = -5.0
    nig_gamma = float(np.sqrt(max((nig_alpha ** 2) - (nig_beta ** 2), 1e-16)))
    nig_var_rate_per_delta = float((nig_alpha ** 2) / (nig_gamma ** 3))
    nig_delta = float(var_rate / max(nig_var_rate_per_delta, 1e-300))
    nig_mu = 0.0

    models: list[ModelSpec] = [
        ModelSpec("GBM", GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": float(target_vol)})),
        ModelSpec(
            "Merton",
            MertonCHF(
                S0=S0,
                r=r,
                q=q,
                divs={},
                params={"vol": merton_vol, "lam": float(merton_lam), "muJ": float(merton_muJ), "sigmaJ": float(merton_sigmaJ)},
            ),
        ),
        ModelSpec(
            "Kou",
            KouCHF(
                S0=S0,
                r=r,
                q=q,
                divs={},
                params={"vol": kou_vol, "lam": float(kou_lam), "p": float(kou_p), "eta1": float(kou_eta1), "eta2": float(kou_eta2)},
            ),
        ),
        ModelSpec(
            "VG",
            VGCHF(
                S0=S0,
                r=r,
                q=q,
                divs={},
                params={"sigma": vg_sigma, "theta": float(vg_theta), "nu": float(vg_nu)},
            ),
        ),
        ModelSpec(
            "CGMY",
            CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": cgmy_C, "G": float(cgmy_G), "M": float(cgmy_M), "Y": float(cgmy_Y)}),
        ),
        ModelSpec(
            "NIG",
            NIGCHF(S0=S0, r=r, q=q, divs={}, params={"alpha": float(nig_alpha), "beta": float(nig_beta), "delta": float(nig_delta), "mu": float(nig_mu)}),
        ),
    ]

    return models


def _weight_k(k: np.ndarray, *, mode: str, k_scale: float = 0.15) -> np.ndarray:
    """Weights for calibration residuals.

    Modes
    -----
    - 'atm'   : downweight wings (stabilizes fit but can miss tails)
    - 'flat'  : equal weight (balanced)
    - 'wings' : upweight wings (forces tail fit)
    """
    k = np.asarray(k, dtype=float)
    mode = str(mode).lower().strip()
    if mode == "flat":
        return np.ones_like(k, dtype=float)
    if mode == "wings":
        # Smoothly increases with |k|. Normalized to 1 at ATM.
        return 1.0 + (np.abs(k) / float(k_scale)) ** 2
    # default: ATM-heavy
    return 1.0 / (1.0 + (np.abs(k) / float(k_scale)) ** 2)


def _model_from_params(*, name: str, S0: float, r: float, q: float, params: dict[str, float]):
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
            params={
                "vol": float(params["vol"]),
                "lam": float(params["lam"]),
                "p": float(params["p"]),
                "eta1": float(params["eta1"]),
                "eta2": float(params["eta2"]),
            },
        )
    if name_u == "CGMY":
        return CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(params["C"]), "G": float(params["G"]), "M": float(params["M"]), "Y": float(params["Y"])})
    if name_u == "NIG":
        return NIGCHF(
            S0=S0,
            r=r,
            q=q,
            divs={},
            params={"alpha": float(params["alpha"]), "beta": float(params["beta"]), "delta": float(params["delta"]), "mu": float(params.get("mu", 0.0))},
        )
    raise ValueError(f"Unknown model name for calibration: {name}")


def _display_model_name(name: str) -> str:
    name_u = str(name).strip().upper()
    if name_u == "CGMY":
        return "CGMY"
    if name_u == "VG":
        return "VG"
    if name_u == "MERTON":
        return "Merton"
    if name_u == "KOU":
        return "Kou"
    if name_u == "NIG":
        return "NIG"
    return name_u.title()


def _average_target_iv(
    *,
    model_specs: list[ModelSpec],
    model_names: list[str],
    k_grid: np.ndarray,
    T: float,
    N: int,
    L: float,
) -> np.ndarray:
    """Compute target IV(k) as the average of the listed models' IV slices."""
    slices = []
    for ms in model_specs:
        if ms.name not in model_names:
            continue
        _k, iv = _iv_slice_for_model(model_spec=ms, k_grid=k_grid, T=T, N=N, L=L)
        slices.append(iv)
    if not slices:
        raise RuntimeError("No models selected to build target")

    arr = np.vstack(slices)
    # average ignoring NaNs
    with np.errstate(invalid="ignore"):
        target = np.nanmean(arr, axis=0)
    return target


def _calibrate_models_to_target(
    *,
    S0: float,
    r: float,
    q: float,
    base_models: list[ModelSpec],
    target_name_list: list[str],
    T_cal: float,
    k_grid: np.ndarray,
    iv_target: np.ndarray,
    N: int,
    L: float,
    verbose: bool,
    weight_mode: str,
    weight_k_scale: float,
) -> list[CalibResult]:
    """Calibrate each selected model so its IV(k) at T_cal matches iv_target."""

    w = _weight_k(k_grid, mode=str(weight_mode), k_scale=float(weight_k_scale))
    finite_mask = np.isfinite(iv_target)
    if not np.any(finite_mask):
        raise RuntimeError("Target IV contains no finite values")

    k_fit = k_grid[finite_mask]
    y = iv_target[finite_mask]
    w_fit = w[finite_mask]

    # Create initial guesses from the existing defaults in _build_models.
    base_by_name = {ms.name: ms.model for ms in base_models}
    out: list[CalibResult] = []

    def eval_iv(name: str, params: dict[str, float]) -> np.ndarray:
        model = _model_from_params(name=name, S0=S0, r=r, q=q, params=params)
        ms = ModelSpec(str(name), model)
        _k, iv = _iv_slice_for_model(model_spec=ms, k_grid=k_fit, T=T_cal, N=N, L=L)
        return iv

    for name in target_name_list:
        name_u = str(name).strip().upper()

        if name_u == "VG":
            base = base_by_name["VG"].params
            x0 = np.array([float(base["sigma"]), float(base["theta"]), float(base["nu"])], dtype=float)
            lb = np.array([1e-6, -2.0, 1e-4], dtype=float)
            ub = np.array([3.0, 0.5, 5.0], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"sigma": float(x[0]), "theta": float(x[1]), "nu": float(x[2])}

        elif name_u == "MERTON":
            base = base_by_name["Merton"].params
            x0 = np.array([float(base["vol"]), float(base["lam"]), float(base["muJ"]), float(base["sigmaJ"])], dtype=float)
            lb = np.array([1e-6, 0.0, -1.0, 1e-4], dtype=float)
            ub = np.array([3.0, 10.0, 0.0, 1.0], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"vol": float(x[0]), "lam": float(x[1]), "muJ": float(x[2]), "sigmaJ": float(x[3])}

        elif name_u == "KOU":
            base = base_by_name["Kou"].params
            # Fit full Kou shape including tail rates (eta1/eta2) so wings can match.
            x0 = np.array([float(base["vol"]), float(base["lam"]), float(base["p"]), float(base["eta1"]), float(base["eta2"])], dtype=float)
            # eta1 must exceed 1 for E[e^Y] to exist on the positive tail.
            lb = np.array([1e-6, 0.0, 1e-3, 1.01, 1e-3], dtype=float)
            ub = np.array([3.0, 10.0, 1.0 - 1e-3, 250.0, 250.0], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"vol": float(x[0]), "lam": float(x[1]), "p": float(x[2]), "eta1": float(x[3]), "eta2": float(x[4])}

        elif name_u == "CGMY":
            base = base_by_name["CGMY"].params
            # Fit positive params in log-space and enforce M>G by parameterizing M = G + exp(d).
            # Also fit Y (tempered-stable activity) so wings can better match.
            C0 = max(float(base["C"]), 1e-16)
            G0 = max(float(base["G"]), 1e-8)
            M0 = max(float(base["M"]), G0 + 1e-8)
            Y0 = float(base.get("Y", 1.3))
            x0 = np.array([np.log(C0), np.log(G0), np.log(max(M0 - G0, 1e-8)), Y0], dtype=float)
            lb = np.array([np.log(1e-12), np.log(1e-6), np.log(1e-8), 0.05], dtype=float)
            ub = np.array([np.log(1e6), np.log(1e3), np.log(1e3), 1.95], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                C = float(np.exp(x[0]))
                G = float(np.exp(x[1]))
                d = float(np.exp(x[2]))
                M = float(G + d)
                Y = float(x[3])
                return {"C": C, "G": G, "M": M, "Y": Y}

        else:
            raise ValueError(f"Unsupported model for calibration: {name}")

        def residuals(x: np.ndarray) -> np.ndarray:
            params = pack(x)
            iv = eval_iv(name_u, params)
            # Replace NaNs with a large penalty.
            bad = ~np.isfinite(iv)
            if np.any(bad):
                iv = iv.copy()
                iv[bad] = 10.0
            return (iv - y) * w_fit

        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=200,
            verbose=2 if verbose else 0,
        )
        params_hat = pack(res.x)
        out.append(CalibResult(name=_display_model_name(name_u), params=params_hat, cost=float(res.cost)))

    return out


def _make_k_grid(*, k_min: float, k_max: float, n: int) -> np.ndarray:
    n = int(n)
    if n < 5:
        raise ValueError("n must be >= 5")
    if n % 2 == 0:
        n += 1  # ensure 0 is included at the midpoint
    k = np.linspace(float(k_min), float(k_max), n, dtype=float)
    # Snap the midpoint to exactly 0.0 (avoids tiny drift from float).
    k[n // 2] = 0.0
    return k


def _iv_slice_for_model(
    *,
    model_spec: ModelSpec,
    k_grid: np.ndarray,
    T: float,
    N: int,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_grid, iv(k)) for maturity T."""

    model = model_spec.model
    pricer = COSPricer(model, N=int(N), L=float(L))

    F = float(model.S0) * float(np.exp((float(model.r) - float(model.q)) * float(T)))
    strikes = F * np.exp(k_grid)

    prices = pricer.european_price(strikes, float(T), is_call=True)
    iv = np.full_like(k_grid, np.nan, dtype=float)
    for i, K in enumerate(strikes):
        iv[i] = _implied_vol_call(
            target_price=float(prices[i]),
            S0=float(model.S0),
            K=float(K),
            r=float(model.r),
            q=float(model.q),
            T=float(T),
        )

    return k_grid, iv


def _atm_metrics(*, k: np.ndarray, iv: np.ndarray, k_rms: float) -> dict[str, float]:
    """Compute ATM metrics using centered finite differences around k=0."""

    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)

    mid = int(len(k) // 2)
    if not (abs(k[mid]) <= 1e-14):
        raise RuntimeError("k-grid midpoint is not at 0")

    dk = float(k[mid + 1] - k[mid])
    if dk <= 0:
        raise RuntimeError("k-grid not increasing")

    iv_m = float(iv[mid - 1])
    iv_0 = float(iv[mid])
    iv_p = float(iv[mid + 1])

    skew = (iv_p - iv_m) / (2.0 * dk)
    curv = (iv_p - 2.0 * iv_0 + iv_m) / (dk ** 2)

    mask = np.abs(k) <= float(k_rms)
    diffs = iv[mask] - iv_0
    smile_rms = float(np.sqrt(np.nanmean(diffs ** 2))) if np.any(mask) else float("nan")

    return {
        "iv_atm": iv_0,
        "skew_atm": float(skew),
        "curv_atm": float(curv),
        "smile_rms": float(smile_rms),
    }


def _parse_maturities(arg: str | None) -> np.ndarray:
    if arg is None or str(arg).strip() == "":
        # Dense near-front, sparse long end.
        return np.array([7 / 365, 14 / 365, 21 / 365, 30 / 365, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0], dtype=float)
    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    return np.array([float(p) for p in parts], dtype=float)


def _numeric_cumulants_k1_to_k4(model, *, T: float, h: float = 1e-3) -> dict[str, float]:
    """Compute cumulants κ1..κ4 of X=ln S_T numerically from log characteristic function.

    For φ(u) = E[e^{i u X}], we have:
        d^n/du^n log φ(u) |_{u=0} = i^n κ_n
    so κ_n = (1/i^n) * derivative.

    We use symmetric finite-difference stencils at u=0 for derivatives up to 4th order.
    """
    T = float(T)
    h = float(h)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")

    u = np.array([-2.0 * h, -h, 0.0, h, 2.0 * h], dtype=float)
    phi = model.char_func(u.astype(complex), T)
    # Guard against numerical zeros
    phi = np.asarray(phi, dtype=complex)
    phi = np.where(np.abs(phi) < 1e-300, 1e-300 + 0j, phi)
    f = np.log(phi)  # complex

    f_m2, f_m1, f_0, f_p1, f_p2 = f
    # Derivatives at 0
    d1 = (f_p1 - f_m1) / (2.0 * h)
    d2 = (f_p1 - 2.0 * f_0 + f_m1) / (h ** 2)
    d3 = (f_p2 - 2.0 * f_p1 + 2.0 * f_m1 - f_m2) / (2.0 * (h ** 3))
    d4 = (f_m2 - 4.0 * f_m1 + 6.0 * f_0 - 4.0 * f_p1 + f_p2) / (h ** 4)

    k1 = (1j) ** (-1) * d1
    k2 = (1j) ** (-2) * d2
    k3 = (1j) ** (-3) * d3
    k4 = (1j) ** (-4) * d4

    # Cumulants should be (close to) real; discard tiny imaginary numerical noise.
    def real(x: complex) -> float:
        return float(np.real(x))

    return {"k1": real(k1), "k2": real(k2), "k3": real(k3), "k4": real(k4)}


def _standardized_cumulants(model, *, T: float, h: float = 1e-3) -> dict[str, float]:
    c = _numeric_cumulants_k1_to_k4(model, T=T, h=h)
    k2 = float(c["k2"])
    k3 = float(c["k3"])
    k4 = float(c["k4"])
    if not np.isfinite(k2) or k2 <= 0.0:
        return {"k2": float("nan"), "skew": float("nan"), "exkurt": float("nan")}
    skew = k3 / (k2 ** 1.5)
    exkurt = k4 / (k2 ** 2)
    return {"k2": k2, "skew": float(skew), "exkurt": float(exkurt)}


def _avg_target_standardized_cumulants(
    *,
    models: list[ModelSpec],
    model_names: list[str],
    T: float,
    h: float,
) -> dict[str, float]:
    vals = []
    for ms in models:
        if ms.name not in model_names:
            continue
        vals.append(_standardized_cumulants(ms.model, T=float(T), h=float(h)))
    if not vals:
        raise RuntimeError("No models selected to build cumulant target")
    k2s = [v["k2"] for v in vals if np.isfinite(v["k2"]) and v["k2"] > 0.0]
    sks = [v["skew"] for v in vals if np.isfinite(v["skew"])]
    eks = [v["exkurt"] for v in vals if np.isfinite(v["exkurt"])]
    if not k2s or not sks or not eks:
        raise RuntimeError("Insufficient finite cumulant target values")
    return {"k2": float(np.mean(k2s)), "skew": float(np.mean(sks)), "exkurt": float(np.mean(eks))}


def _load_models_from_calib_json(path: str) -> list[ModelSpec]:
    """Load VG/Merton/Kou/CGMY model specs from a calibration JSON payload."""
    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    S0 = float(payload.get("S0", 100.0))
    r = float(payload.get("r", 0.02))
    q = float(payload.get("q", 0.0))

    models_payload = payload.get("models", {})
    if not isinstance(models_payload, dict) or not models_payload:
        raise RuntimeError(f"No 'models' found in calibration JSON: {path}")

    specs: list[ModelSpec] = []
    for name, params in models_payload.items():
        if not isinstance(params, dict):
            continue
        # Normalize key, but keep a friendly display name.
        name_u = str(name).strip().upper()
        disp = _display_model_name(name_u)
        model = _model_from_params(name=name_u, S0=S0, r=r, q=q, params={str(k): float(v) for k, v in params.items()})
        specs.append(ModelSpec(disp, model))
    if not specs:
        raise RuntimeError(f"Failed to load any model specs from: {path}")
    return specs


def _calibrate_models_to_cumulants(
    *,
    S0: float,
    r: float,
    q: float,
    base_models: list[ModelSpec],
    target_name_list: list[str],
    T_cal: float,
    target: dict[str, float],
    h: float,
    verbose: bool,
    w_var: float,
    w_skew: float,
    w_exkurt: float,
    iv_target: np.ndarray | None = None,
    k_grid_iv: np.ndarray | None = None,
    iv_weight: float = 0.0,
    iv_N: int = 2**11,
    iv_L: float = 12.0,
) -> list[CalibResult]:
    """Calibrate models to match target standardized cumulants at T_cal.

    Objective matches:
      - log(k2) (variance) for scale
      - skew
      - exkurt
    """
    T_cal = float(T_cal)
    h = float(h)
    target_k2 = float(target["k2"])
    target_skew = float(target["skew"])
    target_exkurt = float(target["exkurt"])

    base_by_name = {ms.name: ms.model for ms in base_models}
    out: list[CalibResult] = []

    def resid_for(name: str, params: dict[str, float]) -> np.ndarray:
        model = _model_from_params(name=name, S0=S0, r=r, q=q, params=params)
        sc = _standardized_cumulants(model, T=T_cal, h=h)
        if not (np.isfinite(sc["k2"]) and np.isfinite(sc["skew"]) and np.isfinite(sc["exkurt"])):
            base = np.array([10.0, 10.0, 10.0], dtype=float)
        else:
            # Scale variance via log to make it dimensionless.
            base = np.array(
                [
                    float(w_var) * np.log(sc["k2"] / max(target_k2, 1e-300)),
                    float(w_skew) * (sc["skew"] - target_skew),
                    float(w_exkurt) * (sc["exkurt"] - target_exkurt),
                ],
                dtype=float,
            )

        # Optional: add IV slice penalty at T_cal to keep IV shape aligned.
        if iv_target is not None and k_grid_iv is not None and float(iv_weight) > 0.0:
            ms = ModelSpec("tmp", model)
            _k, iv = _iv_slice_for_model(model_spec=ms, k_grid=np.asarray(k_grid_iv, dtype=float), T=T_cal, N=int(iv_N), L=float(iv_L))
            iv = np.asarray(iv, dtype=float)
            tgt = np.asarray(iv_target, dtype=float)
            diff = iv - tgt
            bad = ~np.isfinite(diff)
            if np.any(bad):
                diff = diff.copy()
                diff[bad] = 10.0
            return np.concatenate([base, float(iv_weight) * diff.astype(float, copy=False)])

        return base

    for name in target_name_list:
        name_u = str(name).strip().upper()

        if name_u == "VG":
            base = base_by_name["VG"].params
            x0 = np.array([float(base["sigma"]), float(base["theta"]), float(base["nu"])], dtype=float)
            lb = np.array([1e-6, -1.5, 1e-4], dtype=float)
            ub = np.array([1.5, 0.5, 2.0], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"sigma": float(x[0]), "theta": float(x[1]), "nu": float(x[2])}

        elif name_u == "MERTON":
            base = base_by_name["Merton"].params
            x0 = np.array([float(base["vol"]), float(base["lam"]), float(base["muJ"]), float(base["sigmaJ"])], dtype=float)
            lb = np.array([1e-6, 0.0, -0.8, 1e-4], dtype=float)
            ub = np.array([1.5, 10.0, 0.2, 0.8], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"vol": float(x[0]), "lam": float(x[1]), "muJ": float(x[2]), "sigmaJ": float(x[3])}

        elif name_u == "KOU":
            base = base_by_name["Kou"].params
            x0 = np.array([float(base["vol"]), float(base["lam"]), float(base["p"]), float(base["eta1"]), float(base["eta2"]),], dtype=float)
            lb = np.array([1e-6, 0.0, 1e-3, 1.01, 1e-3], dtype=float)
            ub = np.array([1.5, 10.0, 1.0 - 1e-3, 80.0, 80.0], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                return {"vol": float(x[0]), "lam": float(x[1]), "p": float(x[2]), "eta1": float(x[3]), "eta2": float(x[4])}

        elif name_u == "CGMY":
            base = base_by_name["CGMY"].params
            C0 = max(float(base["C"]), 1e-16)
            G0 = max(float(base["G"]), 1e-8)
            M0 = max(float(base["M"]), G0 + 1e-8)
            Y0 = float(base.get("Y", 1.3))
            x0 = np.array([np.log(C0), np.log(G0), np.log(max(M0 - G0, 1e-8)), Y0], dtype=float)
            lb = np.array([np.log(1e-8), np.log(0.05), np.log(1e-3), 0.20], dtype=float)
            ub = np.array([np.log(10.0), np.log(50.0), np.log(50.0), 1.80], dtype=float)

            def pack(x: np.ndarray) -> dict[str, float]:
                C = float(np.exp(x[0]))
                G = float(np.exp(x[1]))
                d = float(np.exp(x[2]))
                M = float(G + d)
                Y = float(x[3])
                return {"C": C, "G": G, "M": M, "Y": Y}

        else:
            raise ValueError(f"Unsupported model for cumulant calibration: {name}")

        def residuals(x: np.ndarray) -> np.ndarray:
            rvec = resid_for(name_u, pack(x))
            bad = ~np.isfinite(rvec)
            if np.any(bad):
                rvec = rvec.copy()
                rvec[bad] = 10.0
            return rvec

        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=250,
            verbose=2 if verbose else 0,
        )
        params_hat = pack(res.x)
        out.append(CalibResult(name=_display_model_name(name_u), params=params_hat, cost=float(res.cost)))

    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Rank Lévy models by skew/smile persistence vs maturity.")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--target-vol", type=float, default=0.20, help="Variance-rate match target (GBM vol)")
    p.add_argument("--maturities", type=str, default=None, help="Comma-separated maturities in years")
    p.add_argument("--k-min", type=float, default=-0.35, help="Min log-moneyness k=ln(K/F)")
    p.add_argument("--k-max", type=float, default=0.35, help="Max log-moneyness k=ln(K/F)")
    p.add_argument("--nk", type=int, default=41, help="# log-moneyness points (forced odd)")
    p.add_argument("--k-rms", type=float, default=0.20, help="RMS window |k|<=k_rms")
    p.add_argument("--N", type=int, default=2**12, help="COS series terms")
    p.add_argument("--L", type=float, default=12.0, help="COS truncation range")
    p.add_argument("--out", type=str, default="figs/skew_persistence_by_model.csv")

    # Optional: calibrate models to a common target.
    calib_group = p.add_mutually_exclusive_group()
    calib_group.add_argument("--calibrate-to-avgshape", action="store_true", help="Calibrate VG/Merton/Kou/CGMY to the average IV(k) at T_cal before running persistence analysis")
    calib_group.add_argument("--calibrate-to-avgcumulants", action="store_true", help="Calibrate VG/Merton/Kou/CGMY to match average standardized cumulants (k2, skew, exkurt) at T_cal")
    p.add_argument("--T-cal", type=float, default=0.25, help="Calibration maturity in years")
    p.add_argument("--calib-N", type=int, default=2**11, help="COS N used during calibration")
    p.add_argument("--calib-L", type=float, default=12.0, help="COS L used during calibration")
    p.add_argument("--calib-verbose", action="store_true", help="Verbose least-squares output")
    p.add_argument("--calib-out-json", type=str, default=None, help="Write calibrated params JSON to this path (default: derived from --out)")
    p.add_argument("--calib-k-min", type=float, default=-0.60, help="Calibration k-grid min ln(K/F)")
    p.add_argument("--calib-k-max", type=float, default=0.60, help="Calibration k-grid max ln(K/F)")
    p.add_argument("--calib-nk", type=int, default=61, help="Calibration k-grid points (forced odd)")
    p.add_argument("--calib-weight", type=str, choices=("atm", "flat", "wings"), default="flat", help="Calibration residual weighting")
    p.add_argument("--calib-weight-k", type=float, default=0.18, help="Scale for weighting in k-space")
    p.add_argument("--calib-cumulant-h", type=float, default=1e-3, help="Finite-difference step for numeric cumulants")
    p.add_argument(
        "--calib-cumulant-target-models",
        type=str,
        default="Merton,Kou,CGMY",
        help="Comma-separated model names used to define the cumulant target (default excludes VG)",
    )
    p.add_argument(
        "--calib-cumulant-target-json",
        type=str,
        default=None,
        help="If set, derive the cumulant target from models stored in this calibration JSON (e.g. avg-shape calibrated params)",
    )
    p.add_argument("--calib-cumulant-iv-weight", type=float, default=0.0, help="If >0, also penalize deviation from target IV(k) at T_cal (hybrid)")
    p.add_argument("--calib-cumulant-iv-k-min", type=float, default=-0.60, help="Hybrid IV penalty k-grid min")
    p.add_argument("--calib-cumulant-iv-k-max", type=float, default=0.60, help="Hybrid IV penalty k-grid max")
    p.add_argument("--calib-cumulant-iv-nk", type=int, default=61, help="Hybrid IV penalty k-grid points (forced odd)")
    p.add_argument("--calib-cumulant-iv-N", type=int, default=2**11, help="COS N used for hybrid IV penalty")
    p.add_argument("--calib-cumulant-iv-L", type=float, default=12.0, help="COS L used for hybrid IV penalty")
    p.add_argument("--calib-cumulant-w-var", type=float, default=1.0, help="Weight for log-variance residual")
    p.add_argument("--calib-cumulant-w-skew", type=float, default=1.0, help="Weight for skewness residual")
    p.add_argument("--calib-cumulant-w-exkurt", type=float, default=0.2, help="Weight for excess-kurtosis residual")
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    S0 = float(args.S0)
    r = float(args.r)
    q = float(args.q)
    target_vol = float(args.target_vol)
    maturities = _parse_maturities(args.maturities)

    k_grid = _make_k_grid(k_min=float(args.k_min), k_max=float(args.k_max), n=int(args.nk))

    models = _build_models(S0=S0, r=r, q=q, target_vol=target_vol)

    # Optional calibration step.
    if bool(args.calibrate_to_avgcumulants):
        names = ["VG", "Merton", "Kou", "CGMY"]
        T_cal = float(args.T_cal)
        h = float(args.calib_cumulant_h)
        target_names = [s.strip() for s in str(args.calib_cumulant_target_models).split(",") if s.strip()]

        # Option B: compute the cumulant target from a previously calibrated set of params.
        if args.calib_cumulant_target_json is not None and str(args.calib_cumulant_target_json).strip() != "":
            target_specs = _load_models_from_calib_json(str(args.calib_cumulant_target_json))
            target = _avg_target_standardized_cumulants(models=target_specs, model_names=target_names, T=T_cal, h=h)
        else:
            target = _avg_target_standardized_cumulants(models=models, model_names=target_names, T=T_cal, h=h)

        # Optional hybrid: also match the IV slice built from the target_specs at T_cal.
        iv_target = None
        k_grid_iv = None
        if float(args.calib_cumulant_iv_weight) > 0.0:
            if args.calib_cumulant_target_json is None or str(args.calib_cumulant_target_json).strip() == "":
                raise ValueError("--calib-cumulant-iv-weight requires --calib-cumulant-target-json")
            # Ensure we have target_specs in this branch.
            if "target_specs" not in locals():
                target_specs = _load_models_from_calib_json(str(args.calib_cumulant_target_json))
            k_grid_iv = _make_k_grid(
                k_min=float(args.calib_cumulant_iv_k_min),
                k_max=float(args.calib_cumulant_iv_k_max),
                n=int(args.calib_cumulant_iv_nk),
            )
            iv_target = _average_target_iv(
                model_specs=target_specs,
                model_names=target_names,
                k_grid=k_grid_iv,
                T=T_cal,
                N=int(args.calib_cumulant_iv_N),
                L=float(args.calib_cumulant_iv_L),
            )
        calib = _calibrate_models_to_cumulants(
            S0=S0,
            r=r,
            q=q,
            base_models=models,
            target_name_list=names,
            T_cal=T_cal,
            target=target,
            h=h,
            verbose=bool(args.calib_verbose),
            w_var=float(args.calib_cumulant_w_var),
            w_skew=float(args.calib_cumulant_w_skew),
            w_exkurt=float(args.calib_cumulant_w_exkurt),
            iv_target=iv_target,
            k_grid_iv=k_grid_iv,
            iv_weight=float(args.calib_cumulant_iv_weight),
            iv_N=int(args.calib_cumulant_iv_N),
            iv_L=float(args.calib_cumulant_iv_L),
        )

        new_specs: list[ModelSpec] = [ModelSpec("GBM", GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": float(target_vol)}))]
        for cr in calib:
            new_specs.append(ModelSpec(cr.name, _model_from_params(name=cr.name, S0=S0, r=r, q=q, params=cr.params)))
        models = new_specs

        out_json = args.calib_out_json
        if out_json is None or str(out_json).strip() == "":
            base, _ext = os.path.splitext(str(args.out))
            out_json = f"{base}_calib_params.json"
        os.makedirs(os.path.dirname(str(out_json)) or ".", exist_ok=True)
        payload = {
            "calib_mode": "avgcumulants",
            "T_cal": float(T_cal),
            "cumulant_h": float(h),
            "target": {k: float(v) for k, v in target.items()},
            "target_models": target_names,
            "weights": {"w_var": float(args.calib_cumulant_w_var), "w_skew": float(args.calib_cumulant_w_skew), "w_exkurt": float(args.calib_cumulant_w_exkurt)},
            "iv_penalty": {
                "weight": float(args.calib_cumulant_iv_weight),
                "k_min": float(args.calib_cumulant_iv_k_min),
                "k_max": float(args.calib_cumulant_iv_k_max),
                "nk": int(args.calib_cumulant_iv_nk),
                "N": int(args.calib_cumulant_iv_N),
                "L": float(args.calib_cumulant_iv_L),
                "target_json": str(args.calib_cumulant_target_json) if args.calib_cumulant_target_json else None,
            },
            "S0": float(S0),
            "r": float(r),
            "q": float(q),
            "target_vol": float(target_vol),
            "models": {cr.name: {k: float(v) for k, v in cr.params.items()} for cr in calib},
        }
        with open(str(out_json), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote calibrated params JSON: {out_json}")

        print("=== Calibration to average standardized cumulants at T_cal ===")
        print(f"T_cal={T_cal:.6f} years | target k2={target['k2']:.6g}, skew={target['skew']:.6g}, exkurt={target['exkurt']:.6g} | h={h:g}")
        for cr in calib:
            print(f"{cr.name}: cost={cr.cost:.6e} params={{{', '.join([f'{k}={v:.6g}' for k, v in cr.params.items()])}}}")
        print("")

    elif bool(args.calibrate_to_avgshape):
        names = ["VG", "Merton", "Kou", "CGMY"]
        T_cal = float(args.T_cal)

        # Use a separate (typically wider) k-grid for calibration so wings are actually fit.
        k_grid_cal = _make_k_grid(k_min=float(args.calib_k_min), k_max=float(args.calib_k_max), n=int(args.calib_nk))

        # Build target from the *current* base shapes.
        iv_target = _average_target_iv(model_specs=models, model_names=names, k_grid=k_grid_cal, T=T_cal, N=int(args.calib_N), L=float(args.calib_L))
        calib = _calibrate_models_to_target(
            S0=S0,
            r=r,
            q=q,
            base_models=models,
            target_name_list=names,
            T_cal=T_cal,
            k_grid=k_grid_cal,
            iv_target=iv_target,
            N=int(args.calib_N),
            L=float(args.calib_L),
            verbose=bool(args.calib_verbose),
            weight_mode=str(args.calib_weight),
            weight_k_scale=float(args.calib_weight_k),
        )

        # Rebuild model specs using calibrated params.
        new_specs: list[ModelSpec] = [ModelSpec("GBM", GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": float(target_vol)}))]
        for cr in calib:
            new_specs.append(ModelSpec(cr.name, _model_from_params(name=cr.name, S0=S0, r=r, q=q, params=cr.params)))
        models = new_specs

        # Persist calibration results for plotting/reuse.
        out_json = args.calib_out_json
        if out_json is None or str(out_json).strip() == "":
            base, ext = os.path.splitext(str(args.out))
            out_json = f"{base}_calib_params.json"
        os.makedirs(os.path.dirname(str(out_json)) or ".", exist_ok=True)
        payload = {
            "T_cal": float(T_cal),
            "k_min": float(np.min(k_grid_cal)),
            "k_max": float(np.max(k_grid_cal)),
            "nk": int(len(k_grid_cal)),
            "calib_weight": str(args.calib_weight),
            "calib_weight_k": float(args.calib_weight_k),
            "S0": float(S0),
            "r": float(r),
            "q": float(q),
            "target_vol": float(target_vol),
            "models": {cr.name: {k: float(v) for k, v in cr.params.items()} for cr in calib},
        }
        with open(str(out_json), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote calibrated params JSON: {out_json}")

        print("=== Calibration to average IV shape at T_cal ===")
        print(
            f"T_cal={T_cal:.6f} years | k-grid=[{float(k_grid_cal.min()):.3f},{float(k_grid_cal.max()):.3f}] nk={len(k_grid_cal)} | weight={args.calib_weight}"
        )
        for cr in calib:
            print(f"{cr.name}: cost={cr.cost:.6e} params={{{', '.join([f'{k}={v:.6g}' for k, v in cr.params.items()])}}}")
        print("")

    # Compute metrics
    rows: list[dict[str, float | str]] = []
    for ms in models:
        for T in maturities:
            k, iv = _iv_slice_for_model(model_spec=ms, k_grid=k_grid, T=float(T), N=int(args.N), L=float(args.L))
            m = _atm_metrics(k=k, iv=iv, k_rms=float(args.k_rms))
            rows.append(
                {
                    "model": ms.name,
                    "T": float(T),
                    **m,
                }
            )

    # Write CSV
    import csv

    fieldnames = ["model", "T", "iv_atm", "skew_atm", "curv_atm", "smile_rms"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Rank by skew persistence: ratio |skew(T_long)| / |skew(T_short)|.
    # Use closest available maturities to 1M and 1Y (and 2Y if available).
    def closest_T(target: float) -> float:
        return float(maturities[np.argmin(np.abs(maturities - target))])

    T_1m = closest_T(30 / 365)
    T_1y = closest_T(1.0)
    T_2y = closest_T(2.0) if np.max(maturities) >= 1.5 else float("nan")

    # Collect per-model skew values
    by_model: dict[str, dict[float, float]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[float(row["T"])] = float(row["skew_atm"])

    ranking = []
    for name, skews in by_model.items():
        s1m = float(skews.get(T_1m, float("nan")))
        s1y = float(skews.get(T_1y, float("nan")))
        s2y = float(skews.get(T_2y, float("nan"))) if np.isfinite(T_2y) else float("nan")
        ratio_1y = abs(s1y) / max(abs(s1m), 1e-14)
        ratio_2y = abs(s2y) / max(abs(s1m), 1e-14) if np.isfinite(s2y) else float("nan")
        ranking.append((name, ratio_1y, ratio_2y, s1m, s1y, s2y))

    ranking.sort(key=lambda x: (-(x[1] if np.isfinite(x[1]) else -1e9)))

    print("=== Skew persistence ranking (higher = more persistent) ===")
    print(f"Target variance rate: vol={target_vol:.4f} (Var/T={target_vol**2:.6f})")
    print(f"Maturities used: 1M≈{T_1m:.6f}y, 1Y≈{T_1y:.6f}y" + (f", 2Y≈{T_2y:.6f}y" if np.isfinite(T_2y) else ""))
    print(f"Wrote CSV: {args.out}")
    print("")
    print("model, |skew(1Y)|/|skew(1M)|, |skew(2Y)|/|skew(1M)|, skew(1M), skew(1Y), skew(2Y)")
    for name, r1y, r2y, s1m, s1y, s2y in ranking:
        r2y_s = f"{r2y:.3f}" if np.isfinite(r2y) else "nan"
        s2y_s = f"{s2y:+.6f}" if np.isfinite(s2y) else "nan"
        print(f"{name}, {r1y:.3f}, {r2y_s}, {s1m:+.6f}, {s1y:+.6f}, {s2y_s}")


if __name__ == "__main__":
    main()
