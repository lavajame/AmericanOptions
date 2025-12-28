"""Synthetic calibration demo: LSMC-generated American prices -> COS-American calibration.

What this does
--------------
1) Generate synthetic American option prices using LSMC under a *simulatable* composite Lévy model
   (e.g. Merton + VG) with:
   - discrete cash dividends
   - negative borrow cost q (e.g. -0.5%)

2) Calibrate the same model class using COS-American prices and analytic sensitivities.

Notes
-----
- This repo does NOT currently include a CGMY Monte Carlo sampler.
  If you want CGMY as the generator, we’d need to implement a CGMY path simulator.
- LSMC prices are noisy; increase n_train/n_price to stabilize calibration.
- COS American sensitivities are computed via the rollback + softmax smoothing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt

# Ensure repo root is on sys.path when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, CompositeLevyCHF  # noqa: E402
from american_options.lsmc import LSMCCompositePricer  # noqa: E402


@dataclass(frozen=True)
class OptionQuote:
    K: float
    T: float
    is_call: bool
    price: float

def _vec_to_components(*, x: np.ndarray, case: str) -> tuple[list[dict], list[str]]:
    """Map parameter vector -> CompositeLevyCHF components.

    Supported cases:
    - "vg": calibrate (VG.theta, VG.sigma, VG.nu)
    - "merton_vg": calibrate (Merton.sigma, Merton.lam, Merton.muJ, Merton.sigmaJ, VG.theta, VG.sigma, VG.nu)
    """
    case = str(case).lower().strip()
    x = np.asarray(x, dtype=float)

    if case == "vg":
        theta, vg_sigma, nu = x.tolist()
        vg_sigma = max(vg_sigma, 1e-12)
        nu = max(nu, 1e-12)
        components = [{"type": "vg", "params": {"theta": theta, "sigma": vg_sigma, "nu": nu}}]
        names = ["VG.theta", "VG.sigma", "VG.nu"]
        return components, names

    if case == "merton_vg":
        sigma, lam, muJ, sigmaJ, theta, vg_sigma, nu = x.tolist()
        lam = max(lam, 1e-12)
        sigma = max(sigma, 1e-12)
        sigmaJ = max(sigmaJ, 1e-12)
        vg_sigma = max(vg_sigma, 1e-12)
        nu = max(nu, 1e-12)
        components = [
            {"type": "merton", "params": {"sigma": sigma, "lam": lam, "muJ": muJ, "sigmaJ": sigmaJ}},
            {"type": "vg", "params": {"theta": theta, "sigma": vg_sigma, "nu": nu}},
        ]
        names = [
            "Merton.sigma",
            "Merton.lam",
            "Merton.muJ",
            "Merton.sigmaJ",
            "VG.theta",
            "VG.sigma",
            "VG.nu",
        ]
        return components, names

    raise ValueError(f"Unknown calibration case: {case}")


def _components_to_model(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], components: list[dict]) -> CompositeLevyCHF:
    return CompositeLevyCHF(S0, r, q, divs, {"components": components})


def _cos_prices_and_jac(
    *,
    x: np.ndarray,
    quotes: list[OptionQuote],
    S0: float,
    r: float,
    q: float,
    divs: dict[float, tuple[float, float]],
    N: int,
    L: float,
    steps: int,
    beta: float,
    case: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (prices, jacobian, param_names) for COS-American under x."""
    components, sens_params = _vec_to_components(x=x, case=case)
    model = _components_to_model(S0=S0, r=r, q=q, divs=divs, components=components)
    pricer = COSPricer(model, N=N, L=L)

    prices = np.zeros(len(quotes), dtype=float)
    jac = np.zeros((len(quotes), len(sens_params)), dtype=float)

    for i, qte in enumerate(quotes):
        px, sens = pricer.american_price(
            np.array([qte.K], dtype=float),
            float(qte.T),
            steps=int(steps),
            is_call=bool(qte.is_call),
            use_softmax=True,
            beta=float(beta),
            return_sensitivities=True,
            sens_method="analytic",
            sens_params=sens_params,
        )
        prices[i] = float(px[0])
        for j, name in enumerate(sens_params):
            jac[i, j] = float(np.asarray(sens[name], dtype=float)[0])

    return prices, jac, sens_params


def main() -> int:
    # Choose which generating model / calibration parameterization to run.
    # "vg" is much more stable under noisy LSMC prices.
    case = os.environ.get("CALIB_CASE", "vg").strip().lower()

    # ------------------------
    # 1) Synthetic market data
    # ------------------------
    generator = os.environ.get("GENERATOR", "LSMC").strip().upper()
    S0 = 100.0
    r = 0.02
    q = -0.005  # borrow cost (negative => benefit to holding the stock)

    # discrete cash dividends (mean, std). LSMC uses mean deterministically.
    divs = {
        0.15: (2.5, 0.0),
    }

    if case == "vg":
        true_x = np.array([
            -0.20,  # VG.theta
            0.12,   # VG.sigma
            0.22,   # VG.nu
        ], dtype=float)
    elif case == "merton_vg":
        true_x = np.array([
            0.14,   # Merton.sigma
            0.8,    # Merton.lam
            -0.08,  # Merton.muJ
            0.20,   # Merton.sigmaJ
            -0.20,  # VG.theta
            0.12,   # VG.sigma
            0.22,   # VG.nu
        ], dtype=float)
    else:
        raise ValueError(f"Unknown case: {case}")

    # Option grid
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    maturities = [0.25, 0.5, 0.75]
    is_call = False

    # LSMC controls (increase for better calibration stability)
    lsmc_steps = 40
    # NOTE: increase these for better stability; American+divs is noisy.
    n_train = int(os.environ.get("LSMC_NTRAIN", "100000"))
    n_price = int(os.environ.get("LSMC_NPRICE", "100000"))
    deg = 3
    seed = 123

    components_true, names = _vec_to_components(x=true_x, case=case)

    quotes: list[OptionQuote] = []

    if generator == "LSMC":
        lsmc = LSMCCompositePricer(S0=S0, r=r, q=q, divs=divs, model_params={}, seed=seed)
        print("Generating synthetic prices via LSMC...")
        for T in maturities:
            for K in strikes:
                _, _, refit_px, _ = lsmc.price_at_tau(
                    K=float(K),
                    tau=float(T),
                    components=components_true,
                    steps=lsmc_steps,
                    n_train=n_train,
                    n_price=n_price,
                    deg=deg,
                    is_call=is_call,
                )
                quotes.append(OptionQuote(K=float(K), T=float(T), is_call=is_call, price=float(refit_px)))
    elif generator == "COS":
        print("Generating synthetic prices via COS-American (noiseless)...")
        model_true = _components_to_model(S0=S0, r=r, q=q, divs=divs, components=components_true)
        pr_true = COSPricer(model_true, N=2048, L=12.0)
        for T in maturities:
            for K in strikes:
                px = pr_true.american_price(
                    np.array([float(K)]),
                    float(T),
                    steps=lsmc_steps,
                    is_call=is_call,
                    use_softmax=True,
                    beta=100.0,
                )[0]
                quotes.append(OptionQuote(K=float(K), T=float(T), is_call=is_call, price=float(px)))
    else:
        raise ValueError("GENERATOR must be LSMC or COS")

    mkt = np.array([q.price for q in quotes], dtype=float)
    print(f"Case: {case}")
    print(f"Built {len(quotes)} quotes (puts={not is_call}), q={q}, divs={divs}")

    # For LSMC-generated data, print the COS-vs-LSMC mismatch at the true params.
    if generator == "LSMC":
        cos_at_truth, _, _ = _cos_prices_and_jac(
            x=true_x,
            quotes=quotes,
            S0=S0,
            r=r,
            q=q,
            divs=divs,
            N=512,
            L=10.0,
            steps=lsmc_steps,
            beta=100.0,
            case=case,
        )
        mismatch = cos_at_truth - mkt
        print(
            f"COS@true vs LSMC market: max abs diff={float(np.max(np.abs(mismatch))):.4e}, "
            f"rmse={float(np.sqrt(np.mean(mismatch**2))):.4e}"
        )

    # ------------------------
    # 2) Calibration (COS)
    # ------------------------
    # Initial guess: perturb the truth.
    if case == "vg":
        x0 = true_x * np.array([0.85, 1.25, 0.90], dtype=float)
        lb = np.array([-2.0, 1e-4, 1e-4], dtype=float)
        ub = np.array([2.0, 1.0, 5.0], dtype=float)
    else:
        x0 = true_x * np.array([1.10, 0.85, 0.70, 1.15, 0.80, 1.20, 0.90], dtype=float)
        lb = np.array([1e-4, 1e-4, -1.0, 1e-4, -2.0, 1e-4, 1e-4], dtype=float)
        ub = np.array([1.0, 10.0, 1.0, 2.0, 2.0, 1.0, 5.0], dtype=float)

    # COS controls
    cos_N = 512
    cos_L = 10.0
    cos_steps = 40
    beta = 100.0

    # Simple weighting: normalize residuals by a price scale.
    scale = np.maximum(1.0, np.abs(mkt))

    # Cache to avoid recomputing price+Jac twice per x.
    cache: dict[str, object] = {"x": None, "px": None, "J": None, "names": None}

    def _eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x_in = np.asarray(x, dtype=float)
        x_cached = cache.get("x")
        if x_cached is not None and np.allclose(x_in, np.asarray(x_cached), rtol=0.0, atol=0.0):
            return (
                np.asarray(cache["px"], dtype=float),
                np.asarray(cache["J"], dtype=float),
                list(cache["names"]),
            )

        px, J, param_names = _cos_prices_and_jac(
            x=x_in,
            quotes=quotes,
            S0=S0,
            r=r,
            q=q,
            divs=divs,
            N=cos_N,
            L=cos_L,
            steps=cos_steps,
            beta=beta,
            case=case,
        )
        cache["x"] = x_in.copy()
        cache["px"] = px
        cache["J"] = J
        cache["names"] = list(param_names)
        return px, J, param_names

    def fun(x: np.ndarray) -> np.ndarray:
        px, _, _ = _eval(x)
        return (px - mkt) / scale

    def jac(x: np.ndarray) -> np.ndarray:
        _, J, _ = _eval(x)
        return J / scale[:, None]

    print("Calibrating using COS-American analytic sensitivities...")
    res = opt.least_squares(
        fun,
        x0,
        jac=jac,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=80,
    )

    x_hat = res.x

    px_hat, _, names = _eval(x_hat)

    err = px_hat - mkt

    print("-")
    print(f"Success: {res.success}  status={res.status}  nfev={res.nfev}  cost={res.cost:.6e}")
    print("Recovered params (true vs fitted):")
    for nm, t, h in zip(names, true_x.tolist(), x_hat.tolist()):
        print(f"  {nm:12s}  true={t: .6f}  fit={h: .6f}  diff={h - t:+.3e}")
    print("-")
    print(f"Pricing fit: max abs err={np.max(np.abs(err)):.4e}, rmse={float(np.sqrt(np.mean(err**2))):.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
