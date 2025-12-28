"""Cross-model calibration demo: generate American prices under Kou, fit CGMY.

Goal
----
- Create *target* American option prices using COS-American under a Kou model.
- Fit a *different* model (CGMY) to match those prices.

Implementation notes
--------------------
- CGMY model here is represented as a `CompositeLevyCHF` with a single CGMY component.
  That lets us reuse the existing Composite analytic sensitivity plumbing:
  - analytic for CGMY.C
  - local finite-difference (on psi) for CGMY.G/M/Y

- This is intentionally a demonstration of identifiability / model mismatch:
  Kou and CGMY are not the same family, so we generally expect imperfect fits.

Usage
-----
    python tools/calibrate_cross_model_kou_to_cgmy_american.py

Optional env vars
-----------------
- STRIKES: comma-separated list (default: 80,90,100,110,120)
- MATS: comma-separated list (default: 0.25,0.5,1.0)
- Q: borrow cost (default: -0.005)
- CALIB_MODEL: cgmy|vg (default: vg)
- CALIB_MODEL: cgmy|vg|merton (default: vg)
- MAX_NFEV: max least-squares evaluations (default: 200)
- PRINT_EVERY: print a status line every N new x evaluations (default: 1)
- PLOT_PATH: where to save the IV plot (default: figs/cross_model_kou_to_cgmy_iv.png)
- CSV_PATH: where to save the IV data (default: figs/cross_model_kou_to_cgmy_iv.csv)

Quote selection
---------------
We build an OTM quote set automatically:
- puts for strikes K <= S0
- calls for strikes K > S0

This makes implied-vol inversion more stable for American prices.
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

from american_options.engine import COSPricer, CompositeLevyCHF, KouCHF  # noqa: E402


@dataclass(frozen=True)
class Quote:
    K: float
    T: float
    is_call: bool
    price: float


def _option_label(is_call: bool) -> str:
    return "C" if is_call else "P"


def _pv_cash_divs(*, divs: dict[float, tuple[float, float]], r: float, T: float) -> float:
    pv = 0.0
    for t, (cash, _prop) in divs.items():
        if float(t) <= float(T):
            pv += float(cash) * float(np.exp(-r * float(t)))
    return pv


def _forward_proxy(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], T: float) -> float:
    # Simple proxy forward used only for plotting x = log-moneyness / sqrt(T).
    # We treat cash dividends as deterministic and subtract their PV.
    prepaid = float(S0) - _pv_cash_divs(divs=divs, r=float(r), T=float(T))
    return float(prepaid * np.exp((float(r) - float(q)) * float(T)))


def _bs_price_forward(*, F: float, K: float, df: float, sigma: float, T: float, is_call: bool) -> float:
    # Black-Scholes price expressed in terms of forward F and discount factor df.
    F = float(F)
    K = float(K)
    df = float(df)
    T = float(T)
    sigma = max(float(sigma), 1e-12)
    if T <= 0.0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic

    from math import erf, log, sqrt

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    vol_sqrt = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _bs_implied_vol_forward(*, price: float, F: float, K: float, df: float, T: float, is_call: bool) -> float:
    # Fast implied vol inversion against European BS (forward form).
    price = float(price)
    F = float(F)
    K = float(K)
    df = float(df)
    T = float(T)

    intrinsic = df * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = df * (F if is_call else K)
    if not (intrinsic - 1e-12 <= price <= upper + 1e-12):
        return float("nan")

    lo, hi = 1e-6, 5.0

    def f(sig: float) -> float:
        return _bs_price_forward(F=F, K=K, df=df, sigma=sig, T=T, is_call=is_call) - price

    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0.0 and tries < 10:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0.0:
        return float("nan")

    sol = opt.root_scalar(f, bracket=(lo, hi), method="brentq", xtol=1e-10, rtol=1e-10, maxiter=200)
    if not sol.converged:
        return float("nan")
    return float(sol.root)


def _parse_csv_floats(s: str, default: list[float]) -> list[float]:
    s = str(s or "").strip()
    if not s:
        return default
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _build_kou(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]]) -> KouCHF:
    # Parameters chosen to create a noticeable skew/smile.
    return KouCHF(
        S0=S0,
        r=r,
        q=q,
        divs=divs,
        params={
            "sigma": 0.16,
            "lam": 1.2,
            "p": 0.35,
            "eta1": 20.0,
            "eta2": 6.0,
        },
    )


def _build_cgmy_model(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], C: float, G: float, M: float, Y: float) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {
            "components": [
                {"type": "cgmy", "params": {"C": float(C), "G": float(G), "M": float(M), "Y": float(Y)}},
            ]
        },
    )


def _build_vg_model(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], theta: float, sigma: float, nu: float) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {
            "components": [
                {"type": "vg", "params": {"theta": float(theta), "sigma": float(sigma), "nu": float(nu)}},
            ]
        },
    )


def _build_merton_model(
    *,
    S0: float,
    r: float,
    q: float,
    divs: dict[float, tuple[float, float]],
    sigma: float,
    lam: float,
    muJ: float,
    sigmaJ: float,
) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {
            "components": [
                {"type": "merton", "params": {"sigma": float(sigma), "lam": float(lam), "muJ": float(muJ), "sigmaJ": float(sigmaJ)}},
            ]
        },
    )


def _price_kou_targets(
    *,
    strikes: list[float],
    maturities: list[float],
    S0: float,
    r: float,
    q: float,
    divs: dict[float, tuple[float, float]],
    N: int,
    L: float,
    steps: int,
    beta: float,
) -> list[Quote]:
    model = _build_kou(S0=S0, r=r, q=q, divs=divs)
    pr = COSPricer(model, N=N, L=L)

    quotes: list[Quote] = []
    for T in maturities:
        for K in strikes:
            is_call = bool(float(K) > float(S0))
            px = pr.american_price(
                np.array([float(K)]),
                float(T),
                steps=int(steps),
                is_call=bool(is_call),
                use_softmax=True,
                beta=float(beta),
            )[0]
            quotes.append(Quote(K=float(K), T=float(T), is_call=bool(is_call), price=float(px)))

    return quotes


def _cgmy_prices_and_jac(
    *,
    x: np.ndarray,
    quotes: list[Quote],
    S0: float,
    r: float,
    q: float,
    divs: dict[float, tuple[float, float]],
    N: int,
    L: float,
    steps: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    C, G, M, Y = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    model = _build_cgmy_model(S0=S0, r=r, q=q, divs=divs, C=C, G=G, M=M, Y=Y)
    pr = COSPricer(model, N=N, L=L)

    sens_params = ["CGMY.C", "CGMY.G", "CGMY.M", "CGMY.Y"]

    prices = np.zeros(len(quotes), dtype=float)
    J = np.zeros((len(quotes), 4), dtype=float)

    for i, qte in enumerate(quotes):
        px, sens = pr.american_price(
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
            J[i, j] = float(np.asarray(sens[name], dtype=float)[0])

    return prices, J


def _vg_prices_and_jac(
    *,
    x: np.ndarray,
    quotes: list[Quote],
    S0: float,
    r: float,
    divs: dict[float, tuple[float, float]],
    N: int,
    L: float,
    steps: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta, sigma, nu, q = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    model = _build_vg_model(S0=S0, r=r, q=q, divs=divs, theta=theta, sigma=sigma, nu=nu)
    pr = COSPricer(model, N=N, L=L)

    sens_params = ["VG.theta", "VG.sigma", "VG.nu", "q"]

    prices = np.zeros(len(quotes), dtype=float)
    J = np.zeros((len(quotes), 4), dtype=float)

    for i, qte in enumerate(quotes):
        px, sens = pr.american_price(
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
            J[i, j] = float(np.asarray(sens[name], dtype=float)[0])

    return prices, J


def _merton_prices_and_jac(
    *,
    x: np.ndarray,
    quotes: list[Quote],
    S0: float,
    r: float,
    divs: dict[float, tuple[float, float]],
    N: int,
    L: float,
    steps: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma, lam, muJ, sigmaJ, q = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    model = _build_merton_model(S0=S0, r=r, q=q, divs=divs, sigma=sigma, lam=lam, muJ=muJ, sigmaJ=sigmaJ)
    pr = COSPricer(model, N=N, L=L)

    sens_params = ["Merton.sigma", "Merton.lam", "Merton.muJ", "Merton.sigmaJ", "q"]

    prices = np.zeros(len(quotes), dtype=float)
    J = np.zeros((len(quotes), len(sens_params)), dtype=float)

    for i, qte in enumerate(quotes):
        px, sens = pr.american_price(
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
            J[i, j] = float(np.asarray(sens[name], dtype=float)[0])

    return prices, J


def main() -> int:
    S0 = 100.0
    r = 0.02
    q = float(os.environ.get("Q", "-0.005"))
    calib_model = str(os.environ.get("CALIB_MODEL", "merton")).strip().lower()

    # cash dividends
    divs = {0.15: (2.5, 0.0)}

    strikes = _parse_csv_floats(os.environ.get("STRIKES", ""), [80.0, 90.0, 100.0, 110.0, 120.0])
    mats = _parse_csv_floats(os.environ.get("MATS", ""), [0.25, 0.5, 1.0])

    # COS knobs
    steps = 40
    beta = 100.0

    max_nfev = int(os.environ.get("MAX_NFEV", "200"))
    print_every = int(os.environ.get("PRINT_EVERY", "1"))
    scipy_verbose = int(os.environ.get("SCIPY_VERBOSE", "2"))

    plot_path = os.environ.get("PLOT_PATH", os.path.join("figs", "cross_model_kou_to_cgmy_iv.png"))
    csv_path = os.environ.get("CSV_PATH", os.path.join("figs", "cross_model_kou_to_cgmy_iv.csv"))

    # Use a decent resolution for targets.
    target_N, target_L = 2048, 12.0
    # Use slightly cheaper settings for calibration iteration.
    calib_N, calib_L = 512, 10.0

    print("Cross-model American calibration: Kou targets -> calibrated fit")
    print(f"S0={S0} r={r} q={q} divs={divs} quote_set=OTM(puts K<=S0, calls K>S0)")
    print(f"Calib model: {calib_model}")
    print(f"Strikes={strikes}")
    print(f"Maturities={mats}")

    print("-")
    print("Generating Kou targets (COS)...")
    quotes = _price_kou_targets(
        strikes=strikes,
        maturities=mats,
        S0=S0,
        r=r,
        q=q,
        divs=divs,
        N=target_N,
        L=target_L,
        steps=steps,
        beta=beta,
    )
    y = np.array([q.price for q in quotes], dtype=float)

    print(f"Built {len(quotes)} OTM quotes")
    for qte in sorted(quotes, key=lambda z: (z.T, z.K)):
        print(f"  T={qte.T:>5.3f} K={qte.K:>7.2f} {_option_label(qte.is_call)}  target={qte.price:>12.6f}")

    if calib_model == "cgmy":
        # Initial CGMY guess (reasonable-ish)
        x0 = np.array([0.8, 5.0, 5.0, 0.7], dtype=float)
        lb = np.array([1e-6, 1e-3, 1e-3, 1e-3], dtype=float)
        ub = np.array([50.0, 200.0, 200.0, 1.95], dtype=float)
        param_labels = ["C", "G", "M", "Y"]
    elif calib_model == "vg":
        # Initial VG guess + q guess.
        # q is calibrated too; per request start with q=0.0.
        x0 = np.array([-0.2, 0.12, 0.22, 0.0], dtype=float)
        lb = np.array([-5.0, 1e-6, 1e-6, -0.25], dtype=float)
        ub = np.array([5.0, 5.0, 10.0, 0.25], dtype=float)
        param_labels = ["theta", "sigma", "nu", "q"]
    elif calib_model == "merton":
        # Initial Merton guess + q guess.
        # q is calibrated too; start with q=0.0.
        x0 = np.array([0.15, 0.8, -0.05, 0.2, 0.0], dtype=float)
        lb = np.array([1e-6, 1e-6, -2.0, 1e-6, -0.25], dtype=float)
        ub = np.array([3.0, 10.0, 2.0, 2.0, 0.25], dtype=float)
        param_labels = ["sigma", "lam", "muJ", "sigmaJ", "q"]
    else:
        raise ValueError(f"Unknown CALIB_MODEL={calib_model!r}, expected 'cgmy', 'vg', or 'merton'")

    # Weight residuals by a price scale.
    scale = np.maximum(1.0, np.abs(y))

    eval_count = {"n": 0}

    def _run_one(x_start: np.ndarray) -> tuple[opt.OptimizeResult, np.ndarray, np.ndarray]:
        cache: dict[str, object] = {"x": None, "px": None, "J": None}

        def _eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_in = np.asarray(x, dtype=float)
            x_cached = cache.get("x")
            if x_cached is not None and np.allclose(x_in, np.asarray(x_cached), rtol=0.0, atol=0.0):
                return np.asarray(cache["px"], dtype=float), np.asarray(cache["J"], dtype=float)

            if calib_model == "cgmy":
                px, J = _cgmy_prices_and_jac(
                    x=x_in,
                    quotes=quotes,
                    S0=S0,
                    r=r,
                    q=q,
                    divs=divs,
                    N=calib_N,
                    L=calib_L,
                    steps=steps,
                    beta=beta,
                )
            elif calib_model == "vg":
                px, J = _vg_prices_and_jac(
                    x=x_in,
                    quotes=quotes,
                    S0=S0,
                    r=r,
                    divs=divs,
                    N=calib_N,
                    L=calib_L,
                    steps=steps,
                    beta=beta,
                )
            else:
                px, J = _merton_prices_and_jac(
                    x=x_in,
                    quotes=quotes,
                    S0=S0,
                    r=r,
                    divs=divs,
                    N=calib_N,
                    L=calib_L,
                    steps=steps,
                    beta=beta,
                )
            cache["x"] = x_in.copy()
            cache["px"] = px
            cache["J"] = J

            eval_count["n"] += 1
            if print_every > 0 and (eval_count["n"] % print_every == 0):
                err = px - y
                rmse = float(np.sqrt(np.mean(err * err)))
                max_abs = float(np.max(np.abs(err)))
                x_str = ",".join(f"{lab}={val:.6g}" for lab, val in zip(param_labels, x_in.tolist()))
                print(f"  eval={eval_count['n']:>4d} x=[{x_str}]  rmse={rmse:.4e}  max|err|={max_abs:.4e}")
            return px, J

        def fun(x: np.ndarray) -> np.ndarray:
            px, _ = _eval(x)
            return (px - y) / scale

        def jac(x: np.ndarray) -> np.ndarray:
            _, J = _eval(x)
            return J / scale[:, None]

        res = opt.least_squares(
            fun,
            np.asarray(x_start, dtype=float),
            jac=jac,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            verbose=scipy_verbose,
        )

        px_hat, _ = _eval(res.x)
        return res, px_hat, jac(res.x)

    print("-")
    print(f"Calibrating {calib_model.upper()} to Kou targets (COS-American sensitivities)...")
    print(f"Settings: calib_N={calib_N} calib_L={calib_L} steps={steps} beta={beta} max_nfev={max_nfev}")
    # Track the initial cost for clearer convergence reporting.
    def _cost_from_px(px: np.ndarray) -> float:
        rr = (np.asarray(px, dtype=float) - y) / scale
        return 0.5 * float(np.sum(rr * rr))

    if calib_model == "cgmy":
        px0, _ = _cgmy_prices_and_jac(
            x=x0,
            quotes=quotes,
            S0=S0,
            r=r,
            q=q,
            divs=divs,
            N=calib_N,
            L=calib_L,
            steps=steps,
            beta=beta,
        )
    elif calib_model == "vg":
        px0, _ = _vg_prices_and_jac(
            x=x0,
            quotes=quotes,
            S0=S0,
            r=r,
            divs=divs,
            N=calib_N,
            L=calib_L,
            steps=steps,
            beta=beta,
        )
    else:
        px0, _ = _merton_prices_and_jac(
            x=x0,
            quotes=quotes,
            S0=S0,
            r=r,
            divs=divs,
            N=calib_N,
            L=calib_L,
            steps=steps,
            beta=beta,
        )
    cost0 = _cost_from_px(px0)

    res, px_hat, _J_hat = _run_one(x0)

    x_hat = res.x
    err = px_hat - y

    print("-")
    print(
        f"Success: {res.success}  status={res.status}  nfev={res.nfev}  cost={res.cost:.6e}  "
        f"cost0={cost0:.6e}  dcost={res.cost - cost0:+.3e}"
    )
    print(f"Termination: {res.message}")
    print(f"Optimality (inf-norm grad): {getattr(res, 'optimality', float('nan')):.3e}")

    print(f"Fitted {calib_model.upper()} params:")
    if calib_model == "cgmy":
        print(f"  C={x_hat[0]:.6g}  G={x_hat[1]:.6g}  M={x_hat[2]:.6g}  Y={x_hat[3]:.6g}")
    elif calib_model == "vg":
        print(f"  theta={x_hat[0]:.6g}  sigma={x_hat[1]:.6g}  nu={x_hat[2]:.6g}  q={x_hat[3]:.6g}")
    else:
        print(f"  sigma={x_hat[0]:.6g}  lam={x_hat[1]:.6g}  muJ={x_hat[2]:.6g}  sigmaJ={x_hat[3]:.6g}  q={x_hat[4]:.6g}")
    print(f"Fit quality: max abs err={float(np.max(np.abs(err))):.4e}, rmse={float(np.sqrt(np.mean(err**2))):.4e}")

    print("-")
    print("Per-quote fit:")
    for qte, px_mkt, px_fit in sorted(zip(quotes, y.tolist(), px_hat.tolist()), key=lambda z: (z[0].T, z[0].K)):
        print(
            f"  T={qte.T:>5.3f} K={qte.K:>7.2f} {_option_label(qte.is_call)}"
            f"  target={px_mkt:>12.6f}  fit={px_fit:>12.6f}  err={px_fit - px_mkt:+.3e}"
        )

    # --- IV plot in time-normalized log-moneyness ---
    try:
        import csv

        import matplotlib.pyplot as plt

        xs: list[float] = []
        iv_target: list[float] = []
        iv_fit: list[float] = []
        Ts: list[float] = []
        Ks: list[float] = []
        types: list[str] = []

        for qte, px_mkt, px_fit in zip(quotes, y.tolist(), px_hat.tolist()):
            T = float(qte.T)
            F = _forward_proxy(S0=S0, r=r, q=q, divs=divs, T=T)
            x = float(np.log(float(qte.K) / F) / np.sqrt(T))
            df = float(np.exp(-r * T))

            iv_mkt = _bs_implied_vol_forward(price=px_mkt, F=F, K=qte.K, df=df, T=T, is_call=qte.is_call)
            iv_mod = _bs_implied_vol_forward(price=px_fit, F=F, K=qte.K, df=df, T=T, is_call=qte.is_call)

            xs.append(x)
            iv_target.append(iv_mkt)
            iv_fit.append(iv_mod)
            Ts.append(T)
            Ks.append(float(qte.K))
            types.append(_option_label(qte.is_call))

        # Save CSV for inspection.
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["T", "K", "type", "x_logmoneyness_sqrtT", "iv_target", "iv_fit", "price_target", "price_fit"])
            for T, K, typ, x, ivt, ivf, pt, pf in zip(Ts, Ks, types, xs, iv_target, iv_fit, y.tolist(), px_hat.tolist()):
                w.writerow([T, K, typ, x, ivt, ivf, pt, pf])

        # Plot style requested:
        # - target: red hollow circles
        # - calibrated: blue X
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        # Connect target points by expiry (helps distinguish expiries visually).
        for T in sorted(set(Ts)):
            idxs = [i for i, t in enumerate(Ts) if abs(float(t) - float(T)) < 1e-12]
            if len(idxs) < 2:
                continue
            idxs_sorted = sorted(idxs, key=lambda i: xs[i])
            x_line = [xs[i] for i in idxs_sorted]
            y_line = [iv_target[i] for i in idxs_sorted]
            ax.plot(x_line, y_line, color="red", alpha=0.5, linewidth=2.5)

        ax.scatter(
            xs,
            iv_target,
            s=40,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label="Target",
        )
        ax.scatter(
            xs,
            iv_fit,
            s=40,
            marker="x",
            color="blue",
            linewidths=1.4,
            label="Calibrated",
        )

        ax.set_xlabel(r"time-normalized log-moneyness $x = \ln(K/F)/\sqrt{T}$")
        ax.set_ylabel("BS implied vol (from American price)")
        ax.set_title("Kou targets vs CGMY fit (OTM quotes)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

        print("-")
        print(f"Saved IV plot: {plot_path}")
        print(f"Saved IV data: {csv_path}")
    except Exception as e:
        print("-")
        print(f"IV plot skipped (matplotlib/csv issue): {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
