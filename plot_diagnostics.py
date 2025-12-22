"""Diagnostic plots for COS pricing vs references.

Produces PNGs in figs/:
- cos_vs_bs_gbm.png: COS European vs Black-Scholes (no divs).
- american_hard_vs_soft.png: American GBM hard max vs softmax rollback and intrinsic.
- levy_vs_equiv_gbm.png: Merton/Kou/VG vs their variance-matched GBM proxies.
- american_dividend_continuation.png: European vs American continuation at t=0.
- american_through_time.png: COS European, American, and FDM comparison through time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import root_scalar
import pandas as pd

# Global number of timesteps for all pricers
NT_GLOBAL = 250

from american_options import (
    GBMCHF,
    MertonCHF,
    KouCHF,
    VGCHF,
    cash_divs_to_proportional_divs,
    equivalent_gbm,
)

# Global override for dividends used by plotting utilities. Set this to an empty dict
# to zero-out dividends globally in all plotting functions, or to a dict of the
# same form as used elsewhere (e.g. {0.25: (0.02, 1e-10)}).
DIVS_OVERRIDE = None

def set_divs_override(divs: dict | None):
    """Set a global override for dividends used in plotting utilities.

    Usage:
        set_divs_override({})   # zero all dividends
        set_divs_override(None) # restore default per-function divs
    """
    global DIVS_OVERRIDE
    DIVS_OVERRIDE = divs


def _get_divs(default_divs: dict) -> dict:
    """Return the global override if present, else the provided default."""
    return DIVS_OVERRIDE if DIVS_OVERRIDE is not None else default_divs
from american_options.engine import COSPricer, forward_price


def bs_call(S, K, r, q, vol, T):
    sigma = vol
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class FDMPricer:
    """Implicit finite difference pricer for GBM with discrete dividends.

    Project-wide convention: dividends are specified as cash amounts in spot currency:
        divs[t] = (D_mean, D_std)
    For these diagnostics, we convert them internally to a proportional form using
    the expected pre-div forward (same approximation as the COS engine).
    """

    def __init__(self, S0, r, q, vol, divs, NS=200, NT=None):
        self.S0 = S0
        self.r = r
        self.q = q
        self.vol = vol
        self.divs = cash_divs_to_proportional_divs(float(S0), float(r), float(q), divs)
        self.NS = NS
        self.NT = NT if NT is not None else NT_GLOBAL

    def price(self, K, T, american=False, is_call=True, debug=False):
        """Implicit backward Euler FD for European/American call/put with discrete proportional dividends.

        - Dividend mapping at exact ex-div steps: V_pre(S) = V_post(S*(1-m)).
        - `is_call=True` prices calls (default); `is_call=False` prices puts.
        - Early exercise applied after dividend mapping.
        """
        K = float(K)
        NS = self.NS
        NT = self.NT
        S_max = self.S0 * 4.0
        S = np.linspace(0.0, S_max, NS + 1)
        dS = S[1] - S[0]
        dt = T / NT if NT > 0 else T

        # Terminal payoff
        if is_call:
            V = np.maximum(S - K, 0.0)
        else:
            V = np.maximum(K - S, 0.0)

        # Precompute coefficients
        sigma2 = self.vol * self.vol
        i = np.arange(1, NS)
        Si = S[i]
        alpha = 0.5 * sigma2 * (Si ** 2) / (dS ** 2)
        beta = (self.r - self.q) * Si / (2.0 * dS)

        lower = dt * (alpha - beta)
        diag = 1.0 + dt * (self.r + 2.0 * alpha)
        upper = dt * (alpha + beta)

        # Dividend indices (forward indices)
        div_indices = set()
        div_index_to_m = {}
        for t_div, (m, _) in self.divs.items():
            if 0.0 < t_div <= T and dt > 0:
                idx = int(round(float(t_div) / float(dt)))
                div_indices.add(idx)
                # If multiple dividends collide on the same index, combine multiplicatively.
                # (This is a reasonable fallback when grid resolution cannot separate ex-div times.)
                if idx in div_index_to_m:
                    prev = float(div_index_to_m[idx])
                    div_index_to_m[idx] = 1.0 - (1.0 - prev) * (1.0 - float(m))
                else:
                    div_index_to_m[idx] = float(m)

        # Backward in time: n from NT-1 down to 0, time t_n = n*dt
        for n in range(NT - 1, -1, -1):
            t_n = n * dt
            # RHS from previous time level
            rhs = V[i]

            # Apply boundary conditions to RHS
            # For calls: S=0 -> 0, S->S_max -> S_max - K*exp(-r*(T-t_n))
            # For puts:  S=0 -> K*exp(-r*(T-t_n)), S->S_max -> 0
            if is_call:
                V_lower = 0.0
                # With discrete proportional dividends, as S->infty the call behaves like
                #   V(t,S) ~ S * exp(-q*(T-t)) * Π_{t_div in (t,T]}(1-m) - K*exp(-r*(T-t))
                # Here we are stepping to the *post-dividend* value at t_n (mapping, if any,
                # happens after this PDE step), so include only dividends strictly after t_n.
                prod_remain = 1.0
                for t_div, (m, _std) in self.divs.items():
                    if (t_n + 1e-12) < float(t_div) <= float(T) + 1e-12:
                        prod_remain *= max(1.0 - float(m), 1e-12)
                V_upper = (S_max * np.exp(-self.q * (T - t_n)) * prod_remain) - K * np.exp(-self.r * (T - t_n))
            else:
                V_lower = K * np.exp(-self.r * (T - t_n))
                V_upper = 0.0
            rhs[0] += lower[0] * V_lower
            rhs[-1] += upper[-1] * V_upper

            # Thomas algorithm
            c_prime = np.zeros_like(upper)
            d_prime = np.zeros_like(rhs)
            # Note: tridiagonal system uses a_i = -lower, b_i = diag, c_i = -upper
            c_prime[0] = -upper[0] / diag[0]
            d_prime[0] = rhs[0] / diag[0]
            for k in range(1, NS - 1):
                denom = diag[k] + lower[k] * c_prime[k - 1]
                c_prime[k] = -upper[k] / denom
                d_prime[k] = (rhs[k] + lower[k] * d_prime[k - 1]) / denom

            V_new = np.zeros(NS + 1)
            V_new[0] = V_lower
            V_new[-1] = V_upper
            V_new[NS - 1] = d_prime[-1]
            for k in range(NS - 2, 0, -1):
                V_new[k] = d_prime[k - 1] - c_prime[k - 1] * V_new[k + 1]

            # Dividend mapping at t_n if ex-div occurs
            if n in div_indices:
                # Map pre-div prices: V_pre(S) = V_post(S*(1-m))
                m = float(div_index_to_m.get(n, 0.0))
                if m > 0.0:
                    # CORRECT dividend mapping: V_pre(S) = V_post(S*(1-m))
                    V_new = np.interp(S * (1.0 - m), S, V_new, left=0.0, right=V_new[-1])

            # Early exercise (intrinsic)
            if american:
                if is_call:
                    intrinsic = np.maximum(S - K, 0.0)
                else:
                    intrinsic = np.maximum(K - S, 0.0)
                V_new = np.maximum(V_new, intrinsic)

            V = V_new

        # Interpolate to S0
        price = float(np.interp(self.S0, S, V))
        if debug:
            return price, S, V
        return price


def ensure_fig_dir():
    os.makedirs("figs", exist_ok=True)


def _invert_vol_for_european_cos_price(
    *,
    target_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    divs: dict,
    is_call: bool,
    N: int,
    L: float,
    vol_lo: float = 1e-6,
    vol_hi: float = 2.0,
    max_hi: float = 8.0,
) -> float:
    if not np.isfinite(target_price) or target_price <= 0.0:
        return float("nan")

    def f(vol: float) -> float:
        model = GBMCHF(S0, r, q, divs, {"vol": float(vol)})
        pricer = COSPricer(model, N=N, L=L)
        price = float(pricer.european_price(np.array([K], dtype=float), T, is_call=is_call)[0])
        return price - target_price

    lo = float(vol_lo)
    hi = float(vol_hi)
    flo = f(lo)
    fhi = f(hi)

    while np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) == np.sign(fhi) and hi < max_hi:
        hi = min(max_hi, hi * 1.5)
        fhi = f(hi)

    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return float("nan")
    if np.sign(flo) == np.sign(fhi):
        return float("nan")

    res = root_scalar(f, bracket=(lo, hi), method="brentq")
    return float(res.root) if res.converged else float("nan")


def plot_cos_vs_bs():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    vol = 0.25
    K = np.linspace(60, 140, 81)
    gbm = GBMCHF(S0, r, q, {}, {"vol": vol})
    pr = COSPricer(gbm, N=512, L=8.0)
    cos_prices = pr.european_price(K, T)
    bs_prices = np.array([bs_call(S0, k, r, q, vol, T) for k in K])
    diff = cos_prices - bs_prices

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(K, bs_prices, label="BS analytic", lw=2)
    ax[0].plot(K, cos_prices, "--", label="COS (GBM)")
    ax[0].set_ylabel("Call price")
    ax[0].legend()
    ax[0].set_title("COS European vs Black-Scholes (GBM, no divs)")

    ax[1].plot(K, diff, label="COS - BS")
    ax[1].axhline(0, color="k", lw=1, alpha=0.7)
    ax[1].set_xlabel("Strike")
    ax[1].set_ylabel("Difference")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("figs/cos_vs_bs_gbm.png", dpi=150)
    plt.close(fig)


def plot_american_hard_vs_soft():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    vol = 0.1
    divs = {0.5: (5.0, 0.0)}

    K = np.linspace(70, 130, 61)
    gbm = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(gbm, N=256, L=8.0)
    amer_hard, euro_hard = pr.american_price(K, T, steps=NT_GLOBAL, beta=50.0, use_softmax=False, return_european=True)
    amer_soft, euro_soft = pr.american_price(K, T, steps=NT_GLOBAL, beta=20.0, use_softmax=True, return_european=True)
    intrinsic = np.maximum(S0 - K, 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K, amer_hard, label="American (hard max)")
    ax.plot(K, amer_soft, "--", label="American (softmax)")
    ax.plot(K, euro_hard, ":", label="European (hard max)")
    ax.plot(K, euro_soft, "*", label="European (softmax)")
    ax.plot(K, intrinsic, "-.", label="Intrinsic")
    ax.set_title("American COS: hard vs softmax")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/american_hard_vs_soft.png", dpi=150)
    plt.close(fig)


def plot_levy_vs_equiv_gbm():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    K = np.linspace(70, 130, 61)

    # Merton (moderate jumps)
    merton = MertonCHF(S0, r, q, {}, {"vol": 0.2, "lam": 0.1, "muJ": -0.05, "sigmaJ": 0.1})
    m_gbm = equivalent_gbm(merton, T)
    pr_m = COSPricer(merton, N=512, L=8.0)
    pr_mg = COSPricer(m_gbm, N=512, L=8.0)
    pm = pr_m.european_price(K, T)
    pmg = pr_mg.european_price(K, T)

    # Kou (small intensity, symmetric-ish)
    kou = KouCHF(S0, r, q, {}, {"vol": 0.2, "lam": 0.05, "p": 0.5, "eta1": 15.0, "eta2": 18.0})
    k_gbm = equivalent_gbm(kou, T)
    pr_k = COSPricer(kou, N=512, L=8.0)
    pr_kg = COSPricer(k_gbm, N=512, L=8.0)
    pk = pr_k.european_price(K, T)
    pkg = pr_kg.european_price(K, T)

    # VG (Gaussian limit mapping)
    vol_vg = 0.25
    theta = -0.5 * vol_vg * vol_vg
    vg = VGCHF(S0, r, q, {}, {"theta": theta, "sigma": vol_vg, "nu": 1e-4})
    vg_gbm = equivalent_gbm(vg, T)
    pr_v = COSPricer(vg, N=512, L=8.0)
    pr_vg = COSPricer(vg_gbm, N=512, L=8.0)
    pv = pr_v.european_price(K, T)
    pvg = pr_vg.european_price(K, T)

    fig, ax = plt.subplots(3, 2, figsize=(10, 10), sharex=True)

    def block(row, prices, proxy, title):
        ax[row, 0].plot(K, prices, label=title)
        ax[row, 0].plot(K, proxy, "--", label="equiv GBM")
        ax[row, 0].set_ylabel("Price")
        ax[row, 0].legend()
        ax[row, 0].set_title(title)

        ax[row, 1].plot(K, prices - proxy, label="diff")
        ax[row, 1].axhline(0, color="k", lw=1)
        ax[row, 1].legend()
        ax[row, 1].set_title(f"{title} - GBM proxy")

    block(0, pm, pmg, "Merton")
    block(1, pk, pkg, "Kou")
    block(2, pv, pvg, "VG (theta=-0.5 sigma^2)")

    ax[2, 0].set_xlabel("Strike")
    ax[2, 1].set_xlabel("Strike")
    fig.tight_layout()
    fig.savefig("figs/levy_vs_equiv_gbm.png", dpi=150)
    plt.close(fig)


def plot_vg_implied_vol_smile_dividend_uncertainty(
    *,
    out_png: str = "figs/vg_iv_smile_div_uncertainty.png",
    out_csv: str = "figs/vg_iv_smile_div_uncertainty.csv",
    S0: float = 100.0,
    T: float = 1.0,
    r: float = 0.02,
    q: float = 0.0,
    strikes: np.ndarray | None = None,
    div_mean: float = 2.0,
    div_times: tuple[float, ...] = (0.25, 0.75),
    div_stds: tuple[float, ...] = (0.0, 0.01, 0.02),
    vg_sigma: float = 0.20,
    vg_theta: float = -0.10,
    vg_nu: float = 0.20,
    is_call: bool = True,
    N: int = NT_GLOBAL,
    L: float = 12.0,
) -> pd.DataFrame:
    """Implied vol smile vs strike for VG European prices under dividend uncertainty."""

    ensure_fig_dir()

    if strikes is None:
        strikes = np.linspace(60.0, 140.0, 33)
    strikes = np.asarray(strikes, dtype=float)

    rows = []
    for div_std in div_stds:
        divs = {float(t): (float(div_mean), float(div_std)) for t in div_times}
        vg_model = VGCHF(S0, r, q, divs, {"sigma": vg_sigma, "theta": vg_theta, "nu": vg_nu})
        vg_pricer = COSPricer(vg_model, N=N, L=L)

        for K in strikes:
            vg_price = float(vg_pricer.european_price(np.array([K], dtype=float), T, is_call=is_call)[0])
            iv = _invert_vol_for_european_cos_price(
                target_price=vg_price,
                S0=S0,
                K=float(K),
                T=T,
                r=r,
                q=q,
                divs=divs,
                is_call=is_call,
                N=N,
                L=L,
            )
            rows.append({"K": float(K), "div_std": float(div_std), "vg_price": vg_price, "iv_gbm": iv})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for div_std in div_stds:
        dfi = df[df["div_std"] == float(div_std)].sort_values("K")
        ax.plot(dfi["K"], dfi["iv_gbm"], label=f"div std = {div_std:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol (GBM via COS inversion)")
    ax.set_title("VG European implied vol smile vs strike\n(discrete dividend uncertainty)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return df


def gbm_mc_european_price_with_uncertain_prop_divs(
    *,
    S0: float,
    K: float | np.ndarray,
    T: float,
    r: float,
    q: float,
    vol: float,
    divs: dict,
    is_call: bool = True,
    n_paths: int = 200_000,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo European price under GBM with uncertain discrete dividends.

    Project-wide convention: divs are cash amounts in spot currency.
    For MC we convert them to the same internal proportional form used by COS.

    Returns (price, standard_error) for each strike in K.
    """
    S = gbm_mc_terminal_spots_with_uncertain_prop_divs(
        S0=S0,
        T=T,
        r=r,
        q=q,
        vol=vol,
        divs=divs,
        n_paths=n_paths,
        seed=seed,
    )
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if is_call:
        payoff = np.maximum(S[:, None] - K[None, :], 0.0)
    else:
        payoff = np.maximum(K[None, :] - S[:, None], 0.0)

    disc = np.exp(-float(r) * float(T))
    price = disc * payoff.mean(axis=0)
    # standard error of discounted payoff
    se = disc * payoff.std(axis=0, ddof=1) / np.sqrt(float(n_paths))
    return price.astype(float), se.astype(float)


def gbm_mc_terminal_spots_with_uncertain_prop_divs(
    *,
    S0: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    divs: dict,
    n_paths: int = 200_000,
    seed: int = 123,
) -> np.ndarray:
    """Simulate terminal spots S_T under GBM with uncertain discrete dividends.

    Project-wide convention: divs are cash amounts in spot currency.
    We convert to internal proportional parameters and then simulate multiplicative drops.
    """
    rng = np.random.default_rng(seed)

    divs_prop = cash_divs_to_proportional_divs(float(S0), float(r), float(q), divs)

    div_items = [(float(t), float(m), float(std)) for t, (m, std) in divs_prop.items() if 0.0 < float(t) <= float(T)]
    div_items.sort(key=lambda z: z[0])

    times = [0.0] + [t for (t, _, _) in div_items] + [float(T)]
    dts = np.diff(np.array(times, dtype=float))

    S = np.full(int(n_paths), float(S0), dtype=float)

    drift_per_year = float(r) - float(q) - 0.5 * float(vol) ** 2
    vol_sqrt = float(vol)

    for idx, dt in enumerate(dts, start=1):
        if dt > 0.0:
            Z = rng.standard_normal(S.shape[0])
            S *= np.exp(drift_per_year * dt + vol_sqrt * np.sqrt(dt) * Z)

        if idx < len(times) - 1:
            _, m, std = div_items[idx - 1]
            if m != 0.0 or std != 0.0:
                Zd = rng.standard_normal(S.shape[0])
                lnD = np.log(max(1.0 - m, 1e-12)) - 0.5 * (std ** 2) + std * Zd
                S *= np.exp(lnD)

    return S


def plot_gbm_mc_vs_cos_dividend_uncertainty(
    *,
    out_png: str = "figs/gbm_mc_vs_cos_div_uncertainty.png",
    out_csv: str = "figs/gbm_mc_vs_cos_div_uncertainty.csv",
    S0: float = 100.0,
    T: float = 0.3,
    r: float = 0.0,
    q: float = 0.0,
    vol: float = 0.10,
    strikes: np.ndarray | None = None,
    div_mean: float = 3.0,
    div_times: tuple[float, ...] = (0.25, 0.75),
    div_stds: tuple[float, ...] | None = (0.0, 2.0, 4.0, 8.0),
    div_std: float | None = None,
    is_call: bool = True,
    mc_paths: int = 1_000_000,
    mc_seed: int = 123,
    cos_N: int = 512,
    cos_L: float = 10.0,
    iv_cos_N: int = NT_GLOBAL,
    iv_cos_L: float = 12.0,
    use_otm_for_iv: bool = True,
) -> pd.DataFrame:
    """Compare GBM MC vs GBM COS under uncertain *cash* dividends.

    Also computes an "equivalent" implied vol by inverting to a deterministic-dividend GBM (std=0)
    for both MC and COS target prices.
    """
    ensure_fig_dir()

    # Back-compat: if div_std provided, use it as a single scenario.
    if div_std is not None:
        div_stds_use = (float(div_std),)
    else:
        div_stds_use = tuple(float(x) for x in (div_stds or (0.0,)))

    # Auto strike grid: centered around the deterministic-cash-dividend forward F_det,
    # with width scaled by vol*sqrt(T). This keeps the strike range sensible as T/vol change.
    # Note: dividends are cash amounts; we must not treat div_mean as a proportional drop.
    divs_det = {float(t): (float(div_mean), 0.0) for t in div_times if 0.0 < float(t) <= float(T) + 1e-12}
    F_det = float(forward_price(float(S0), float(r), float(q), float(T), divs_det))

    if strikes is None:
        n_sigma = 3.0
        width = float(vol) * float(np.sqrt(max(T, 1e-16)))
        K_lo = F_det * float(np.exp(-n_sigma * width))
        K_hi = F_det * float(np.exp(n_sigma * width))
        strikes = np.linspace(K_lo, K_hi, 33)
    strikes = np.asarray(strikes, dtype=float)

    rows = []

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for scenario_idx, div_std_s in enumerate(div_stds_use):
        divs_unc = {float(t): (float(div_mean), float(div_std_s)) for t in div_times if 0.0 < float(t) <= float(T) + 1e-12}

        # Simulate once; compute both call/put prices for OTM IV extraction.
        S_T = gbm_mc_terminal_spots_with_uncertain_prop_divs(
            S0=S0,
            T=T,
            r=r,
            q=q,
            vol=vol,
            divs=divs_unc,
            n_paths=mc_paths,
            seed=mc_seed + 97 * scenario_idx,
        )
        disc = np.exp(-float(r) * float(T))
        call_payoff = np.maximum(S_T[:, None] - strikes[None, :], 0.0)
        put_payoff = np.maximum(strikes[None, :] - S_T[:, None], 0.0)
        mc_call = disc * call_payoff.mean(axis=0)
        mc_put = disc * put_payoff.mean(axis=0)
        mc_call_se = disc * call_payoff.std(axis=0, ddof=1) / np.sqrt(float(mc_paths))
        mc_put_se = disc * put_payoff.std(axis=0, ddof=1) / np.sqrt(float(mc_paths))

        # For the price panel keep a single instrument curve (default: whatever caller requested)
        if is_call:
            mc_price, mc_se = mc_call, mc_call_se
        else:
            mc_price, mc_se = mc_put, mc_put_se

        gbm_unc = GBMCHF(S0, r, q, divs_unc, {"vol": vol})
        pr_unc = COSPricer(gbm_unc, N=cos_N, L=cos_L)
        cos_call = np.asarray(pr_unc.european_price(strikes, T, is_call=True), dtype=float)
        cos_put = np.asarray(pr_unc.european_price(strikes, T, is_call=False), dtype=float)
        cos_price = cos_call if is_call else cos_put

        iv_mc = []
        iv_cos = []
        iv_opt_type = []
        for idx, K in enumerate(strikes):
            if use_otm_for_iv:
                # Use deterministic forward as moneyness split to avoid low-vega ITM inversion noise.
                use_call = bool(float(K) >= F_det)
            else:
                use_call = bool(is_call)

            target_mc = float(mc_call[idx] if use_call else mc_put[idx])
            target_cos = float(cos_call[idx] if use_call else cos_put[idx])
            iv_opt_type.append("C" if use_call else "P")

            iv_mc.append(
                _invert_vol_for_european_cos_price(
                    target_price=target_mc,
                    S0=S0,
                    K=float(K),
                    T=T,
                    r=r,
                    q=q,
                    divs=divs_det,
                    is_call=use_call,
                    N=iv_cos_N,
                    L=iv_cos_L,
                )
            )
            iv_cos.append(
                _invert_vol_for_european_cos_price(
                    target_price=target_cos,
                    S0=S0,
                    K=float(K),
                    T=T,
                    r=r,
                    q=q,
                    divs=divs_det,
                    is_call=use_call,
                    N=iv_cos_N,
                    L=iv_cos_L,
                )
            )
        iv_mc = np.asarray(iv_mc, dtype=float)
        iv_cos = np.asarray(iv_cos, dtype=float)

        for idx, K in enumerate(strikes):
            rows.append(
                {
                    "div_std": float(div_std_s),
                    "K": float(K),
                    "iv_option": iv_opt_type[idx],
                    "mc_price": float(mc_price[idx]),
                    "mc_se": float(mc_se[idx]),
                    "mc_call": float(mc_call[idx]),
                    "mc_put": float(mc_put[idx]),
                    "mc_call_se": float(mc_call_se[idx]),
                    "mc_put_se": float(mc_put_se[idx]),
                    "cos_price": float(cos_price[idx]),
                    "cos_call": float(cos_call[idx]),
                    "cos_put": float(cos_put[idx]),
                    "abs_diff": float(abs(mc_price[idx] - cos_price[idx])),
                    "iv_det_gbm_from_mc": float(iv_mc[idx]),
                    "iv_det_gbm_from_cos": float(iv_cos[idx]),
                }
            )

        label_suffix = f"std={div_std_s:.3f}"
        ax[0].plot(strikes, cos_price, lw=2, label=f"COS {label_suffix}")
        ax[0].errorbar(
            strikes,
            mc_price,
            yerr=2.0 * mc_se,
            fmt="o",
            ms=3,
            capsize=2,
            alpha=0.8,
            label=f"MC ±2SE {label_suffix}",
        )
        ax[1].plot(strikes, iv_cos, lw=2, label=f"IV from COS {label_suffix}")
        ax[1].plot(strikes, iv_mc, "--", lw=1.5, label=f"IV from MC {label_suffix}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    ax[0].set_ylabel("European price")
    ax[0].set_title("GBM: MC vs COS with uncertain cash dividends (sweep div std)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(ncol=2, fontsize=9)

    ax[1].set_xlabel("Strike K")
    ax[1].set_ylabel("Implied vol (det-div GBM inversion, OTM)" if use_otm_for_iv else "Implied vol (det-div GBM inversion)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return df


def plot_american_dividend_continuation():
    """Compare European continuation (standalone COS) vs American continuation through dividend rollback.

    Plots continuation value at intermediate times during backward induction through dividends.
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    _default_divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}  # cash divs at t=0.25, 0.75
    divs = _get_divs(_default_divs)
    K = 100.0  # ATM strike
    vol = 0.25
    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(model, N=512, L=8.0)

    # Grid of spot prices to evaluate continuation
    S_eval = np.linspace(70, 130, 121)

    # European continuation: compute European price from each spot to maturity T
    euro_vals = []
    for s in S_eval:
        m_temp = GBMCHF(s, r, q, divs, {"vol": vol})
        pr_temp = COSPricer(m_temp, N=256, L=8.0)
        p = pr_temp.european_price(np.array([K]), T)[0]
        euro_vals.append(p)
    euro_vals = np.array(euro_vals)

    # American continuation: compute American price from each spot to maturity T
    amer_vals = []
    for s in S_eval:
        m_temp = GBMCHF(s, r, q, divs, {"vol": vol})
        pr_temp = COSPricer(m_temp, N=256, L=8.0)
        p = pr_temp.american_price(np.array([K]), T, steps=NT_GLOBAL, use_softmax=False)[0]
        amer_vals.append(p)
    amer_vals = np.array(amer_vals)

    # Intrinsic value
    intrinsic = np.maximum(S_eval - K, 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_eval, euro_vals, color="orange", lw=2, label="European (COS)")
    ax.plot(S_eval, amer_vals, "bx", markersize=4, label="American (rollback)")
    ax.plot(S_eval, amer_vals, "r-", lw=2, alpha=0.5, label="American (cont)")
    ax.plot(S_eval, intrinsic, "k--", lw=1, alpha=0.5, label="Intrinsic")

    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Option value")
    ax.set_title(f"Continuation value: European vs American (K={K}, T={T}, divs at t=0.25, 0.75)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figs/american_dividend_continuation.png", dpi=150)
    plt.close(fig)


def plot_american_through_time():
    """Plot option values through time (t=0→T) comparing COS vs FDM methods.

    Uses cached trajectory from COS backward induction for efficiency.
    
    Compares:
    - COS European (standalone)
    - COS European (extracted from American rollback)
    - COS American (from cached trajectory)
    - FDM American (fast approximation at key points)
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    _default_divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}
    divs = _get_divs(_default_divs)
    K = np.array([100.0])
    vol = 0.25

    # Run COS backward induction once with trajectory caching (both American and continuation)
    pricer = COSPricer(GBMCHF(S0, r, q, divs, {"vol": vol}), N=128, L=8.0)
    amer_prices, euro_prices, cos_amer_trajectory, cos_cont_trajectory = pricer.american_price(
        K, T, steps=NT_GLOBAL, return_european=True, return_trajectory=True, return_continuation_trajectory=True
    )
    
    # Extract times and values from trajectories
    cos_times = np.array([t for t, _ in cos_amer_trajectory])
    cos_american = np.array([p for _, p in cos_amer_trajectory])
    cos_continuation = np.array([p for _, p in cos_cont_trajectory])
    
    # Compute standalone European at trajectory times
    cos_euro_standalone = []
    for t in cos_times:
        tau = T - t
        euro_sa = pricer.european_price(K, tau)[0]
        cos_euro_standalone.append(euro_sa)
    cos_euro_standalone = np.array(cos_euro_standalone)
    
    # For validation, compute standalone European at trajectory times (should match continuation)
    cos_euro_standalone = []
    for t in cos_times:
        tau = T - t
        euro_sa = pricer.european_price(K, tau)[0]
        cos_euro_standalone.append(euro_sa)
    cos_euro_standalone = np.array(cos_euro_standalone)
    
    # FDM approximation at trajectory times (fast due to simple formula)
    fdm = FDMPricer(S0, r, q, vol, divs)
    fdm_american = np.array([fdm.price(K[0], T - t, american=True) for t in cos_times])

    # Plot (focus on continuation through time)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(cos_times, cos_continuation, label="COS Continuation (European from rollback)", linewidth=2, color="cyan", linestyle="--")
    ax.plot(cos_times, cos_american, label="COS American", linewidth=2.5, color="red", linestyle="-", marker="o", markersize=3)
    # Optional: show standalone European to confirm it matches continuation
    ax.plot(cos_times, cos_euro_standalone, label="COS European (standalone)", linewidth=1.5, color="orange", linestyle=":", alpha=0.7)

    # Overlay LSMC (train/refit) from CSV if present
    csvp = "figs/continuation_through_time.csv"
    if os.path.exists(csvp):
        try:
            df = pd.read_csv(csvp)
            if "lsmc_american_train" in df.columns:
                ax.plot(df.time, df.lsmc_american_train, label="LSMC American (train)", color="green", linestyle="-.")
                ax.plot(df.time, df.lsmc_american_refit, label="LSMC American (refit)", color="lime", linestyle=":")
                if "lsmc_cont_train_at_S0" in df.columns:
                    ax.plot(df.time, df.lsmc_cont_train_at_S0, label="LSMC cont@S0 (train)", color="green", linestyle="--", alpha=0.7)
                if "lsmc_cont_refit_at_S0" in df.columns:
                    ax.plot(df.time, df.lsmc_cont_refit_at_S0, label="LSMC cont@S0 (refit)", color="lime", linestyle="--", alpha=0.7)
            # plot any heavy refit columns present (e.g., lsmc_refit_500000)
            for c in df.columns:
                if c.startswith('lsmc_refit_') and c not in ('lsmc_american_refit', 'lsmc_american_train'):
                    n = c.split('_')[-1]
                    ax.plot(df.time, df[c], label=f"LSMC all-in refit ({n} paths)", color='black', linestyle='--', linewidth=1.6)
                if c.startswith('lsmc_cont_refit_at_S0_'):
                    n = c.split('_')[-1]
                    ax.plot(df.time, df[c], label=f"LSMC cont@S0 refit ({n} paths)", color='black', linestyle=':', alpha=0.7)
        except Exception:
            pass

    # Mark dividend times
    for t_div in [0.25, 0.75]:
        if t_div <= T:
            ax.axvline(x=t_div, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Time (t)", fontsize=11)
    ax.set_ylabel("Option Value", fontsize=11)
    ax.set_title(f"Continuation Through Time (Cached COS Trajectories)\n(S0={S0}, K={K[0]}, T={T}, σ={vol}, r={r}, q={q})", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figs/american_through_time.png", dpi=150)
    plt.close(fig)


def plot_continuation_through_time(do_lsmc=False):
    """Combined function: exports CSV and plots continuation through time.

    - Produces figs/continuation_through_time.csv
    - Produces figs/continuation_through_time.png
    - If do_lsmc=True, includes LSMC runs and plots; otherwise, skips them.
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {0.25: (2.0, 1e-6), 0.75: (2.0, 1e-6)}
    K = np.array([100.0])
    vol = 0.25

    N=2**9
    L=10.0
    base_model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pricer = COSPricer(base_model, N=N, L=L)
    # Use coarse grid plus fine windows around dividend dates
    eps = 1e-6
    coarse = np.linspace(eps, T-eps, 51)
    fine1 = np.linspace(0.24, 0.26, 5)
    fine2 = np.linspace(0.74, 0.76, 5)
    # round to avoid duplicates from float construction (e.g. two representations of 0.25)
    times = np.unique(np.round(np.concatenate([coarse, fine1, fine2]), 12))

    cos_euro_standalone = []
    cos_euro_from_rb = []
    cos_american = []
    fdm_euro = []
    fdm_american = []
    if do_lsmc:
        from american_options.lsmc import LSMCPricer
        lsmc = LSMCPricer(S0, r, q, divs, vol, seed=2025)
        lsmc_amer_train = []
        lsmc_amer_refit = []
        lsmc_cont_train = []
        lsmc_cont_refit = []
    for t in times:
        tau = T - t
        # Shift *cash* dividends to remaining horizon
        divs_shift = {dt - t: (Dm, Ds) for dt, (Dm, Ds) in divs.items() if dt > t + 1e-12}

        # If a shifted dividend is extremely close to tau=0, treat it as immediate.
        # With cash dividends, approximate an immediate dividend by subtracting its *mean* from spot.
        S0_eff = S0
        if divs_shift and NT_GLOBAL > 0 and tau > 0:
            dt_nom = tau / NT_GLOBAL
            # Treat dividends within ~one timestep as immediate to avoid step-placement artifacts.
            eps_immediate = 2.00 * dt_nom
            immediate_times = sorted([td for td in divs_shift.keys() if td <= eps_immediate])
            if immediate_times:
                for td in immediate_times:
                    Dm, _Ds = divs_shift[td]
                    S0_eff = float(max(1e-12, float(S0_eff) - float(Dm)))
                divs_shift = {td: v for td, v in divs_shift.items() if td > eps_immediate}

        # COS with shifted dividends
        model_t = GBMCHF(S0_eff, r, q, divs_shift, {"vol": vol})
        pricer_t = COSPricer(model_t, N=N, L=L)

        euro_sa = pricer_t.european_price(K, tau)[0]
        cos_euro_standalone.append(euro_sa)
        # Use a "harder" softmax (larger beta) so the smooth exercise operator
        # stays numerically close to the hard-max / FDM reference.
        amer_tau, euro_tau = pricer_t.american_price(K, tau, steps=NT_GLOBAL, beta=100.0, use_softmax=True, return_european=True)
        cos_american.append(amer_tau[0])
        cos_euro_from_rb.append(euro_tau[0])

        # FDM with shifted dividends
        fdm_t = FDMPricer(S0_eff, r, q, vol, divs_shift, NS=400, NT=NT_GLOBAL)
        fdm_euro.append(fdm_t.price(K[0], tau, american=False))
        fdm_american.append(fdm_t.price(K[0], tau, american=True))
        if do_lsmc:
            # LSMC pricing for this remaining horizon
            # Use a per-time pricer so we can incorporate S0_eff/divs_shift (esp. when an immediate dividend is applied).
            from american_options.lsmc import LSMCPricer
            lsmc_t = LSMCPricer(S0_eff, r, q, divs_shift, vol, seed=2025)
            tr_price, tr_cont, rf_price, rf_cont = lsmc_t.price_at_tau(K[0], tau, steps=NT_GLOBAL, n_train=2000, n_price=2000, deg=3)
            lsmc_amer_train.append(tr_price)
            lsmc_amer_refit.append(rf_price)
            lsmc_cont_train.append(tr_cont if tr_cont is not None else np.nan)
            lsmc_cont_refit.append(rf_cont if rf_cont is not None else np.nan)

    data = {
        "time": times,
        "cos_euro_standalone": cos_euro_standalone,
        "cos_euro_from_rollback": cos_euro_from_rb,
        "fdm_euro": fdm_euro,
        "cos_american": cos_american,
        "fdm_american": fdm_american,
    }
    if do_lsmc:
        data["lsmc_american_train"] = lsmc_amer_train
        data["lsmc_american_refit"] = lsmc_amer_refit
        data["lsmc_cont_train_at_S0"] = lsmc_cont_train
        data["lsmc_cont_refit_at_S0"] = lsmc_cont_refit

    df = pd.DataFrame(data)
    os.makedirs("figs", exist_ok=True)
    df.to_csv("figs/continuation_through_time.csv", index=False)
    print("Saved dataframe:", "figs/continuation_through_time.csv")
    do_print_rows = False
    if do_print_rows:
        # Show rows around dividend dates for inspection
        print("\nRows near first dividend (t~0.25):")
        print(df[(df.time >= 0.22) & (df.time <= 0.28)].to_string(index=False))
        print("\nRows near second dividend (t~0.75):")
        print(df[(df.time >= 0.72) & (df.time <= 0.78)].to_string(index=False))

    # Now plot using the just-generated DataFrame
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Europeans ---

    # COS standalone (unique color + X marker)
    ax.plot(df.time, df.cos_euro_standalone,
            label="COS European (standalone)",
            color="#000000", marker="x", markersize=7,
            linestyle="None", linewidth=0)

    # COS from rollback (semi-opaque line)
    ax.plot(df.time, df.cos_euro_from_rollback,
            label="COS European (from rollback)",
            color="#1f77b4", linewidth=1.0, linestyle="-", alpha=0.5)

    # FDM European (hollow circle, same color)
    ax.plot(df.time, df.fdm_euro,
            label="FDM European",
            color="#1f77b4", marker="o", markersize=6,
            markerfacecolor="none", linestyle="None", linewidth=0)


    # --- Americans ---

    # COS American (semi-opaque line)
    ax.plot(df.time, df.cos_american,
            label="COS American",
            color="#ff0e0e", linewidth=1.0, linestyle="-", alpha=0.5)

    # FDM American (hollow circle, same color)
    ax.plot(df.time, df.fdm_american,
            label="FDM American",
            color="#ff0e0e", marker="o", markersize=7,
            markerfacecolor="none", linestyle="None", linewidth=0)



    # ax.plot(df.time, df.cos_euro_standalone, label="COS European (standalone)", linewidth=1.8, color="orange", linestyle=":")
    # ax.plot(df.time, df.cos_euro_from_rollback, label="COS European (from rollback)", linewidth=2, color="cyan", linestyle="--")
    # ax.plot(df.time, df.fdm_euro, label="FDM European", linewidth=1.8, color="purple", linestyle=":")
    # ax.plot(df.time, df.cos_american, label="COS American", linewidth=2.5, color="red",alpha=0.5)
    # ax.plot(df.time, df.fdm_american, label="FDM American", linewidth=2, color="magenta", linestyle="--")



    if do_lsmc:
        if "lsmc_american_train" in df.columns:
            ax.plot(df.time, df.lsmc_american_train, label="LSMC American (train)", color="green", linestyle="-.")
            ax.plot(df.time, df.lsmc_american_refit, label="LSMC American (refit)", color="lime", linestyle=":")
            if "lsmc_cont_train_at_S0" in df.columns:
                ax.plot(df.time, df.lsmc_cont_train_at_S0, label="LSMC cont@S0 (train)", color="green", linestyle="--", alpha=0.7)
            if "lsmc_cont_refit_at_S0" in df.columns:
                ax.plot(df.time, df.lsmc_cont_refit_at_S0, label="LSMC cont@S0 (refit)", color="lime", linestyle="--", alpha=0.7)
        # plot any heavy refit columns present (e.g., lsmc_refit_500000)
        for c in df.columns:
            if c.startswith('lsmc_refit_') and c not in ('lsmc_american_refit', 'lsmc_american_train'):
                n = c.split('_')[-1]
                ax.plot(df.time, df[c], label=f"LSMC all-in refit ({n} paths)", color='black', linestyle='--', linewidth=1.6)
            if c.startswith('lsmc_cont_refit_at_S0_'):
                n = c.split('_')[-1]
                ax.plot(df.time, df[c], label=f"LSMC cont@S0 refit ({n} paths)", color='black', linestyle=':', alpha=0.7)

    # Mark dividend times
    for t_div in [0.25, 0.75]:
        if t_div <= T:
            ax.axvline(x=t_div, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Time (t)", fontsize=11)
    ax.set_ylabel("Option Value", fontsize=11)
    ax.set_title(f"Continuation + References Through Time\n(S0={S0}, K={K[0]}, T={T}, σ={vol}, r={r}, q={q})", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figs/continuation_through_time.png", dpi=150)
    plt.close(fig)


def plot_forward_parity_through_time():
    """Plot model-free forward vs COS-implied forward via put-call parity as time-from-now.

    Here the x-axis is maturity $\tau$ (years from now), i.e. we roll forward by
    increasing the horizon from 0→T.

    Project-wide convention: dividends are cash amounts in spot currency.

    We compute a model-free expected forward using the same approximation as the engine:
        F_model(0,\tau) = forward_price(S0, r, q, \tau, divs_cash_up_to_tau)

    Put-call parity implies:
        F_implied = exp(r * \tau) * (C - P) + K
    """
    ensure_fig_dir()
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    r = 0.0 # override
    _default_divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}
    divs = _get_divs(_default_divs)
    vol = 0.1
    K0 = 100.0

    # Maturity grid tau in (0, T) (avoid tau=0)
    eps = 1e-6
    coarse = np.linspace(eps, T - eps, 41)
    fine1 = np.linspace(0.24, 0.26, 21)
    fine2 = np.linspace(0.74, 0.76, 21)
    taus = np.unique(np.round(np.concatenate([coarse, fine1, fine2]), 12))

    def mean_factor_from_cash_divs(divs_cash: dict, tau: float) -> float:
        divs_tau_cash = {t_div: v for t_div, v in divs_cash.items() if 0.0 < float(t_div) <= float(tau) + 1e-12}
        divs_tau_prop = cash_divs_to_proportional_divs(float(S0), float(r), float(q), divs_tau_cash)
        p = 1.0
        for _t, (m, _std) in divs_tau_prop.items():
            p *= max(1.0 - float(m), 1e-12)
        return float(p)

    rows = []
    for tau in taus:
        # Include cash dividends that occur before maturity tau
        divs_tau = {t_div: v for t_div, v in divs.items() if 0.0 < float(t_div) <= float(tau) + 1e-12}
        prod = mean_factor_from_cash_divs(divs, float(tau))

        # Model-free forward / prepaid forward (consistent with engine approximation)
        f_model = float(forward_price(float(S0), float(r), float(q), float(tau), divs_tau))
        pf_model = float(np.exp(-r * tau) * f_model)

        # COS European call/put at maturity tau
        model_tau = GBMCHF(S0, r, q, divs_tau, {"vol": vol})
        pr = COSPricer(model_tau, N=512, L=8.0)
        call = float(pr.european_price(np.array([K0]), tau, is_call=True)[0])
        put = float(pr.european_price(np.array([K0]), tau, is_call=False)[0])

        pf_implied = float(call - put + K0 * np.exp(-r * tau))
        f_implied = float(np.exp(r * tau) * (call - put) + K0)

        rows.append({
            "tau": tau,
            "K": float(K0),
            "div_prod": prod,
            "forward_model": f_model,
            "forward_implied": f_implied,
            "prepaid_model": pf_model,
            "prepaid_implied": pf_implied,
            "call": call,
            "put": put,
            "parity_residual_prepaid": float(pf_implied - pf_model),
        })

    df = pd.DataFrame(rows).sort_values("tau")
    df.to_csv("figs/forward_parity_through_time.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(df.tau, df.forward_model, label="Model-free forward", lw=2)
    ax[0].plot(df.tau, df.forward_implied, "--", label="Implied via COS parity", lw=2)
    ax[0].set_ylabel("Forward (undiscounted)")
    ax[0].set_title("Forward vs maturity: model-free vs COS-implied (put-call parity)")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df.tau, df.call, label="COS European Call", color="tab:orange")
    ax[1].plot(df.tau, df.put, label="COS European Put", color="tab:blue")
    ax[1].set_xlabel("Maturity (years from now)")
    ax[1].set_ylabel("Option price")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.3)

    # Mark dividend times
    for t_div in sorted([t for t in divs.keys() if 0.0 < t <= T]):
        ax[0].axvline(x=t_div, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax[1].axvline(x=t_div, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    fig.tight_layout()
    fig.savefig("figs/forward_parity_through_time.png", dpi=150)
    plt.close(fig)
    print("Saved:", "figs/forward_parity_through_time.png")
    print("Saved:", "figs/forward_parity_through_time.csv")




def run_focused_lsmc(refit_times, n_paths=500000, steps=None, deg=3):
    """Run heavy LSMC refit (all-in) at a focused set of times and append results to CSV.

    refit_times: list of times t in [0,T] at which to run a refit LSMC (tau=T-t)
    n_paths: number of Monte Carlo paths used in the refit pass
    steps: number of timesteps in the LSMC
    deg: polynomial degree for basis
    """
    if steps is None:
        steps = NT_GLOBAL
    csvp = "figs/continuation_through_time.csv"
    if not os.path.exists(csvp):
        raise RuntimeError(f"CSV not found: {csvp}; run export_continuation_dataframe() first")
    df = pd.read_csv(csvp)

    # prepare columns
    col_price = f"lsmc_refit_{n_paths}"
    col_cont = f"lsmc_cont_refit_at_S0_{n_paths}"
    df[col_price] = np.nan
    df[col_cont] = np.nan

    S0 = 100.0
    r = 0.02
    q = 0.0
    vol = 0.25
    T = 1.0
    divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}
    from american_options.lsmc import LSMCPricer

    lsmc = LSMCPricer(S0, r, q, divs, vol, seed=2025)

    for t in refit_times:
        tau = T - t
        idx = df['time'].sub(t).abs().idxmin()
        print(f"Running LSMC refit for t={t} (tau={tau}), idx={idx}, n_paths={n_paths}, steps={steps}")
        # ...existing code...
        try:
            import time as _time
            t0 = _time.time()
            price, cont0 = lsmc._refit_on_price(100.0, tau, steps, n_price=n_paths, deg=deg)
            dt = _time.time() - t0
            cont_str = f"{cont0:.6f}" if (cont0 is not None) else "None"
            print(f"  Completed in {dt:.1f}s: price={price:.6f}, cont0={cont_str}")
            df.loc[idx, col_price] = price
            df.loc[idx, col_cont] = cont0 if cont0 is not None else np.nan
        except Exception as e:
            print(f"  LSMC refit failed at t={t}: {e}")
            df.loc[idx, col_price] = np.nan
            df.loc[idx, col_cont] = np.nan

    df.to_csv(csvp, index=False)
    print(f"Updated CSV with refit LSMC columns: {col_price}, {col_cont}")
    # regenerate plots to include overlay
    plot_continuation_through_time()
    plot_american_through_time()
    print("Regenerated plots with heavy LSMC overlay")


def run_cgmy_examples():
    """Run a small set of CGMY examples (COS European + American rollback) and save results.

    Prints prices and saves results to `figs/cgmy_examples.csv` so you can compare to Bowen & Oosterlee.
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    Ks = [90.0, 100.0, 110.0]
    param_sets = [
        {"name": "cgmy1", "C": 0.02, "G": 5.0, "M": 5.0, "Y": 0.5},
        {"name": "cgmy2", "C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2},
        {"name": "cgmy3", "C": 0.10, "G": 3.0, "M": 3.0, "Y": 0.5},
    ]

    rows = []
    from american_options import CGMYCHF
    for ps in param_sets:
        params = {k: v for k, v in ps.items() if k != "name"}
        model = CGMYCHF(S0, r, q, {}, params)
        pr = COSPricer(model, N=512, L=8.0)
        for K in Ks:
            eu = pr.european_price(np.array([K]), T)[0]
            am = pr.american_price(np.array([K]), T, steps=NT_GLOBAL, use_softmax=False)[0]
            rows.append({"name": ps["name"], "C": ps["C"], "G": ps["G"], "M": ps["M"], "Y": ps["Y"], "K": K, "european_cos": eu, "american_cos": am})
            print(f"{ps['name']} K={K}: European={eu:.6f}, American={am:.6f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs("figs", exist_ok=True)
    df.to_csv("figs/cgmy_examples.csv", index=False)
    print("Saved CGMY examples to figs/cgmy_examples.csv")


if __name__ == "__main__":
    print(f"Running COS American continuation plots with NT_GLOBAL={NT_GLOBAL}")
    ensure_fig_dir()
    plot_gbm_mc_vs_cos_dividend_uncertainty()
    if True:
        plot_forward_parity_through_time()
        plot_cos_vs_bs()
        plot_american_hard_vs_soft()
        plot_levy_vs_equiv_gbm()
        plot_american_dividend_continuation()
        plot_american_through_time()
        plot_continuation_through_time()
        run_cgmy_examples()
        print("Saved plots and CGMY examples")
