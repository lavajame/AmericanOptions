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
import pandas as pd

from american_options import (
    GBMCHF,
    MertonCHF,
    KouCHF,
    VGCHF,
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
from american_options.engine import COSPricer


def bs_call(S, K, r, q, vol, T):
    sigma = vol
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class FDMPricer:
    """Implicit finite difference pricer for GBM with proportional discrete dividends."""

    def __init__(self, S0, r, q, vol, divs, NS=200, NT=200):
        self.S0 = S0
        self.r = r
        self.q = q
        self.vol = vol
        self.divs = divs
        self.NS = NS
        self.NT = NT

    def price(self, K, T, american=False, debug=False):
        """Implicit backward Euler FD for European/American call with discrete proportional dividends.

        - Dividend mapping at exact ex-div steps: V_pre(S) = V_post(S*(1-m)).
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
        V = np.maximum(S - K, 0.0)

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
        for t_div, (m, _) in self.divs.items():
            if 0.0 < t_div <= T and dt > 0:
                div_indices.add(int(round(t_div / dt)))

        # Backward in time: n from NT-1 down to 0, time t_n = n*dt
        for n in range(NT - 1, -1, -1):
            t_n = n * dt
            # RHS from previous time level
            rhs = V[i]

            # Apply boundary conditions to RHS
            # S=0: V=0; S=S_max: V ~ S_max - K * exp(-r*(T - t_n))
            V_upper = S_max - K * np.exp(-self.r * (T - t_n))
            rhs[0] += lower[0] * 0.0
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
            V_new[0] = 0.0
            V_new[-1] = V_upper
            V_new[NS - 1] = d_prime[-1]
            for k in range(NS - 2, 0, -1):
                V_new[k] = d_prime[k - 1] - c_prime[k - 1] * V_new[k + 1]

            # Dividend mapping at t_n if ex-div occurs
            if n in div_indices:
                # Map pre-div prices: V_pre(S) = V_post(S*(1-m))
                m = None
                # find dividend magnitude at this t_n
                for t_div, (md, _) in self.divs.items():
                    if abs(t_div - t_n) <= 0.5 * dt:
                        m = md
                        break
                if m is not None and m > 0:
                    S_post = S * (1.0 - m)
                    V_new = np.interp(S, S_post, V_new, left=0.0, right=V_new[-1])

            # Early exercise
            if american:
                V_new = np.maximum(V_new, S - K)

            V = V_new

        # Interpolate to S0
        price = float(np.interp(self.S0, S, V))
        if debug:
            return price, S, V
        return price


def ensure_fig_dir():
    os.makedirs("figs", exist_ok=True)


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
    S0, r, q, T = 100.0, 0.02, 0.01, 1.0
    vol = 0.25
    K = np.linspace(70, 130, 61)
    gbm = GBMCHF(S0, r, q, {}, {"vol": vol})
    pr = COSPricer(gbm, N=256, L=8.0)
    amer_hard = pr.american_price(K, T, steps=80, beta=50.0, use_softmax=False)
    amer_soft = pr.american_price(K, T, steps=80, beta=20.0, use_softmax=True)
    euro = pr.european_price(K, T)
    intrinsic = np.maximum(S0 - K, 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K, amer_hard, label="American (hard max)")
    ax.plot(K, amer_soft, "--", label="American (softmax)")
    ax.plot(K, euro, ":", label="European")
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


def plot_american_dividend_continuation():
    """Compare European continuation (standalone COS) vs American continuation through dividend rollback.

    Plots continuation value at intermediate times during backward induction through dividends.
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    _default_divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}  # divs at t=0.25, 0.75
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
        p = pr_temp.american_price(np.array([K]), T, steps=100, use_softmax=False)[0]
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
    _default_divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}
    divs = _get_divs(_default_divs)
    K = np.array([100.0])
    vol = 0.25

    # Run COS backward induction once with trajectory caching (both American and continuation)
    pricer = COSPricer(GBMCHF(S0, r, q, divs, {"vol": vol}), N=128, L=8.0)
    amer_prices, euro_prices, cos_amer_trajectory, cos_cont_trajectory = pricer.american_price(
        K, T, steps=40, return_european=True, return_trajectory=True, return_continuation_trajectory=True
    )
    
    # Extract times and values from trajectories
    cos_times = np.array([t for t, _ in cos_amer_trajectory])
    cos_american = np.array([p for _, p in cos_amer_trajectory])
    cos_continuation = np.array([p for _, p in cos_cont_trajectory])
    
    # Compute standalone European at trajectory times
    cos_euro_standalone = []
    for t in cos_times:
        tau = T - t
        if tau < 1e-6:
            euro_sa = max(S0 - K[0], 0.0)
        else:
            euro_sa = pricer.european_price(K, tau)[0]
        cos_euro_standalone.append(euro_sa)
    cos_euro_standalone = np.array(cos_euro_standalone)
    
    # For validation, compute standalone European at trajectory times (should match continuation)
    cos_euro_standalone = []
    for t in cos_times:
        tau = T - t
        if tau < 1e-6:
            euro_sa = max(S0 - K[0], 0.0)
        else:
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


def plot_continuation_through_time():
    """New combined plot focusing only on continuation through time.

    Plots COS American vs COS continuation (European-from-rollback) across t in [0, T].
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}  # divs at t=0.25, 0.75
    K = np.array([100.0])
    vol = 0.25

    pricer = COSPricer(GBMCHF(S0, r, q, divs, {"vol": vol}), N=128, L=8.0)
    prices, euro, amer_traj, cont_traj = pricer.american_price(
        K, T, steps=40, return_european=True, return_trajectory=True, return_continuation_trajectory=True
    )

    times = np.array([t for t, _ in amer_traj])
    amer_vals = np.array([p for _, p in amer_traj])
    cont_vals = np.array([p for _, p in cont_traj])

    # COS European standalone through time (should match continuation)
    cos_euro_standalone = []
    for t in times:
        tau = T - t
        if tau < 1e-6:
            cos_euro_standalone.append(max(S0 - K[0], 0.0))
        else:
            cos_euro_standalone.append(pricer.european_price(K, tau)[0])
    cos_euro_standalone = np.array(cos_euro_standalone)

    # FDM European and American via implicit solver
    fdm = FDMPricer(S0, r, q, vol, divs, NS=200, NT=200)
    fdm_euro = []
    fdm_amer = []
    for t in times:
        tau = T - t
        if tau < 1e-6:
            intrinsic = max(S0 - K[0], 0.0)
            fdm_euro.append(intrinsic)
            fdm_amer.append(intrinsic)
        else:
            fdm_euro.append(fdm.price(K[0], tau, american=False))
            fdm_amer.append(fdm.price(K[0], tau, american=True))
    fdm_euro = np.array(fdm_euro)
    fdm_amer = np.array(fdm_amer)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(times, cos_euro_standalone, label="COS European (standalone)", linewidth=1.8, color="orange", linestyle=":")
    ax.plot(times, cont_vals, label="COS European (from rollback)", linewidth=2, color="cyan", linestyle="--")
    ax.plot(times, fdm_euro, label="FDM European", linewidth=1.8, color="purple", linestyle=":")
    ax.plot(times, amer_vals, label="COS American", linewidth=2.5, color="red")
    ax.plot(times, fdm_amer, label="FDM American", linewidth=2, color="magenta", linestyle="--")

    # Attempt to overlay LSMC series from the exported CSV (if available)
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
    ax.set_title(f"Continuation + References Through Time\n(S0={S0}, K={K[0]}, T={T}, σ={vol}, r={r}, q={q})", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figs/continuation_through_time.png", dpi=150)
    plt.close(fig)


def export_continuation_dataframe():
    """Produce a dataframe of the five time series through time and save to CSV.

    Columns: time, cos_euro_standalone, cos_euro_from_rollback, fdm_euro, cos_american, fdm_american
    Uses per-timestep pricing to avoid trajectory artifacts.
    """
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}
    K = np.array([100.0])
    vol = 0.25

    base_model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pricer = COSPricer(base_model, N=128, L=8.0)
    # Use coarse grid plus fine windows around dividend dates
    coarse = np.linspace(0, T, 41)
    fine1 = np.linspace(0.24, 0.26, 21)
    fine2 = np.linspace(0.74, 0.76, 21)
    times = np.unique(np.concatenate([coarse, fine1, fine2]))

    cos_euro_standalone = []
    cos_euro_from_rb = []
    cos_american = []
    fdm_euro = []
    fdm_american = []
    # LSMC pricer
    from american_options.lsmc import LSMCPricer
    lsmc = LSMCPricer(S0, r, q, divs, vol, seed=2025)
    lsmc_amer_train = []
    lsmc_amer_refit = []
    lsmc_cont_train = []
    lsmc_cont_refit = []
    # FDM helpers
    def spot_after_future_divs(S, divs_dict, t_current):
        S_eff = S
        for t_div, (m, _) in sorted(divs_dict.items()):
            if t_div > t_current + 1e-12:
                S_eff *= (1.0 - m)
        return S_eff

    for t in times:
        tau = T - t
        # Shift dividends to remaining horizon
        divs_shift = {dt - t: (m, s) for dt, (m, s) in divs.items() if dt > t + 1e-12}

        # COS with shifted dividends
        model_t = GBMCHF(S0, r, q, divs_shift, {"vol": vol})
        pricer_t = COSPricer(model_t, N=128, L=8.0)
        euro_sa = max(S0 - K[0], 0.0) if tau < 1e-6 else pricer_t.european_price(K, tau)[0]
        cos_euro_standalone.append(euro_sa)

        amer_tau, euro_tau = pricer_t.american_price(K, tau, steps=60, beta=20.0, use_softmax=True, return_european=True)
        cos_american.append(amer_tau[0])
        cos_euro_from_rb.append(euro_tau[0])

        # FDM with shifted dividends
        fdm_t = FDMPricer(S0, r, q, vol, divs_shift, NS=400, NT=max(200, int(600 * tau / T) if T > 0 else 600))
        if tau < 1e-6:
            intrinsic = max(S0 - K[0], 0.0)
            fdm_euro.append(intrinsic)
            fdm_american.append(intrinsic)
            lsmc_amer_train.append(intrinsic)
            lsmc_amer_refit.append(intrinsic)
            lsmc_cont_train.append(intrinsic)
            lsmc_cont_refit.append(intrinsic)
        else:
            fdm_euro.append(fdm_t.price(K[0], tau, american=False))
            fdm_american.append(fdm_t.price(K[0], tau, american=True))
            # LSMC pricing for this remaining horizon
            tr_price, tr_cont, rf_price, rf_cont = lsmc.price_at_tau(K[0], tau, steps=60, n_train=2000, n_price=2000, deg=3)
            lsmc_amer_train.append(tr_price)
            lsmc_amer_refit.append(rf_price)
            lsmc_cont_train.append(tr_cont if tr_cont is not None else np.nan)
            lsmc_cont_refit.append(rf_cont if rf_cont is not None else np.nan)

    df = pd.DataFrame({
        "time": times,
        "cos_euro_standalone": cos_euro_standalone,
        "cos_euro_from_rollback": cos_euro_from_rb,
        "fdm_euro": fdm_euro,
        "cos_american": cos_american,
        "fdm_american": fdm_american,
        "lsmc_american_train": lsmc_amer_train,
        "lsmc_american_refit": lsmc_amer_refit,
        "lsmc_cont_train_at_S0": lsmc_cont_train,
        "lsmc_cont_refit_at_S0": lsmc_cont_refit,
    })
    os.makedirs("figs", exist_ok=True)
    df.to_csv("figs/continuation_through_time.csv", index=False)
    print("Saved dataframe:", "figs/continuation_through_time.csv")
    # Show rows around dividend dates for inspection
    print("\nRows near first dividend (t~0.25):")
    print(df[(df.time >= 0.22) & (df.time <= 0.28)].to_string(index=False))
    print("\nRows near second dividend (t~0.75):")
    print(df[(df.time >= 0.72) & (df.time <= 0.78)].to_string(index=False))


def run_focused_lsmc(refit_times, n_paths=500000, steps=60, deg=3):
    """Run heavy LSMC refit (all-in) at a focused set of times and append results to CSV.

    refit_times: list of times t in [0,T] at which to run a refit LSMC (tau=T-t)
    n_paths: number of Monte Carlo paths used in the refit pass
    steps: number of timesteps in the LSMC
    deg: polynomial degree for basis
    """
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
    divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}
    from american_options.lsmc import LSMCPricer

    lsmc = LSMCPricer(S0, r, q, divs, vol, seed=2025)

    for t in refit_times:
        tau = T - t
        idx = df['time'].sub(t).abs().idxmin()
        print(f"Running LSMC refit for t={t} (tau={tau}), idx={idx}, n_paths={n_paths}, steps={steps}")
        if tau < 1e-8:
            intrinsic = max(S0 - 100.0, 0.0)
            df.loc[idx, col_price] = intrinsic
            df.loc[idx, col_cont] = intrinsic
            continue
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
            am = pr.american_price(np.array([K]), T, steps=200, use_softmax=False)[0]
            rows.append({"name": ps["name"], "C": ps["C"], "G": ps["G"], "M": ps["M"], "Y": ps["Y"], "K": K, "european_cos": eu, "american_cos": am})
            print(f"{ps['name']} K={K}: European={eu:.6f}, American={am:.6f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs("figs", exist_ok=True)
    df.to_csv("figs/cgmy_examples.csv", index=False)
    print("Saved CGMY examples to figs/cgmy_examples.csv")


if __name__ == "__main__":
    ensure_fig_dir()
    plot_cos_vs_bs()
    plot_american_hard_vs_soft()
    plot_levy_vs_equiv_gbm()
    plot_american_dividend_continuation()
    plot_american_through_time()
    plot_continuation_through_time()
    run_cgmy_examples()
    print("Saved plots and CGMY examples")
