"""Profile COS vs FDM time complexity (single vs multi-strike).

Produces a 1x2 diagnostic plot in figs/:
- Left: European call (COS vs FDM)
- Right: American call (COS rollback vs FDM)

For each method, we sweep resolution knobs:
- COS: N (number of COS terms), fixed L=10
- FDM: (NS, NT) grid sizes

We compute log10 absolute error vs baselines:
- European: analytic GBM call under the repo's deterministic cash->proportional dividend approximation
- American: high-resolution FDM (no closed form with discrete dividends)

Run:
    python profile_cos_vs_fdm_time_complexity.py
"""

from __future__ import annotations

import os
import sys
import time
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# Allow running via: `python tools/profile_cos_vs_fdm_time_complexity.py`
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options import GBMCHF, cash_divs_to_proportional_divs
from american_options.engine import COSPricer, forward_price

# Reuse the diagnostic FDM pricer that already matches the repo's cash-dividend convention.
from plot_diagnostics import FDMPricer, ensure_fig_dir


VERBOSE = True


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


@dataclass(frozen=True)
class Scenario:
    S0: float = 100.0
    # Chosen to make the accuracy/time trade-off non-trivial:
    # higher vol and longer maturity widen the truncation domain and slow convergence.
    r: float = 0.01
    q: float = 0.00
    vol: float = 0.10
    T: float = 0.75
    divs: dict[float, tuple[float, float]] = None  # cash divs: t -> (D_mean, D_std)

    def __post_init__(self):
        if self.divs is None:
            # Multiple discrete cash dividends across the horizon.
            object.__setattr__(
                self,
                "divs",
                {
                    0.5: (4.0, 0.0),
                    1.0: (4.0, 0.0),
                    1.5: (4.0, 0.0),
                },
            )


def _time_and_value(fn):
    """Return (wall_time_seconds, value) for a single run."""
    t0 = time.perf_counter()
    val = fn()
    return float(time.perf_counter() - t0), val


def _cos_price_european(sc: Scenario, K: np.ndarray, *, N: int, L: float) -> np.ndarray:
    model = GBMCHF(sc.S0, sc.r, sc.q, sc.divs, {"vol": sc.vol})
    pricer = COSPricer(model, N=int(N), L=float(L))
    return np.asarray(pricer.european_price(np.asarray(K, dtype=float), sc.T, is_call=True), dtype=float)

def _cos_price_american(sc: Scenario, K: np.ndarray, *, N: int, L: float, steps: int) -> np.ndarray:
    model = GBMCHF(sc.S0, sc.r, sc.q, sc.divs, {"vol": sc.vol})
    pricer = COSPricer(model, N=int(N), L=float(L))
    prices = pricer.american_price(np.asarray(K, dtype=float), sc.T, steps=int(steps), is_call=True, use_softmax=False)
    return np.asarray(prices, dtype=float)


def _fdm_price_european(sc: Scenario, K: np.ndarray, *, NS: int, NT: int) -> np.ndarray:
    fdm = FDMPricer(sc.S0, sc.r, sc.q, sc.vol, sc.divs, NS=int(NS), NT=int(NT))
    out = []
    for k in np.asarray(K, dtype=float).ravel():
        out.append(float(fdm.price(float(k), sc.T, american=False, is_call=True)))
    return np.asarray(out, dtype=float)

def _fdm_price_american(sc: Scenario, K: np.ndarray, *, NS: int, NT: int) -> np.ndarray:
    fdm = FDMPricer(sc.S0, sc.r, sc.q, sc.vol, sc.divs, NS=int(NS), NT=int(NT))
    out = []
    for k in np.asarray(K, dtype=float).ravel():
        out.append(float(fdm.price(float(k), sc.T, american=True, is_call=True)))
    return np.asarray(out, dtype=float)


def _log10_abs_err(x: np.ndarray, ref: np.ndarray, *, eps: float = 1e-16) -> np.ndarray:
    return np.log10(np.maximum(np.abs(np.asarray(x, dtype=float) - np.asarray(ref, dtype=float)), eps))


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _bs_call_price(S: float | np.ndarray, K: float | np.ndarray, r: float, q: float, vol: float, T: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = float(vol)
    tau = float(T)

    if tau <= 0.0:
        return np.maximum(S - K, 0.0)
    if sigma <= 0.0:
        # Deterministic forward
        F = S * np.exp((float(r) - float(q)) * tau)
        return np.exp(-float(r) * tau) * np.maximum(F - K, 0.0)

    sqrtT = np.sqrt(tau)
    d1 = (np.log(S / K) + (float(r) - float(q) + 0.5 * sigma * sigma) * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * np.exp(-float(q) * tau) * _norm_cdf(d1) - K * np.exp(-float(r) * tau) * _norm_cdf(d2)


def _analytic_baseline_prices(sc: Scenario, K: np.ndarray) -> np.ndarray:
    """Fast baseline under the same deterministic-dividend approximation used in the engine.

    For deterministic (std=0) discrete dividends, after cash->proportional conversion the model
    implies a multiplicative spot scaling by Π(1-m). Under GBM, that keeps S_T lognormal, so a
    Black-Scholes call with spot S0_eff = S0 * Π(1-m) matches this approximation.
    """
    divs_prop = cash_divs_to_proportional_divs(float(sc.S0), float(sc.r), float(sc.q), dict(sc.divs))
    prod = 1.0
    for t_div, (m, _std) in divs_prop.items():
        if 0.0 < float(t_div) <= float(sc.T) + 1e-12:
            prod *= max(1.0 - float(m), 1e-12)
    S0_eff = float(sc.S0) * float(prod)
    return np.asarray(_bs_call_price(S0_eff, np.asarray(K, dtype=float), sc.r, sc.q, sc.vol, sc.T), dtype=float)


def main() -> None:
    ensure_fig_dir()

    sc = Scenario()

    # Multi-strike count for the scalability comparison.
    # Keep this modest: FDM loops strikes in this diagnostic.
    K_num = 100

    # Time discretization for American COS rollback.
    cos_steps_american = 200

    # [a] single strike and [b] multi-strike
    # Use a forward-centered grid so the moneyness range stays meaningful under divs.
    F0 = float(forward_price(float(sc.S0), float(sc.r), float(sc.q), float(sc.T), dict(sc.divs)))
    K1 = np.asarray([F0], dtype=float)  # ATM-forward

    width = float(sc.vol) * float(np.sqrt(max(sc.T, 1e-16)))
    K_lo = F0 * float(np.exp(-2.5 * width))
    K_hi = F0 * float(np.exp(+2.5 * width))
    K_multi = np.linspace(K_lo, K_hi, int(K_num), dtype=float)

    cos_L = 10.0

    # Baselines
    # - European: analytic GBM call under the same deterministic cash->proportional approximation.
    _log("[baseline] European analytic baseline...")
    ref_eu_1 = _analytic_baseline_prices(sc, K1)
    ref_eu_multi = _analytic_baseline_prices(sc, K_multi)
    _log("[baseline] European analytic baseline done")

    # - American: COS-American rollback baseline (cheaper than very high-res references).
    am_base_steps = 200
    am_base_N = 2**11
    _log(f"[baseline] American COS baseline steps={am_base_steps}, N={am_base_N}...")
    am_model = GBMCHF(sc.S0, sc.r, sc.q, sc.divs, {"vol": sc.vol})
    am_pricer = COSPricer(am_model, N=int(am_base_N), L=float(cos_L))
    ref_am_1 = np.asarray(
        am_pricer.american_price(np.asarray(K1, dtype=float), sc.T, steps=int(am_base_steps), is_call=True, use_softmax=False),
        dtype=float,
    )
    ref_am_multi = np.asarray(
        am_pricer.american_price(
            np.asarray(K_multi, dtype=float), sc.T, steps=int(am_base_steps), is_call=True, use_softmax=False
        ),
        dtype=float,
    )
    _log("[baseline] American COS baseline done")

    # Sweeps
    # NOTE: European COS converges fast here; keep a compact sweep.
    cos_Ns_euro = [2**k for k in range(3, 8)]  # 8 .. 4096
    # American rollback is more expensive; keep an even smaller sweep.
    cos_Ns_amer = [2**k for k in range(3, 11)]  # 8 .. 256

    # FDM grid sweep: increase both space and time grids together.
    # Keep this modest: FDM is O(NS*NT) per strike here (we loop strikes), so large grids
    # get very expensive for multi-strike runs.
    # Use a simple increasing NS=NT sweep.
    fdm_grids = [50*(i+1) for i in range(10)]

    def profile_case(*, is_american: bool):
        _log("[profile] Starting " + ("American" if is_american else "European") + " case")
        if is_american:
            cos_price = lambda N, K: _cos_price_american(sc, K, N=N, L=cos_L, steps=cos_steps_american)
            fdm_price = lambda NS, NT, K: _fdm_price_american(sc, K, NS=NS, NT=NT)
            ref_1 = ref_am_1
            ref_multi = ref_am_multi
            cos_Ns = cos_Ns_amer
        else:
            cos_price = lambda N, K: _cos_price_european(sc, K, N=N, L=cos_L)
            fdm_price = lambda NS, NT, K: _fdm_price_european(sc, K, NS=NS, NT=NT)
            ref_1 = ref_eu_1
            ref_multi = ref_eu_multi
            cos_Ns = cos_Ns_euro

        cos_pts_1: list[tuple[float, float, str]] = []
        cos_pts_multi: list[tuple[float, float, str]] = []
        fdm_pts_1: list[tuple[float, float, str]] = []
        fdm_pts_multi: list[tuple[float, float, str]] = []

        # COS profiling
        for N in cos_Ns:
            _log(f"[profile] COS N={N} ({'AM' if is_american else 'EU'})...")
            t1, p1 = _time_and_value(lambda: cos_price(N, K1))
            tm, pm = _time_and_value(lambda: cos_price(N, K_multi))

            e1 = float(_log10_abs_err(p1, ref_1)[0])
            em = float(np.max(_log10_abs_err(pm, ref_multi)))

            cos_pts_1.append((t1, e1, f"N={N}"))
            cos_pts_multi.append((tm / float(K_num), em, f"N={N}"))

        # FDM profiling
        for grid in fdm_grids:
            NS = int(grid)
            NT = int(grid)
            _log(f"[profile] FDM NS=NT={NS} ({'AM' if is_american else 'EU'})...")
            t1, p1 = _time_and_value(lambda: fdm_price(NS, NT, K1))
            tm, pm = _time_and_value(lambda: fdm_price(NS, NT, K_multi))

            e1 = float(_log10_abs_err(p1, ref_1)[0])
            em = float(np.max(_log10_abs_err(pm, ref_multi)))

            fdm_pts_1.append((t1, e1, f"NS=NT={NS}"))
            fdm_pts_multi.append((tm / float(K_num), em, f"NS=NT={NS}"))

        return cos_pts_1, cos_pts_multi, fdm_pts_1, fdm_pts_multi

    euro = profile_case(is_american=False)
    amer = profile_case(is_american=True)

    _log("[plot] Rendering figure...")

    def xy(pts: list[tuple[float, float, str]]):
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        return x, y

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panels = [
        ("European call", euro, "log10 absolute error vs analytic baseline"),
        ("American call", amer, "log10 absolute error vs high-res FDM baseline"),
    ]

    for ax, (title, (cos_pts_1, cos_pts_multi, fdm_pts_1, fdm_pts_multi), ylab) in zip(axes, panels):
        cos_x1, cos_y1 = xy(cos_pts_1)
        cos_xm, cos_ym = xy(cos_pts_multi)
        fdm_x1, fdm_y1 = xy(fdm_pts_1)
        fdm_xm, fdm_ym = xy(fdm_pts_multi)

        ax.plot(cos_x1, cos_y1, "o-", label="COS (1 strike)")
        ax.plot(cos_xm, cos_ym, "^-", label=f"COS ({K_num} strikes, per-strike time)")
        ax.plot(fdm_x1, fdm_y1, "s-", label="FDM (1 strike)")
        ax.plot(fdm_xm, fdm_ym, "D-", label=f"FDM ({K_num} strikes, per-strike time)")

        # Light annotations to aid reading.
        for x, y, lab in cos_pts_1:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, alpha=0.65)
        for x, y, lab in fdm_pts_1:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, alpha=0.65)

        ax.set_xscale("log")
        ax.set_xlabel("Wall time per priced strike (seconds, log scale)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("log10 absolute error vs baseline")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    fig.suptitle(
        "COS vs FDM: time vs accuracy (cash dividends)\n"
        f"S0={sc.S0}, T={sc.T}, r={sc.r}, q={sc.q}, vol={sc.vol}, div_times={sorted(sc.divs.keys())}",
        fontsize=11,
    )

    fig.tight_layout()
    out_png = "figs/cos_vs_fdm_time_complexity.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    _log(f"[done] Saved: {out_png}")

    print("Saved:", out_png, flush=True)


if __name__ == "__main__":
    main()
