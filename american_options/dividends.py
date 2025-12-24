"""Dividend utilities.

Project-wide convention: discrete dividends are provided as cash amounts (mean, std)
in spot currency. Internally we convert to proportional multiplicative factors.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

def cash_divs_to_proportional_divs(
    S0: float,
    r: float,
    q: float,
    divs_cash: Dict[float, Tuple[float, float]],
) -> Dict[float, Tuple[float, float]]:
    """Convert cash discrete dividends into internal proportional-dividend parameters.

    External project-wide convention (cash dividends):
        divs_cash[t] = (D_mean, D_std)
    where D_* are in *spot currency* paid at ex-div time t.

    Internal convention used by COS/LSMC/FDM code paths:
        divs_prop[t] = (m_mean, std_log)
    with multiplicative factor applied at ex-div:
        ln D_factor = ln(1-m_mean) - 0.5*std_log^2 + std_log * Z
    so E[D_factor] = (1-m_mean).

    We approximate a cash dividend as a proportional drop relative to the *expected* pre-div forward:
        m_mean ≈ D_mean / E[S_{t-}]
        Var[D_factor] ≈ (D_std / E[S_{t-}])^2
    and match this variance by choosing std_log via the lognormal moment relation.

    Note: this is an approximation (cash dividends are additive in reality).
    """
    if not divs_cash:
        return {}

    # Sort by time; build expected pre-div level recursively using mean impacts.
    items = [(float(t), float(Dm), float(Ds)) for t, (Dm, Ds) in divs_cash.items()]
    items.sort(key=lambda z: z[0])

    divs_prop: Dict[float, Tuple[float, float]] = {}
    mean_factor = 1.0
    for t, D_mean, D_std in items:
        if t <= 0.0:
            continue
        if D_mean < 0.0 or D_std < 0.0:
            raise ValueError("Cash dividends require non-negative mean/std")

        expected_pre = float(S0) * float(np.exp((r - q) * t)) * float(mean_factor)
        if expected_pre <= 0.0:
            raise ValueError("Invalid expected pre-div level while converting dividends")

        m_mean = float(D_mean) / expected_pre
        if m_mean >= 1.0:
            raise ValueError(f"Cash dividend too large at t={t}: D_mean={D_mean} vs E_pre={expected_pre}")
        if m_mean < 0.0:
            raise ValueError("Cash dividend mean cannot imply negative proportional drop")

        one_minus_m = max(1.0 - m_mean, 1e-12)
        rel_std = float(D_std) / expected_pre

        if rel_std <= 0.0:
            std_log = 0.0
        else:
            # For X ~ LogNormal with mean = mu_x and log-std = s:
            # Var[X] = (exp(s^2) - 1) * mu_x^2
            # Here mu_x := E[D_factor] = (1-m_mean) and we want Var[D_factor] ≈ rel_std^2.
            ratio = (rel_std * rel_std) / (one_minus_m * one_minus_m)
            std_log = float(np.sqrt(np.log(1.0 + max(ratio, 0.0))))

        divs_prop[t] = (m_mean, std_log)
        mean_factor *= one_minus_m

    return divs_prop

def _dividend_adjustment(T: float, divs: Dict[float, Tuple[float, float]]) -> Tuple[float, np.ndarray]:
    """
        Return the cumulative log product adjustment and an array of (mean, var) pairs.

        Internal dividend convention (proportional lognormal factor):
        - Each dividend event at time t has mean proportional drop `m` and a log-factor uncertainty `std_log`.
        - We model the dividend multiplicative factor as:
                ln D = ln(1-m) - 0.5 * std^2 + std * Z,  Z ~ N(0,1)
            so that E[D] = (1-m) while Var[ln D] = std^2.

        We return:
        - sum_log = Σ (ln(1-m) - 0.5*std^2) for t<=T
        - params = [(m, std^2), ...] for t<=T
    """
    sum_log = 0.0
    params = []
    for t, (m, std) in divs.items():
        if t <= T:
                        var = float(std) ** 2
                        sum_log += np.log(max(1.0 - m, 1e-12)) - 0.5 * var
                        params.append((m, var))
    return sum_log, np.array(params)


def _dividend_adjustment_window(t0: float, t1: float, divs: Dict[float, Tuple[float, float]]) -> Tuple[float, np.ndarray]:
    """Dividend log-moment adjustment for dividends in (t0, t1].

    Uses the same internal proportional-dividend convention as `_dividend_adjustment`.
    This helper is required for pricing over sub-intervals when dividends are specified
    in absolute time (from 0).
    """
    t0 = float(t0)
    t1 = float(t1)
    if t1 <= t0:
        return 0.0, np.zeros((0, 2), dtype=float)

    sum_log = 0.0
    params: list[tuple[float, float]] = []
    for t, (m, std) in divs.items():
        tt = float(t)
        if t0 < tt <= t1:
            var = float(std) ** 2
            sum_log += np.log(max(1.0 - float(m), 1e-12)) - 0.5 * var
            params.append((float(m), var))
    return sum_log, np.array(params, dtype=float)


def _forward_price_from_prop_divs(S0: float, r: float, q: float, T: float,
                                 divs_prop: Dict[float, Tuple[float, float]]) -> float:
    """Internal forward using internal proportional dividends."""
    sum_log, div_params = _dividend_adjustment(T, divs_prop)
    var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
    # E[S_T] = S0 * exp((r-q)T) * Π E[D_factor]. With our convention E[D_factor]=(1-m),
    # which equals exp(sum_log + 0.5*var_div) since sum_log includes -0.5*var_div.
    return S0 * np.exp((r - q) * T + sum_log + 0.5 * var_div)


def forward_price(S0: float, r: float, q: float, T: float,
                  divs: Dict[float, Tuple[float, float]]) -> float:
    """
    Risk‑neutral forward under the project-wide *cash dividend* convention.

    divs[t] = (D_mean, D_std) in spot currency.

    We convert cash dividends to internal proportional parameters using the expected pre-div forward,
    then compute:
        F ≈ S0 * exp((r - q) * T) * Π E[D_factor]
    where E[D_factor] = (1 - m_mean).
    """
    divs_prop = cash_divs_to_proportional_divs(S0, r, q, divs)
    return _forward_price_from_prop_divs(S0, r, q, T, divs_prop)



