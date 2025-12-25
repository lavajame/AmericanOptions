"""Dividend utilities.

Project-wide convention: discrete dividends are provided as cash amounts (mean, std)
in spot currency.

Uncertain dividends are handled via an Inverse Gaussian (IG) *process* applied as an
independent log-factor on the proportional dividend drop. This preserves positivity
of the multiplicative factor while keeping inputs intuitive in cash space.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _ig_scale_from_mean_std(mean: float, std: float) -> float:
    """Return the IG scale (lambda) from mean and standard deviation.

    NumPy's `Generator.wald(mean, scale)` uses the Inverse Gaussian parameterization:
      E[X] = mean,  Var[X] = mean^3 / scale.
    """
    mean = float(mean)
    std = float(std)
    if mean <= 0.0:
        raise ValueError("Inverse Gaussian requires mean > 0")
    if std <= 0.0:
        raise ValueError("Inverse Gaussian scale is undefined for std<=0")
    return (mean**3) / (std**2)


def _ig_laplace_log(*, s: complex | np.ndarray, mean: float, scale: float) -> np.ndarray:
    """Log Laplace transform of IG: log E[exp(-s X)].

    For X ~ IG(mean=mu, scale=lambda):
      E[e^{-sX}] = exp((lambda/mu) * (1 - sqrt(1 + 2*mu^2*s/lambda)))

    Works for complex s (via complex sqrt).
    """
    mu = float(mean)
    lam = float(scale)
    s = np.asarray(s, dtype=complex)
    return (lam / mu) * (1.0 - np.sqrt(1.0 + (2.0 * (mu**2) * s) / lam))


def dividend_char_factor(u: np.ndarray, T: float, divs_prop: Dict[float, Tuple[float, float]]) -> np.ndarray:
    """Characteristic multiplier from proportional dividends up to time T.

    Internal representation:
      divs_prop[t] = (m_mean, rel_std)

    Deterministic mean drop is handled by `m_mean` via the factor (1-m_mean).
    Uncertainty is modeled as a *positive* log-factor driven by an inverse Gaussian
    centered at the relative cash mean.

    For each dividend event we define a random log factor:
      ln D_factor = ln(1-m) + J - log(E[e^J])
    where J := -(X - m), X ~ IG(mean=m, std=rel_std).
    This normalization enforces E[D_factor] = (1-m).
    """
    u = np.asarray(u, dtype=complex)
    T = float(T)
    if not divs_prop:
        return np.ones_like(u)

    out = np.ones_like(u)
    for t, (m, rel_std) in divs_prop.items():
        tt = float(t)
        if not (0.0 < tt <= T):
            continue
        m = float(m)
        rel_std = float(rel_std)
        if m <= 0.0:
            continue
        ln1m = float(np.log(max(1.0 - m, 1e-12)))
        if rel_std <= 0.0:
            out *= np.exp(1j * u * ln1m)
            continue

        mu = m
        lam = _ig_scale_from_mean_std(mu, rel_std)

        # J = -(X - mu) = mu - X. Need:
        #   E[e^J] = E[e^{mu - X}] = exp(mu) * E[e^{-X}]  (Laplace at s=1).
        log_LT_1 = _ig_laplace_log(s=1.0 + 0j, mean=mu, scale=lam)
        log_EexpJ = float(mu + np.real(log_LT_1))

        # phi_J(u) = E[e^{i u J}] = exp(i u mu) * E[e^{-i u X}]  (Laplace at s=i u).
        log_LT_iu = _ig_laplace_log(s=1j * u, mean=mu, scale=lam)
        phi_J = np.exp(1j * u * mu + log_LT_iu)

        out *= np.exp(1j * u * (ln1m - log_EexpJ)) * phi_J

    return out


def dividend_char_factor_window(
    u: np.ndarray,
    t0: float,
    t1: float,
    divs_prop: Dict[float, Tuple[float, float]],
) -> np.ndarray:
    """Dividend characteristic multiplier for dividends in (t0, t1]."""
    u = np.asarray(u, dtype=complex)
    t0 = float(t0)
    t1 = float(t1)
    if not divs_prop or t1 <= t0:
        return np.ones_like(u)
    out = np.ones_like(u)
    for t, (m, rel_std) in divs_prop.items():
        tt = float(t)
        if not (t0 < tt <= t1):
            continue
        # Reuse the up-to-T helper by constructing a tiny dict.
        out *= dividend_char_factor(u, tt, {tt: (m, rel_std)})
    return out


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
        divs_prop[t] = (m_mean, rel_std)
    where:
        m_mean  ~= D_mean / E[S_{t-}]   (dimensionless mean proportional drop)
        rel_std ~= D_std  / E[S_{t-}]   (dimensionless cash stdev)

    Pricing engines treat the dividend as an independent *multiplicative* log-factor
    driven by an inverse Gaussian (see :func:`dividend_char_factor`).

    We approximate a cash dividend as a proportional drop relative to the *expected* pre-div forward:
        m_mean ≈ D_mean / E[S_{t-}]
        D_std / E[S_{t-}]

    Note: this is still an approximation (cash dividends are additive in reality).
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

        rel_std = float(D_std) / expected_pre

        divs_prop[t] = (m_mean, rel_std)
        mean_factor *= max(1.0 - m_mean, 1e-12)

    return divs_prop


def _dividend_adjustment(T: float, divs: Dict[float, Tuple[float, float]]) -> Tuple[float, np.ndarray]:
    """
        Return a deterministic log-mean term and an array of (mean, var) pairs.

        This helper remains for backward-compat with older code paths that expect
        a "sum_log" plus a variance proxy.

        Under the IG dividend uncertainty convention, we return:
        - sum_log = Σ ln(1-m) for t<=T
        - params = [(m, rel_std^2), ...] for t<=T
    """
    sum_log = 0.0
    params = []
    for t, (m, std) in divs.items():
        if float(t) <= float(T):
            var = float(std) ** 2
            sum_log += float(np.log(max(1.0 - float(m), 1e-12)))
            params.append((float(m), var))
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
            sum_log += float(np.log(max(1.0 - float(m), 1e-12)))
            params.append((float(m), var))
    return sum_log, np.array(params, dtype=float)


def _forward_price_from_prop_divs(S0: float, r: float, q: float, T: float,
                                 divs_prop: Dict[float, Tuple[float, float]]) -> float:
    """Internal forward using internal proportional dividends."""
    # E[S_T] = S0 * exp((r-q)T) * Π E[D_factor]. By construction E[D_factor] = (1-m).
    prod = 1.0
    for t, (m, _std) in divs_prop.items():
        if 0.0 < float(t) <= float(T):
            prod *= max(1.0 - float(m), 1e-12)
    return float(S0) * float(np.exp((r - q) * float(T))) * float(prod)


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



