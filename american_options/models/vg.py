"""Variance-Gamma characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, dividend_char_factor

class VGCHF(CharacteristicFunction):
    """Variance‑Gamma (Madan‑Carr‑Chang) model."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])

        # characteristic function of VG increments under risk-neutral drift choice
        # MGF factor: (1 - i theta nu u + 0.5 sigma^2 nu u^2)^(-T/nu)
        # Drift correction (COS.pdf): choose omega such that E[e^{X_T + omega T}] = 1
        # so that E[S_T] = S0 * exp((r-q)T) * Π(1-m). For VG:
        # omega = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu)
        denom_at_1 = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
        if denom_at_1 <= 0:
            # numerical safety: shift a little (shouldn't normally happen for sensible params)
            denom_at_1 = max(1e-12, denom_at_1)
        omega = (1.0 / nu) * np.log(denom_at_1)
        mu = np.log(self.S0) + (self.r - self.q + omega) * T
        phi_base = (1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * (nu) * (u ** 2)) ** (-T / nu)
        phi = np.exp(1j * u * mu) * phi_base
        return phi * dividend_char_factor(u, T, self.divs)

    def _var2(self, T: float) -> float:
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])
        # Var[X_t] = t*(sigma^2 + theta^2 * nu)
        return T * (sigma ** 2 + theta ** 2 * nu)

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Characteristic of log-return increment over dt."""
        u = np.asarray(u, dtype=complex)
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])

        denom_at_1 = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
        if denom_at_1 <= 0:
            denom_at_1 = max(1e-12, denom_at_1)
        omega = (1.0 / nu) * np.log(denom_at_1)
        mu_term = (self.r - self.q + omega) * dt

        inside = 1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * (nu) * (u ** 2)
        phi = np.exp(1j * u * mu_term) * (inside ** (-dt / nu))
        return phi * dividend_char_factor(u, dt, self.divs)

    def increment_char_and_grad(
        self,
        u: np.ndarray,
        dt: float,
        *,
        params: list[str] | None = None,
        method: str = "analytic",
        rel_step: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        method = str(method).lower().strip()
        if method == "fd":
            return super().increment_char_and_grad(u, dt, params=params, method=method, rel_step=rel_step)

        u = np.asarray(u, dtype=complex)
        dt = float(dt)
        if params is None:
            params = self.param_names()

        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])

        denom_at_1 = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
        if denom_at_1 <= 0:
            denom_at_1 = max(1e-12, denom_at_1)
        omega = (1.0 / nu) * np.log(denom_at_1)
        mu_term = (self.r - self.q + omega) * dt

        inside = 1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * nu * (u ** 2)
        log_inside = np.log(inside)
        phi_base = np.exp(1j * u * mu_term) * np.exp((-dt / nu) * log_inside)
        phi = phi_base * dividend_char_factor(u, dt, self.divs)

        grad: dict[str, np.ndarray] = {}

        # dlogphi/dtheta
        if "theta" in params:
            domega = -1.0 / denom_at_1
            dmu = dt * domega
            dlog_inside = (-1j * nu * u) / inside
            dlogphi = 1j * u * dmu - (dt / nu) * dlog_inside
            grad["theta"] = phi * dlogphi

        # dlogphi/dsigma
        if "sigma" in params:
            domega = -(sigma) / denom_at_1
            dmu = dt * domega
            dlog_inside = (sigma * nu * (u ** 2)) / inside
            dlogphi = 1j * u * dmu - (dt / nu) * dlog_inside
            grad["sigma"] = phi * dlogphi

        # dlogphi/dnu
        if "nu" in params:
            ddenom = -(theta) - 0.5 * sigma * sigma
            domega = (-(1.0 / (nu ** 2)) * np.log(denom_at_1)) + (1.0 / nu) * (ddenom / denom_at_1)
            dmu = dt * domega
            dinside = (-1j * theta * u) + 0.5 * (sigma ** 2) * (u ** 2)
            dlog_inside = dinside / inside
            d_dt_over_nu = -dt / (nu ** 2)
            dlogphi = 1j * u * dmu - d_dt_over_nu * log_inside - (dt / nu) * dlog_inside
            grad["nu"] = phi * dlogphi

        return phi, grad

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        exp_divs, _ = _dividend_adjustment(T, self.divs)
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])
        denom_at_1 = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
        if denom_at_1 <= 0:
            denom_at_1 = max(1e-12, denom_at_1)
        omega = (1.0 / nu) * np.log(denom_at_1)
        # Approximate dividend mean shift via Σ ln(1-m).
        mu = np.log(self.S0) + (self.r - self.q + omega) * T + exp_divs

        k1 = theta * T
        k2 = (sigma ** 2 + (theta ** 2) * nu) * T
        k4 = (3.0 * (sigma ** 4) * nu + 12.0 * (sigma ** 2) * (theta ** 2) * (nu ** 2) + 6.0 * (theta ** 4) * (nu ** 3)) * T

        c1 = mu + k1
        c2 = k2
        c4 = k4
        return float(c1), float(c2), float(c4)



