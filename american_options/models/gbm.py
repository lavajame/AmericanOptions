"""GBM (Black-Scholes) characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, dividend_char_factor

class GBMCHF(CharacteristicFunction):
    """Black‑Scholes / GBM characteristic function."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        """
        GBM characteristic function with discrete dividends.
        """
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        # Base GBM (no dividends)
        mu = np.log(self.S0) + (self.r - self.q - 0.5 * (vol ** 2)) * T
        phi = np.exp(1j * u * mu - 0.5 * (u ** 2) * ((vol ** 2) * T))
        # Multiply by dividend characteristic factor (handles mean + uncertainty)
        return phi * dividend_char_factor(u, T, self.divs)

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Characteristic of log-return increment over dt: E[e^{i u (ln S_{t+dt}-ln S_t)}]."""
        # For GBM, increment characteristic equals exp(-0.5 u^2 sigma^2 dt + i u (r - q - 0.5 sigma^2) dt)
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        mu_term = (self.r - self.q - 0.5 * (vol ** 2)) * float(dt)
        phi = np.exp(1j * u * mu_term - 0.5 * (u ** 2) * ((vol ** 2) * float(dt)))
        return phi * dividend_char_factor(u, float(dt), self.divs)

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

        vol = float(self.params["vol"])
        mu_term = (self.r - self.q - 0.5 * (vol ** 2)) * dt
        phi_base = np.exp(1j * u * mu_term - 0.5 * (u ** 2) * ((vol ** 2) * dt))
        phi = phi_base * dividend_char_factor(u, dt, self.divs)

        grad: dict[str, np.ndarray] = {}
        if "vol" in params:
            dlog_dvol = (-1j * u * vol * dt) - (u ** 2) * vol * dt
            grad["vol"] = phi * dlog_dvol
        return phi, grad

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        vol = float(self.params["vol"])
        c2 = (vol ** 2) * T + var_div
        c1 = np.log(self.S0) + (self.r - self.q) * T + sum_log - 0.5 * ((vol ** 2) * T)
        c4 = 0.0
        return float(c1), float(c2), float(c4)

    def _var2(self, T: float) -> float:
        vol = float(self.params["vol"])
        return (vol ** 2) * T



