"""Kou double-exponential jump-diffusion characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, dividend_char_factor

class KouCHF(CharacteristicFunction):
    """Kou double‑exponential jump‑diffusion."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))

        # Characteristic of jump Y (additive log-jump) E[e^{i u Y}]:
        # For double-exponential on Y:
        # E[e^{i u Y}] = p * eta1 / (eta1 - i u) + (1-p) * eta2 / (eta2 + i u)
        phi_jump = p * (eta1 / (eta1 - 1j * u)) + (1 - p) * (eta2 / (eta2 + 1j * u))
        # E[e^{Y}] for compensation (kappa)
        kappa = p * (eta1 / (eta1 - 1.0)) + (1 - p) * (eta2 / (eta2 + 1.0)) - 1.0
        mu = np.log(self.S0) + (self.r - self.q) * T - lam * kappa * T - 0.5 * (vol ** 2) * T
        phi = np.exp(1j * u * mu - 0.5 * (u ** 2) * (vol ** 2) * T + lam * T * (phi_jump - 1.0))
        return phi * dividend_char_factor(u, T, self.divs)

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Characteristic of log-return increment over dt."""
        u = np.asarray(u, dtype=complex)
        dt = float(dt)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))

        phi_jump = p * (eta1 / (eta1 - 1j * u)) + (1.0 - p) * (eta2 / (eta2 + 1j * u))
        kappa = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0)) - 1.0
        mu_term = (self.r - self.q) * dt - lam * kappa * dt - 0.5 * (vol ** 2) * dt
        exponent = 1j * u * mu_term - 0.5 * (u ** 2) * (vol ** 2) * dt + lam * dt * (phi_jump - 1.0)
        phi = np.exp(exponent)
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

        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))

        phi_jump = p * (eta1 / (eta1 - 1j * u)) + (1.0 - p) * (eta2 / (eta2 + 1j * u))
        kappa = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0)) - 1.0
        mu_term = (self.r - self.q) * dt - lam * kappa * dt - 0.5 * (vol ** 2) * dt
        exponent = 1j * u * mu_term - 0.5 * (u ** 2) * (vol ** 2) * dt + lam * dt * (phi_jump - 1.0)
        phi_base = np.exp(exponent)
        phi = phi_base * dividend_char_factor(u, dt, self.divs)

        grad: dict[str, np.ndarray] = {}

        if "vol" in params:
            dlog = (-(u ** 2 + 1j * u) * vol * dt)
            grad["vol"] = phi * dlog

        if "lam" in params:
            dlog = dt * (phi_jump - 1.0 - 1j * u * kappa)
            grad["lam"] = phi * dlog

        if "p" in params:
            dphi_dp = (eta1 / (eta1 - 1j * u)) - (eta2 / (eta2 + 1j * u))
            dkappa_dp = (eta1 / (eta1 - 1.0)) - (eta2 / (eta2 + 1.0))
            dlog = lam * dt * (dphi_dp - 1j * u * dkappa_dp)
            grad["p"] = phi * dlog

        if "eta1" in params:
            dphi = p * ((-1j * u) / ((eta1 - 1j * u) ** 2))
            dkappa = p * ((-1.0) / ((eta1 - 1.0) ** 2))
            dlog = lam * dt * (dphi - 1j * u * dkappa)
            grad["eta1"] = phi * dlog

        if "eta2" in params:
            dphi = (1.0 - p) * ((1j * u) / ((eta2 + 1j * u) ** 2))
            dkappa = (1.0 - p) * ((1.0) / ((eta2 + 1.0) ** 2))
            dlog = lam * dt * (dphi - 1j * u * dkappa)
            grad["eta2"] = phi * dlog

        return phi, grad

    def _var2(self, T: float) -> float:
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))
        var_jumps = lam * T * (p * 2.0 / (eta1 ** 2) + (1 - p) * 2.0 / (eta2 ** 2))
        return (vol ** 2) * T + var_jumps

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        sum_log, _ = _dividend_adjustment(T, self.divs)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))
        kappa = p * (eta1 / (eta1 - 1.0)) + (1 - p) * (eta2 / (eta2 + 1.0)) - 1.0

        EY = p * (1.0 / eta1) - (1 - p) * (1.0 / eta2)
        EY2 = 2.0 * p * (1.0 / (eta1 ** 2)) + 2.0 * (1 - p) * (1.0 / (eta2 ** 2))
        EY4 = 24.0 * p * (1.0 / (eta1 ** 4)) + 24.0 * (1 - p) * (1.0 / (eta2 ** 4))

        c1 = np.log(self.S0) + (self.r - self.q) * T + sum_log - lam * kappa * T - 0.5 * (vol ** 2) * T + lam * T * EY
        c2 = (vol ** 2) * T + lam * T * EY2
        c4 = lam * T * EY4
        return float(c1), float(c2), float(c4)



