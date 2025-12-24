"""Merton jump-diffusion characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment

class MertonCHF(CharacteristicFunction):
    """Merton jump‑diffusion (Gaussian jumps)."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        muJ = float(self.params.get("muJ", 0.0))
        sigmaJ = float(self.params.get("sigmaJ", 0.0))
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

        # Jump mgf: E[e^{Y}] for Y ~ N(muJ, sigmaJ^2)
        kappa = np.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0
        mu = np.log(self.S0) + (self.r - self.q) * T + sum_log - lam * kappa * T - 0.5 * (vol ** 2) * T
        var = (vol ** 2) * T + lam * T * (muJ ** 2 + sigmaJ ** 2) + var_div
        # jump characteristic for additive log-jump: E[e^{i u Y}] = exp(i u muJ - 0.5 u^2 sigmaJ^2)
        phi_jump = np.exp(1j * u * muJ - 0.5 * (u ** 2) * sigmaJ ** 2)
        phi = np.exp(1j * u * mu - 0.5 * (u ** 2) * (vol ** 2) * T + lam * T * (phi_jump - 1.0))
        # Add dividend uncertainty as an independent Gaussian log-factor.
        if var_div > 0.0:
            phi *= np.exp(-0.5 * (u ** 2) * var_div)
        return phi

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        muJ = float(self.params.get("muJ", 0.0))
        sigmaJ = float(self.params.get("sigmaJ", 0.0))
        sum_log, div_params = _dividend_adjustment(dt, self.divs)
        # phi_jump for dt
        phi_jump = np.exp(1j * u * muJ - 0.5 * (u ** 2) * sigmaJ ** 2)
        psi = -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump - 1.0) + 1j * u * ((self.r - self.q) + (sum_log / max(dt, 1e-16)) - 0.5 * (vol ** 2))
        return np.exp(psi * dt)

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
        muJ = float(self.params.get("muJ", 0.0))
        sigmaJ = float(self.params.get("sigmaJ", 0.0))
        sum_log, _ = _dividend_adjustment(dt, self.divs)

        phi_jump = np.exp(1j * u * muJ - 0.5 * (u ** 2) * (sigmaJ ** 2))
        drift_rate = (self.r - self.q) + (sum_log / max(dt, 1e-16)) - 0.5 * (vol ** 2)
        psi = -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump - 1.0) + 1j * u * drift_rate
        phi = np.exp(psi * dt)

        grad: dict[str, np.ndarray] = {}
        # dpsi/dvol = -(u^2 + i u) * vol
        if "vol" in params:
            dpsi = -(u ** 2 + 1j * u) * vol
            grad["vol"] = phi * (dt * dpsi)
        if "lam" in params:
            dpsi = (phi_jump - 1.0)
            grad["lam"] = phi * (dt * dpsi)
        if "muJ" in params:
            dphi_jump = (1j * u) * phi_jump
            dpsi = lam * dphi_jump
            grad["muJ"] = phi * (dt * dpsi)
        if "sigmaJ" in params:
            dphi_jump = (-(u ** 2) * sigmaJ) * phi_jump
            dpsi = lam * dphi_jump
            grad["sigmaJ"] = phi * (dt * dpsi)
        return phi, grad

    def _var2(self, T: float) -> float:
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        muJ = float(self.params.get("muJ", 0.0))
        sigmaJ = float(self.params.get("sigmaJ", 0.0))
        return (vol ** 2) * T + lam * T * (muJ ** 2 + sigmaJ ** 2)

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        sum_log, _ = _dividend_adjustment(T, self.divs)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        muJ = float(self.params.get("muJ", 0.0))
        sigmaJ = float(self.params.get("sigmaJ", 0.0))
        kappa = np.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0
        EY = muJ
        EY2 = muJ ** 2 + sigmaJ ** 2
        EY4 = muJ ** 4 + 6.0 * muJ ** 2 * sigmaJ ** 2 + 3.0 * sigmaJ ** 4
        c1 = np.log(self.S0) + (self.r - self.q) * T + sum_log - lam * kappa * T - 0.5 * (vol ** 2) * T + lam * T * EY
        c2 = (vol ** 2) * T + lam * T * EY2
        c4 = lam * T * EY4
        return float(c1), float(c2), float(c4)



