"""Normal Inverse Gaussian (NIG) characteristic function model.

We model the log-price as an exponential-L\u00e9vy process:
    ln S_T = ln S0 + (r-q)T + sum_log - psi(-i)T + X_T
where X_T is an NIG L\u00e9vy process with characteristic exponent per unit time:
    psi(u) = i u mu + delta * (gamma - sqrt(alpha^2 - (beta + i u)^2))
    gamma = sqrt(alpha^2 - beta^2)

The martingale correction uses psi(-i) so that, absent dividends/events,
E[S_T] = S0 * exp((r-q)T).

Parameter constraints for the risk-neutral correction to be real-valued:
- alpha > |beta|
- alpha > |beta + 1| (finite exp-moment at 1)
- delta > 0

References: Barndorff-Nielsen (1997) and standard exponential-L\u00e9vy option pricing texts.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, dividend_char_factor


class NIGCHF(CharacteristicFunction):
    """Normal Inverse Gaussian (NIG) exponential-L\u00e9vy model."""

    @staticmethod
    def _validate(alpha: float, beta: float, delta: float) -> None:
        if not np.isfinite(alpha) or not np.isfinite(beta) or not np.isfinite(delta):
            raise ValueError("NIG parameters must be finite")
        if alpha <= 0.0:
            raise ValueError("NIG requires alpha > 0")
        if delta <= 0.0:
            raise ValueError("NIG requires delta > 0")
        if alpha <= abs(beta):
            raise ValueError("NIG requires alpha > |beta|")
        # Need finite exp-moment at 1 for the martingale correction psi(-i).
        if alpha <= abs(beta + 1.0):
            raise ValueError("NIG requires alpha > |beta+1| for risk-neutral martingale correction")

    @staticmethod
    def _psi_unit(u: np.ndarray, *, alpha: float, beta: float, delta: float, mu: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        gamma = np.sqrt((alpha * alpha) - (beta * beta) + 0j)
        sqrt_term = np.sqrt((alpha * alpha) - (beta + 1j * u) ** 2 + 0j)
        return 1j * u * mu + delta * (gamma - sqrt_term)

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        T = float(T)

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        delta = float(self.params["delta"])
        mu = float(self.params.get("mu", 0.0))
        self._validate(alpha, beta, delta)

        psi_vals = self._psi_unit(u, alpha=alpha, beta=beta, delta=delta, mu=mu)
        psi_minus_i = self._psi_unit(-1j, alpha=alpha, beta=beta, delta=delta, mu=mu)

        mu_base = np.log(self.S0) + (self.r - self.q) * T
        exponent = 1j * u * mu_base + (psi_vals - 1j * u * psi_minus_i) * T
        phi = np.exp(exponent)
        return phi * dividend_char_factor(u, T, self.divs)

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        dt = float(dt)

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        delta = float(self.params["delta"])
        mu = float(self.params.get("mu", 0.0))
        self._validate(alpha, beta, delta)

        psi_vals = self._psi_unit(u, alpha=alpha, beta=beta, delta=delta, mu=mu)
        psi_minus_i = self._psi_unit(-1j, alpha=alpha, beta=beta, delta=delta, mu=mu)

        exponent = 1j * u * (self.r - self.q) * dt + (psi_vals - 1j * u * psi_minus_i) * dt
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

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        delta = float(self.params["delta"])
        mu = float(self.params.get("mu", 0.0))
        self._validate(alpha, beta, delta)

        gamma = np.sqrt((alpha * alpha) - (beta * beta) + 0j)
        sqrt_u = np.sqrt((alpha * alpha) - (beta + 1j * u) ** 2 + 0j)
        sqrt_mi = np.sqrt((alpha * alpha) - (beta + 1.0) ** 2 + 0j)  # u = -i -> beta + i u = beta + 1

        psi_vals = 1j * u * mu + delta * (gamma - sqrt_u)
        psi_minus_i = mu + delta * (gamma - sqrt_mi)

        exponent = 1j * u * (self.r - self.q) * dt + (psi_vals - 1j * u * psi_minus_i) * dt
        phi_base = np.exp(exponent)
        phi = phi_base * dividend_char_factor(u, dt, self.divs)

        grad: dict[str, np.ndarray] = {}

        if "mu" in params:
            dpsi_u = 1j * u
            dpsi_mi = 1.0
            dlogphi = dt * (dpsi_u - 1j * u * dpsi_mi)
            grad["mu"] = phi * dlogphi

        if "delta" in params:
            dpsi_u = (gamma - sqrt_u)
            dpsi_mi = (gamma - sqrt_mi)
            dlogphi = dt * (dpsi_u - 1j * u * dpsi_mi)
            grad["delta"] = phi * dlogphi

        if "alpha" in params:
            dgamma = alpha / gamma
            dsqrt_u = alpha / sqrt_u
            dsqrt_mi = alpha / sqrt_mi
            dpsi_u = delta * (dgamma - dsqrt_u)
            dpsi_mi = delta * (dgamma - dsqrt_mi)
            dlogphi = dt * (dpsi_u - 1j * u * dpsi_mi)
            grad["alpha"] = phi * dlogphi

        if "beta" in params:
            dgamma = -beta / gamma
            dsqrt_u = -(beta + 1j * u) / sqrt_u
            dsqrt_mi = -(beta + 1.0) / sqrt_mi
            dpsi_u = delta * (dgamma - dsqrt_u)
            dpsi_mi = delta * (dgamma - dsqrt_mi)
            dlogphi = dt * (dpsi_u - 1j * u * dpsi_mi)
            grad["beta"] = phi * dlogphi

        return phi, grad

    def _var2(self, T: float) -> float:
        # Var[ln S_T] excluding dividend uncertainty proxy.
        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        delta = float(self.params["delta"])
        self._validate(alpha, beta, delta)
        gamma = float(np.sqrt((alpha * alpha) - (beta * beta)))
        var_rate = float(delta) * (alpha * alpha) / (gamma ** 3)
        return float(T) * var_rate

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        # c1/c2/c4 for ln(S_T). Use dividend mean shift and include a variance proxy
        # for dividend uncertainty (same convention as GBMCHF/CGMYCHF).
        sum_log, div_params = _dividend_adjustment(float(T), self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        delta = float(self.params["delta"])
        mu = float(self.params.get("mu", 0.0))
        self._validate(alpha, beta, delta)

        gamma = float(np.sqrt((alpha * alpha) - (beta * beta)))

        # NIG cumulants for X_T
        k1 = float(mu) * float(T) + float(delta) * float(T) * (beta / gamma)
        k2 = float(delta) * float(T) * (alpha * alpha) / (gamma ** 3)
        k4 = 3.0 * float(delta) * float(T) * (alpha * alpha) * ((alpha * alpha) + 4.0 * (beta * beta)) / (gamma ** 7)

        # Risk-neutral correction subtracts psi(-i) from drift.
        gamma_c = float(np.sqrt((alpha * alpha) - ((beta + 1.0) ** 2)))
        psi_minus_i = float(mu) + float(delta) * (gamma - gamma_c)

        c1 = np.log(self.S0) + (self.r - self.q) * float(T) + sum_log + k1 - psi_minus_i * float(T)
        c2 = k2 + var_div
        c4 = k4
        return float(c1), float(c2), float(c4)
