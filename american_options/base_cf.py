"""Characteristic-function base model API.

Contains:
- CharacteristicFunction base class
- dividend-aware increment handling
- parameter-sensitivity hooks
- convenience wrappers for COS pricing
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .dividends import cash_divs_to_proportional_divs, dividend_char_factor, dividend_char_factor_window
from .events import DiscreteEventJump

class CharacteristicFunction:
    """
    Base class for all models.
    Subclasses must implement :meth:`char_func`.
    """

    def __init__(self,
                 S0: float,
                 r: float,
                 q: float,
                 divs: Dict[float, Tuple[float, float]],
                 params: Dict[str, Any]):
        self.S0 = S0
        self.r = r
        self.q = q
        # Project-wide convention: `divs` are cash dividends (mean, std) in spot currency.
        # Convert once at model construction to internal proportional parameters.
        self.divs = cash_divs_to_proportional_divs(S0, r, q, divs)
        self.params = params

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        """
        Placeholder characteristic function. Subclasses must override this.
        """
        raise NotImplementedError

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Default increment characteristic for time-homogeneous models: derive from
        the model's `char_func` by removing the dependence on ln(S0).
        Subclasses may override for improved numerical accuracy.
        """
        u = np.asarray(u, dtype=complex)
        return self.char_func(u, dt) * np.exp(-1j * u * np.log(self.S0))

    def increment_char_interval(self, u: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """Characteristic of log-return increment over [t0, t1].

        Default implementation supports proportional dividends scheduled in absolute time
        by reweighting the base `increment_char(u, dt)`.

                Notes
                -----
                - For models without discrete dividends, this reduces to time-homogeneous increments.
                - With dividends specified in absolute time, `increment_char(u, dt)` treats dividend
                    times as if they were relative to 0; this method corrects to the true (t0, t1] window.
        """
        dt = float(t1) - float(t0)
        if dt <= 0.0:
            u = np.asarray(u, dtype=complex)
            return np.ones_like(u)

        u = np.asarray(u, dtype=complex)
        phi = self.increment_char(u, dt)

        if not getattr(self, "divs", None):
            return phi

        # Base increment_char(u, dt) includes dividends with absolute times <= dt.
        # For a true interval [t0, t1], include only dividends with absolute times in (t0, t1].
        old = dividend_char_factor(u, dt, self.divs)
        new = dividend_char_factor_window(u, t0, t1, self.divs)
        # Avoid 0/0 issues (shouldn't happen for valid params, but guard anyway).
        corr = np.where(np.abs(old) > 0.0, new / old, 1.0 + 0j)
        return phi * corr

    # ----------------------------------------------------------------------- #
    # 2.2b. Model-parameter sensitivities (characteristic function gradients)
    # ----------------------------------------------------------------------- #
    def param_names(self) -> list[str]:
        """Return model parameter names for sensitivity reporting.

        Default: keys of `self.params` in deterministic order.
        Composite / wrapped models may override.
        """
        if hasattr(self, "params") and isinstance(self.params, dict):
            return [str(k) for k in sorted(self.params.keys(), key=lambda x: str(x))]
        return []

    def increment_char_and_grad(
        self,
        u: np.ndarray,
        dt: float,
        *,
        params: list[str] | None = None,
        method: str = "analytic",
        rel_step: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return (phi_inc, grad) for the increment CF over dt.

        grad maps param name -> d(phi_inc)/d(param) evaluated at the model's current params.

        Parameters
        ----------
        method:
            - "analytic": require an analytic implementation (subclasses override).
            - "fd": central finite differences on `self.params` (generic fallback).
        """
        method = str(method).lower().strip()
        if method == "fd":
            return self._increment_char_and_grad_fd(u, dt, params=params, rel_step=rel_step)
        raise NotImplementedError("Analytic sensitivities not implemented for this model")

    def _increment_char_and_grad_fd(
        self,
        u: np.ndarray,
        dt: float,
        *,
        params: list[str] | None = None,
        rel_step: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Finite-difference gradient of increment_char (central differences).

        This mutates `self.params` temporarily (restored before returning).
        """
        u = np.asarray(u, dtype=complex)
        dt = float(dt)
        phi0 = self.increment_char(u, dt)

        if params is None:
            params = self.param_names()

        grad: dict[str, np.ndarray] = {}
        if not params:
            return phi0, grad

        if not (hasattr(self, "params") and isinstance(self.params, dict)):
            return phi0, grad

        for name in params:
            if name not in self.params:
                continue
            base = self.params[name]
            if not isinstance(base, (int, float, np.floating)):
                continue
            base_f = float(base)
            h = float(rel_step) * max(1.0, abs(base_f))
            if h == 0.0:
                h = float(rel_step)

            try:
                self.params[name] = base_f + h
                phi_p = self.increment_char(u, dt)
                self.params[name] = base_f - h
                phi_m = self.increment_char(u, dt)
            finally:
                self.params[name] = base

            grad[name] = (phi_p - phi_m) / (2.0 * h)

        return phi0, grad

    def char_func_and_grad(
        self,
        u: np.ndarray,
        T: float,
        *,
        params: list[str] | None = None,
        method: str = "analytic",
        rel_step: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return (phi, grad) for phi(u;T) = E[e^{i u ln S_T}]."""
        u = np.asarray(u, dtype=complex)
        T = float(T)
        phi_inc, g_inc = self.increment_char_and_grad(u, T, params=params, method=method, rel_step=rel_step)
        shift = np.exp(1j * u * np.log(self.S0))
        phi = phi_inc * shift
        grad: dict[str, np.ndarray] = {k: v * shift for k, v in g_inc.items()}
        return phi, grad

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        """Return (c1, c2, c4) cumulants of ln(S_T).

        Used for COS truncation domain selection via
        a = c1 - L * sqrt(c2 + sqrt(c4)), b = c1 + L * sqrt(c2 + sqrt(c4)).

        Subclasses should override with analytic cumulants when available.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # 2.3. American rollback with softmax
    # ----------------------------------------------------------------------- #
    def european_price(self,
                       K: np.ndarray,
                       T: float,
                       is_call: bool = True,
                       N: int = 512,
                       L: float = 10.0,
                       event: DiscreteEventJump | None = None,
                       payoff_coeffs: str = "classic",
                       return_sensitivities: bool = False,
                       sens_method: str = "analytic",
                       sens_params: list[str] | None = None):
        """European option price via COS.

        This is a convenience wrapper around :class:`COSPricer`.
        """
        K = np.atleast_1d(K).astype(float)
        from .cos.pricer import COSPricer
        pricer = COSPricer(self, N=N, L=L)
        return pricer.european_price(
            K,
            T,
            is_call=is_call,
            event=event,
            payoff_coeffs=payoff_coeffs,
            return_sensitivities=return_sensitivities,
            sens_method=sens_method,
            sens_params=sens_params,
        )

    def american_price(self,
                       K: np.ndarray,
                       T: float,
                       steps: int = 50,
                       beta: float = 20.0,
                       N: int = 512,
                       L: float = 10.0,
                       use_softmax: bool = False,
                       return_european: bool = False,
                       event: DiscreteEventJump | None = None,
                       return_sensitivities: bool = False,
                       sens_method: str = "analytic",
                       sens_params: list[str] | None = None):
        """
        American call price via generic COS backward induction (Bowen/Zhang-Oosterlee style).
        Dispatches to the COSPricer which accepts the model (self) as input.

        Parameters
        ----------
        use_softmax: if True, uses differentiable softmax for early exercise (enables smooth sensitivities)
        return_european: if True, also return the European COS price for comparison/diagnostics
        """
        K = np.atleast_1d(K).astype(float)
        from .cos.pricer import COSPricer
        pricer = COSPricer(self, N=N, L=L)
        return pricer.american_price(
            K,
            T,
            steps=steps,
            beta=beta,
            use_softmax=use_softmax,
            return_european=return_european,
            event=event,
            return_sensitivities=return_sensitivities,
            sens_method=sens_method,
            sens_params=sens_params,
        )



