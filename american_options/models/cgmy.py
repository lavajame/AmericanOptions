"""CGMY characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from scipy.special import gamma as sp_gamma

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment

class CGMYCHF(CharacteristicFunction):
    """CGMY (Carr–Geman–Madan–Yor) class of tempered stable processes.

    Parameters (in `params` dict): C, G, M, Y
    """

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        from scipy.special import gamma as sp_gamma

        u = np.asarray(u, dtype=complex)
        C = float(self.params.get("C", 0.02))
        G = float(self.params.get("G", 5.0))
        M = float(self.params.get("M", 5.0))
        Y = float(self.params.get("Y", 0.5))

        # dividend adjustments (sum of log(1-m)) and variance from discrete divs
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

        # characteristic exponent (per unit time) for CGMY:
        # psi(u) = C * Gamma(-Y) * [ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ]
        # Use stable form for large M, G: M^Y * ((1 - i u/M)^Y - 1) = M^Y * expm1(Y * log1p(-i u/M))
        gamma_m = sp_gamma(-Y)
        def psi_unit(z):
            # Use log1p and expm1 to avoid catastrophic cancellation when z/M is small
            # For extremely large M, G, the term M^Y * expm1(...) can still overflow if C is huge.
            # However, C * M^Y is what matters.
            # C * M^Y = (sigma^2 / (2 * Gamma(2-Y) * M^(Y-2))) * M^Y = sigma^2 * M^2 / (2 * Gamma(2-Y))
            # This grows as M^2.
            # To handle extremely large M, G, we can use the Taylor expansion of expm1(Y * log1p(x))
            # for small x = -1j * z / M.
            # expm1(Y * log1p(x)) = Y*x + 0.5*Y*(Y-1)*x^2 + ...
            # M^Y * expm1(...) = Y * x * M^Y + 0.5 * Y * (Y-1) * x^2 * M^Y + ...
            # = Y * (-1j * z) * M^(Y-1) + 0.5 * Y * (Y-1) * (-z^2) * M^(Y-2) + ...
            
            x_m = -1j * z / M
            x_g = 1j * z / G
            
            # Threshold for Taylor expansion to avoid precision loss or overflow
            if np.all(np.abs(x_m) < 1e-4) and np.all(np.abs(x_g) < 1e-4):
                # Second order Taylor: Y*x + 0.5*Y*(Y-1)*x^2
                # M^Y * (Y*x + 0.5*Y*(Y-1)*x^2) = Y * (-1j*z) * M^(Y-1) - 0.5*Y*(Y-1)*z^2 * M^(Y-2)
                term_m = Y * (-1j * z) * np.power(M, Y - 1.0) - 0.5 * Y * (Y - 1.0) * (z**2) * np.power(M, Y - 2.0)
                term_g = Y * (1j * z) * np.power(G, Y - 1.0) - 0.5 * Y * (Y - 1.0) * (z**2) * np.power(G, Y - 2.0)
            else:
                term_m = np.power(M, Y) * np.expm1(Y * np.log1p(x_m))
                term_g = np.power(G, Y) * np.expm1(Y * np.log1p(x_g))
                
            res = C * gamma_m * (term_m + term_g)
            return res

        # compute psi values and the special psi(-i) used to enforce exp-moment
        psi_vals = psi_unit(u)
        psi_minus_i = psi_unit(-1j)

        # choose mu so that E[e^{X_T}] = exp((r-q)T + sum_log)
        # mu = ln S0 + (r-q)T + sum_log - psi(-i)T
        # For large M, G, psi(-i) should approach 0.5 * sigma^2.
        # Let's check the exponent: 1j * u * mu + psi(u) * T
        # = 1j * u * (ln S0 + (r-q)T + sum_log - psi(-i)T) + psi(u) * T
        # = 1j * u * (ln S0 + (r-q)T + sum_log) + (psi(u) - 1j * u * psi(-i)) * T
        
        mu_base = np.log(self.S0) + (self.r - self.q) * T + sum_log
        exponent = 1j * u * mu_base + (psi_vals - 1j * u * psi_minus_i) * T
        
        phi = np.exp(exponent)
        # Add dividend uncertainty as an independent Gaussian log-factor.
        if var_div > 0.0:
            phi *= np.exp(-0.5 * (u ** 2) * var_div)
        return phi

    def _var2(self, T: float) -> float:
        # Var[X_T] = c2
        return self.cumulants(T)[1]

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        from scipy.special import gamma as sp_gamma
        C = float(self.params.get("C", 0.02))
        G = float(self.params.get("G", 5.0))
        M = float(self.params.get("M", 5.0))
        Y = float(self.params.get("Y", 0.5))
        sum_log, _ = _dividend_adjustment(T, self.divs)

        # c_n = C * T * Gamma(n - Y) * (M^(Y-n) + G^(Y-n))
        # For c1, we must include the risk-neutral drift correction:
        # mu = ln S0 + (r-q)T + sum_log - psi(-i)T
        # c1 = mu + E[X_1]*T = mu + C * T * Gamma(1-Y) * (M^(Y-1) - G^(Y-1))
        
        # Use log-space for stable power calculations: M^(Y-n) = exp((Y-n)*ln M)
        def stable_pow(base, exp):
            return np.exp(exp * np.log(base))

        # psi(-i) = C * Gamma(-Y) * ( (M-1)^Y - M^Y + (G+1)^Y - G^Y )
        # Use stable expm1/log1p form for psi(-i)
        gamma_m = sp_gamma(-Y)
        term_m_rn = stable_pow(M, Y) * np.expm1(Y * np.log1p(-1.0 / M))
        term_g_rn = stable_pow(G, Y) * np.expm1(Y * np.log1p(1.0 / G))
        psi_minus_i = C * gamma_m * (term_m_rn + term_g_rn)

        mu = np.log(self.S0) + (self.r - self.q) * T + sum_log - psi_minus_i * T
        
        # c1_jump = C * T * Gamma(1-Y) * (M^(Y-1) - G^(Y-1))
        c1_jump = C * T * sp_gamma(1.0 - Y) * (stable_pow(M, Y - 1.0) - stable_pow(G, Y - 1.0))
        c1 = mu + c1_jump
        
        # c2 = C * T * Gamma(2-Y) * (M^(Y-2) + G^(Y-2))
        c2 = C * T * sp_gamma(2.0 - Y) * (stable_pow(M, Y - 2.0) + stable_pow(G, Y - 2.0))
        
        # c4 = C * T * Gamma(4-Y) * (M^(Y-4) + G^(Y-4))
        c4 = C * T * sp_gamma(4.0 - Y) * (stable_pow(M, Y - 4.0) + stable_pow(G, Y - 4.0))
        
        return float(c1), float(c2), float(c4)



