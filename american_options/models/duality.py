"""Put-call duality wrapper model."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..base_cf import CharacteristicFunction

class DualModel(CharacteristicFunction):
    """
    Put-Call Duality model wrapper.
    C(S0, K, r, q, T, phi) = P(K, S0, q, r, T, phi_dual)
    where phi_dual(u) = phi(-u - i) / phi(-i).
    """

    def __init__(self, base_model: CharacteristicFunction, dual_S0: float):
        super().__init__(
            S0=dual_S0,
            r=base_model.q,
            q=base_model.r,
            divs={},  # Discrete dividends duality is complex
            params=base_model.params
        )
        self.base_model = base_model

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u_complex = -u - 1j
        phi_num = self.base_model.char_func(u_complex, T)
        phi_den = self.base_model.char_func(-1j, T)
        # The base model char_func includes exp(i u_complex ln S0_base)
        # phi_num / phi_den = [phi_inc(-u-i) * exp(i(-u-i)ln S0_base)] / [phi_inc(-i) * exp(i(-i)ln S0_base)]
        #                  = [phi_inc(-u-i) / phi_inc(-i)] * exp(-i u ln S0_base)
        # We want the dual char_func to be [phi_inc(-u-i) / phi_inc(-i)] * exp(i u ln S0_dual)
        # So we multiply by exp(i u ln S0_base) * exp(i u ln S0_dual)
        return (phi_num / phi_den) * np.exp(1j * u * (np.log(self.S0) + np.log(self.base_model.S0)))

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        u_complex = -u - 1j
        # phi_inc_dual(u) = phi_inc_base(-u-i) / phi_inc_base(-i)
        phi_num = self.base_model.increment_char(u_complex, dt)
        phi_den = self.base_model.increment_char(-1j, dt)
        return phi_num / phi_den

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        # Use base model cumulants for truncation range, but shift c1
        c1, c2, c4 = self.base_model.cumulants(T)
        # c1_base = ln S0_base + (r_base - q_base + omega) T
        # c1_dual = ln S0_dual + (q_base - r_base - omega) T
        # Shift = (ln S0_dual - ln S0_base) + (q - r - omega) T - (r - q + omega) T
        #       = (ln S0_dual - ln S0_base) - 2 * (r - q + omega) T
        # Note: (r - q + omega) T is exactly (c1_base - ln S0_base)
        shift = np.log(self.S0) - np.log(self.base_model.S0) - 2.0 * (c1 - np.log(self.base_model.S0))
        return c1 + shift, c2, c4

    def _var2(self, T: float) -> float:
        # Use base model variance
        if hasattr(self.base_model, '_var2'):
            return self.base_model._var2(T)
        return 0.0


# --------------------------------------------------------------------------- #
# 2.x COS-based pricer
# --------------------------------------------------------------------------- #

