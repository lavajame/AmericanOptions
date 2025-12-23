from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class DiscreteEventJump:
    """Single discrete binary event jump at a known time.

    The event applies an instantaneous multiplicative factor to spot:
        S_{t+} = S_{t-} * U   with prob p
        S_{t+} = S_{t-} * D   with prob (1-p)

    In log-space, this is an additive jump J in {ln U, ln D}.

    Risk-neutral handling
    --------------------
    For discounted spot to remain a martingale across the jump time, the jump
    factor must have mean 1 under the pricing measure:
        E[factor] = p*U + (1-p)*D = 1.

    If `ensure_martingale=True`, we apply a compensator that normalizes the
    factors so the mean is 1 while keeping p unchanged:
        U_rn = U / M,  D_rn = D / M,  where M = p*U + (1-p)*D.

    This matches the “subtract ln(M)” adjustment in log-space.
    """

    time: float
    p: float
    u: float
    d: float
    ensure_martingale: bool = True

    def __post_init__(self) -> None:
        t = float(self.time)
        p = float(self.p)
        u = float(self.u)
        d = float(self.d)

        if not np.isfinite(t) or t < 0.0:
            raise ValueError("Event time must be finite and >= 0")
        if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
            raise ValueError("Event probability p must be in (0, 1)")
        if not np.isfinite(u) or u <= 0.0:
            raise ValueError("Event up factor u must be finite and > 0")
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError("Event down factor d must be finite and > 0")

        m = p * u + (1.0 - p) * d
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError("Event mean factor must be finite and > 0")

    @property
    def mean_factor(self) -> float:
        p = float(self.p)
        return p * float(self.u) + (1.0 - p) * float(self.d)

    @property
    def u_eff(self) -> float:
        if not self.ensure_martingale:
            return float(self.u)
        return float(self.u) / float(self.mean_factor)

    @property
    def d_eff(self) -> float:
        if not self.ensure_martingale:
            return float(self.d)
        return float(self.d) / float(self.mean_factor)

    def phi(self, u: np.ndarray) -> np.ndarray:
        """Characteristic function factor for the event jump J.

        Returns E[exp(i*u*J)] for J in {ln(U_eff), ln(D_eff)}.
        """
        u_arr = np.asarray(u, dtype=complex)
        p = float(self.p)
        ln_u = np.log(self.u_eff)
        ln_d = np.log(self.d_eff)
        return p * np.exp(1j * u_arr * ln_u) + (1.0 - p) * np.exp(1j * u_arr * ln_d)

    def log_cumulants(self) -> tuple[float, float, float]:
        """(c1, c2, c4) cumulants of the *log jump* J.

        For a 2-point distribution this is exact:
        - c1 = E[J]
        - c2 = Var[J]
        - c4 = 4th cumulant = E[(J-μ)^4] - 3 Var[J]^2
        """
        p = float(self.p)
        j1 = float(np.log(self.u_eff))
        j2 = float(np.log(self.d_eff))
        mu = p * j1 + (1.0 - p) * j2
        v = p * (j1 - mu) ** 2 + (1.0 - p) * (j2 - mu) ** 2
        m4 = p * (j1 - mu) ** 4 + (1.0 - p) * (j2 - mu) ** 4
        c4 = m4 - 3.0 * (v ** 2)
        return float(mu), float(v), float(c4)
