from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class DiscreteEventJump:
    """Single discrete binary event jump at a known time.

    The event is specified in *log-jump* space:

        J = u   with probability p
        J = d   with probability (1-p)

    where typically u > 0 (up move) and d < 0 (down move).

    The spot jump is multiplicative:
        S_{t+} = S_{t-} * exp(J)

    Risk-neutral handling
    --------------------
    For discounted spot to remain a martingale across the jump time, the jump
    factor must have mean 1 under the pricing measure:
        E[exp(J)] = p*exp(u) + (1-p)*exp(d) = 1.

    If `ensure_martingale=True`, we apply a compensator that normalizes the
    factors so the mean is 1 while keeping p unchanged:
        exp(u_rn) = exp(u) / M,
        exp(d_rn) = exp(d) / M,
        where M = E[exp(J)].

    In log space this is simply:
        u_rn = u - ln(M),
        d_rn = d - ln(M).
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
            raise ValueError("Event up log-jump u must be finite and > 0")
        if not np.isfinite(d) or d >= 0.0:
            raise ValueError("Event down log-jump d must be finite and < 0")

        m = p * float(np.exp(u)) + (1.0 - p) * float(np.exp(d))
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError("Event mean factor must be finite and > 0")

    @property
    def mean_factor(self) -> float:
        p = float(self.p)
        return p * float(np.exp(self.u)) + (1.0 - p) * float(np.exp(self.d))

    @property
    def u_eff(self) -> float:
        """Effective up *factor* (exp of log-jump), after martingale normalization if enabled."""
        return float(np.exp(self.u_log_eff))

    @property
    def d_eff(self) -> float:
        """Effective down *factor* (exp of log-jump), after martingale normalization if enabled."""
        return float(np.exp(self.d_log_eff))

    @property
    def u_log_eff(self) -> float:
        """Effective up log-jump, after martingale normalization if enabled."""
        if not self.ensure_martingale:
            return float(self.u)
        return float(self.u) - float(np.log(self.mean_factor))

    @property
    def d_log_eff(self) -> float:
        """Effective down log-jump, after martingale normalization if enabled."""
        if not self.ensure_martingale:
            return float(self.d)
        return float(self.d) - float(np.log(self.mean_factor))

    def phi(self, u: np.ndarray) -> np.ndarray:
        """Characteristic function factor for the event jump J.

        Returns E[exp(i*u*J)] for J in {u_eff_log, d_eff_log}.
        """
        u_arr = np.asarray(u, dtype=complex)
        p = float(self.p)
        ju = float(self.u_log_eff)
        jd = float(self.d_log_eff)
        return p * np.exp(1j * u_arr * ju) + (1.0 - p) * np.exp(1j * u_arr * jd)

    def log_cumulants(self) -> tuple[float, float, float]:
        """(c1, c2, c4) cumulants of the *log jump* J.

        For a 2-point distribution this is exact:
        - c1 = E[J]
        - c2 = Var[J]
        - c4 = 4th cumulant = E[(J-μ)^4] - 3 Var[J]^2
        """
        p = float(self.p)
        j1 = float(self.u_log_eff)
        j2 = float(self.d_log_eff)
        mu = p * j1 + (1.0 - p) * j2
        v = p * (j1 - mu) ** 2 + (1.0 - p) * (j2 - mu) ** 2
        m4 = p * (j1 - mu) ** 4 + (1.0 - p) * (j2 - mu) ** 4
        c4 = m4 - 3.0 * (v ** 2)
        return float(mu), float(v), float(c4)
