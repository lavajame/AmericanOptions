"""
Vectorised option-pricing engine

Provides:
* GBM, Merton JD, Kou JD, VG characteristic functions
* Discrete proportional dividends & borrow spread
* COS European pricing
* American rollback with softmax
* Sensitivity calculations
* Volatility inversion helper
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any
import scipy.optimize as opt

from .events import DiscreteEventJump


# --------------------------------------------------------------------------- #
# 1. Helpers – dividends & forward
# --------------------------------------------------------------------------- #

def cash_divs_to_proportional_divs(
    S0: float,
    r: float,
    q: float,
    divs_cash: Dict[float, Tuple[float, float]],
) -> Dict[float, Tuple[float, float]]:
    """Convert cash discrete dividends into internal proportional-dividend parameters.

    External project-wide convention (cash dividends):
        divs_cash[t] = (D_mean, D_std)
    where D_* are in *spot currency* paid at ex-div time t.

    Internal convention used by COS/LSMC/FDM code paths:
        divs_prop[t] = (m_mean, std_log)
    with multiplicative factor applied at ex-div:
        ln D_factor = ln(1-m_mean) - 0.5*std_log^2 + std_log * Z
    so E[D_factor] = (1-m_mean).

    We approximate a cash dividend as a proportional drop relative to the *expected* pre-div forward:
        m_mean ≈ D_mean / E[S_{t-}]
        Var[D_factor] ≈ (D_std / E[S_{t-}])^2
    and match this variance by choosing std_log via the lognormal moment relation.

    Note: this is an approximation (cash dividends are additive in reality).
    """
    if not divs_cash:
        return {}

    # Sort by time; build expected pre-div level recursively using mean impacts.
    items = [(float(t), float(Dm), float(Ds)) for t, (Dm, Ds) in divs_cash.items()]
    items.sort(key=lambda z: z[0])

    divs_prop: Dict[float, Tuple[float, float]] = {}
    mean_factor = 1.0
    for t, D_mean, D_std in items:
        if t <= 0.0:
            continue
        if D_mean < 0.0 or D_std < 0.0:
            raise ValueError("Cash dividends require non-negative mean/std")

        expected_pre = float(S0) * float(np.exp((r - q) * t)) * float(mean_factor)
        if expected_pre <= 0.0:
            raise ValueError("Invalid expected pre-div level while converting dividends")

        m_mean = float(D_mean) / expected_pre
        if m_mean >= 1.0:
            raise ValueError(f"Cash dividend too large at t={t}: D_mean={D_mean} vs E_pre={expected_pre}")
        if m_mean < 0.0:
            raise ValueError("Cash dividend mean cannot imply negative proportional drop")

        one_minus_m = max(1.0 - m_mean, 1e-12)
        rel_std = float(D_std) / expected_pre

        if rel_std <= 0.0:
            std_log = 0.0
        else:
            # For X ~ LogNormal with mean = mu_x and log-std = s:
            # Var[X] = (exp(s^2) - 1) * mu_x^2
            # Here mu_x := E[D_factor] = (1-m_mean) and we want Var[D_factor] ≈ rel_std^2.
            ratio = (rel_std * rel_std) / (one_minus_m * one_minus_m)
            std_log = float(np.sqrt(np.log(1.0 + max(ratio, 0.0))))

        divs_prop[t] = (m_mean, std_log)
        mean_factor *= one_minus_m

    return divs_prop

def _dividend_adjustment(T: float, divs: Dict[float, Tuple[float, float]]) -> Tuple[float, np.ndarray]:
    """
        Return the cumulative log product adjustment and an array of (mean, var) pairs.

        Internal dividend convention (proportional lognormal factor):
        - Each dividend event at time t has mean proportional drop `m` and a log-factor uncertainty `std_log`.
        - We model the dividend multiplicative factor as:
                ln D = ln(1-m) - 0.5 * std^2 + std * Z,  Z ~ N(0,1)
            so that E[D] = (1-m) while Var[ln D] = std^2.

        We return:
        - sum_log = Σ (ln(1-m) - 0.5*std^2) for t<=T
        - params = [(m, std^2), ...] for t<=T
    """
    sum_log = 0.0
    params = []
    for t, (m, std) in divs.items():
        if t <= T:
                        var = float(std) ** 2
                        sum_log += np.log(max(1.0 - m, 1e-12)) - 0.5 * var
                        params.append((m, var))
    return sum_log, np.array(params)


def _forward_price_from_prop_divs(S0: float, r: float, q: float, T: float,
                                 divs_prop: Dict[float, Tuple[float, float]]) -> float:
    """Internal forward using internal proportional dividends."""
    sum_log, div_params = _dividend_adjustment(T, divs_prop)
    var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
    # E[S_T] = S0 * exp((r-q)T) * Π E[D_factor]. With our convention E[D_factor]=(1-m),
    # which equals exp(sum_log + 0.5*var_div) since sum_log includes -0.5*var_div.
    return S0 * np.exp((r - q) * T + sum_log + 0.5 * var_div)


def forward_price(S0: float, r: float, q: float, T: float,
                  divs: Dict[float, Tuple[float, float]]) -> float:
    """
    Risk‑neutral forward under the project-wide *cash dividend* convention.

    divs[t] = (D_mean, D_std) in spot currency.

    We convert cash dividends to internal proportional parameters using the expected pre-div forward,
    then compute:
        F ≈ S0 * exp((r - q) * T) * Π E[D_factor]
    where E[D_factor] = (1 - m_mean).
    """
    divs_prop = cash_divs_to_proportional_divs(S0, r, q, divs)
    return _forward_price_from_prop_divs(S0, r, q, T, divs_prop)


def softmax(a: np.ndarray, b: np.ndarray, beta: float = 20.0) -> np.ndarray:
    """Numerically stable differentiable approximation of max(a, b) using log-sum-exp."""
    ba = beta * a
    bb = beta * b
    m = np.maximum(ba, bb)
    s = m + np.log(np.exp(ba - m) + np.exp(bb - m))
    return (a + b) / 2.0 + 0.5 * s / beta


# --------------------------------------------------------------------------- #
# 2. Abstract base class
# --------------------------------------------------------------------------- #
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
                       event: DiscreteEventJump | None = None) -> np.ndarray:
        """European option price via COS.

        This is a convenience wrapper around :class:`COSPricer`.
        """
        K = np.atleast_1d(K).astype(float)
        pricer = COSPricer(self, N=N, L=L)
        return pricer.european_price(K, T, is_call=is_call, event=event)

    def american_price(self,
                       K: np.ndarray,
                       T: float,
                       steps: int = 50,
                       beta: float = 20.0,
                       N: int = 512,
                       L: float = 10.0,
                       use_softmax: bool = False,
                       return_european: bool = False,
                       event: DiscreteEventJump | None = None):
        """
        American call price via generic COS backward induction (Bowen/Zhang-Oosterlee style).
        Dispatches to the COSPricer which accepts the model (self) as input.

        Parameters
        ----------
        use_softmax: if True, uses differentiable softmax for early exercise (enables smooth sensitivities)
        return_european: if True, also return the European COS price for comparison/diagnostics
        """
        K = np.atleast_1d(K).astype(float)
        pricer = COSPricer(self, N=N, L=L)
        return pricer.american_price(K, T, steps=steps, beta=beta, use_softmax=use_softmax, return_european=return_european, event=event)


# --------------------------------------------------------------------------- #
# 2.x Put-Call Duality
# --------------------------------------------------------------------------- #

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
class COSPricer:
    """Generic COS pricer that takes a CharacteristicFunction instance and prices options."""

    def __init__(self, model: CharacteristicFunction, N: int = 512, L: float = 10.0, M: int = None):
        self.model = model
        self.N = int(N)
        self.L = float(L)
        self.M = int(M) if M is not None else max(2 * self.N, 512)

    def european_price(self, K: np.ndarray, T: float, is_call: bool = True, event: DiscreteEventJump | None = None) -> np.ndarray:
        """COS European pricing using the Fang–Oosterlee COS method.

        Vectorized over strikes K.

        Notes
        -----
        - When ``is_call=True`` prices calls with payoff ``max(S_T - K, 0)``.
        - When ``is_call=False`` prices puts with payoff ``max(K - S_T, 0)`` directly
          (not via put-call parity).
        """
        K = np.atleast_1d(K).astype(float)
        # COS truncation domain from cumulants
        try:
            c1, c2, c4 = self.model.cumulants(T)
            if event is not None and 0.0 < float(event.time) <= float(T):
                e1, e2, e4 = event.log_cumulants()
                c1, c2, c4 = float(c1 + e1), float(c2 + e2), float(c4 + e4)
            c2 = float(max(c2, 0.0))
            c4 = float(max(c4, 0.0))
            width = self.L * np.sqrt(c2 + np.sqrt(c4))
            a = float(c1 - width)
            b = float(c1 + width)
        except Exception:
            # Fallback: variance-based symmetric interval (kept for models without cumulants)
            sum_log, div_params = _dividend_adjustment(T, self.model.divs)
            var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
            var = float(self.model._var2(T) + var_div)
            F = _forward_price_from_prop_divs(self.model.S0, self.model.r, self.model.q, T, self.model.divs)
            if event is not None and 0.0 < float(event.time) <= float(T):
                # Forward multiplier for the remaining horizon (mean factor = 1 if ensure_martingale=True)
                F *= float(event.mean_factor if not event.ensure_martingale else 1.0)
                var += float(event.log_cumulants()[1])
            mu = float(np.log(F) - 0.5 * var)
            width = self.L * np.sqrt(max(var, 1e-16))
            a, b = mu - width, mu + width

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval: a={a}, b={b}")

        N = self.N
        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)
        phi = self.model.char_func(u.flatten(), T).reshape((N,))
        if event is not None and 0.0 < float(event.time) <= float(T):
            phi = phi * event.phi(u.flatten())
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        # Payoff coefficients (call/put) in log-space over [a, b]
        # Call: f(x)=max(exp(x)-K,0) on [ln K, b]
        # Put:  f(x)=max(K-exp(x),0) on [a, ln K]
        logK = np.clip(np.log(K), a, b).reshape((1, -1))
        K_reshaped = K.reshape((1, -1))
        if is_call:
            c = logK
            d = b
        else:
            c = np.full_like(logK, a)
            d = logK

        theta_c = u * (c - a)
        theta_d = u * (d - a)
        sin_c = np.sin(theta_c)
        sin_d = np.sin(theta_d)
        cos_c = np.cos(theta_c)
        cos_d = np.cos(theta_d)

        psi = np.zeros((N, len(K)))
        chi = np.zeros((N, len(K)))

        k0 = (k.flatten() == 0)
        psi[k0, :] = (d - c)
        chi[k0, :] = (np.exp(d) - np.exp(c))

        non0 = ~k0
        u_nz = u[non0]
        denom = 1.0 + (u_nz ** 2)
        psi[non0, :] = (sin_d[non0, :] - sin_c[non0, :]) / u_nz
        chi[non0, :] = (
            (np.exp(d) * (cos_d[non0, :] + u_nz * sin_d[non0, :])
             - np.exp(c) * (cos_c[non0, :] + u_nz * sin_c[non0, :]))
            / denom
        )

        if is_call:
            Vk = (2.0 / (b - a)) * (chi - K_reshaped * psi)
        else:
            Vk = (2.0 / (b - a)) * (K_reshaped * psi - chi)

        # COS summation rule: k=0 term has weight 1/2
        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * T) * mat_sum

    def american_price(self, K: np.ndarray, T: float, steps: int = 50, is_call: bool = True, beta: float = 20.0, use_softmax: bool = False, return_european: bool = False, return_trajectory: bool = False, return_continuation_trajectory: bool = False, rollback_debug: bool = False, diagnostics_file: str = None, event: DiscreteEventJump | None = None):
        """COS-based backward induction for American options.

        - Maintain coefficients across steps.
        - Reconstruct continuation in price space, compare with intrinsic.
        - For calls with q > 0, we use a parity-based rollback to improve stability,
          but for general robustness (especially with fat tails), pricing the dual put is preferred.
        """
        K = np.atleast_1d(K).astype(float)
        N = self.N
        M = self.M
        prices = np.zeros_like(K)
        euro_prices = None
        if return_european:
            euro_prices = self.european_price(K, T, is_call=is_call, event=event)

        trajectory_cache = [] if return_trajectory else None
        cont_trajectory_cache = [] if return_continuation_trajectory else None

        # For calls with no discrete dividends and non-positive continuous yield,
        # early exercise is not optimal in standard models, so American == European.
        # The numerical rollback can otherwise introduce small max-operator artifacts.
        if is_call and (not self.model.divs) and float(self.model.q) <= 0.0 and not (return_trajectory or return_continuation_trajectory):
            eu = self.european_price(K, T, is_call=True)
            if return_european:
                return eu, eu
            return eu

        # Put-Call Duality for American Calls with continuous yield only.
        # Discrete dividends require explicit rollback mapping; do not use duality shortcut.
        # Also, trajectory caching relies on running the full rollback, so skip the shortcut
        # when any trajectory output is requested.
        if (
            is_call
            and float(self.model.q) > 0.0
            and not self.model.divs
            and not (return_trajectory or return_continuation_trajectory)
        ):
            # C(S0, K, r, q, T) = P(K, S0, q, r, T) under dual measure
            # We price a put with strike = S0, spot = K, r_dual = q, q_dual = r
            dual_prices = np.zeros_like(K)
            for j, Kval in enumerate(K):
                dual_model = DualModel(self.model, dual_S0=Kval)
                dual_pricer = COSPricer(dual_model, N=self.N, L=self.L, M=self.M)
                # Price a put with strike = self.model.S0
                p = dual_pricer.american_price(np.array([self.model.S0]), T, steps=steps, is_call=False, 
                                               beta=beta, use_softmax=use_softmax)
                dual_prices[j] = p[0]
            
            if return_european:
                return dual_prices, euro_prices
            return dual_prices

        # With discrete dividends, parity-based call rollback is fragile and can create
        # discontinuities near ex-div dates (especially when the horizon changes and
        # a dividend moves close to t=0). In that case, price calls directly.
        direct_call_with_divs = bool(is_call) and bool(self.model.divs)

        # Truncation range for entire horizon T
        exp_divs, div_params = _dividend_adjustment(T, self.model.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        var = self.model._var2(T) + var_div
        if event is not None and 0.0 < float(event.time) <= float(T):
            var += float(event.log_cumulants()[1])
        # Truncation safety: cap variance used for COS truncation to avoid astronomic grids
        if hasattr(self.model, 'params') and 'safety_trunc_var' in self.model.params:
            VAR_TRUNC = float(self.model.params.get('safety_trunc_var'))
        else:
            try:
                from american_options import CGMYCHF
                is_cgmy = isinstance(self.model, CGMYCHF)
            except Exception:
                is_cgmy = False
            VAR_TRUNC = 4.0 if is_cgmy else 10.0
        var_trunc = min(var, VAR_TRUNC)
        F = _forward_price_from_prop_divs(self.model.S0, self.model.r, self.model.q, T, self.model.divs)
        if event is not None and 0.0 < float(event.time) <= float(T) and not event.ensure_martingale:
            F *= float(event.mean_factor)
        mu = np.log(F) - 0.5 * var_trunc
        a = mu - self.L * np.sqrt(max(var_trunc, 1e-16))
        b = mu + self.L * np.sqrt(max(var_trunc, 1e-16))

        x_grid = np.linspace(a, b, M)
        dx = x_grid[1] - x_grid[0]
        S_grid = np.exp(x_grid)

        # Precompute cosine basis and u-grid
        k = np.arange(N).reshape((-1, 1))
        cos_k_x = np.cos(k * np.pi * (x_grid - a) / (b - a))  # shape (N, M)
        sin_k_x = np.sin(k * np.pi * (x_grid - a) / (b - a))  # shape (N, M)
        u = (k * np.pi / (b - a)).flatten()

        # Build a rollback time grid that includes dividend times exactly.
        #
        # Previous behavior snapped dividends to an integer index via round(t_div / dt).
        # When this routine is called repeatedly for slightly different horizons (e.g. tau=T-t
        # in diagnostics), dt changes slightly and the rounding can jump by 1, creating visible
        # spikes/discontinuities around ex-div dates.
        #
        # Here we instead split the grid at each dividend time, then sub-step each interval.
        # This makes the dividend mapping occur at a deterministic time boundary.
        if steps <= 0:
            t_steps = np.array([0.0, float(T)])
            dt_nominal = float(T)
        else:
            dt_nominal = float(T) / float(steps)
            div_times = sorted([float(t) for t in self.model.divs.keys() if 0.0 < float(t) <= float(T)])
            evt_time = float(event.time) if (event is not None) else None
            event_times = [evt_time] if (evt_time is not None and 0.0 < evt_time <= float(T)) else []
            # Interval boundaries
            t_knots = sorted(set([0.0] + div_times + event_times + [float(T)]))
            t_steps_list = [0.0]
            for t0, t1 in zip(t_knots[:-1], t_knots[1:]):
                interval = float(t1) - float(t0)
                if interval <= 0.0:
                    continue
                n_seg = max(1, int(round(interval / dt_nominal)))
                seg = t0 + interval * (np.arange(1, n_seg + 1, dtype=float) / float(n_seg))
                t_steps_list.extend(seg.tolist())
            t_steps = np.array(t_steps_list, dtype=float)

        # Dividend lookup with tolerance (times should be present in t_steps, but guard float noise).
        div_time_m = [(float(t_div), float(m)) for t_div, (m, _) in self.model.divs.items() if 0.0 < float(t_div) <= float(T)]
        div_tol = max(1e-12, 1e-10 * dt_nominal)

        evt_tol = div_tol
        evt_time = float(event.time) if (event is not None) else None

        # Endpoint trapezoidal weights for projection
        w = np.ones_like(x_grid)
        w[0] = 0.5
        w[-1] = 0.5

        for j, Kval in enumerate(K):
            # Terminal payoff
            if direct_call_with_divs:
                V_grid = np.maximum(S_grid - Kval, 0.0)
            else:
                # Puts are priced directly. Calls without discrete dividends use a parity-based
                # rollback on put coefficients for stability.
                V_grid = np.maximum(Kval - S_grid, 0.0)
            
            coeffs = (2.0 / (b - a)) * (cos_k_x @ (V_grid * w)) * dx

            for step_idx in range(len(t_steps) - 1, 0, -1):
                dt = float(t_steps[step_idx] - t_steps[step_idx - 1])
                phi_dt = self.model.increment_char(u, dt)
                coeffs_mod = coeffs.copy()
                coeffs_mod[0] *= 0.5
                
                # Spectral damping window (optional)
                window = np.ones(N)
                
                # Reconstruct continuation value
                term_cos = (np.real(phi_dt).reshape((-1, 1)) * coeffs_mod.reshape((-1, 1)) * window.reshape((-1, 1)))
                cont = (term_cos * cos_k_x).sum(axis=0)
                term_sin = (-np.imag(phi_dt).reshape((-1, 1)) * coeffs_mod.reshape((-1, 1)) * window.reshape((-1, 1)))
                cont += (term_sin * sin_k_x).sum(axis=0)
                cont *= np.exp(-self.model.r * dt)

                # Dividend mapping
                t_current = float(t_steps[step_idx - 1])
                for t_div, m_div in div_time_m:
                    if abs(t_div - t_current) <= div_tol and m_div > 0.0:
                        shift = np.log(1.0 - m_div)
                        # Correct mapping for proportional discrete dividends:
                        # V_pre(x) = V_post(x + log(1-m))
                        cont = np.interp(x_grid + shift, x_grid, cont, left=cont[0], right=0.0)
                        break

                # Discrete event mapping (binary jump) at known time
                if event is not None and evt_time is not None and abs(evt_time - t_current) <= evt_tol:
                    shift_u = float(np.log(event.u_eff))
                    shift_d = float(np.log(event.d_eff))
                    cont_u = np.interp(x_grid + shift_u, x_grid, cont, left=cont[0], right=0.0)
                    cont_d = np.interp(x_grid + shift_d, x_grid, cont, left=cont[0], right=0.0)
                    cont = float(event.p) * cont_u + (1.0 - float(event.p)) * cont_d

                remaining_tau = float(T) - t_current
                
                if direct_call_with_divs:
                    intrinsic = S_grid - Kval
                    if use_softmax:
                        xdiff = beta * (intrinsic - cont)
                        m_stable = np.maximum(0.0, xdiff)
                        V_prev = cont + (m_stable + np.log(np.exp(-m_stable) + np.exp(xdiff - m_stable))) / beta
                    else:
                        V_prev = np.maximum(intrinsic, cont)
                    V_prev = np.maximum(V_prev, 0.0)
                    V_prev_call = V_prev

                elif is_call:
                    # Convert put-continuation to call-continuation via parity
                    sum_log_total, _ = _dividend_adjustment(T, self.model.divs)
                    sum_log_up_to_t, _ = _dividend_adjustment(t_current, self.model.divs)
                    sum_log_rem = sum_log_total - sum_log_up_to_t
                    spot_adj = S_grid * np.exp(-self.model.q * remaining_tau + sum_log_rem)
                    if event is not None and evt_time is not None and not event.ensure_martingale:
                        # If event is in the remaining horizon, include its mean factor in the forward-like adjustment
                        if float(t_current) < float(evt_time) <= float(T):
                            spot_adj = spot_adj * float(event.mean_factor)
                    
                    cont_call = cont + spot_adj - Kval * np.exp(-self.model.r * remaining_tau)
                    intrinsic = S_grid - Kval
                    
                    if use_softmax:
                        xdiff = beta * (intrinsic - cont_call)
                        m_stable = np.maximum(0.0, xdiff)
                        V_prev_call = cont_call + (m_stable + np.log(np.exp(-m_stable) + np.exp(xdiff - m_stable))) / beta
                    else:
                        V_prev_call = np.maximum(intrinsic, cont_call)
                    V_prev_call = np.maximum(V_prev_call, 0.0)
                    
                    # Reproject back to put coefficients (parity)
                    V_prev = V_prev_call - spot_adj + Kval * np.exp(-self.model.r * remaining_tau)
                    # Note: Do NOT clip V_prev to 0 here if q > 0, as it can be negative.
                    # However, for the COS method to be stable, we might need to handle the e^x growth.
                else:
                    # Direct put exercise
                    intrinsic = Kval - S_grid
                    if use_softmax:
                        xdiff = beta * (intrinsic - cont)
                        m_stable = np.maximum(0.0, xdiff)
                        V_prev = cont + (m_stable + np.log(np.exp(-m_stable) + np.exp(xdiff - m_stable))) / beta
                    else:
                        V_prev = np.maximum(intrinsic, cont)
                    V_prev = np.maximum(V_prev, 0.0)

                # Cache trajectories at S0 for first strike only
                if (return_trajectory or return_continuation_trajectory) and j == 0:
                    x0 = float(np.log(self.model.S0))
                    if return_continuation_trajectory:
                        if direct_call_with_divs:
                            cont_at_s0 = float(np.interp(x0, x_grid, cont))
                        else:
                            cont_at_s0 = float(np.interp(x0, x_grid, cont_call if is_call else cont))
                        cont_trajectory_cache.append((t_current, cont_at_s0))
                    if return_trajectory:
                        if direct_call_with_divs:
                            exercise_at_s0 = float(np.interp(x0, x_grid, V_prev_call))
                        else:
                            exercise_at_s0 = float(np.interp(x0, x_grid, V_prev_call if is_call else V_prev))
                        trajectory_cache.append((t_current, exercise_at_s0))

                coeffs = (2.0 / (b - a)) * (cos_k_x @ (V_prev * w)) * dx

            # Final value at t=0
            V0 = 0.5 * coeffs[0] * cos_k_x[0] + (coeffs[1:, None] * cos_k_x[1:]).sum(axis=0)
            val0 = float(np.interp(np.log(self.model.S0), x_grid, V0))
            
            if direct_call_with_divs:
                prices[j] = val0

            elif is_call:
                # Convert back to call price
                sum_log_total, _ = _dividend_adjustment(T, self.model.divs)
                forward_adj = self.model.S0 * np.exp(-self.model.q * T + sum_log_total)
                if event is not None and 0.0 < float(event.time) <= float(T) and not event.ensure_martingale:
                    forward_adj *= float(event.mean_factor)
                prices[j] = val0 + forward_adj - Kval * np.exp(-self.model.r * T)
            else:
                prices[j] = val0

        if return_trajectory and trajectory_cache is not None:
            trajectory_cache.sort(key=lambda z: z[0])

        if return_continuation_trajectory and cont_trajectory_cache is not None:
            cont_trajectory_cache.sort(key=lambda z: z[0])

        out = [prices]
        if return_european:
            out.append(euro_prices)
        if return_trajectory:
            out.append(trajectory_cache)
        if return_continuation_trajectory:
            out.append(cont_trajectory_cache)
        return out[0] if len(out) == 1 else tuple(out)

    # ----------------------------------------------------------------------- #
    # 2.5. Helper – variance over [0,T] for COS truncation
    # ----------------------------------------------------------------------- #
    def _var2(self, T: float) -> float:
        """
        Return Var[ln S_T] (default approximation: diffusion variance).
        Override if the model has jumps or different structure.
        """
        vol = float(self.params.get("vol", 0.0))
        return (vol ** 2) * T


# --------------------------------------------------------------------------- #
# 3. Concrete model implementations
# --------------------------------------------------------------------------- #
class GBMCHF(CharacteristicFunction):
    """Black‑Scholes / GBM characteristic function."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        """
        φ(u) = exp(i u μ - 0.5 u^2 σ^2 T) where μ = ln S0 + (r - q)T + Σ ln(1-m) - 0.5 σ^2 T
        Includes dividend multiplicative mean via Σ ln(1-m) and dividend variance via approximate add to variance.
        """
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        mu = np.log(self.S0) + (self.r - self.q) * T + sum_log - 0.5 * (vol ** 2) * T
        var = (vol ** 2) * T + var_div
        return np.exp(1j * u * mu - 0.5 * (u ** 2) * var)

    def increment_char(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Characteristic of log-return increment over dt: E[e^{i u (ln S_{t+dt}-ln S_t)}]."""
        # For GBM, increment characteristic equals exp(-0.5 u^2 sigma^2 dt + i u (r - q - 0.5 sigma^2) dt)
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        sum_log, div_params = _dividend_adjustment(dt, self.divs)
        var = (vol ** 2) * dt + (float(np.sum(div_params[:, 1])) if div_params.size else 0.0)
        # drift excluding ln S0: (r - q) dt + sum_log - 0.5 sigma^2 dt
        mu_term = (self.r - self.q) * dt + sum_log - 0.5 * (vol ** 2) * dt
        return np.exp(1j * u * mu_term - 0.5 * (u ** 2) * var)

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        vol = float(self.params["vol"])
        c2 = (vol ** 2) * T + var_div
        c1 = np.log(self.S0) + (self.r - self.q) * T + sum_log - 0.5 * c2
        c4 = 0.0
        return float(c1), float(c2), float(c4)

    def _var2(self, T: float) -> float:
        vol = float(self.params["vol"])
        return (vol ** 2) * T


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


class KouCHF(CharacteristicFunction):
    """Kou double‑exponential jump‑diffusion."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        vol = float(self.params["vol"])
        lam = float(self.params.get("lam", 0.0))
        p = float(self.params.get("p", 0.5))
        eta1 = float(self.params.get("eta1", 10.0))
        eta2 = float(self.params.get("eta2", 5.0))
        exp_divs, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

        # Characteristic of jump Y (additive log-jump) E[e^{i u Y}]:
        # For double-exponential on Y:
        # E[e^{i u Y}] = p * eta1 / (eta1 - i u) + (1-p) * eta2 / (eta2 + i u)
        phi_jump = p * (eta1 / (eta1 - 1j * u)) + (1 - p) * (eta2 / (eta2 + 1j * u))
        # E[e^{Y}] for compensation (kappa)
        kappa = p * (eta1 / (eta1 - 1.0)) + (1 - p) * (eta2 / (eta2 + 1.0)) - 1.0
        mu = np.log(self.S0) + (self.r - self.q) * T + exp_divs - lam * kappa * T - 0.5 * (vol ** 2) * T
        var_jumps = lam * T * (p * 2.0 / (eta1 ** 2) + (1 - p) * 2.0 / (eta2 ** 2))
        var = (vol ** 2) * T + var_jumps + var_div
        phi = np.exp(1j * u * mu - 0.5 * (u ** 2) * (vol ** 2) * T + lam * T * (phi_jump - 1.0))
        # Add dividend uncertainty as an independent Gaussian log-factor.
        if var_div > 0.0:
            phi *= np.exp(-0.5 * (u ** 2) * var_div)
        return phi

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


class VGCHF(CharacteristicFunction):
    """Variance‑Gamma (Madan‑Carr‑Chang) model."""

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])
        exp_divs, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

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
        mu = np.log(self.S0) + (self.r - self.q + omega) * T + exp_divs
        phi_base = (1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * (nu) * (u ** 2)) ** (-T / nu)
        phi = np.exp(1j * u * mu) * phi_base
        # Add dividend uncertainty as an independent Gaussian log-factor.
        if var_div > 0.0:
            phi *= np.exp(-0.5 * (u ** 2) * var_div)
        return phi

    def _var2(self, T: float) -> float:
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])
        # Var[X_t] = t*(sigma^2 + theta^2 * nu)
        return T * (sigma ** 2 + theta ** 2 * nu)

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        exp_divs, _ = _dividend_adjustment(T, self.divs)
        theta = float(self.params["theta"])
        sigma = float(self.params["sigma"])
        nu = float(self.params["nu"])
        denom_at_1 = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
        if denom_at_1 <= 0:
            denom_at_1 = max(1e-12, denom_at_1)
        omega = (1.0 / nu) * np.log(denom_at_1)
        mu = np.log(self.S0) + (self.r - self.q + omega) * T + exp_divs

        k1 = theta * T
        k2 = (sigma ** 2 + (theta ** 2) * nu) * T
        k4 = (3.0 * (sigma ** 4) * nu + 12.0 * (sigma ** 2) * (theta ** 2) * (nu ** 2) + 6.0 * (theta ** 4) * (nu ** 3)) * T

        c1 = mu + k1
        c2 = k2
        c4 = k4
        return float(c1), float(c2), float(c4)


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

# --------------------------------------------------------------------------- #
# 4. Volatility inversion helper
# --------------------------------------------------------------------------- #
def invert_vol_for_american_price(american_price: float,
                                  S0: float,
                                  r: float,
                                  q: float,
                                  T: float,
                                  divs: Dict[float, Tuple[float, float]],
                                  K: float,
                                  target_eps: float = 1e-8,
                                  max_iter: int = 100) -> float:
    """
    Given an American price and other inputs, find the GBM volatility that reproduces it.
    Uses a bracketed root-finding on vol (Brent).
    """
    american_price = float(american_price)
    K = float(K)

    def f(vol: float) -> float:
        vol = max(vol, 1e-12)
        model = GBMCHF(S0, r, q, divs, {"vol": vol})
        price = model.american_price(np.array([K]), T)[0]
        return price - american_price

    # bracket find: start with small to moderate bounds
    lo, hi = 1e-6, 2.0
    flo, fhi = f(lo), f(hi)
    trials = 0
    while flo * fhi > 0 and trials < max_iter:
        hi *= 2.0
        fhi = f(hi)
        trials += 1
    if flo * fhi > 0:
        raise RuntimeError("Failed to bracket root for implied volatility")
    sol = opt.root_scalar(f, bracket=[lo, hi], method="brentq", xtol=target_eps, maxiter=max_iter)
    if not sol.converged:
        raise RuntimeError("Root-finding failed to converge")
    return sol.root


# --------------------------------------------------------------------------- #
# 4b. Nearest-GBM helper for Lévy models
# --------------------------------------------------------------------------- #
def equivalent_gbm(model: CharacteristicFunction, T: float) -> GBMCHF:
    """
    Return a GBMCHF whose variance matches the model's Var[ln S_T]/T at maturity T.
    This is useful for comparing a Lévy model to its nearest-diffusion proxy.
    """
    var = float(model._var2(T))
    if T <= 0:
        raise ValueError("T must be positive")
    vol_eq = np.sqrt(max(var / T, 0.0))
    return GBMCHF(model.S0, model.r, model.q, model.divs, {"vol": vol_eq})
