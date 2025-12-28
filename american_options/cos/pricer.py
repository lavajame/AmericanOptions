"""COS pricing engine (European + American rollback)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, _dividend_adjustment_window, _forward_price_from_prop_divs
from ..events import DiscreteEventJump
from .. import numerics
from ..models.duality import DualModel

class COSPricer:
    """Generic COS pricer that takes a CharacteristicFunction instance and prices options."""

    # Shared caches across COSPricer instances.
    # This matters because many call sites construct a new COSPricer per price request.
    _AMERICAN_CTX_CACHE: "OrderedDict[Hashable, Dict[str, Any]]" = OrderedDict()
    _AMERICAN_CTX_CACHE_MAX = 12

    def __init__(self, model: CharacteristicFunction, N: int = 512, L: float = 10.0, M: int = None):
        self.model = model
        self.N = int(N)
        self.L = float(L)
        self.M = int(M) if M is not None else max(2 * self.N, 512)

        # Interpolation used for dividend/event shift mappings on the x_grid.
        # Default is linear (robust). Options:
        #   - 'linear'
        #   - 'pchip' (shape-preserving monotone cubic Hermite; recommended if you want smaller M)
        #   - 'cubic' (Catmull–Rom; can overshoot near kinks)
        params = getattr(model, "params", None)
        if isinstance(params, dict):
            self.shift_interp = str(params.get("shift_interp", "linear")).lower().strip()
        else:
            self.shift_interp = "linear"

    @staticmethod
    def _float_key(x: float, ndp: int = 14) -> float:
        return float(round(float(x), ndp))

    @classmethod
    def _model_cache_key(cls, model: CharacteristicFunction) -> tuple:
        # Model identity for caching: type + core scalars + params + internal div schedule.
        # (We avoid id(model) so new model instances still hit the cache.)
        params_items: list[tuple[str, str]] = []
        if hasattr(model, "params") and isinstance(model.params, dict):
            for k, v in sorted(model.params.items(), key=lambda kv: str(kv[0])):
                params_items.append((str(k), repr(v)))

        div_items: list[tuple[float, float, float]] = []
        if hasattr(model, "divs") and isinstance(model.divs, dict):
            for t, (m, std) in sorted(model.divs.items(), key=lambda kv: float(kv[0])):
                div_items.append((cls._float_key(t), cls._float_key(m), cls._float_key(std)))

        return (
            type(model).__name__,
            cls._float_key(model.S0),
            cls._float_key(model.r),
            cls._float_key(model.q),
            tuple(params_items),
            tuple(div_items),
        )

    @classmethod
    def _event_cache_key(cls, event: Optional[DiscreteEventJump]) -> tuple:
        if event is None:
            return (None,)
        # Use effective log-jumps (after martingale normalization) because those drive rollback mapping.
        return (
            "DiscreteEventJump",
            cls._float_key(event.time),
            cls._float_key(event.p),
            cls._float_key(event.u_log_eff),
            cls._float_key(event.d_log_eff),
            bool(getattr(event, "ensure_martingale", True)),
            cls._float_key(getattr(event, "mean_factor", 1.0)),
        )

    @classmethod
    def _american_context_key(
        cls,
        model: CharacteristicFunction,
        *,
        T: float,
        steps: int,
        N: int,
        M: int,
        L: float,
        event: Optional[DiscreteEventJump],
    ) -> tuple:
        return (
            "american_ctx",
            cls._model_cache_key(model),
            cls._event_cache_key(event),
            cls._float_key(T),
            int(steps),
            int(N),
            int(M),
            cls._float_key(L),
        )

    @staticmethod
    def _build_shift_operator(x_grid: np.ndarray, shift: float, *, kind: str = "linear") -> Dict[str, np.ndarray]:
        """Precompute interpolation indices/weights for y(x+shift) on the x_grid.

        Parameters
        ----------
        x_grid:
            1D monotone increasing grid in log-space.
        shift:
            Log-shift applied to the argument: evaluate y(x + shift).
        kind:
            'linear' (default), 'pchip' (shape-preserving monotone cubic Hermite),
            or 'cubic' (Catmull–Rom on 4 neighboring points).

        Notes
        -----
        - Linear is monotone/robust near kinks.
                - PCHIP is shape-preserving on monotone data and avoids overshoot.
                - Catmull–Rom can reduce error for smooth regions, but may overshoot near sharp
                    kinks; we fall back to linear near the domain edges.
        """
        x_grid = np.asarray(x_grid, dtype=float)
        kind = str(kind).lower().strip()
        xq = x_grid + float(shift)

        idx_lin = np.searchsorted(x_grid, xq, side="right") - 1
        idx_lin = np.clip(idx_lin, 0, len(x_grid) - 2)
        x0 = x_grid[idx_lin]
        x1 = x_grid[idx_lin + 1]
        denom = (x1 - x0)
        wgt_lin = (xq - x0) / denom

        left_mask = xq <= x_grid[0]
        right_mask = xq >= x_grid[-1]

        if kind == "linear" or len(x_grid) < 4:
            return {
                "kind": "linear",
                "idx": idx_lin.astype(np.int64, copy=False),
                "wgt": wgt_lin.astype(float, copy=False),
                "left_mask": left_mask,
                "right_mask": right_mask,
            }

        if kind == "pchip":
            # PCHIP needs node-derivatives computed from values at apply-time, but the
            # query cell indices and local coordinates (t in [0,1]) are value-independent.
            h = float(x_grid[1] - x_grid[0])
            return {
                "kind": "pchip",
                "idx": idx_lin.astype(np.int64, copy=False),
                "t": wgt_lin.astype(float, copy=False),
                "h": h,
                "left_mask": left_mask,
                "right_mask": right_mask,
            }

        # Fall back to Catmull–Rom cubic.
        kind = "cubic"

        # Cubic Catmull–Rom interpolation on uniform-ish grids.
        # Use linear fallback near edges where idx-1 or idx+2 is out of range.
        idx_c = np.clip(idx_lin, 1, len(x_grid) - 3)
        edge_mask = (idx_lin < 1) | (idx_lin > (len(x_grid) - 3))
        xc0 = x_grid[idx_c]
        xc1 = x_grid[idx_c + 1]
        tc = (xq - xc0) / (xc1 - xc0)

        t = tc.astype(float, copy=False)
        t2 = t * t
        t3 = t2 * t

        w0 = (-0.5 * t + 1.0 * t2 - 0.5 * t3)
        w1 = (1.0 - 2.5 * t2 + 1.5 * t3)
        w2 = (0.5 * t + 2.0 * t2 - 1.5 * t3)
        w3 = (-0.5 * t2 + 0.5 * t3)

        return {
            "kind": "cubic",
            "idx_lin": idx_lin.astype(np.int64, copy=False),
            "wgt_lin": wgt_lin.astype(float, copy=False),
            "idx": idx_c.astype(np.int64, copy=False),
            "w0": w0.astype(float, copy=False),
            "w1": w1.astype(float, copy=False),
            "w2": w2.astype(float, copy=False),
            "w3": w3.astype(float, copy=False),
            "edge_mask": edge_mask,
            "left_mask": left_mask,
            "right_mask": right_mask,
        }

    @staticmethod
    def _apply_shift_operator(values: np.ndarray, op: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply a precomputed shift operator to a 2D grid (nK, M)."""
        vals = np.asarray(values, dtype=float)
        if vals.ndim == 1:
            vals = vals[None, :]

        kind = str(op.get("kind", "linear")).lower().strip()
        left_mask = op["left_mask"]
        right_mask = op["right_mask"]

        if kind == "linear":
            idx = op["idx"]
            wgt = op["wgt"]
            out = (1.0 - wgt)[None, :] * vals[:, idx] + wgt[None, :] * vals[:, idx + 1]
        elif kind == "pchip":
            # Shape-preserving monotone cubic Hermite interpolation (PCHIP).
            # Assumes uniform x_grid (true in this engine: x_grid=linspace(a,b,M)).
            idx = op["idx"]
            t = op["t"]
            h = float(op["h"])

            # Compute node derivatives per row using the Fritsch–Carlson method.
            # vals is shape (nK, M)
            nK, m = vals.shape
            if m < 2:
                out = vals.copy()
            else:
                delta = (vals[:, 1:] - vals[:, :-1]) / max(h, 1e-300)  # (nK, m-1)
                d = np.zeros((nK, m), dtype=float)

                if m == 2:
                    d[:, 0] = delta[:, 0]
                    d[:, 1] = delta[:, 0]
                else:
                    # Interior nodes
                    d0 = delta[:, :-1]
                    d1 = delta[:, 1:]
                    prod = d0 * d1
                    hm = np.zeros_like(prod)
                    mask = prod > 0.0
                    hm[mask] = (2.0 * d0[mask] * d1[mask]) / (d0[mask] + d1[mask])
                    d[:, 1:-1] = hm

                    # Endpoints (standard PCHIP endpoint formula with limiting)
                    d_left = (2.0 * delta[:, 0] - delta[:, 1])
                    # If d_left has wrong sign, set to 0. If it is too large in magnitude, cap.
                    wrong = (d_left * delta[:, 0]) <= 0.0
                    d_left[wrong] = 0.0
                    cap = (delta[:, 0] * delta[:, 1] < 0.0) & (np.abs(d_left) > 3.0 * np.abs(delta[:, 0]))
                    d_left[cap] = 3.0 * delta[:, 0][cap]
                    d[:, 0] = d_left

                    d_right = (2.0 * delta[:, -1] - delta[:, -2])
                    wrong = (d_right * delta[:, -1]) <= 0.0
                    d_right[wrong] = 0.0
                    cap = (delta[:, -1] * delta[:, -2] < 0.0) & (np.abs(d_right) > 3.0 * np.abs(delta[:, -1]))
                    d_right[cap] = 3.0 * delta[:, -1][cap]
                    d[:, -1] = d_right

                # Evaluate cubic Hermite on each query point.
                # Hermite basis
                tt = t.astype(float, copy=False)
                tt2 = tt * tt
                tt3 = tt2 * tt
                h00 = (2.0 * tt3 - 3.0 * tt2 + 1.0)
                h10 = (tt3 - 2.0 * tt2 + tt)
                h01 = (-2.0 * tt3 + 3.0 * tt2)
                h11 = (tt3 - tt2)

                y0 = vals[:, idx]
                y1 = vals[:, idx + 1]
                m0 = d[:, idx]
                m1 = d[:, idx + 1]

                out = (
                    h00[None, :] * y0
                    + h10[None, :] * (max(h, 1e-300) * m0)
                    + h01[None, :] * y1
                    + h11[None, :] * (max(h, 1e-300) * m1)
                )
        else:
            idx = op["idx"]
            w0 = op["w0"]
            w1 = op["w1"]
            w2 = op["w2"]
            w3 = op["w3"]
            edge_mask = op["edge_mask"]

            out = (
                w0[None, :] * vals[:, idx - 1]
                + w1[None, :] * vals[:, idx]
                + w2[None, :] * vals[:, idx + 1]
                + w3[None, :] * vals[:, idx + 2]
            )

            # Linear fallback near edges
            if np.any(edge_mask):
                idx_lin = op["idx_lin"]
                wgt_lin = op["wgt_lin"]
                out[:, edge_mask] = (1.0 - wgt_lin[edge_mask])[None, :] * vals[:, idx_lin[edge_mask]] + wgt_lin[edge_mask][None, :] * vals[:, idx_lin[edge_mask] + 1]

        if np.any(left_mask):
            out[:, left_mask] = vals[:, 0][:, None]
        if np.any(right_mask):
            out[:, right_mask] = 0.0
        return out

    def _get_or_build_american_context(self, T: float, *, steps: int, event: DiscreteEventJump | None) -> Dict[str, Any]:
        """Build (or fetch) strike-independent rollback context for American pricing."""
        key = self._american_context_key(self.model, T=float(T), steps=int(steps), N=self.N, M=self.M, L=self.L, event=event)
        cache = self._AMERICAN_CTX_CACHE
        ctx = cache.get(key)
        if ctx is not None:
            # LRU bump
            cache.move_to_end(key)
            return ctx

        N = self.N
        M = self.M

        # Truncation range for entire horizon T
        exp_divs, div_params = _dividend_adjustment(float(T), self.model.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        var = self.model._var2(float(T)) + var_div
        if event is not None and 0.0 < float(event.time) <= float(T):
            var += float(event.log_cumulants()[1])

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

        F = _forward_price_from_prop_divs(self.model.S0, self.model.r, self.model.q, float(T), self.model.divs)
        if event is not None and 0.0 < float(event.time) <= float(T) and not event.ensure_martingale:
            F *= float(event.mean_factor)
        mu = float(np.log(F) - 0.5 * var_trunc)
        a = float(mu - self.L * np.sqrt(max(var_trunc, 1e-16)))
        b = float(mu + self.L * np.sqrt(max(var_trunc, 1e-16)))

        x_grid = np.linspace(a, b, M)
        dx = float(x_grid[1] - x_grid[0])
        S_grid = np.exp(x_grid)

        k = np.arange(N).reshape((-1, 1))
        cos_k_x = np.cos(k * np.pi * (x_grid - a) / (b - a))
        sin_k_x = np.sin(k * np.pi * (x_grid - a) / (b - a))
        u = (k * np.pi / (b - a)).flatten()

        # Time grid including dividends/events exactly
        if steps <= 0:
            t_steps = np.array([0.0, float(T)], dtype=float)
            dt_nominal = float(T)
        else:
            dt_nominal = float(T) / float(steps)
            div_times = sorted([float(t) for t in self.model.divs.keys() if 0.0 < float(t) <= float(T)])
            evt_time = float(event.time) if (event is not None) else None
            event_times = [evt_time] if (evt_time is not None and 0.0 < evt_time <= float(T)) else []
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

        div_tol = max(1e-12, 1e-10 * float(dt_nominal))
        evt_tol = div_tol
        evt_time = float(event.time) if (event is not None) else None

        # Endpoint trapezoidal weights for projection
        w = np.ones_like(x_grid)
        w[0] = 0.5
        w[-1] = 0.5

        # Precompute shift operators for dividends (by t_current) and event u/d.
        div_shift_ops_by_time: Dict[float, Dict[str, np.ndarray]] = {}
        for t_div, (m, _std) in self.model.divs.items():
            t_div = float(t_div)
            if 0.0 < t_div <= float(T) and float(m) > 0.0:
                shift = float(np.log(max(1.0 - float(m), 1e-12)))
                div_shift_ops_by_time[t_div] = self._build_shift_operator(x_grid, shift, kind=self.shift_interp)

        evt_ops = None
        if event is not None and evt_time is not None and 0.0 < evt_time <= float(T):
            evt_ops = (
                self._build_shift_operator(x_grid, float(event.u_log_eff), kind=self.shift_interp),
                self._build_shift_operator(x_grid, float(event.d_log_eff), kind=self.shift_interp),
            )

        # Precompute per-step increment CFs + discount.
        step_info = []
        for step_idx in range(len(t_steps) - 1, 0, -1):
            dt = float(t_steps[step_idx] - t_steps[step_idx - 1])
            t_current = float(t_steps[step_idx - 1])
            phi_dt = self.model.increment_char(u, dt)
            step_info.append(
                (
                    dt,
                    t_current,
                    np.real(phi_dt).astype(float, copy=False),
                    np.imag(phi_dt).astype(float, copy=False),
                    float(np.exp(-self.model.r * dt)),
                )
            )

        # Dividend prefix sums for parity conversions.
        # Under the project convention, mean dividend factor is (1-m); uncertainty is handled
        # in the CF, so these sums track only Σ ln(1-m).
        sum_log_total = 0.0
        div_log_terms: list[tuple[float, float]] = []
        for t, (m, std) in sorted(self.model.divs.items(), key=lambda kv: float(kv[0])):
            t = float(t)
            if 0.0 < t <= float(T):
                term = float(np.log(max(1.0 - float(m), 1e-12)))
                div_log_terms.append((t, term))
                sum_log_total += term

        # Map each t_current to sum_log_up_to_t via a sweep.
        sum_log_up_to: Dict[float, float] = {}
        running = 0.0
        j = 0
        # Use sorted unique times from t_steps (excluding T) for stable lookup.
        t_sorted = sorted(set(float(t) for t in t_steps[:-1]))
        for t_cur in t_sorted:
            while j < len(div_log_terms) and div_log_terms[j][0] <= t_cur + 1e-15:
                running += div_log_terms[j][1]
                j += 1
            sum_log_up_to[t_cur] = running

        # Interp-at-S0 weights (fixed across strikes)
        x0 = float(np.log(self.model.S0))
        ix0 = int(np.searchsorted(x_grid, x0, side="right") - 1)
        ix0 = max(0, min(ix0, len(x_grid) - 2))
        wgt0 = float((x0 - x_grid[ix0]) / (x_grid[ix0 + 1] - x_grid[ix0]))

        ctx = {
            "a": a,
            "b": b,
            "x_grid": x_grid,
            "S_grid": S_grid,
            "dx": dx,
            "w": w,
            "cos_k_x": cos_k_x,
            "sin_k_x": sin_k_x,
            "u": u,
            "t_steps": t_steps,
            "dt_nominal": float(dt_nominal),
            "div_tol": float(div_tol),
            "evt_tol": float(evt_tol),
            "evt_time": evt_time,
            "div_shift_ops_by_time": div_shift_ops_by_time,
            "evt_ops": evt_ops,
            "step_info": step_info,
            "sum_log_total": float(sum_log_total),
            "sum_log_up_to": sum_log_up_to,
            "ix0": int(ix0),
            "wgt0": float(wgt0),
        }

        # LRU insert
        cache[key] = ctx
        cache.move_to_end(key)
        while len(cache) > int(self._AMERICAN_CTX_CACHE_MAX):
            cache.popitem(last=False)
        return ctx

    def european_price(
        self,
        K: np.ndarray,
        T: float,
        is_call: bool = True,
        event: DiscreteEventJump | None = None,
        payoff_coeffs: str = "classic",
        return_sensitivities: bool = False,
        sens_method: str = "analytic",
        sens_params: list[str] | None = None,
    ) -> np.ndarray:
        """COS European pricing using the Fang–Oosterlee COS method.

        Vectorized over strikes K.

        Notes
        -----
        - When ``is_call=True`` prices calls with payoff ``max(S_T - K, 0)``.
        - When ``is_call=False`` prices puts with payoff ``max(K - S_T, 0)`` directly
          (not via put-call parity).
        """
        K = np.atleast_1d(K).astype(float)

        payoff_coeffs = str(payoff_coeffs).lower().strip()
        if payoff_coeffs not in {"classic", "lefloch"}:
            raise ValueError("payoff_coeffs must be one of: 'classic', 'lefloch'")

        if payoff_coeffs == "lefloch":
            if return_sensitivities:
                raise NotImplementedError("Sensitivities are currently implemented only for payoff_coeffs='classic'")
            return self._european_price_lefloch(K, T, is_call=is_call, event=event)

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
        if not return_sensitivities:
            phi = self.model.char_func(u.flatten(), T).reshape((N,))
            if event is not None and 0.0 < float(event.time) <= float(T):
                phi = phi * event.phi(u.flatten())
            re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))
        else:
            phi, dphi = self.model.char_func_and_grad(u.flatten(), T, params=sens_params, method=sens_method)
            phi = phi.reshape((N,))
            if event is not None and 0.0 < float(event.time) <= float(T):
                # event is parameter-independent
                phi = phi * event.phi(u.flatten())
                dphi = {k: v * event.phi(u.flatten()) for k, v in dphi.items()}
            phase = np.exp(-1j * u.flatten() * a)
            re_phi = np.real(phi.reshape((-1, 1)) * phase.reshape((-1, 1)))
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
        disc = float(np.exp(-self.model.r * T))
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        price = disc * mat_sum

        if not return_sensitivities:
            return price

        # dPrice/dp = disc * sum_k w_k * Re(dphi_k * exp(-i u_k a)) * Vk
        sens: dict[str, np.ndarray] = {}
        if sens_params is None:
            sens_params = self.model.param_names()
        for p in sens_params:
            if p not in dphi:
                continue
            dphi_p = dphi[p].reshape((N,))
            d_re_phi = np.real(dphi_p.reshape((-1, 1)) * phase.reshape((-1, 1)))
            sens[p] = disc * np.sum(weights * d_re_phi * Vk, axis=0)

        return price, sens

    def _european_price_lefloch(self, K: np.ndarray, T: float, *, is_call: bool, event: DiscreteEventJump | None) -> np.ndarray:
        """European pricing using Le Floc'h (2020) improved COS payoff coefficients.

        Reference
        ---------
        Fabien Le Floc'h, "More Robust Pricing of European Options Based on Fourier Cosine
        Series Expansions" (arXiv:2005.13248v2).

        Notes
        -----
        - Computes the put price with improved coefficients (Eq. (8)-(10)) and uses
          put-call parity for calls, as recommended in the paper.
        - Uses the log-moneyness variable y = ln(S_T / F) with F = E[S_T] and z = ln(K/F).
        """
        K = np.atleast_1d(K).astype(float)
        T = float(T)

        disc = float(np.exp(-float(self.model.r) * T))
        # Risk-neutral forward under internal proportional-dividend convention.
        F = float(_forward_price_from_prop_divs(self.model.S0, self.model.r, self.model.q, T, self.model.divs))
        if event is not None and 0.0 < float(event.time) <= float(T):
            F *= float(event.mean_factor if not event.ensure_martingale else 1.0)
        F = float(max(F, 1e-300))
        logF = float(np.log(F))

        # Truncation interval for y = ln(S_T/F): shift the ln(S_T) cumulant center by -ln(F).
        try:
            c1, c2, c4 = self.model.cumulants(T)
            if event is not None and 0.0 < float(event.time) <= float(T):
                e1, e2, e4 = event.log_cumulants()
                c1, c2, c4 = float(c1 + e1), float(c2 + e2), float(c4 + e4)
            c1 = float(c1) - logF
            c2 = float(max(c2, 0.0))
            c4 = float(max(c4, 0.0))
            width = float(self.L) * float(np.sqrt(c2 + np.sqrt(c4)))
            a = float(c1 - width)
            b = float(c1 + width)
        except Exception:
            # Fallback: variance-based symmetric interval for y
            sum_log, div_params = _dividend_adjustment(T, self.model.divs)
            var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
            var = float(self.model._var2(T) + var_div)
            if event is not None and 0.0 < float(event.time) <= float(T):
                var += float(event.log_cumulants()[1])
            mu_y = float(-0.5 * var)
            width = float(self.L) * float(np.sqrt(max(var, 1e-16)))
            a, b = float(mu_y - width), float(mu_y + width)

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval (lefloch): a={a}, b={b}")

        N = int(self.N)
        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)

        # Characteristic function for y = ln(S_T/F): phi_y(u) = E[e^{i u (ln S_T - ln F)}]
        phi = self.model.char_func(u.flatten(), T).reshape((N,))
        if event is not None and 0.0 < float(event.time) <= float(T):
            phi = phi * event.phi(u.flatten())
        phi = phi * np.exp(-1j * u.flatten() * logF)
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        # z = ln(K/F)
        z = np.log(np.clip(K, 1e-300, np.inf) / F).reshape((1, -1))
        exp_a = float(np.exp(a))
        exp_z = np.exp(z)
        z_minus_a = z - float(a)

        # Pre-fill Vk with zeros; handle out-of-range strikes per the paper.
        Vk = np.zeros((N, len(K)), dtype=float)

        # In-range mask (a <= z <= b)
        in_range = (z >= float(a)) & (z <= float(b))
        if np.any(in_range):
            # V0 (Eq. 8)
            V0 = (2.0 * F / (b - a)) * (exp_a - exp_z + exp_z * z_minus_a)

            # k=0 term
            Vk[0, :] = V0[0, :]

            # k>=1 terms (Eq. 9)
            if N > 1:
                u_nz = u[1:, :]  # (N-1,1)
                theta = u_nz * z_minus_a  # (N-1,M)
                sin_t = np.sin(theta)
                cos_t = np.cos(theta)
                denom = 1.0 + (u_nz ** 2)
                term1 = (exp_a - exp_z * (cos_t + u_nz * sin_t)) / denom
                term2 = exp_z * (sin_t / u_nz)
                Vk[1:, :] = (2.0 * F / (b - a)) * (term1 + term2)

            # Zero out out-of-range columns (we will override prices directly below)
            Vk = Vk * in_range.astype(float)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        put = disc * np.sum(weights * re_phi * Vk, axis=0)

        # Out-of-range handling (paper guidance)
        z_flat = z.flatten()
        low = z_flat < float(a)
        high = z_flat > float(b)
        if np.any(low):
            put[low] = 0.0
        if np.any(high):
            put[high] = disc * np.maximum(K[high] - F, 0.0)

        put = np.maximum(put, 0.0)

        if is_call:
            call = put + disc * (F - K)
            return np.maximum(call, 0.0)
        return put

    def _truncation_interval(self, T: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> tuple[float, float]:
        """Return COS truncation interval [a, b] for ln(S_T) given an optional spot override.

        For exponential-Levy style models, changing the conditioning spot shifts the first cumulant
        by ln(spot / S0) while higher cumulants remain (approximately) spot independent.
        """
        spot_eff = float(self.model.S0 if spot is None else spot)
        try:
            c1, c2, c4 = self.model.cumulants(T)
            c1 = float(c1) + float(np.log(spot_eff / float(self.model.S0)))
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
            # scale the forward by spot/S0 when conditioning spot differs
            F0 = _forward_price_from_prop_divs(self.model.S0, self.model.r, self.model.q, T, self.model.divs)
            F = float(F0) * float(spot_eff / float(self.model.S0))
            if event is not None and 0.0 < float(event.time) <= float(T):
                F *= float(event.mean_factor if not event.ensure_martingale else 1.0)
                var += float(event.log_cumulants()[1])
            mu = float(np.log(F) - 0.5 * var)
            width = self.L * np.sqrt(max(var, 1e-16))
            a, b = mu - width, mu + width

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval: a={a}, b={b}")
        return a, b

    def _phi_at_spot(self, u: np.ndarray, T: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Characteristic function of ln(S_T) for an optional conditioning spot.

        Uses increment characteristic function to avoid re-instantiating models:
        phi_S(u;T) = phi_inc(u;T) * exp(i u ln(S)).
        """
        spot_eff = float(self.model.S0 if spot is None else spot)
        u_flat = np.asarray(u, dtype=complex).flatten()
        phi = self.model.increment_char(u_flat, T) * np.exp(1j * u_flat * np.log(spot_eff))
        if event is not None and 0.0 < float(event.time) <= float(T):
            phi = phi * event.phi(u_flat)
        return phi

    def _truncation_interval_interval(self, t0: float, t1: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> tuple[float, float]:
        """Return COS truncation interval [a, b] for ln(S_{t1}) conditional on S_{t0}=spot."""
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        dt = t1 - t0
        spot_eff = float(self.model.S0 if spot is None else spot)

        try:
            c1, c2, c4 = self.model.cumulants(dt)
            # Correct for spot conditioning
            c1 = float(c1) + float(np.log(spot_eff / float(self.model.S0)))

            # Correct for discrete dividends in the interval (t0, t1]
            if getattr(self.model, "divs", None):
                old_sum_log, old_params = _dividend_adjustment(dt, self.model.divs)
                old_var = float(np.sum(old_params[:, 1])) if old_params.size else 0.0
                new_sum_log, new_params = _dividend_adjustment_window(t0, t1, self.model.divs)
                new_var = float(np.sum(new_params[:, 1])) if new_params.size else 0.0
                c1 = float(c1 + (new_sum_log - old_sum_log))
                c2 = float(c2 + (new_var - old_var))

            if event is not None and t0 < float(event.time) <= t1:
                e1, e2, e4 = event.log_cumulants()
                c1, c2, c4 = float(c1 + e1), float(c2 + e2), float(c4 + e4)
            c2 = float(max(c2, 0.0))
            c4 = float(max(c4, 0.0))
            width = self.L * np.sqrt(c2 + np.sqrt(c4))
            a = float(c1 - width)
            b = float(c1 + width)
        except Exception:
            # Fallback: variance-based symmetric interval
            var = float(self.model._var2(dt))
            sum_log, params = _dividend_adjustment_window(t0, t1, getattr(self.model, "divs", {}) or {})
            var_div = float(np.sum(params[:, 1])) if params.size else 0.0
            var = float(var + var_div)
            F = spot_eff * float(np.exp((self.model.r - self.model.q) * dt + sum_log + 0.5 * var_div))
            if event is not None and t0 < float(event.time) <= t1:
                F *= float(event.mean_factor if not event.ensure_martingale else 1.0)
                var += float(event.log_cumulants()[1])
            mu = float(np.log(max(F, 1e-300)) - 0.5 * var)
            width = self.L * np.sqrt(max(var, 1e-16))
            a, b = mu - width, mu + width

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval: a={a}, b={b}")
        return a, b

    def _phi_interval_at_spot(self, u: np.ndarray, t0: float, t1: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Characteristic function of ln(S_{t1}) conditional on S_{t0}=spot."""
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        spot_eff = float(self.model.S0 if spot is None else spot)
        u_flat = np.asarray(u, dtype=complex).flatten()
        phi_inc = self.model.increment_char_interval(u_flat, t0, t1)
        phi = phi_inc * np.exp(1j * u_flat * np.log(spot_eff))
        if event is not None and t0 < float(event.time) <= t1:
            phi = phi * event.phi(u_flat)
        return phi

    def _truncation_interval_increment_interval(self, t0: float, t1: float, event: DiscreteEventJump | None = None) -> tuple[float, float]:
        """Return COS truncation interval [a,b] for the log-return increment Y = ln(S_{t1}/S_{t0}).

        This is spot-independent and is used to cheaply re-evaluate digitals/AONs for many spots.
        """
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        dt = float(t1 - t0)

        try:
            c1_abs, c2, c4 = self.model.cumulants(dt)
            # Convert absolute ln(S) cumulant into increment cumulant by removing ln(S0).
            c1 = float(c1_abs) - float(np.log(float(self.model.S0)))

            # Correct for discrete dividends in the interval (t0, t1].
            if getattr(self.model, "divs", None):
                old_sum_log, old_params = _dividend_adjustment(dt, self.model.divs)
                old_var = float(np.sum(old_params[:, 1])) if old_params.size else 0.0
                new_sum_log, new_params = _dividend_adjustment_window(t0, t1, self.model.divs)
                new_var = float(np.sum(new_params[:, 1])) if new_params.size else 0.0
                c1 = float(c1 + (new_sum_log - old_sum_log))
                c2 = float(c2 + (new_var - old_var))

            if event is not None and t0 < float(event.time) <= t1:
                e1, e2, e4 = event.log_cumulants()
                c1, c2, c4 = float(c1 + e1), float(c2 + e2), float(c4 + e4)

            c2 = float(max(c2, 0.0))
            c4 = float(max(c4, 0.0))
            width = float(self.L) * float(np.sqrt(c2 + np.sqrt(c4)))
            a = float(c1 - width)
            b = float(c1 + width)
        except Exception:
            # Fallback: variance-based symmetric interval for increment
            var = float(self.model._var2(dt))
            sum_log, params = _dividend_adjustment_window(t0, t1, getattr(self.model, "divs", {}) or {})
            var_div = float(np.sum(params[:, 1])) if params.size else 0.0
            var = float(var + var_div)

            # Increment forward: E[S_{t1}/S_{t0}] under our proportional-div convention.
            F_inc = float(np.exp((self.model.r - self.model.q) * dt + sum_log + 0.5 * var_div))
            if event is not None and t0 < float(event.time) <= t1:
                F_inc *= float(event.mean_factor if not event.ensure_martingale else 1.0)
                var += float(event.log_cumulants()[1])

            mu = float(np.log(max(F_inc, 1e-300)) - 0.5 * var)
            width = float(self.L) * float(np.sqrt(max(var, 1e-16)))
            a, b = float(mu - width), float(mu + width)

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval: a={a}, b={b}")
        return a, b

    @dataclass(frozen=True)
    class _IncrementIntervalBasis:
        a: float
        b: float
        dt: float
        w_re_phi: np.ndarray  # shape (N,)
        disc: float

        def digital(self, pricer: "COSPricer", barrier: float, spot: float) -> float:
            c_rel = float(np.log(max(float(barrier), 1e-300) / max(float(spot), 1e-300)))
            Vk = pricer._cos_coeff_indicator_left(self.a, self.b, np.array([c_rel]), pricer.N)
            return float(self.disc * float(np.dot(self.w_re_phi, Vk[:, 0])))

        def aon(self, pricer: "COSPricer", barrier: float, spot: float) -> float:
            c_rel = float(np.log(max(float(barrier), 1e-300) / max(float(spot), 1e-300)))
            Vk = pricer._cos_coeff_exp_indicator_left(self.a, self.b, np.array([c_rel]), pricer.N)
            base = float(self.disc * float(np.dot(self.w_re_phi, Vk[:, 0])))
            return float(max(float(spot), 1e-300) * base)

    def _make_increment_interval_basis(self, t0: float, t1: float, *, event: DiscreteEventJump | None = None) -> "COSPricer._IncrementIntervalBasis":
        """Precompute spot-independent COS basis for increment payoffs over (t0,t1]."""
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        dt = float(t1 - t0)

        a, b = self._truncation_interval_increment_interval(t0, t1, event=event)
        N = int(self.N)
        k = np.arange(N).reshape((-1, 1))
        u = (k * np.pi / (b - a)).flatten()  # (N,)

        phi_inc = self.model.increment_char_interval(u.astype(complex), t0, t1)
        if event is not None and t0 < float(event.time) <= t1:
            phi_inc = phi_inc * event.phi(u)

        re_phi = np.real(phi_inc * np.exp(-1j * u * a))
        weights = np.ones(N, dtype=float)
        weights[0] = 0.5
        w_re_phi = weights * re_phi
        disc = float(np.exp(-float(self.model.r) * dt))
        return COSPricer._IncrementIntervalBasis(a=float(a), b=float(b), dt=float(dt), w_re_phi=w_re_phi, disc=disc)

    def digital_put_price_interval(self, B: np.ndarray, t0: float, t1: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Time-t0 price of a digital put paying 1{S_{t1} < B}."""
        B = np.atleast_1d(B).astype(float)
        a, b = self._truncation_interval_interval(t0, t1, spot=spot, event=event)
        N = self.N
        dt = float(t1) - float(t0)

        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)
        phi = self._phi_interval_at_spot(u.flatten(), t0, t1, spot=spot, event=event).reshape((N,))
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        c = np.log(np.clip(B, 1e-300, np.inf)).reshape((1, -1))
        Vk = self._cos_coeff_indicator_left(a, b, c, N)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * dt) * mat_sum

    def asset_or_nothing_put_price_interval(self, B: np.ndarray, t0: float, t1: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Time-t0 price of an asset-or-nothing put paying S_{t1} * 1{S_{t1} < B}."""
        B = np.atleast_1d(B).astype(float)
        a, b = self._truncation_interval_interval(t0, t1, spot=spot, event=event)
        N = self.N
        dt = float(t1) - float(t0)

        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)
        phi = self._phi_interval_at_spot(u.flatten(), t0, t1, spot=spot, event=event).reshape((N,))
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        c = np.log(np.clip(B, 1e-300, np.inf)).reshape((1, -1))
        Vk = self._cos_coeff_exp_indicator_left(a, b, c, N)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * dt) * mat_sum

    def european_price_interval(
        self,
        K: np.ndarray,
        t0: float,
        t1: float,
        *,
        is_call: bool = True,
        spot: float | None = None,
        event: DiscreteEventJump | None = None,
        payoff_coeffs: str = "classic",
    ) -> np.ndarray:
        """Time-t0 European option price with maturity t1.

        Dividend-aware interval version of `european_price`.
        Vectorized over strikes K.
        """
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        dt = float(t1 - t0)

        K = np.atleast_1d(K).astype(float)

        payoff_coeffs = str(payoff_coeffs).lower().strip()
        if payoff_coeffs not in {"classic", "lefloch"}:
            raise ValueError("payoff_coeffs must be one of: 'classic', 'lefloch'")
        if payoff_coeffs == "lefloch":
            return self._european_price_interval_lefloch(K, t0, t1, is_call=is_call, spot=spot, event=event)

        a, b = self._truncation_interval_interval(t0, t1, spot=spot, event=event)

        N = self.N
        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)
        phi = self._phi_interval_at_spot(u.flatten(), t0, t1, spot=spot, event=event).reshape((N,))
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        logK = np.clip(np.log(np.clip(K, 1e-300, np.inf)), a, b).reshape((1, -1))
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

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * dt) * mat_sum

    def _european_price_interval_lefloch(
        self,
        K: np.ndarray,
        t0: float,
        t1: float,
        *,
        is_call: bool,
        spot: float | None,
        event: DiscreteEventJump | None,
    ) -> np.ndarray:
        """Interval version of Le Floc'h (2020) improved COS payoff coefficients."""
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        dt = float(t1 - t0)

        K = np.atleast_1d(K).astype(float)
        disc = float(np.exp(-float(self.model.r) * dt))
        spot_eff = float(self.model.S0 if spot is None else spot)

        # Conditional forward F = E[S_{t1} | S_{t0}=spot]
        sum_log, params = _dividend_adjustment_window(t0, t1, getattr(self.model, "divs", {}) or {})
        var_div = float(np.sum(params[:, 1])) if params.size else 0.0
        F = float(spot_eff * np.exp((float(self.model.r) - float(self.model.q)) * dt + sum_log + 0.5 * var_div))
        if event is not None and t0 < float(event.time) <= t1:
            F *= float(event.mean_factor if not event.ensure_martingale else 1.0)
        F = float(max(F, 1e-300))
        logF = float(np.log(F))

        # Start from ln(S_{t1}) truncation and shift to y = ln(S_{t1}/F).
        a_x, b_x = self._truncation_interval_interval(t0, t1, spot=spot, event=event)
        a = float(a_x - logF)
        b = float(b_x - logF)

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError(f"Invalid COS truncation interval (lefloch interval): a={a}, b={b}")

        N = int(self.N)
        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)

        # CF of y via conditional CF of ln(S_{t1})
        phi_x = self._phi_interval_at_spot(u.flatten(), t0, t1, spot=spot, event=event).reshape((N,))
        phi = phi_x * np.exp(-1j * u.flatten() * logF)
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        z = np.log(np.clip(K, 1e-300, np.inf) / F).reshape((1, -1))
        exp_a = float(np.exp(a))
        exp_z = np.exp(z)
        z_minus_a = z - float(a)

        Vk = np.zeros((N, len(K)), dtype=float)
        in_range = (z >= float(a)) & (z <= float(b))
        if np.any(in_range):
            V0 = (2.0 * F / (b - a)) * (exp_a - exp_z + exp_z * z_minus_a)
            Vk[0, :] = V0[0, :]
            if N > 1:
                u_nz = u[1:, :]
                theta = u_nz * z_minus_a
                sin_t = np.sin(theta)
                cos_t = np.cos(theta)
                denom = 1.0 + (u_nz ** 2)
                term1 = (exp_a - exp_z * (cos_t + u_nz * sin_t)) / denom
                term2 = exp_z * (sin_t / u_nz)
                Vk[1:, :] = (2.0 * F / (b - a)) * (term1 + term2)
            Vk = Vk * in_range.astype(float)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        put = disc * np.sum(weights * re_phi * Vk, axis=0)

        z_flat = z.flatten()
        low = z_flat < float(a)
        high = z_flat > float(b)
        if np.any(low):
            put[low] = 0.0
        if np.any(high):
            put[high] = disc * np.maximum(K[high] - F, 0.0)

        put = np.maximum(put, 0.0)

        if is_call:
            call = put + disc * (F - K)
            return np.maximum(call, 0.0)
        return put

    def american_put_price_eep_from_boundary(
        self,
        K: float,
        T: float,
        t_grid: np.ndarray,
        boundary: np.ndarray,
        *,
        t0: float = 0.0,
        spot: float | None = None,
        event: DiscreteEventJump | None = None,
    ) -> tuple[float, float, float]:
        """American put via Andersen Eq. 11 using a supplied exercise boundary.

        Returns (american, european, eep) at time t0 for spot S(t0)=spot.

        Notes
        -----
        - This does not *solve* for the boundary; it evaluates Eq. 11 given (t_grid, boundary).
        - Uses trapezoid integration over u.
        """
        K = float(K)
        t0 = float(t0)
        T = float(T)
        if not (0.0 <= t0 < T):
            raise ValueError("Require 0 <= t0 < T")

        t_grid = np.asarray(t_grid, dtype=float).reshape((-1,))
        boundary = np.asarray(boundary, dtype=float).reshape((-1,))
        if t_grid.shape != boundary.shape:
            raise ValueError("t_grid and boundary must have the same shape")
        if t_grid.size < 2:
            raise ValueError("t_grid must have at least two points")
        if np.any(np.diff(t_grid) <= 0.0):
            raise ValueError("t_grid must be strictly increasing")
        if float(t_grid[0]) < t0 - 1e-15 or float(t_grid[-1]) > T + 1e-15:
            raise ValueError("t_grid must cover [t0, T]")

        # Restrict to integration nodes in [t0, T]
        mask = (t_grid >= t0) & (t_grid <= T)
        u_nodes = t_grid[mask]
        b_nodes = boundary[mask]
        if u_nodes.size < 2:
            raise ValueError("Need at least two integration nodes in [t0, T]")

        # European put at time t0
        if t0 == 0.0:
            euro = float(self.european_price(np.array([K]), T, is_call=False, event=event)[0])
        else:
            euro = float(self.european_price_interval(np.array([K]), t0, T, is_call=False, spot=spot, event=event)[0])

        # EEP integrands at u_nodes: prices at time t0 of (digital put, AON put) with strike boundary(u).
        # At u=t0 the interval length is 0, and the strict indicator {S(u)<B(u)} gives 0 at S=B,
        # so we set the first integrand point to 0 explicitly and start from the next node.
        dig = np.zeros_like(u_nodes, dtype=float)
        aon = np.zeros_like(u_nodes, dtype=float)
        for k in range(1, int(u_nodes.size)):
            u = float(u_nodes[k])
            b = float(b_nodes[k])
            dig[k] = float(self.digital_put_price_interval(np.array([b]), t0, u, spot=spot, event=event)[0])
            aon[k] = float(self.asset_or_nothing_put_price_interval(np.array([b]), t0, u, spot=spot, event=event)[0])

        r = float(self.model.r)
        q = float(self.model.q)
        integrand = (r * K) * dig - q * aon
        eep = float(np.trapezoid(integrand, u_nodes))
        amer = float(euro + eep)
        return amer, euro, eep

    def solve_american_put_boundary_eep(
        self,
        K: float,
        T: float,
        *,
        steps: int = 80,
        t_grid: np.ndarray | None = None,
        spot_eps: float = 1e-10,
        bisect_tol: float = 1e-8,
        max_bisect_iter: int = 80,
        use_andersen_dividend_upper_bound: bool = True,
        enforce_prediv_zero: bool = True,
        prediv_epsilon: float = 1e-8,
        dividend_time_tol: float = 1e-12,
        enforce_monotone: bool = True,
        event: DiscreteEventJump | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the American put exercise boundary via Andersen Eq. 11 + value matching.

        Computes a discrete boundary curve B(t_i) on a time grid using backward bisection.
        EEP integrals are evaluated with interval digital/AON COS primitives (dividend-aware).
        """
        K = float(K)
        T = float(T)
        if not (T > 0.0):
            raise ValueError("T must be > 0")

        if t_grid is None:
            if steps <= 0:
                base = np.array([0.0, T], dtype=float)
            else:
                base = np.linspace(0.0, T, int(steps) + 1, dtype=float)
            div_times = []
            if getattr(self.model, "divs", None):
                div_times = [float(t) for t in self.model.divs.keys() if 0.0 < float(t) < T]
            prediv_times = []
            if enforce_prediv_zero and div_times:
                eps = float(prediv_epsilon)
                for td in div_times:
                    tpd = float(td) - eps
                    if 0.0 < tpd < T:
                        prediv_times.append(tpd)
            evt_times = []
            if event is not None and 0.0 < float(event.time) < T:
                evt_times = [float(event.time)]
            t_grid = np.array(sorted(set(list(base) + div_times + prediv_times + evt_times + [0.0, T])), dtype=float)
        else:
            t_grid = np.asarray(t_grid, dtype=float).reshape((-1,))
            # If user provided a grid that includes dividend times, optionally insert pre-div nodes.
            if enforce_prediv_zero and getattr(self.model, "divs", None):
                eps = float(prediv_epsilon)
                extra = []
                for td in self.model.divs.keys():
                    td = float(td)
                    tpd = td - eps
                    if 0.0 < tpd < T:
                        extra.append(tpd)
                if extra:
                    t_grid = np.array(sorted(set(list(t_grid) + extra)), dtype=float)

        if t_grid.size < 2:
            raise ValueError("t_grid must have at least two points")
        if abs(float(t_grid[0])) > 1e-14 or abs(float(t_grid[-1]) - T) > 1e-12:
            raise ValueError("t_grid must start at 0 and end at T")
        if np.any(np.diff(t_grid) <= 0.0):
            raise ValueError("t_grid must be strictly increasing")

        dividend_times = set()
        if getattr(self.model, "divs", None):
            dividend_times = {float(t) for t in self.model.divs.keys()}

        has_discrete_divs = bool(dividend_times)

        prediv_times = set()
        if enforce_prediv_zero and dividend_times:
            eps = float(prediv_epsilon)
            for td in dividend_times:
                tpd = float(td) - eps
                if 0.0 < tpd < T:
                    prediv_times.add(tpd)

        r = float(self.model.r)
        q = float(self.model.q)
        mu = r - q
        n = int(t_grid.size)
        B = np.empty(n, dtype=float)
        B[-1] = K

        # Andersen (with dividends) Lemma 2, Eq. (10): an analytical upper bound for the boundary
        # in each dividend interval (t_{k-1}, t_k). This does not change the model, it only tightens
        # the bisection bracket / initial guess.
        def _andersen_upper_bound_before_next_div(t: float) -> float | None:
            if (not use_andersen_dividend_upper_bound) or (not getattr(self.model, "divs", None)):
                return None

            # Next dividend strictly after t.
            next_div = None
            for td in sorted(dividend_times):
                if float(td) > float(t) + float(dividend_time_tol):
                    next_div = float(td)
                    break
            if next_div is None:
                return None

            # Mean proportional drop c_i in Andersen notation (our m_mean).
            c_i = float(self.model.divs.get(next_div, (0.0, 0.0))[0])
            c_i = float(np.clip(c_i, 0.0, 1.0 - 1e-15))
            if c_i <= 0.0:
                return None

            # Previous dividend time (or 0.0) defining interval start.
            prev_div = 0.0
            for td in sorted(dividend_times):
                if float(td) < next_div - float(dividend_time_tol):
                    prev_div = float(td)
                else:
                    break

            # Only applies in the open interval before the dividend.
            if not (prev_div - 1e-15 <= t < next_div - float(dividend_time_tol)):
                return None

            if float(mu) > 0.0:
                # t_i* = max(t_i + ln(1-c_i)/mu, t_{i-1})
                t_star = max(next_div + float(np.log(1.0 - c_i)) / float(mu), prev_div)
            else:
                t_star = prev_div

            # For t in [t_{i-1}, t_i*], the bound is simply K.
            if float(t) <= float(t_star) + float(dividend_time_tol):
                return float(K)

            # For t in (t_i*, t_i), Eq. (10) gives a tighter upper bound.
            dt = float(next_div - t)
            if dt <= 0.0:
                return float(K)

            num = 1.0 - float(np.exp(-r * dt))
            denom = 1.0 - float(np.exp(-(mu - r) * dt)) * (1.0 - c_i)
            if denom <= 0.0:
                return float(K)
            ub = float(K) * (num / denom)
            # Always cap to [0, K].
            return float(np.clip(ub, 0.0, float(K)))

        def _in_time_set(t: float, times: set[float]) -> bool:
            if not times:
                return False
            for td in times:
                if abs(float(t) - float(td)) <= float(dividend_time_tol):
                    return True
            return False

        for i in range(n - 2, -1, -1):
            ti = float(t_grid[i])

            if enforce_prediv_zero and _in_time_set(ti, prediv_times):
                B[i] = 0.0
                continue

            # With discrete dividends (and especially when enforcing pre-div boundary=0), the boundary can be non-monotone.
            enforce_monotone_step = bool(enforce_monotone) and (not has_discrete_divs)
            if enforce_monotone_step:
                if _in_time_set(float(t_grid[i]), dividend_times) or _in_time_set(float(t_grid[i + 1]), dividend_times):
                    enforce_monotone_step = False
                if _in_time_set(float(t_grid[i]), prediv_times) or _in_time_set(float(t_grid[i + 1]), prediv_times):
                    enforce_monotone_step = False

            hi = float(min(K, B[i + 1])) if enforce_monotone_step else float(K)
            if has_discrete_divs:
                ub = _andersen_upper_bound_before_next_div(ti)
                if ub is not None:
                    hi = float(min(hi, ub))
            lo = float(min(hi, max(float(spot_eps), 1e-300)))

            # Cache spot-independent COS pieces for all future integration nodes.
            # This is the dominant speed win: bisection iterations only recompute Vk(c_rel) per node.
            future_u = np.asarray(t_grid[i + 1 :], dtype=float)
            future_B = np.asarray(B[i + 1 :], dtype=float)
            future_bases = [self._make_increment_interval_basis(ti, float(uj), event=event) for uj in future_u]

            def residual(s: float) -> float:
                spot = float(max(s, 1e-300))
                euro = float(self.european_price_interval(np.array([K]), ti, T, is_call=False, spot=spot, event=event)[0])

                fj = np.empty_like(future_u, dtype=float)
                for idx in range(int(future_u.size)):
                    dig = future_bases[idx].digital(self, float(future_B[idx]), spot)
                    aon = future_bases[idx].aon(self, float(future_B[idx]), spot)
                    fj[idx] = (r * K) * dig - q * aon

                # Trapezoid over [ti] U future_u with f(ti)=0 by convention.
                u_nodes = np.concatenate(([ti], future_u))
                f_nodes = np.concatenate(([0.0], fj))
                eep = float(np.trapezoid(f_nodes, u_nodes))
                amer = float(euro + eep)
                return (K - spot) - amer

            flo = residual(lo)
            fhi = residual(hi)

            if flo == 0.0:
                root = lo
            elif fhi == 0.0:
                root = hi
            elif flo * fhi > 0.0:
                grid = np.linspace(lo, hi, 25)
                fgrid = [residual(float(x)) for x in grid]
                bracketed = False
                for a0, b0, fa0, fb0 in zip(grid[:-1], grid[1:], fgrid[:-1], fgrid[1:]):
                    if fa0 == 0.0:
                        lo, hi = float(a0), float(a0)
                        bracketed = True
                        break
                    if fa0 * fb0 < 0.0:
                        lo, hi = float(a0), float(b0)
                        flo, fhi = float(fa0), float(fb0)
                        bracketed = True
                        break
                if not bracketed:
                    raise RuntimeError(f"Could not bracket boundary root at t={ti}: f(lo)={flo}, f(hi)={fhi}")

            if lo == hi:
                root = lo
            else:
                a = lo
                b = hi
                fa = flo
                fb = fhi
                root = 0.5 * (a + b)
                for _ in range(int(max_bisect_iter)):
                    root = 0.5 * (a + b)
                    fm = residual(root)
                    if abs(fm) <= float(bisect_tol) or (b - a) <= float(bisect_tol) * max(1.0, abs(root)):
                        break
                    if fa * fm <= 0.0:
                        b = root
                        fb = fm
                    else:
                        a = root
                        fa = fm

            B[i] = float(min(root, hi))
            if enforce_monotone_step:
                B[i] = float(min(B[i], B[i + 1]))

        return t_grid, B

    @staticmethod
    def _cos_coeff_indicator_left(a: float, b: float, c: np.ndarray, N: int) -> np.ndarray:
        """COS coefficients for f(x)=1_{x < c} on [a,b].

        Returns Vk with shape (N, nC), matching the convention used in `european_price`.
        """
        c = np.asarray(c, dtype=float).reshape((1, -1))
        k = np.arange(N, dtype=float).reshape((-1, 1))
        u = k * np.pi / (b - a)
        # normalize c into [a,b]
        c = np.clip(c, a, b)
        d = (c - a) / (b - a)
        Vk = np.zeros((N, c.shape[1]), dtype=float)
        Vk[0, :] = 2.0 * (c - a) / (b - a)
        non0 = (k.flatten() != 0)
        if np.any(non0):
            kk = k[non0]
            Vk[non0, :] = 2.0 * np.sin(kk * np.pi * d) / (kk * np.pi)
        return Vk

    @staticmethod
    def _cos_coeff_exp_indicator_left(a: float, b: float, c: np.ndarray, N: int) -> np.ndarray:
        """COS coefficients for f(x)=exp(x) * 1_{x < c} on [a,b].

        This reuses the same integral structure as the vanilla payoff `chi` term.
        Returns Vk with shape (N, nC).
        """
        c = np.asarray(c, dtype=float).reshape((1, -1))
        c = np.clip(c, a, b)

        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)

        # Integral of exp(x)*cos(u*(x-a)) from x=a to x=c
        theta_c = u * (c - a)
        sin_c = np.sin(theta_c)
        cos_c = np.cos(theta_c)

        Vk = np.zeros((N, c.shape[1]), dtype=float)
        k0 = (k.flatten() == 0)
        Vk[k0, :] = (2.0 / (b - a)) * (np.exp(c) - np.exp(a))

        non0 = ~k0
        if np.any(non0):
            u_nz = u[non0]
            denom = 1.0 + (u_nz ** 2)
            # upper bound contribution minus lower bound at a (theta=0, sin=0, cos=1)
            integ = (np.exp(c) * (cos_c[non0, :] + u_nz * sin_c[non0, :]) - np.exp(a) * (1.0 + 0.0 * u_nz)) / denom
            Vk[non0, :] = (2.0 / (b - a)) * np.real(integ)
        return Vk

    def digital_put_price(self, B: np.ndarray, T: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Price a European digital put paying 1{S_T < B}.

        Vectorized over barriers/strikes B.
        """
        B = np.atleast_1d(B).astype(float)
        a, b = self._truncation_interval(T, spot=spot, event=event)
        N = self.N

        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)
        phi = self._phi_at_spot(u.flatten(), T, spot=spot, event=event).reshape((N,))
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        c = np.log(np.clip(B, 1e-300, np.inf)).reshape((1, -1))
        Vk = self._cos_coeff_indicator_left(a, b, c, N)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * T) * mat_sum

    def asset_or_nothing_put_price(self, B: np.ndarray, T: float, spot: float | None = None, event: DiscreteEventJump | None = None) -> np.ndarray:
        """Price an asset-or-nothing put paying S_T * 1{S_T < B}.

        Vectorized over barriers/strikes B.
        """
        B = np.atleast_1d(B).astype(float)
        a, b = self._truncation_interval(T, spot=spot, event=event)
        N = self.N

        k = np.arange(N).reshape((-1, 1))
        u = k * np.pi / (b - a)  # (N,1)
        phi = self._phi_at_spot(u.flatten(), T, spot=spot, event=event).reshape((N,))
        re_phi = np.real(phi.reshape((-1, 1)) * np.exp(-1j * u * a))

        c = np.log(np.clip(B, 1e-300, np.inf)).reshape((1, -1))
        Vk = self._cos_coeff_exp_indicator_left(a, b, c, N)

        weights = np.ones((N, 1))
        weights[0, 0] = 0.5
        mat_sum = np.sum(weights * re_phi * Vk, axis=0)
        return np.exp(-self.model.r * T) * mat_sum

    def american_price(
        self,
        K: np.ndarray,
        T: float,
        steps: int = 50,
        is_call: bool | np.ndarray = True,
        beta: float = 20.0,
        use_softmax: bool = False,
        return_european: bool = False,
        return_trajectory: bool = False,
        return_continuation_trajectory: bool = False,
        return_boundary: bool = False,
        rollback_debug: bool = False,
        diagnostics_file: str = None,
        event: DiscreteEventJump | None = None,
        return_sensitivities: bool = False,
        sens_method: str = "analytic",
        sens_params: list[str] | None = None,
        *,
        return_grid_snapshot: bool = False,
        snapshot_time: float | None = None,
    ):
        """COS-based backward induction for American options.

        - Maintain coefficients across steps.
        - Reconstruct continuation in price space, compare with intrinsic.
        - For calls with q > 0, we use a parity-based rollback to improve stability,
          but for general robustness (especially with fat tails), pricing the dual put is preferred.
        """
        K = np.atleast_1d(K).astype(float)

        if return_sensitivities and (return_european or return_trajectory or return_continuation_trajectory):
            raise ValueError(
                "return_sensitivities cannot be combined with return_european/trajectory outputs yet"
            )

        # Mixed put/call vector support: price puts and calls separately, reusing cached context.
        is_call_arr = np.asarray(is_call)
        if is_call_arr.ndim > 0 and is_call_arr.size not in (0, 1):
            if is_call_arr.shape[0] != K.shape[0]:
                raise ValueError("If is_call is an array, it must match K shape")
            if return_trajectory or return_continuation_trajectory or return_european or return_boundary:
                raise ValueError("Mixed is_call with trajectory/european outputs is not supported")
            out = np.empty_like(K, dtype=float)
            call_mask = is_call_arr.astype(bool)
            put_mask = ~call_mask

            if return_sensitivities:
                sens_params_eff = sens_params if sens_params is not None else self.model.param_names()
                d_out: dict[str, np.ndarray] = {p: np.empty_like(out, dtype=float) for p in sens_params_eff}

                if np.any(put_mask):
                    prices_put, d_put = self.american_price(
                        K[put_mask],
                        T,
                        steps=steps,
                        is_call=False,
                        beta=beta,
                        use_softmax=use_softmax,
                        event=event,
                        return_sensitivities=True,
                        sens_method=sens_method,
                        sens_params=sens_params_eff,
                    )
                    out[put_mask] = np.asarray(prices_put, dtype=float)
                    for p in sens_params_eff:
                        d_out[p][put_mask] = np.asarray(d_put[p], dtype=float)

                if np.any(call_mask):
                    prices_call, d_call = self.american_price(
                        K[call_mask],
                        T,
                        steps=steps,
                        is_call=True,
                        beta=beta,
                        use_softmax=use_softmax,
                        event=event,
                        return_sensitivities=True,
                        sens_method=sens_method,
                        sens_params=sens_params_eff,
                    )
                    out[call_mask] = np.asarray(prices_call, dtype=float)
                    for p in sens_params_eff:
                        d_out[p][call_mask] = np.asarray(d_call[p], dtype=float)

                return out, d_out

            if np.any(put_mask):
                out[put_mask] = np.asarray(
                    self.american_price(
                        K[put_mask],
                        T,
                        steps=steps,
                        is_call=False,
                        beta=beta,
                        use_softmax=use_softmax,
                        event=event,
                    ),
                    dtype=float,
                )
            if np.any(call_mask):
                out[call_mask] = np.asarray(
                    self.american_price(
                        K[call_mask],
                        T,
                        steps=steps,
                        is_call=True,
                        beta=beta,
                        use_softmax=use_softmax,
                        event=event,
                    ),
                    dtype=float,
                )
            return out

        # Scalar is_call
        is_call = bool(is_call)
        N = self.N
        M = self.M
        prices = np.zeros_like(K)
        euro_prices = None
        if return_european:
            euro_prices = self.european_price(K, T, is_call=is_call, event=event)

        trajectory_cache = [] if return_trajectory else None
        cont_trajectory_cache = [] if return_continuation_trajectory else None
        boundary_nodes: list[tuple[float, np.ndarray]] | None = [] if return_boundary else None

        # For calls with no discrete dividends and non-positive continuous yield,
        # early exercise is not optimal in standard models, so American == European.
        # The numerical rollback can otherwise introduce small max-operator artifacts.
        if is_call and (not self.model.divs) and float(self.model.q) <= 0.0 and not (return_trajectory or return_continuation_trajectory):
            if return_sensitivities:
                eu, d_eu = self.european_price(
                    K,
                    T,
                    is_call=True,
                    event=event,
                    return_sensitivities=True,
                    sens_method=sens_method,
                    sens_params=sens_params,
                )
                return eu, d_eu

            eu = self.european_price(K, T, is_call=True, event=event)
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
            and K.size <= 8
            and not return_sensitivities
        ):
            # C(S0, K, r, q, T) = P(K, S0, q, r, T) under dual measure
            # We price a put with strike = S0, spot = K, r_dual = q, q_dual = r
            dual_prices = np.zeros_like(K)
            for j, Kval in enumerate(K):
                dual_model = DualModel(self.model, dual_S0=Kval)
                dual_pricer = COSPricer(dual_model, N=self.N, L=self.L, M=self.M)
                # Price a put with strike = self.model.S0
                p = dual_pricer.american_price(
                    np.array([self.model.S0]),
                    T,
                    steps=steps,
                    is_call=False,
                    beta=beta,
                    use_softmax=use_softmax,
                )
                dual_prices[j] = p[0]

            if return_european:
                return dual_prices, euro_prices
            return dual_prices

        # With discrete dividends, parity-based call rollback is fragile and can create
        # discontinuities near ex-div dates (especially when the horizon changes and
        # a dividend moves close to t=0). In that case, price calls directly.
        direct_call_with_divs = bool(is_call) and bool(self.model.divs)

        ctx = self._get_or_build_american_context(float(T), steps=int(steps), event=event)
        a = float(ctx["a"])
        b = float(ctx["b"])
        x_grid = ctx["x_grid"]
        S_grid = ctx["S_grid"]
        dx = float(ctx["dx"])
        w = ctx["w"]
        cos_k_x = ctx["cos_k_x"]
        sin_k_x = ctx["sin_k_x"]
        u = ctx["u"]
        t_steps = ctx["t_steps"]
        div_tol = float(ctx["div_tol"])
        evt_tol = float(ctx["evt_tol"])
        evt_time = ctx["evt_time"]
        div_shift_ops_by_time = ctx["div_shift_ops_by_time"]
        evt_ops = ctx["evt_ops"]
        step_info = ctx["step_info"]
        sum_log_total = float(ctx["sum_log_total"])
        sum_log_up_to = ctx["sum_log_up_to"]
        ix0 = int(ctx["ix0"])
        wgt0 = float(ctx["wgt0"])

        # Optional diagnostics snapshot (for inspecting the value function on x_grid).
        grid_snapshot = None
        snapshot_time_val = None if snapshot_time is None else float(snapshot_time)
        # Tolerance tied to the nominal step, but always at least 1e-12.
        snapshot_tol = float(max(1e-12, 1e-10 * float(ctx.get("dt_nominal", 1.0))))

        # Terminal payoff on grid for all strikes (shape: (nK, M))
        Kcol = K.reshape((-1, 1))
        if direct_call_with_divs:
            V_grid = np.maximum(S_grid.reshape((1, -1)) - Kcol, 0.0)
        else:
            # Work in put-space for stability; parity conversions handle call output.
            V_grid = np.maximum(Kcol - S_grid.reshape((1, -1)), 0.0)

        # Initial projection to cosine coefficients (shape: (N, nK))
        coeffs = (2.0 / (b - a)) * (cos_k_x @ (V_grid * w).T) * dx

        sens_params_eff: list[str] = []
        if return_sensitivities:
            sens_params_eff = list(self.model.param_names() if sens_params is None else sens_params)
            # Derivative of terminal payoff w.r.t model params is zero (frozen grid assumption)
            d_coeffs: dict[str, np.ndarray] = {p: np.zeros_like(coeffs, dtype=float) for p in sens_params_eff}

        # Stack bases once to allow a single matmul per step:
        # cont = disc * [term_cos; term_sin]^T @ [cos_k_x; sin_k_x]
        # This avoids scaling the large (N,M) basis matrices each step.
        cos_sin_stack = np.vstack([cos_k_x, sin_k_x])  # (2N, M)
        term_stack = np.empty((2 * N, K.shape[0]), dtype=float)  # (2N, nK)

        for dt, t_current, re_phi, im_phi, disc_dt in step_info:
            # Gradients of increment CF for this step (if requested)
            if return_sensitivities:
                u_flat = u.astype(complex, copy=False)
                phi_dt_c = re_phi.astype(float, copy=False) + 1j * im_phi.astype(float, copy=False)
                _, dphi_dt = self.model.increment_char_and_grad(u_flat, float(dt), params=sens_params_eff, method=sens_method)
                d_re_phi: dict[str, np.ndarray] = {}
                d_im_phi: dict[str, np.ndarray] = {}
                for p in sens_params_eff:
                    if p in dphi_dt:
                        d_re_phi[p] = np.real(dphi_dt[p]).astype(float, copy=False)
                        d_im_phi[p] = np.imag(dphi_dt[p]).astype(float, copy=False)
                    else:
                        d_re_phi[p] = np.zeros_like(re_phi, dtype=float)
                        d_im_phi[p] = np.zeros_like(im_phi, dtype=float)
            # Reconstruct continuation value for all strikes (shape: (nK, M))
            coeffs_mod = coeffs.copy()
            coeffs_mod[0, :] *= 0.5

            # Fill stacked (2N, nK) term matrix (cheap; scales with N*nK)
            term_stack[:N, :] = (re_phi.reshape((-1, 1)) * coeffs_mod)
            term_stack[N:, :] = ((-im_phi).reshape((-1, 1)) * coeffs_mod)

            cont = (term_stack.T @ cos_sin_stack) * disc_dt

            if return_sensitivities:
                d_cont: dict[str, np.ndarray] = {}
                for p in sens_params_eff:
                    d_coeffs_mod = d_coeffs[p].copy()
                    d_coeffs_mod[0, :] *= 0.5
                    # Stack derivative terms
                    term_stack[:N, :] = (d_re_phi[p].reshape((-1, 1)) * coeffs_mod) + (re_phi.reshape((-1, 1)) * d_coeffs_mod)
                    term_stack[N:, :] = ((-d_im_phi[p]).reshape((-1, 1)) * coeffs_mod) + ((-im_phi).reshape((-1, 1)) * d_coeffs_mod)
                    d_cont[p] = (term_stack.T @ cos_sin_stack) * disc_dt

            # Dividend mapping (same shift for all strikes)
            # (times should coincide with t_steps; tolerate float noise)
            for t_div, op in div_shift_ops_by_time.items():
                if abs(float(t_div) - float(t_current)) <= div_tol:
                    cont = self._apply_shift_operator(cont, op)
                    if return_sensitivities:
                        for p in sens_params_eff:
                            d_cont[p] = self._apply_shift_operator(d_cont[p], op)
                    break

            # Discrete event mapping (binary jump) at known time (same shift for all strikes)
            if event is not None and evt_time is not None and abs(evt_time - t_current) <= evt_tol and evt_ops is not None:
                op_u, op_d = evt_ops
                cont_u = self._apply_shift_operator(cont, op_u)
                cont_d = self._apply_shift_operator(cont, op_d)
                cont = float(event.p) * cont_u + (1.0 - float(event.p)) * cont_d
                if return_sensitivities:
                    for p in sens_params_eff:
                        dcu = self._apply_shift_operator(d_cont[p], op_u)
                        dcd = self._apply_shift_operator(d_cont[p], op_d)
                        d_cont[p] = float(event.p) * dcu + (1.0 - float(event.p)) * dcd

            remaining_tau = float(T) - t_current

            # Optional: when returning boundaries, refine the boundary location by doing
            # a local root-find in log-space x for g(x)=intrinsic(x)-continuation(x)
            # around the sign-change cell. This improves boundary accuracy for small M
            # without changing the rollback grid used for pricing.
            cont_eval_at_x: Callable[[int, float], float] | None = None
            if return_boundary and boundary_nodes is not None:
                # Base (pre-div/event mapping) continuation evaluation from coefficients.
                # cont_pre_j(x) = disc_dt * sum_k coeffs_mod[k,j] * (re_phi[k]*cos(u_k*(x-a)) - im_phi[k]*sin(u_k*(x-a)))
                # where u_k = k*pi/(b-a).
                u_vec = np.asarray(u, dtype=float)
                re_phi_vec = np.asarray(re_phi, dtype=float)
                im_phi_vec = np.asarray(im_phi, dtype=float)

                def _cont_pre_at_x(j: int, x: float) -> float:
                    theta = u_vec * (float(x) - float(a))
                    c = coeffs_mod[:, int(j)]
                    return float(disc_dt) * float(np.sum(c * (re_phi_vec * np.cos(theta) - im_phi_vec * np.sin(theta))))

                def _shift_eval(val_fn: Callable[[int, float], float], j: int, x: float, shift: float) -> float:
                    xq = float(x) + float(shift)
                    if xq <= float(a):
                        return float(val_fn(j, float(a)))
                    if xq >= float(b):
                        return 0.0
                    return float(val_fn(j, xq))

                # Match the exact mapping order applied to cont on-grid: dividend shift then event mapping.
                mapped_fn: Callable[[int, float], float] = _cont_pre_at_x

                # Dividend mapping at this step (if any)
                shift_div = None
                for t_div, (m, _std) in self.model.divs.items():
                    if abs(float(t_div) - float(t_current)) <= div_tol and float(m) > 0.0:
                        shift_div = float(np.log(max(1.0 - float(m), 1e-12)))
                        break
                if shift_div is not None:
                    prev_fn = mapped_fn

                    def mapped_fn(j: int, x: float, *, _prev=prev_fn, _sd=shift_div) -> float:
                        return _shift_eval(_prev, j, x, _sd)

                # Discrete event mapping at this step (if any)
                if event is not None and evt_time is not None and abs(evt_time - t_current) <= evt_tol:
                    prev_fn = mapped_fn
                    p_evt = float(event.p)
                    u_shift = float(event.u_log_eff)
                    d_shift = float(event.d_log_eff)

                    def mapped_fn(j: int, x: float, *, _prev=prev_fn, _p=p_evt, _us=u_shift, _ds=d_shift) -> float:
                        return _p * _shift_eval(_prev, j, x, _us) + (1.0 - _p) * _shift_eval(_prev, j, x, _ds)

                cont_eval_at_x = mapped_fn

            def _boundary_from_value_matching(
                S: np.ndarray,
                intrinsic_nm: np.ndarray,
                cont_nm: np.ndarray,
                *,
                exercise_at_low_spot: bool,
            ) -> np.ndarray:
                """Return per-row boundary spot where intrinsic == continuation.

                Parameters
                ----------
                S:
                    1D spot grid (ascending).
                intrinsic_nm, cont_nm:
                    Arrays shaped (nK, M).
                exercise_at_low_spot:
                    True for puts (exercise region at low S), False for calls.

                Returns
                -------
                boundary:
                    1D array length nK. NaN means 'no exercise region on grid'.
                """
                S = np.asarray(S, dtype=float)
                g = np.asarray(intrinsic_nm, dtype=float) - np.asarray(cont_nm, dtype=float)
                if g.ndim == 1:
                    g = g[None, :]
                nK, m = g.shape
                mask = (g >= 0.0)
                out_b = np.full((nK,), np.nan, dtype=float)

                # Helper: refine boundary in x using bisection if a continuation evaluator is available.
                def _refine_in_x(j: int, idx_lo: int, idx_hi: int, *, call_like: bool) -> float | None:
                    if cont_eval_at_x is None:
                        return None
                    if idx_lo < 0 or idx_hi >= m or idx_lo >= idx_hi:
                        return None

                    x_lo = float(x_grid[idx_lo])
                    x_hi = float(x_grid[idx_hi])
                    K_j = float(Kcol[int(j), 0])
                    pvK = float(K_j * np.exp(-self.model.r * remaining_tau))

                    # For parity-call path, continuation is cont_call(x) = cont_put(x) + spot_adj(x) - pvK.
                    # For direct-call-with-divs path, continuation is already in call-space, so the caller
                    # supplies cont_nm accordingly and call_like=True will use intrinsic S-K and cont_eval_at_x.
                    def _g_at_x(x: float) -> float:
                        Sx = float(np.exp(x))
                        if call_like:
                            intrinsic_x = Sx - K_j
                            # Determine whether cont_nm is call continuation. In the parity case, we must add spot_adj.
                            if (not direct_call_with_divs) and bool(is_call):
                                sum_log_rem = float(sum_log_total - float(sum_log_up_to.get(float(t_current), 0.0)))
                                spot_mult = float(np.exp(-self.model.q * remaining_tau + sum_log_rem))
                                if event is not None and evt_time is not None and (not event.ensure_martingale):
                                    if float(t_current) < float(evt_time) <= float(T):
                                        spot_mult *= float(event.mean_factor)
                                cont_x = float(cont_eval_at_x(int(j), float(x))) + (Sx * spot_mult) - pvK
                            else:
                                cont_x = float(cont_eval_at_x(int(j), float(x)))
                            return float(intrinsic_x - cont_x)
                        else:
                            intrinsic_x = K_j - Sx
                            cont_x = float(cont_eval_at_x(int(j), float(x)))
                            return float(intrinsic_x - cont_x)

                    g_lo = _g_at_x(x_lo)
                    g_hi = _g_at_x(x_hi)

                    # Ensure we have a bracket. If not, fall back to grid interpolation.
                    if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
                        return None
                    if g_lo == 0.0:
                        return float(np.exp(x_lo))
                    if g_hi == 0.0:
                        return float(np.exp(x_hi))
                    if g_lo * g_hi > 0.0:
                        return None

                    # Bisection in x.
                    xl, xh = x_lo, x_hi
                    gl, gh = g_lo, g_hi
                    for _ in range(60):
                        xm = 0.5 * (xl + xh)
                        gm = _g_at_x(xm)
                        if not np.isfinite(gm):
                            break
                        if abs(xh - xl) <= 1e-12:
                            return float(np.exp(xm))
                        if gm == 0.0:
                            return float(np.exp(xm))
                        if gl * gm <= 0.0:
                            xh, gh = xm, gm
                        else:
                            xl, gl = xm, gm
                    return float(np.exp(0.5 * (xl + xh)))

                if exercise_at_low_spot:
                    any_ex = np.any(mask, axis=1)
                    # last index where mask is True (exercise optimal)
                    rev = mask[:, ::-1]
                    last_true = (m - 1) - np.argmax(rev, axis=1)
                    for j in range(nK):
                        if not bool(any_ex[j]):
                            continue
                        idx = int(last_true[j])
                        if idx >= m - 1:
                            out_b[j] = float(S[-1])
                            continue
                        refined = _refine_in_x(j, idx, idx + 1, call_like=False)
                        if refined is not None:
                            out_b[j] = float(refined)
                            continue
                        # Fallback: linear interpolation in spot.
                        g0 = float(g[j, idx])
                        g1 = float(g[j, idx + 1])
                        x0 = float(S[idx])
                        x1 = float(S[idx + 1])
                        denom = (g1 - g0)
                        if abs(denom) <= 1e-16:
                            out_b[j] = x0
                        else:
                            out_b[j] = x0 + (0.0 - g0) * (x1 - x0) / denom
                else:
                    any_ex = np.any(mask, axis=1)
                    first_true = np.argmax(mask, axis=1)
                    for j in range(nK):
                        if not bool(any_ex[j]):
                            continue
                        idx = int(first_true[j])
                        if idx <= 0:
                            out_b[j] = float(S[0])
                            continue
                        refined = _refine_in_x(j, idx - 1, idx, call_like=True)
                        if refined is not None:
                            out_b[j] = float(refined)
                            continue
                        # Fallback: linear interpolation in spot.
                        g0 = float(g[j, idx - 1])
                        g1 = float(g[j, idx])
                        x0 = float(S[idx - 1])
                        x1 = float(S[idx])
                        denom = (g1 - g0)
                        if abs(denom) <= 1e-16:
                            out_b[j] = x1
                        else:
                            out_b[j] = x0 + (0.0 - g0) * (x1 - x0) / denom

                return out_b

            if direct_call_with_divs:
                intrinsic = S_grid.reshape((1, -1)) - Kcol

                if return_boundary and boundary_nodes is not None:
                    bnd = _boundary_from_value_matching(S_grid, intrinsic, cont, exercise_at_low_spot=False)
                    boundary_nodes.append((float(t_current), bnd))

                if use_softmax:
                    V_prev = numerics.SOFTMAX_FN(intrinsic, cont, beta)
                    if return_sensitivities:
                        x = float(beta) * (intrinsic - cont)
                        # derivative of softmax_pair wrt cont: sigmoid(-beta*(a-b))
                        w_cont = 1.0 / (1.0 + np.exp(np.clip(x, -60.0, 60.0)))
                else:
                    V_prev = np.maximum(intrinsic, cont)
                    if return_sensitivities:
                        w_cont = (cont >= intrinsic).astype(float)
                V_prev = np.maximum(V_prev, 0.0)
                if return_sensitivities:
                    # derivative through max(.,0)
                    w_pos = (V_prev > 0.0).astype(float)
                    for p in sens_params_eff:
                        dVp = w_cont * d_cont[p]
                        dVp = w_pos * dVp
                        d_cont[p] = dVp
                V_prev_call = V_prev

                # Snapshot after exercise decision (call space)
                if (
                    return_grid_snapshot
                    and grid_snapshot is None
                    and (snapshot_time_val is None or abs(float(t_current) - snapshot_time_val) <= snapshot_tol)
                ):
                    grid_snapshot = {
                        "t": float(t_current),
                        "x_grid": np.asarray(x_grid, dtype=float).copy(),
                        "S_grid": np.asarray(S_grid, dtype=float).copy(),
                        "intrinsic": np.asarray(intrinsic[0], dtype=float).copy(),
                        "continuation": np.asarray(cont[0], dtype=float).copy(),
                        "value": np.asarray(V_prev_call[0], dtype=float).copy(),
                        "space": "call",
                    }

            elif is_call:
                # Convert put-continuation to call-continuation via parity
                sum_log_rem = float(sum_log_total - float(sum_log_up_to.get(float(t_current), 0.0)))
                spot_adj = S_grid * np.exp(-self.model.q * remaining_tau + sum_log_rem)
                if event is not None and evt_time is not None and not event.ensure_martingale:
                    if float(t_current) < float(evt_time) <= float(T):
                        spot_adj = spot_adj * float(event.mean_factor)

                cont_call = cont + spot_adj.reshape((1, -1)) - Kcol * float(np.exp(-self.model.r * remaining_tau))
                intrinsic = S_grid.reshape((1, -1)) - Kcol

                if return_boundary and boundary_nodes is not None:
                    bnd = _boundary_from_value_matching(S_grid, intrinsic, cont_call, exercise_at_low_spot=False)
                    boundary_nodes.append((float(t_current), bnd))

                if use_softmax:
                    V_prev_call = numerics.SOFTMAX_FN(intrinsic, cont_call, beta)
                    if return_sensitivities:
                        x = float(beta) * (intrinsic - cont_call)
                        w_cont = 1.0 / (1.0 + np.exp(np.clip(x, -60.0, 60.0)))
                else:
                    V_prev_call = np.maximum(intrinsic, cont_call)
                    if return_sensitivities:
                        w_cont = (cont_call >= intrinsic).astype(float)
                V_prev_call = np.maximum(V_prev_call, 0.0)

                # Reproject back to put-space (parity)
                V_prev = V_prev_call - spot_adj.reshape((1, -1)) + Kcol * float(np.exp(-self.model.r * remaining_tau))

                # Snapshot after exercise decision (call space) using cont_call
                if (
                    return_grid_snapshot
                    and grid_snapshot is None
                    and (snapshot_time_val is None or abs(float(t_current) - snapshot_time_val) <= snapshot_tol)
                ):
                    grid_snapshot = {
                        "t": float(t_current),
                        "x_grid": np.asarray(x_grid, dtype=float).copy(),
                        "S_grid": np.asarray(S_grid, dtype=float).copy(),
                        "intrinsic": np.asarray(intrinsic[0], dtype=float).copy(),
                        "continuation": np.asarray(cont_call[0], dtype=float).copy(),
                        "value": np.asarray(V_prev_call[0], dtype=float).copy(),
                        "space": "call",
                    }

                if return_sensitivities:
                    w_pos = (V_prev_call > 0.0).astype(float)
                    for p in sens_params_eff:
                        dVcall = w_pos * (w_cont * d_cont[p])
                        # parity subtraction/addition are param-independent under frozen grid
                        d_cont[p] = dVcall
            else:
                intrinsic = Kcol - S_grid.reshape((1, -1))

                if return_boundary and boundary_nodes is not None:
                    bnd = _boundary_from_value_matching(S_grid, intrinsic, cont, exercise_at_low_spot=True)
                    boundary_nodes.append((float(t_current), bnd))

                if use_softmax:
                    V_prev = numerics.SOFTMAX_FN(intrinsic, cont, beta)
                    if return_sensitivities:
                        x = float(beta) * (intrinsic - cont)
                        w_cont = 1.0 / (1.0 + np.exp(np.clip(x, -60.0, 60.0)))
                else:
                    V_prev = np.maximum(intrinsic, cont)
                    if return_sensitivities:
                        w_cont = (cont >= intrinsic).astype(float)
                V_prev = np.maximum(V_prev, 0.0)
                if return_sensitivities:
                    w_pos = (V_prev > 0.0).astype(float)
                    for p in sens_params_eff:
                        dVp = w_cont * d_cont[p]
                        dVp = w_pos * dVp
                        d_cont[p] = dVp

                # Snapshot after exercise decision (put space)
                if (
                    return_grid_snapshot
                    and grid_snapshot is None
                    and (snapshot_time_val is None or abs(float(t_current) - snapshot_time_val) <= snapshot_tol)
                ):
                    grid_snapshot = {
                        "t": float(t_current),
                        "x_grid": np.asarray(x_grid, dtype=float).copy(),
                        "S_grid": np.asarray(S_grid, dtype=float).copy(),
                        "intrinsic": np.asarray(intrinsic[0], dtype=float).copy(),
                        "continuation": np.asarray(cont[0], dtype=float).copy(),
                        "value": np.asarray(V_prev[0], dtype=float).copy(),
                        "space": "put",
                    }

            # Cache trajectories at S0 for the first strike only
            if (return_trajectory or return_continuation_trajectory):
                def _interp_at_s0(arr_1d: np.ndarray) -> float:
                    return float((1.0 - wgt0) * arr_1d[ix0] + wgt0 * arr_1d[ix0 + 1])

                if return_continuation_trajectory and cont_trajectory_cache is not None:
                    if direct_call_with_divs:
                        cont_at_s0 = _interp_at_s0(cont[0])
                    else:
                        cont_at_s0 = _interp_at_s0((cont_call if is_call else cont)[0])
                    cont_trajectory_cache.append((t_current, cont_at_s0))
                if return_trajectory and trajectory_cache is not None:
                    if direct_call_with_divs:
                        ex_at_s0 = _interp_at_s0(V_prev_call[0])
                    else:
                        ex_at_s0 = _interp_at_s0((V_prev_call if is_call else V_prev)[0])
                    trajectory_cache.append((t_current, ex_at_s0))

            # Re-project value function back to cosine coefficients
            coeffs = (2.0 / (b - a)) * (cos_k_x @ (V_prev * w).T) * dx
            if return_sensitivities:
                for p in sens_params_eff:
                    d_coeffs[p] = (2.0 / (b - a)) * (cos_k_x @ (d_cont[p] * w).T) * dx

        # Final value at t=0 for all strikes: reconstruct and interpolate at S0
        coeffs_mod = coeffs.copy()
        coeffs_mod[0, :] *= 0.5
        V0_grid = coeffs_mod.T @ cos_k_x  # (nK, M)

        val0 = (1.0 - wgt0) * V0_grid[:, ix0] + wgt0 * V0_grid[:, ix0 + 1]

        # If a snapshot was requested but not captured inside the loop (e.g. snapshot_time=0.0),
        # provide a t=0 snapshot from the reconstructed grid in the appropriate space.
        if return_grid_snapshot and grid_snapshot is None and (snapshot_time_val is None or abs(0.0 - snapshot_time_val) <= snapshot_tol):
            if direct_call_with_divs or bool(is_call):
                # V0_grid is in put-space for parity path; reconstruct call-space value for diagnostics.
                if direct_call_with_divs:
                    V0_diag = V0_grid
                    intrinsic0 = np.maximum(S_grid.reshape((1, -1)) - Kcol, 0.0)
                    grid_snapshot = {
                        "t": 0.0,
                        "x_grid": np.asarray(x_grid, dtype=float).copy(),
                        "S_grid": np.asarray(S_grid, dtype=float).copy(),
                        "intrinsic": np.asarray((S_grid - float(K[0])), dtype=float).copy(),
                        "continuation": np.asarray(V0_diag[0], dtype=float).copy(),
                        "value": np.asarray(V0_diag[0], dtype=float).copy(),
                        "space": "call",
                    }
                else:
                    # Build call-space value at t=0 from parity using the known forward adjustment.
                    forward_adj = float(self.model.S0 * np.exp(-self.model.q * T + sum_log_total))
                    if event is not None and 0.0 < float(event.time) <= float(T) and not event.ensure_martingale:
                        forward_adj *= float(event.mean_factor)
                    # At t=0, remaining_tau = T.
                    spot_adj0 = S_grid * np.exp(-self.model.q * T + sum_log_total)
                    if event is not None and evt_time is not None and not event.ensure_martingale:
                        if 0.0 < float(evt_time) <= float(T):
                            spot_adj0 = spot_adj0 * float(event.mean_factor)
                    cont_call0 = V0_grid + spot_adj0.reshape((1, -1)) - Kcol * float(np.exp(-self.model.r * T))
                    V_call0 = np.maximum(S_grid.reshape((1, -1)) - Kcol, cont_call0)
                    grid_snapshot = {
                        "t": 0.0,
                        "x_grid": np.asarray(x_grid, dtype=float).copy(),
                        "S_grid": np.asarray(S_grid, dtype=float).copy(),
                        "intrinsic": np.asarray((S_grid - float(K[0])), dtype=float).copy(),
                        "continuation": np.asarray(cont_call0[0], dtype=float).copy(),
                        "value": np.asarray(V_call0[0], dtype=float).copy(),
                        "space": "call",
                    }
            else:
                # Put space
                grid_snapshot = {
                    "t": 0.0,
                    "x_grid": np.asarray(x_grid, dtype=float).copy(),
                    "S_grid": np.asarray(S_grid, dtype=float).copy(),
                    "intrinsic": np.asarray((float(K[0]) - S_grid), dtype=float).copy(),
                    "continuation": np.asarray(V0_grid[0], dtype=float).copy(),
                    "value": np.asarray(V0_grid[0], dtype=float).copy(),
                    "space": "put",
                }

        if return_sensitivities:
            d_val0: dict[str, np.ndarray] = {}
            for p in sens_params_eff:
                dc = d_coeffs[p].copy()
                dc[0, :] *= 0.5
                dV0_grid = dc.T @ cos_k_x
                d_val0[p] = (1.0 - wgt0) * dV0_grid[:, ix0] + wgt0 * dV0_grid[:, ix0 + 1]

        if direct_call_with_divs:
            prices[:] = val0
        elif is_call:
            forward_adj = self.model.S0 * np.exp(-self.model.q * T + sum_log_total)
            if event is not None and 0.0 < float(event.time) <= float(T) and not event.ensure_martingale:
                forward_adj *= float(event.mean_factor)
            prices[:] = val0 + forward_adj - K * np.exp(-self.model.r * T)
        else:
            prices[:] = val0

        if return_sensitivities:
            # Parity adjustments are parameter-independent under frozen grid.
            return (prices, d_val0)

        if return_trajectory and trajectory_cache is not None:
            trajectory_cache.sort(key=lambda z: z[0])

        if return_continuation_trajectory and cont_trajectory_cache is not None:
            cont_trajectory_cache.sort(key=lambda z: z[0])

        boundary_curve = None
        if return_boundary and boundary_nodes is not None:
            boundary_nodes.sort(key=lambda z: z[0])
            t_bnd = np.array([t for t, _b in boundary_nodes], dtype=float)
            bnd_mat = np.vstack([np.asarray(b, dtype=float).reshape((1, -1)) for _t, b in boundary_nodes])
            boundary_curve = (t_bnd, bnd_mat)

        out = [prices]
        if return_european:
            out.append(euro_prices)
        if return_trajectory:
            out.append(trajectory_cache)
        if return_continuation_trajectory:
            out.append(cont_trajectory_cache)
        if return_boundary:
            out.append(boundary_curve)
        if return_grid_snapshot:
            out.append(grid_snapshot)
        return out[0] if len(out) == 1 else tuple(out)

    def estimate_next_exercise_node_from_boundary(
        self,
        t_grid: np.ndarray,
        boundary: np.ndarray,
        *,
        is_call: bool,
        strike_index: int = 0,
        spot: float | None = None,
    ) -> tuple[float, float, float]:
        """Heuristic: pick a 'most likely next exercise node' from a boundary curve.

        This is *not* a first-passage / optimal stopping probability. It uses marginal
        probabilities at each node: p_i ≈ P(S_{t_i} in exercise region), then selects the
        time with the largest incremental increase dp_i = max(p_i - p_{i-1}, 0).

        Returns
        -------
        (t_star, b_star, dp_star)
            t_star: node time
            b_star: boundary level at that node
            dp_star: incremental probability mass at that node under this heuristic
        """
        t = np.asarray(t_grid, dtype=float).reshape((-1,))
        bnd = np.asarray(boundary, dtype=float)
        if bnd.ndim == 2:
            b = bnd[:, int(strike_index)].reshape((-1,))
        else:
            b = bnd.reshape((-1,))
        if t.shape[0] != b.shape[0]:
            raise ValueError("t_grid and boundary must have the same length")

        spot_eff = float(self.model.S0 if spot is None else spot)

        # Compute marginal exercise probability at each node via COS digitals.
        p = np.full_like(t, np.nan, dtype=float)
        for i, ti in enumerate(t):
            if not (ti > 0.0):
                # At t=0, marginal distribution is degenerate at spot.
                if np.isnan(b[i]):
                    p[i] = 0.0
                else:
                    if bool(is_call):
                        p[i] = float(spot_eff >= float(b[i]))
                    else:
                        p[i] = float(spot_eff <= float(b[i]))
                continue

            if np.isnan(b[i]) or float(b[i]) <= 0.0:
                p[i] = 0.0 if not is_call else 0.0
                continue

            prob_put = float(self.digital_put_price(np.array([float(b[i])]), float(ti), spot=spot_eff)[0])
            if bool(is_call):
                p[i] = 1.0 - prob_put
            else:
                p[i] = prob_put

        # Incremental probability mass (crude hazard proxy)
        dp = np.zeros_like(p)
        dp[0] = max(float(p[0]), 0.0)
        for i in range(1, len(p)):
            if np.isnan(p[i]) or np.isnan(p[i - 1]):
                dp[i] = 0.0
            else:
                dp[i] = max(float(p[i] - p[i - 1]), 0.0)

        if float(np.max(dp)) <= 0.0:
            # No mass assigned; return earliest node with a finite boundary if present.
            finite = np.where(np.isfinite(b))[0]
            if finite.size == 0:
                return (float(t[0]), float("nan"), 0.0)
            i0 = int(finite[0])
            return (float(t[i0]), float(b[i0]), 0.0)

        i_star = int(np.argmax(dp))
        return (float(t[i_star]), float(b[i_star]), float(dp[i_star]))

    # ----------------------------------------------------------------------- #
    # 2.5. Helper – variance over [0,T] for COS truncation
    # ----------------------------------------------------------------------- #
    def _var2(self, T: float) -> float:
        """
        Return Var[ln S_T] (default approximation: diffusion variance).
        Override if the model has jumps or different structure.
        """
        params = getattr(self.model, "params", {})
        if not isinstance(params, dict):
            params = {}
        sigma = float(params.get("sigma", params.get("vol", 0.0)))
        return (sigma ** 2) * T



