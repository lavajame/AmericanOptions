"""Composite exponential-Levy characteristic function model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_cf import CharacteristicFunction
from ..dividends import _dividend_adjustment, dividend_char_factor

class CompositeLevyCHF(CharacteristicFunction):
    """Composite exponential-Lévy model built from independent components.

    This model represents:
        ln S_T = ln S0 + (r-q)T + sum_log  - psi_mix(-i)T  + X_T
    where X_T is a Lévy process with characteristic exponent
        E[e^{i u X_T}] = exp(psi_mix(u) T)
    and psi_mix(u) is the sum of component exponents.

    Parameters
    ----------
    params["components"] : list
        A list of components. Each component is a dict of the form:
            {"type": "merton"|"vg"|"kou"|"cgmy"|"gbm", "params": {...}}
        Component parameters match the corresponding *model* params in this module.

    Notes
    -----
    - This avoids double-counting deterministic drift: (r-q) and discrete-dividend
      mean adjustment are applied once, while *martingale corrections* from each
      Lévy component are applied via the combined psi_mix(-i).
    - Because :class:`CharacteristicFunction` provides a default `increment_char`,
      this composite model works in the COS American rollback without extra hooks.
    """

    _SUPPORTED_TYPES = {"merton", "vg", "kou", "cgmy", "gbm"}

    def __init__(self,
                 S0: float,
                 r: float,
                 q: float,
                 divs: Dict[float, Tuple[float, float]],
                 params: Dict[str, Any]):
        super().__init__(S0, r, q, divs, params)
        comps = params.get("components", None)
        if comps is None:
            raise ValueError("CompositeLevyCHF requires params['components']")
        if not isinstance(comps, (list, tuple)) or len(comps) == 0:
            raise ValueError("params['components'] must be a non-empty list")

        normed = []
        for c in comps:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                c = {"type": c[0], "params": c[1]}
            if not isinstance(c, dict):
                raise ValueError("Each component must be a dict (or (type, params) tuple)")
            ctype = str(c.get("type", "")).lower().strip()
            cparams = c.get("params", {})
            if ctype not in self._SUPPORTED_TYPES:
                raise ValueError(f"Unsupported component type: {ctype}. Supported: {sorted(self._SUPPORTED_TYPES)}")
            if not isinstance(cparams, dict):
                raise ValueError("component['params'] must be a dict")
            normed.append({"type": ctype, "params": dict(cparams)})

        self._components = normed

    @staticmethod
    def _psi_unit_merton(u: np.ndarray, *, vol: float, lam: float, muJ: float, sigmaJ: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        phi_jump = np.exp(1j * u * muJ - 0.5 * (u ** 2) * (sigmaJ ** 2))
        return -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump - 1.0)

    @staticmethod
    def _psi_unit_kou(u: np.ndarray, *, vol: float, lam: float, p: float, eta1: float, eta2: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        phi_jump = p * (eta1 / (eta1 - 1j * u)) + (1.0 - p) * (eta2 / (eta2 + 1j * u))
        return -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump - 1.0)

    @staticmethod
    def _psi_unit_vg(u: np.ndarray, *, theta: float, sigma: float, nu: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        # psi(u) = -(1/nu) * ln(1 - i theta nu u + 0.5 sigma^2 nu u^2)
        inside = 1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * nu * (u ** 2)
        return -(1.0 / nu) * np.log(inside)

    @staticmethod
    def _psi_unit_cgmy(u: np.ndarray, *, C: float, G: float, M: float, Y: float) -> np.ndarray:
        from scipy.special import gamma as sp_gamma

        u = np.asarray(u, dtype=complex)
        gamma_m = sp_gamma(-Y)

        # Use stable expm1/log1p expansion similar to CGMYCHF.
        def psi_unit(z):
            x_m = -1j * z / M
            x_g = 1j * z / G

            if np.all(np.abs(x_m) < 1e-4) and np.all(np.abs(x_g) < 1e-4):
                term_m = Y * (-1j * z) * np.power(M, Y - 1.0) - 0.5 * Y * (Y - 1.0) * (z ** 2) * np.power(M, Y - 2.0)
                term_g = Y * (1j * z) * np.power(G, Y - 1.0) - 0.5 * Y * (Y - 1.0) * (z ** 2) * np.power(G, Y - 2.0)
            else:
                term_m = np.power(M, Y) * np.expm1(Y * np.log1p(x_m))
                term_g = np.power(G, Y) * np.expm1(Y * np.log1p(x_g))
            return C * gamma_m * (term_m + term_g)

        return psi_unit(u)

    def _psi_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        total = np.zeros_like(u, dtype=complex)
        for comp in self._components:
            ctype = comp["type"]
            p = comp["params"]
            if ctype == "merton":
                total += self._psi_unit_merton(
                    u,
                    vol=float(p.get("vol", 0.0)),
                    lam=float(p.get("lam", 0.0)),
                    muJ=float(p.get("muJ", 0.0)),
                    sigmaJ=float(p.get("sigmaJ", 0.0)),
                )
            elif ctype == "kou":
                total += self._psi_unit_kou(
                    u,
                    vol=float(p.get("vol", 0.0)),
                    lam=float(p.get("lam", 0.0)),
                    p=float(p.get("p", 0.5)),
                    eta1=float(p.get("eta1", 10.0)),
                    eta2=float(p.get("eta2", 5.0)),
                )
            elif ctype == "vg":
                total += self._psi_unit_vg(
                    u,
                    theta=float(p.get("theta", 0.0)),
                    sigma=float(p.get("sigma", 0.0)),
                    nu=float(p.get("nu", 0.0)),
                )
            elif ctype == "cgmy":
                total += self._psi_unit_cgmy(
                    u,
                    C=float(p.get("C", 0.02)),
                    G=float(p.get("G", 5.0)),
                    M=float(p.get("M", 5.0)),
                    Y=float(p.get("Y", 0.5)),
                )
            elif ctype == "gbm":
                vol = float(p.get("vol", 0.0))
                total += -0.5 * (u ** 2) * (vol ** 2)
            else:
                raise RuntimeError(f"Unhandled component type: {ctype}")
        return total

    @staticmethod
    def _type_label(ctype: str) -> str:
        ctype = str(ctype).lower().strip()
        if ctype == "vg":
            return "VG"
        if ctype == "cgmy":
            return "CGMY"
        if ctype == "gbm":
            return "GBM"
        if ctype == "merton":
            return "Merton"
        if ctype == "kou":
            return "Kou"
        return ctype

    def _component_param_name_map(self) -> dict[str, tuple[int, str]]:
        """Map composite parameter names -> (component_index, component_param_key)."""
        counts: dict[str, int] = {}
        for comp in self._components:
            counts[comp["type"]] = counts.get(comp["type"], 0) + 1

        seen: dict[str, int] = {}
        name_map: dict[str, tuple[int, str]] = {}
        for idx, comp in enumerate(self._components):
            ctype = comp["type"]
            label = self._type_label(ctype)
            seen[ctype] = seen.get(ctype, 0) + 1
            prefix = label if counts.get(ctype, 0) == 1 else f"{label}{seen[ctype]}"
            for k in sorted(comp["params"].keys(), key=lambda x: str(x)):
                name_map[f"{prefix}.{k}"] = (idx, str(k))
        return name_map

    def param_names(self) -> list[str]:
        return list(self._component_param_name_map().keys())

    def _increment_char_and_grad_fd_composite(
        self,
        u: np.ndarray,
        dt: float,
        *,
        params: list[str] | None = None,
        rel_step: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Finite-difference gradient over nested component params."""
        u = np.asarray(u, dtype=complex)
        dt = float(dt)
        phi0 = self.increment_char(u, dt)

        name_map = self._component_param_name_map()
        if params is None:
            params = list(name_map.keys())
        params = [p for p in params if p in name_map]

        grad: dict[str, np.ndarray] = {}
        for name in params:
            comp_idx, key = name_map[name]
            comp_params = self._components[comp_idx]["params"]
            if key not in comp_params:
                continue
            base = comp_params[key]
            if not isinstance(base, (int, float, np.floating)):
                continue
            base_f = float(base)
            h = float(rel_step) * max(1.0, abs(base_f))
            if h == 0.0:
                h = float(rel_step)

            try:
                comp_params[key] = base_f + h
                phi_p = self.increment_char(u, dt)
                comp_params[key] = base_f - h
                phi_m = self.increment_char(u, dt)
            finally:
                comp_params[key] = base

            grad[name] = (phi_p - phi_m) / (2.0 * h)

        return phi0, grad

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
            return self._increment_char_and_grad_fd_composite(u, dt, params=params, rel_step=rel_step)

        u = np.asarray(u, dtype=complex)
        dt = float(dt)
        # Increment CF excludes ln(S0).
        mu_inc = (self.r - self.q) * dt

        name_map = self._component_param_name_map()
        if params is None:
            params_eff = list(name_map.keys())
        else:
            params_eff = [p for p in params if p in name_map]

        psi_u = np.zeros_like(u, dtype=complex)
        psi_mi = 0.0 + 0.0j
        dpsi_u: dict[str, np.ndarray] = {p: np.zeros_like(u, dtype=complex) for p in params_eff}
        dpsi_mi: dict[str, complex] = {p: 0.0 + 0.0j for p in params_eff}

        for idx, comp in enumerate(self._components):
            ctype = comp["type"]
            p = comp["params"]

            if ctype == "merton":
                vol = float(p.get("vol", 0.0))
                lam = float(p.get("lam", 0.0))
                muJ = float(p.get("muJ", 0.0))
                sigmaJ = float(p.get("sigmaJ", 0.0))
                phi_jump_u = np.exp(1j * u * muJ - 0.5 * (u ** 2) * (sigmaJ ** 2))
                psi_u += -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump_u - 1.0)
                u_mi = -1j
                phi_jump_mi = np.exp(1j * u_mi * muJ - 0.5 * (u_mi ** 2) * (sigmaJ ** 2))
                psi_mi += (-0.5 * (u_mi ** 2) * (vol ** 2) + lam * (phi_jump_mi - 1.0))

                for name in params_eff:
                    comp_idx, key = name_map[name]
                    if comp_idx != idx:
                        continue
                    if key == "vol":
                        dpsi_u[name] += -(u ** 2) * vol
                        dpsi_mi[name] += -(u_mi ** 2) * vol
                    elif key == "lam":
                        dpsi_u[name] += (phi_jump_u - 1.0)
                        dpsi_mi[name] += (phi_jump_mi - 1.0)
                    elif key == "muJ":
                        dpsi_u[name] += lam * phi_jump_u * (1j * u)
                        dpsi_mi[name] += lam * phi_jump_mi * (1j * u_mi)
                    elif key == "sigmaJ":
                        dpsi_u[name] += lam * phi_jump_u * (-(u ** 2) * sigmaJ)
                        dpsi_mi[name] += lam * phi_jump_mi * (-(u_mi ** 2) * sigmaJ)
            elif ctype == "kou":
                vol = float(p.get("vol", 0.0))
                lam = float(p.get("lam", 0.0))
                pp = float(p.get("p", 0.5))
                eta1 = float(p.get("eta1", 10.0))
                eta2 = float(p.get("eta2", 5.0))
                a_u = eta1 / (eta1 - 1j * u)
                b_u = eta2 / (eta2 + 1j * u)
                phi_jump_u = pp * a_u + (1.0 - pp) * b_u
                psi_u += -0.5 * (u ** 2) * (vol ** 2) + lam * (phi_jump_u - 1.0)
                u_mi = -1j
                a_mi = eta1 / (eta1 - 1j * u_mi)
                b_mi = eta2 / (eta2 + 1j * u_mi)
                phi_jump_mi = pp * a_mi + (1.0 - pp) * b_mi
                psi_mi += (-0.5 * (u_mi ** 2) * (vol ** 2) + lam * (phi_jump_mi - 1.0))

                da_u_deta1 = (-1j * u) / ((eta1 - 1j * u) ** 2)
                db_u_deta2 = (1j * u) / ((eta2 + 1j * u) ** 2)
                da_mi_deta1 = (-1j * u_mi) / ((eta1 - 1j * u_mi) ** 2)
                db_mi_deta2 = (1j * u_mi) / ((eta2 + 1j * u_mi) ** 2)

                for name in params_eff:
                    comp_idx, key = name_map[name]
                    if comp_idx != idx:
                        continue
                    if key == "vol":
                        dpsi_u[name] += -(u ** 2) * vol
                        dpsi_mi[name] += -(u_mi ** 2) * vol
                    elif key == "lam":
                        dpsi_u[name] += (phi_jump_u - 1.0)
                        dpsi_mi[name] += (phi_jump_mi - 1.0)
                    elif key == "p":
                        dpsi_u[name] += lam * (a_u - b_u)
                        dpsi_mi[name] += lam * (a_mi - b_mi)
                    elif key == "eta1":
                        dpsi_u[name] += lam * (pp * da_u_deta1)
                        dpsi_mi[name] += lam * (pp * da_mi_deta1)
                    elif key == "eta2":
                        dpsi_u[name] += lam * ((1.0 - pp) * db_u_deta2)
                        dpsi_mi[name] += lam * ((1.0 - pp) * db_mi_deta2)
            elif ctype == "vg":
                theta = float(p.get("theta", 0.0))
                sigma = float(p.get("sigma", 0.0))
                nu = float(p.get("nu", 0.0))
                inside_u = 1.0 - 1j * theta * nu * u + 0.5 * (sigma ** 2) * nu * (u ** 2)
                psi_u += -(1.0 / nu) * np.log(inside_u)
                u_mi = -1j
                inside_mi = 1.0 - 1j * theta * nu * u_mi + 0.5 * (sigma ** 2) * nu * (u_mi ** 2)
                psi_mi += (-(1.0 / nu) * np.log(inside_mi))

                for name in params_eff:
                    comp_idx, key = name_map[name]
                    if comp_idx != idx:
                        continue
                    if key == "theta":
                        dpsi_u[name] += (1j * u) / inside_u
                        dpsi_mi[name] += (1j * u_mi) / inside_mi
                    elif key == "sigma":
                        dpsi_u[name] += -(sigma * (u ** 2)) / inside_u
                        dpsi_mi[name] += -(sigma * (u_mi ** 2)) / inside_mi
                    elif key == "nu":
                        dA_u = (-1j * theta * u) + 0.5 * (sigma ** 2) * (u ** 2)
                        dA_mi = (-1j * theta * u_mi) + 0.5 * (sigma ** 2) * (u_mi ** 2)
                        dpsi_u[name] += (1.0 / (nu ** 2)) * np.log(inside_u) - (1.0 / nu) * (dA_u / inside_u)
                        dpsi_mi[name] += (1.0 / (nu ** 2)) * np.log(inside_mi) - (1.0 / nu) * (dA_mi / inside_mi)
            elif ctype == "gbm":
                vol = float(p.get("vol", 0.0))
                psi_u += -0.5 * (u ** 2) * (vol ** 2)
                u_mi = -1j
                psi_mi += (-0.5 * (u_mi ** 2) * (vol ** 2))
                for name in params_eff:
                    comp_idx, key = name_map[name]
                    if comp_idx != idx:
                        continue
                    if key == "vol":
                        dpsi_u[name] += -(u ** 2) * vol
                        dpsi_mi[name] += -(u_mi ** 2) * vol
            elif ctype == "cgmy":
                # Hybrid: analytic for C, FD for others to avoid heavy symbolic derivatives.
                C = float(p.get("C", 0.02))
                G = float(p.get("G", 5.0))
                M = float(p.get("M", 5.0))
                Y = float(p.get("Y", 0.5))
                psi_u_comp = self._psi_unit_cgmy(u, C=C, G=G, M=M, Y=Y)
                psi_u += psi_u_comp
                psi_mi_comp = complex(self._psi_unit_cgmy(-1j, C=C, G=G, M=M, Y=Y))
                psi_mi += psi_mi_comp

                for name in params_eff:
                    comp_idx, key = name_map[name]
                    if comp_idx != idx:
                        continue
                    if key == "C":
                        # psi is linear in C
                        dpsi_u[name] += psi_u_comp / max(C, 1e-300)
                        dpsi_mi[name] += psi_mi_comp / max(C, 1e-300)
                    else:
                        # Local FD on the CGMY exponent only
                        base_val = float(p.get(key))
                        h = float(rel_step) * max(1.0, abs(base_val))
                        if h == 0.0:
                            h = float(rel_step)

                        p_p = dict(p)
                        p_m = dict(p)
                        p_p[key] = base_val + h
                        p_m[key] = base_val - h
                        psi_p = self._psi_unit_cgmy(u, C=float(p_p.get("C", C)), G=float(p_p.get("G", G)), M=float(p_p.get("M", M)), Y=float(p_p.get("Y", Y)))
                        psi_m = self._psi_unit_cgmy(u, C=float(p_m.get("C", C)), G=float(p_m.get("G", G)), M=float(p_m.get("M", M)), Y=float(p_m.get("Y", Y)))
                        dpsi_u[name] += (psi_p - psi_m) / (2.0 * h)

                        psi_p_mi = complex(self._psi_unit_cgmy(-1j, C=float(p_p.get("C", C)), G=float(p_p.get("G", G)), M=float(p_p.get("M", M)), Y=float(p_p.get("Y", Y))))
                        psi_m_mi = complex(self._psi_unit_cgmy(-1j, C=float(p_m.get("C", C)), G=float(p_m.get("G", G)), M=float(p_m.get("M", M)), Y=float(p_m.get("Y", Y))))
                        dpsi_mi[name] += (psi_p_mi - psi_m_mi) / (2.0 * h)
            else:
                raise RuntimeError(f"Unhandled component type: {ctype}")

        exponent_inc = 1j * u * mu_inc + (psi_u - 1j * u * psi_mi) * dt
        phi_inc = np.exp(exponent_inc) * dividend_char_factor(u, dt, self.divs)

        grad: dict[str, np.ndarray] = {}
        if params_eff:
            for name in params_eff:
                dlog = dt * (dpsi_u[name] - 1j * u * dpsi_mi[name])
                grad[name] = phi_inc * dlog

        return phi_inc, grad

    def char_func(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)

        psi_vals = self._psi_unit(u)
        psi_minus_i = self._psi_unit(-1j)

        mu_base = np.log(self.S0) + (self.r - self.q) * T
        exponent = 1j * u * mu_base + (psi_vals - 1j * u * psi_minus_i) * T
        phi = np.exp(exponent)
        return phi * dividend_char_factor(u, T, self.divs)

    def cumulants(self, T: float) -> Tuple[float, float, float]:
        from scipy.special import gamma as sp_gamma

        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0

        # Sum cumulants of X_T across components.
        k1 = 0.0
        k2 = 0.0
        k4 = 0.0
        for comp in self._components:
            ctype = comp["type"]
            p = comp["params"]
            if ctype == "merton":
                lam = float(p.get("lam", 0.0))
                muJ = float(p.get("muJ", 0.0))
                sigmaJ = float(p.get("sigmaJ", 0.0))
                vol = float(p.get("vol", 0.0))
                EY = muJ
                EY2 = muJ ** 2 + sigmaJ ** 2
                EY4 = muJ ** 4 + 6.0 * muJ ** 2 * sigmaJ ** 2 + 3.0 * sigmaJ ** 4
                k1 += lam * T * EY
                k2 += (vol ** 2) * T + lam * T * EY2
                k4 += lam * T * EY4
            elif ctype == "kou":
                lam = float(p.get("lam", 0.0))
                vol = float(p.get("vol", 0.0))
                pp = float(p.get("p", 0.5))
                eta1 = float(p.get("eta1", 10.0))
                eta2 = float(p.get("eta2", 5.0))
                EY = pp * (1.0 / eta1) - (1.0 - pp) * (1.0 / eta2)
                EY2 = 2.0 * pp * (1.0 / (eta1 ** 2)) + 2.0 * (1.0 - pp) * (1.0 / (eta2 ** 2))
                EY4 = 24.0 * pp * (1.0 / (eta1 ** 4)) + 24.0 * (1.0 - pp) * (1.0 / (eta2 ** 4))
                k1 += lam * T * EY
                k2 += (vol ** 2) * T + lam * T * EY2
                k4 += lam * T * EY4
            elif ctype == "vg":
                theta = float(p.get("theta", 0.0))
                sigma = float(p.get("sigma", 0.0))
                nu = float(p.get("nu", 0.0))
                k1 += theta * T
                k2 += (sigma ** 2 + (theta ** 2) * nu) * T
                k4 += (3.0 * (sigma ** 4) * nu + 12.0 * (sigma ** 2) * (theta ** 2) * (nu ** 2) + 6.0 * (theta ** 4) * (nu ** 3)) * T
            elif ctype == "cgmy":
                C = float(p.get("C", 0.02))
                G = float(p.get("G", 5.0))
                M = float(p.get("M", 5.0))
                Y = float(p.get("Y", 0.5))
                # c_n = C * T * Gamma(n - Y) * (M^(Y-n) + (-1)^n * G^(Y-n))
                # For cumulants of X: c1 has (M^(Y-1) - G^(Y-1)), even orders add.
                def stable_pow(base: float, exp: float) -> float:
                    return float(np.exp(exp * np.log(base)))
                k1 += C * T * float(sp_gamma(1.0 - Y)) * (stable_pow(M, Y - 1.0) - stable_pow(G, Y - 1.0))
                k2 += C * T * float(sp_gamma(2.0 - Y)) * (stable_pow(M, Y - 2.0) + stable_pow(G, Y - 2.0))
                k4 += C * T * float(sp_gamma(4.0 - Y)) * (stable_pow(M, Y - 4.0) + stable_pow(G, Y - 4.0))
            elif ctype == "gbm":
                vol = float(p.get("vol", 0.0))
                # X is Brownian with zero drift.
                k2 += (vol ** 2) * T
            else:
                raise RuntimeError(f"Unhandled component type: {ctype}")

        # Martingale drift correction from combined exponent.
        # psi(-i) should be real for valid parameter sets; strip tiny numerical imaginary parts.
        psi_minus_i = complex(self._psi_unit(-1j))
        psi_minus_i_real = float(np.real_if_close(psi_minus_i, tol=1e6))
        mu = np.log(self.S0) + (self.r - self.q) * T + sum_log - psi_minus_i_real * T
        mu = float(np.real_if_close(mu, tol=1e6))

        c1 = float(mu + k1)
        c2 = float(k2 + var_div)
        c4 = float(k4)
        return c1, c2, c4

    def _var2(self, T: float) -> float:
        # Keep consistency with existing jump models: exclude dividend variance.
        _, c2, _ = self.cumulants(T)
        sum_log, div_params = _dividend_adjustment(T, self.divs)
        var_div = float(np.sum(div_params[:, 1])) if div_params.size else 0.0
        return float(max(c2 - var_div, 0.0))


