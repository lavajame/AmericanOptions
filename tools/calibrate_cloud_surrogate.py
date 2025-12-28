"""Hermite-RBF surrogate calibration for an American option quote cloud.

This tool fits a (model params, q) set to a cloud of American option prices by:
- computing COS-American prices and analytic sensitivities (softmax rollback)
- forming a scalar loss (default: weighted log-price L2)
- iteratively fitting a Hermite RBF surrogate on (loss, gradient)
- proposing new candidates by minimizing the surrogate (multi-start L-BFGS-B)

It is intentionally lightweight and self-contained.

CSV format
----------
The input CSV is expected to contain at least:
- K: strike
- T: maturity (year fraction)
- is_call: 1/0 or True/False
- price: market price
Optional:
- weight: per-quote weight (default 1.0)

Example
-------
python tools/calibrate_cloud_surrogate.py --synthetic --case merton_vg_q

python tools/calibrate_cloud_surrogate.py --quotes quotes.csv --case merton_vg_q \
  --S0 100 --r 0.02 --div "0.15:2.5:0" --steps 40 --N 512 --L 10 --beta 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.optimize as opt

# Ensure repo root is on sys.path when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, CompositeLevyCHF  # noqa: E402


def _option_label(is_call: bool) -> str:
    return "C" if bool(is_call) else "P"


def _pv_cash_divs(*, divs: dict[float, tuple[float, float]], r: float, T: float) -> float:
    pv = 0.0
    for t, (cash, _prop) in divs.items():
        if float(t) <= float(T):
            pv += float(cash) * float(np.exp(-float(r) * float(t)))
    return float(pv)


def _forward_proxy(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], T: float) -> float:
    # Simple proxy forward used only for plotting x = log-moneyness / sqrt(T).
    # We treat cash dividends as deterministic and subtract their PV.
    prepaid = float(S0) - _pv_cash_divs(divs=divs, r=float(r), T=float(T))
    return float(prepaid * np.exp((float(r) - float(q)) * float(T)))


def _bs_price_forward(*, F: float, K: float, df: float, sigma: float, T: float, is_call: bool) -> float:
    # Black-Scholes price expressed in terms of forward F and discount factor df.
    F = float(F)
    K = float(K)
    df = float(df)
    T = float(T)
    sigma = max(float(sigma), 1e-12)
    if T <= 0.0:
        intrinsic = max(F - K, 0.0) if bool(is_call) else max(K - F, 0.0)
        return df * intrinsic

    from math import erf, log, sqrt

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    vol_sqrt = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    if bool(is_call):
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _bs_implied_vol_forward(*, price: float, F: float, K: float, df: float, T: float, is_call: bool) -> float:
    # Fast implied vol inversion against European BS (forward form).
    price = float(price)
    F = float(F)
    K = float(K)
    df = float(df)
    T = float(T)

    intrinsic = df * (max(F - K, 0.0) if bool(is_call) else max(K - F, 0.0))
    upper = df * (F if bool(is_call) else K)
    if not (intrinsic - 1e-12 <= price <= upper + 1e-12):
        return float("nan")

    lo, hi = 1e-6, 5.0

    def f(sig: float) -> float:
        return _bs_price_forward(F=F, K=K, df=df, sigma=sig, T=T, is_call=bool(is_call)) - price

    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0.0 and tries < 10:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0.0:
        return float("nan")

    sol = opt.root_scalar(f, bracket=(lo, hi), method="brentq", xtol=1e-10, rtol=1e-10, maxiter=200)
    if not sol.converged:
        return float("nan")
    return float(sol.root)


@dataclass(frozen=True)
class OptionQuote:
    K: float
    T: float
    is_call: bool
    price: float
    weight: float = 1.0


def _parse_bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean: {x!r}")


def load_quotes_csv(path: str) -> list[OptionQuote]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        cols = {c.strip() for c in reader.fieldnames}
        required = {"K", "T", "is_call", "price"}
        missing = sorted(required - cols)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        out: list[OptionQuote] = []
        for row in reader:
            K = float(row["K"])
            T = float(row["T"])
            is_call = _parse_bool(row["is_call"])
            price = float(row["price"])
            w = float(row.get("weight", "1.0"))
            out.append(OptionQuote(K=K, T=T, is_call=is_call, price=price, weight=w))

    if len(out) == 0:
        raise ValueError("No quotes loaded")
    return out


def parse_divs(specs: Iterable[str]) -> dict[float, tuple[float, float]]:
    """Parse dividend specs of the form "t:D:std"."""
    divs: dict[float, tuple[float, float]] = {}
    for s in specs:
        s = str(s).strip()
        if not s:
            continue
        parts = s.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Bad div spec {s!r}, expected t:D[:std]")
        t = float(parts[0])
        D = float(parts[1])
        std = float(parts[2]) if len(parts) == 3 else 0.0
        divs[float(t)] = (float(D), float(std))
    return divs


def _format_phys(*, phys: dict[str, float], names: list[str]) -> str:
    parts: list[str] = []
    for k in names:
        if k in phys:
            parts.append(f"{k}={phys[k]: .6g}")
    return " ".join(parts)


class TraceWriter:
    def __init__(self, path: str | None, *, phys_names: list[str], z_dim: int):
        self.path = path
        self.phys_names = list(phys_names)
        self.z_dim = int(z_dim)
        self._fh = None
        self._writer = None

        if self.path is not None:
            self._fh = open(self.path, "w", newline="")
            fieldnames = [
                "stage",
                "iter",
                "tag",
                "f",
            ]
            fieldnames += [f"z{i}" for i in range(self.z_dim)]
            fieldnames += self.phys_names
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
            self._writer.writeheader()

    def write(self, *, stage: str, it: int, tag: str, f: float, z: np.ndarray, phys: dict[str, float]) -> None:
        if self._writer is None:
            return
        z = np.asarray(z, dtype=float).reshape((-1,))
        row: dict[str, object] = {
            "stage": str(stage),
            "iter": int(it),
            "tag": str(tag),
            "f": float(f),
        }
        for i in range(self.z_dim):
            row[f"z{i}"] = float(z[i])
        for k in self.phys_names:
            row[k] = float(phys.get(k, np.nan))
        self._writer.writerow(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


def _softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse of softplus for y>0 (stable for large y)."""
    y = np.asarray(y, dtype=float)
    # y = log(1+exp(x)) => exp(y) - 1 = exp(x) => x = log(exp(y)-1)
    # Use: log(expm1(y)) when y is small; y + log1p(-exp(-y)) when y large.
    out = np.empty_like(y)
    small = y < 20.0
    out[small] = np.log(np.expm1(y[small]))
    out[~small] = y[~small] + np.log1p(-np.exp(-y[~small]))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class ParamTransform:
    """Unconstrained vector z -> physical params (including q) with Jacobian diag."""

    def __init__(self, case: str):
        self.case = str(case).lower().strip()
        if self.case not in {"merton_vg_q", "merton_q"}:
            raise ValueError("Supported cases: merton_q, merton_vg_q")

        if self.case == "merton_q":
            # z ordering (unconstrained):
            # 0: q
            # 1: log(Merton.sigma)
            # 2: log(Merton.lam)
            # 3: Merton.muJ
            # 4: log(Merton.sigmaJ)
            self.names_phys = [
                "q",
                "Merton.sigma",
                "Merton.lam",
                "Merton.muJ",
                "Merton.sigmaJ",
            ]
        else:
            # z ordering (unconstrained):
            # 0: q
            # 1: log(Merton.sigma)
            # 2: log(Merton.lam)
            # 3: Merton.muJ
            # 4: log(Merton.sigmaJ)
            # 5: VG.theta_free (used to enforce VG mgf constraint at u=-i)
            # 6: log(VG.sigma)
            # 7: log(VG.nu)
            self.names_phys = [
                "q",
                "Merton.sigma",
                "Merton.lam",
                "Merton.muJ",
                "Merton.sigmaJ",
                "VG.theta",
                "VG.sigma",
                "VG.nu",
            ]

    @property
    def dim(self) -> int:
        return len(self.names_phys)

    def z_to_phys(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=float).reshape((-1,))
        if z.size != self.dim:
            raise ValueError(f"z has wrong dim: {z.size}, expected {self.dim}")

        q = float(z[0])
        m_sigma = float(np.exp(z[1]))
        m_lam = float(np.exp(z[2]))
        m_muJ = float(z[3])
        m_sigmaJ = float(np.exp(z[4]))

        out = {
            "q": q,
            "Merton.sigma": m_sigma,
            "Merton.lam": m_lam,
            "Merton.muJ": m_muJ,
            "Merton.sigmaJ": m_sigmaJ,
        }

        if self.case == "merton_q":
            return out

        # VG params: theta is just z[5] directly (no constraint)
        vg_theta = float(z[5])
        vg_sigma = float(np.exp(z[6]))
        vg_nu = float(np.exp(z[7]))
        out["VG.theta"] = vg_theta
        out["VG.sigma"] = vg_sigma
        out["VG.nu"] = vg_nu
        return out

    def phys_and_jac(self, z: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        """Return (phys_dict, J) where J[i,j] = d phys_i / d z_j.

        phys_i ordering matches self.names_phys.
        """
        z = np.asarray(z, dtype=float).reshape((-1,))
        if z.size != self.dim:
            raise ValueError(f"z has wrong dim: {z.size}, expected {self.dim}")

        phys = self.z_to_phys(z)
        J = np.zeros((self.dim, self.dim), dtype=float)

        # q
        J[0, 0] = 1.0
        # log-positive params
        J[1, 1] = float(phys["Merton.sigma"])  # d exp / d log = exp
        J[2, 2] = float(phys["Merton.lam"])
        J[3, 3] = 1.0
        J[4, 4] = float(phys["Merton.sigmaJ"])

        if self.case == "merton_q":
            return phys, J

        # VG sigma/nu
        vg_sigma = float(phys["VG.sigma"])
        vg_nu = float(phys["VG.nu"])
        J[6, 6] = vg_sigma
        J[7, 7] = vg_nu

        # VG theta is just z[5] directly (no transform)
        J[5, 5] = 1.0

        return phys, J

    def components_from_phys(self, phys: dict[str, float]) -> list[dict]:
        comps = [
            {
                "type": "merton",
                "params": {
                    "sigma": float(phys["Merton.sigma"]),
                    "lam": float(phys["Merton.lam"]),
                    "muJ": float(phys["Merton.muJ"]),
                    "sigmaJ": float(phys["Merton.sigmaJ"]),
                },
            },
        ]

        if self.case == "merton_q":
            return comps

        comps.append(
            {
                "type": "vg",
                "params": {
                    "theta": float(phys["VG.theta"]),
                    "sigma": float(phys["VG.sigma"]),
                    "nu": float(phys["VG.nu"]),
                },
            }
        )
        return comps

    def sens_param_names(self) -> list[str]:
        # CompositeLevyCHF uses these labels
        if self.case == "merton_q":
            return [
                "q",
                "Merton.sigma",
                "Merton.lam",
                "Merton.muJ",
                "Merton.sigmaJ",
            ]

        return [
            "q",
            "Merton.sigma",
            "Merton.lam",
            "Merton.muJ",
            "Merton.sigmaJ",
            "VG.theta",
            "VG.sigma",
            "VG.nu",
        ]


def _group_quotes(quotes: list[OptionQuote]) -> dict[tuple[float, bool], list[int]]:
    groups: dict[tuple[float, bool], list[int]] = {}
    for i, q in enumerate(quotes):
        key = (float(q.T), bool(q.is_call))
        groups.setdefault(key, []).append(i)
    return groups


class CloudObjective:
    def __init__(
        self,
        *,
        transform: ParamTransform,
        quotes: list[OptionQuote],
        S0: float,
        r: float,
        divs: dict[float, tuple[float, float]],
        N: int,
        L: float,
        steps: int,
        beta: float,
        eps_price: float,
        sigma_floor: float,
        sigma_penalty: float,
        sigma_penalty_scale: float,
        sens_method: str,
        use_european: bool = False,
        otm_threshold: float = 0.0,
    ):
        self.transform = transform
        self.quotes = quotes
        self.S0 = float(S0)
        self.r = float(r)
        self.divs = dict(divs)
        self.N = int(N)
        self.L = float(L)
        self.steps = int(steps)
        self.beta = float(beta)
        self.eps_price = float(eps_price)
        self.sigma_floor = float(sigma_floor)
        self.sigma_penalty = float(sigma_penalty)
        self.sigma_penalty_scale = float(sigma_penalty_scale)
        self.sens_method = str(sens_method)
        self.use_european = bool(use_european)
        self.otm_threshold = float(otm_threshold)

        self._groups = _group_quotes(quotes)
        self._mkt_prices = np.array([q.price for q in quotes], dtype=float)
        self._weights = np.array([q.weight for q in quotes], dtype=float)
        
        # Compute OTM mask: deep OTM options have normalized |log-moneyness| > threshold
        # Normalized moneyness: log(K/F) / (sigma_est * sqrt(T))
        # Use a rough vol estimate (0.2) for filtering
        self._otm_mask = np.zeros(len(quotes), dtype=bool)
        if otm_threshold > 0.0:
            sigma_est = 0.2  # rough vol estimate for filtering
            for i, q in enumerate(quotes):
                F = _forward_proxy(S0=S0, r=r, q=0.0, divs=divs, T=q.T)  # use q=0 for forward approx
                log_m = np.log(q.K / F)
                normalized_m = abs(log_m / (sigma_est * np.sqrt(q.T)))
                self._otm_mask[i] = normalized_m > otm_threshold

    def _prices_and_sens(self, phys: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        q = float(phys["q"])
        model = CompositeLevyCHF(
            self.S0,
            self.r,
            q,
            self.divs,
            {"components": self.transform.components_from_phys(phys)},
        )
        pricer = COSPricer(model, N=self.N, L=self.L)

        sens_params = self.transform.sens_param_names()
        px = np.zeros(len(self.quotes), dtype=float)
        sens: dict[str, np.ndarray] = {p: np.zeros(len(self.quotes), dtype=float) for p in sens_params}

        if self.use_european:
            # European pricing (faster, for deep OTM where early exercise premium ~ 0)
            for (T, is_call), idxs in self._groups.items():
                Ks = np.array([self.quotes[i].K for i in idxs], dtype=float)
                prices_T, d_T = pricer.european_price(
                    Ks,
                    float(T),
                    is_call=bool(is_call),
                    return_sensitivities=True,
                    sens_method=self.sens_method,
                    sens_params=sens_params,
                )
                prices_T = np.asarray(prices_T, dtype=float)
                px[idxs] = prices_T
                for p in sens_params:
                    sens[p][idxs] = np.asarray(d_T[p], dtype=float)
        else:
            # American pricing (full model)
            for (T, is_call), idxs in self._groups.items():
                Ks = np.array([self.quotes[i].K for i in idxs], dtype=float)
                prices_T, d_T = pricer.american_price(
                    Ks,
                    float(T),
                    steps=self.steps,
                    is_call=bool(is_call),
                    use_softmax=True,
                    beta=self.beta,
                    return_sensitivities=True,
                    sens_method=self.sens_method,
                    sens_params=sens_params,
                )
                prices_T = np.asarray(prices_T, dtype=float)
                px[idxs] = prices_T
                for p in sens_params:
                    sens[p][idxs] = np.asarray(d_T[p], dtype=float)

        return px, sens

    def _sigma_floor_penalty(self, phys: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Return (penalty_value, dpenalty/dphys for affected params)."""
        if self.sigma_penalty <= 0.0 or self.sigma_floor <= 0.0:
            return 0.0, {}

        scale = max(self.sigma_penalty_scale, 1e-12)
        # Apply to any diffusion-vol keys present for the chosen case.
        affected = [k for k in ("Merton.sigma", "VG.sigma") if k in phys]

        val = 0.0
        grad: dict[str, float] = {}
        for name in affected:
            s = float(phys[name])
            x = (self.sigma_floor - s) / scale
            sp = float(_softplus(np.array([x]))[0])
            sig = float(_sigmoid(np.array([x]))[0])
            # penalty = w * softplus(x)^2
            val += self.sigma_penalty * (sp * sp)
            # d/ds softplus(x) = sigmoid(x) * d x/d s = sigmoid(x) * (-1/scale)
            d_sp_ds = sig * (-1.0 / scale)
            grad[name] = grad.get(name, 0.0) + self.sigma_penalty * (2.0 * sp * d_sp_ds)

        return float(val), grad

    def loss_and_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape((-1,))
        phys, J = self.transform.phys_and_jac(z)

        try:
            model_px, sens = self._prices_and_sens(phys)
        except Exception:
            # Invalid parameter region (e.g. moment/mgf conditions violated, numerics blew up).
            # Return a big penalty; keep gradient finite so optimizers/surrogates don't crash.
            return 1e12, np.zeros_like(z)

        eps = self.eps_price
        mkt = self._mkt_prices
        w = self._weights
        
        # If in European mode with OTM filtering, only use OTM options
        if self.use_european and self.otm_threshold > 0.0:
            mask = self._otm_mask
            model_px = model_px[mask]
            mkt = mkt[mask]
            w = w[mask]

        # residual in log space
        log_model = np.log(np.clip(model_px, 0.0, np.inf) + eps)
        log_mkt = np.log(np.clip(mkt, 0.0, np.inf) + eps)
        r = (log_model - log_mkt)

        # objective: 0.5 * sum (w*r)^2
        wr = w * r
        f = 0.5 * float(np.sum(wr * wr))

        # gradient w.r.t physical params
        inv = 1.0 / (model_px + eps)
        grad_phys: dict[str, float] = {}
        for p in self.transform.sens_param_names():
            dp = np.asarray(sens[p], dtype=float)
            if self.use_european and self.otm_threshold > 0.0:
                dp = dp[self._otm_mask]
            # d f / d p = sum_i (w_i^2 * r_i * d r_i/dp)
            # where d r_i/dp = (1/(px_i+eps)) * dpx_i/dp
            grad_phys[p] = float(np.sum((w * w) * r * inv * dp))

        # add sigma-floor penalty
        pen, dpen_dphys = self._sigma_floor_penalty(phys)
        f += float(pen)
        for name, dval in dpen_dphys.items():
            grad_phys[name] = grad_phys.get(name, 0.0) + float(dval)

        # chain to z: g_z = J^T * g_phys
        g_phys_vec = np.array([grad_phys.get(name, 0.0) for name in self.transform.names_phys], dtype=float)
        g = J.T @ g_phys_vec

        return float(f), g


class HermiteRBF:
    """Hermite RBF surrogate for scalar f(x) with gradient constraints.

    Supports two kernels:
    - 'gaussian': phi(r) = exp(-(eps*r)^2)
    - 'multiquadric': phi(r) = sqrt(eps^2 + r^2)  (linear asymptote)

    Interpolant:
        s(x) = sum_j a_j phi(||x-xj||) + sum_j b_j · grad_x phi(||x-xj||)
    Fit solves the Hermite system with Tikhonov regularization.
    """

    def __init__(self, *, eps: float, reg: float = 1e-10, kernel: str = "gaussian"):
        self.eps = float(eps)
        self.reg = float(reg)
        self.kernel = str(kernel).lower().strip()
        if self.kernel not in {"gaussian", "multiquadric"}:
            raise ValueError(f"kernel must be 'gaussian' or 'multiquadric', got {self.kernel}")
        self.X: np.ndarray | None = None
        self.a: np.ndarray | None = None
        self.b: np.ndarray | None = None  # (n,d)

    def fit(self, X: np.ndarray, f: np.ndarray, g: np.ndarray) -> "HermiteRBF":
        X = np.asarray(X, dtype=float)
        f = np.asarray(f, dtype=float).reshape((-1,))
        g = np.asarray(g, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, d = X.shape
        if f.shape != (n,):
            raise ValueError("f shape mismatch")
        if g.shape != (n, d):
            raise ValueError("g shape mismatch")

        diff = X[:, None, :] - X[None, :, :]  # (n,n,d)
        r2 = np.sum(diff * diff, axis=2)  # (n,n)
        e2 = self.eps * self.eps
        
        if self.kernel == "gaussian":
            # Gaussian: phi = exp(-(eps*r)^2)
            Phi = np.exp(-e2 * r2)  # (n,n)
            # grad: -2 eps^2 (x_i - x_j) phi
            grad = (-2.0 * e2) * diff * Phi[:, :, None]  # (n,n,d)
            # Hessian: (4 eps^4 (dx_l dx_k) - 2 eps^2 delta_lk) phi
            I = np.eye(d, dtype=float)
            outer = diff[:, :, :, None] * diff[:, :, None, :]  # (n,n,d,d)
            H_t = (4.0 * (e2 * e2) * outer - 2.0 * e2 * I[None, None, :, :]) * Phi[:, :, None, None]  # (n,n,d,d)
        else:
            # Multiquadric: phi = sqrt(eps^2 + r^2)
            Phi = np.sqrt(e2 + r2)  # (n,n)
            # grad: (x_i - x_j) / phi
            # handle r=0 case
            Phi_safe = np.where(Phi > 1e-16, Phi, 1e-16)
            grad = diff / Phi_safe[:, :, None]  # (n,n,d)
            # Hessian: (I / phi) - (dx_l dx_k) / phi^3
            I = np.eye(d, dtype=float)
            outer = diff[:, :, :, None] * diff[:, :, None, :]  # (n,n,d,d)
            Phi3 = Phi_safe[:, :, None, None] ** 3
            H_t = (I[None, None, :, :] / Phi_safe[:, :, None, None]) - (outer / Phi3)  # (n,n,d,d)

        G = grad.reshape((n, n * d))
        G2 = grad.transpose(0, 2, 1).reshape((n * d, n))
        H = H_t.transpose(0, 2, 1, 3).reshape((n * d, n * d))

        A = np.block([
            [Phi, G],
            [G2, H],
        ])

        y = np.concatenate([f, g.reshape((-1,))], axis=0)

        # regularize
        A = A + self.reg * np.eye(A.shape[0], dtype=float)

        try:
            coeff = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            coeff, *_ = np.linalg.lstsq(A, y, rcond=None)

        a = coeff[:n]
        b_flat = coeff[n:]
        b = b_flat.reshape((n, d))

        self.X = X
        self.a = a
        self.b = b
        return self

    def predict(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.X is None or self.a is None or self.b is None:
            raise RuntimeError("Model not fit")
        X = self.X
        a = self.a
        b = self.b
        x = np.asarray(x, dtype=float).reshape((-1,))
        n, d = X.shape
        if x.size != d:
            raise ValueError("x dim mismatch")

        diff = x[None, :] - X  # (n,d)
        r2 = np.sum(diff * diff, axis=1)  # (n,)
        e2 = self.eps * self.eps
        
        if self.kernel == "gaussian":
            # Gaussian
            phi = np.exp(-e2 * r2)  # (n,)
            grad_phi = (-2.0 * e2) * diff * phi[:, None]  # (n,d)
            # Hessian per-center: (d,d)
            I = np.eye(d, dtype=float)
            outer = diff[:, :, None] * diff[:, None, :]  # (n,d,d)
            H = (4.0 * (e2 * e2) * outer - 2.0 * e2 * I[None, :, :]) * phi[:, None, None]
        else:
            # Multiquadric
            phi = np.sqrt(e2 + r2)  # (n,)
            phi_safe = np.where(phi > 1e-16, phi, 1e-16)
            grad_phi = diff / phi_safe[:, None]  # (n,d)
            # Hessian per-center: (d,d)
            I = np.eye(d, dtype=float)
            outer = diff[:, :, None] * diff[:, None, :]  # (n,d,d)
            phi3 = phi_safe[:, None, None] ** 3
            H = (I[None, :, :] / phi_safe[:, None, None]) - (outer / phi3)

        f_hat = float(np.sum(a * phi) + np.sum(b * grad_phi))

        # grad: sum_j a_j grad_phi_j + sum_j H_j @ b_j
        grad_hat = np.sum(a[:, None] * grad_phi, axis=0) + np.sum(H @ b[:, :, None], axis=0).reshape((-1,))
        return f_hat, grad_hat


def _default_eps_from_X(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return 1.0
    # median pairwise distance
    diff = X[:, None, :] - X[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    vals = r[np.triu_indices(r.shape[0], k=1)]
    med = float(np.median(vals))
    if not np.isfinite(med) or med <= 0.0:
        return 1.0
    return 1.0 / med


def _make_synthetic_quotes(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], steps: int) -> tuple[list[OptionQuote], np.ndarray]:
    # truth for merton+vg
    vg_sigma = 0.12
    vg_nu = 0.22
    vg_theta = -0.20
    true_z = np.array([
        q,
        np.log(0.14),   # Merton.sigma
        np.log(0.8),    # Merton.lam
        -0.08,          # Merton.muJ
        np.log(0.20),   # Merton.sigmaJ
        vg_theta,       # VG.theta
        np.log(vg_sigma),   # VG.sigma
        np.log(vg_nu),      # VG.nu
    ], dtype=float)

    tfm = ParamTransform("merton_vg_q")
    phys = tfm.z_to_phys(true_z)
    model = CompositeLevyCHF(S0, r, phys["q"], divs, {"components": tfm.components_from_phys(phys)})
    pr = COSPricer(model, N=512, L=10.0)

    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    maturities = [0.25, 0.5, 0.75]

    quotes: list[OptionQuote] = []
    for T in maturities:
        for K in strikes:
            for is_call in (False, True):
                px = float(
                    pr.american_price(
                        np.array([float(K)]),
                        float(T),
                        steps=int(steps),
                        is_call=bool(is_call),
                        use_softmax=True,
                        beta=100.0,
                    )[0]
                )
                quotes.append(OptionQuote(K=float(K), T=float(T), is_call=bool(is_call), price=px, weight=1.0))

    return quotes, true_z


def _make_synthetic_quotes_kou(
    *,
    S0: float,
    r: float,
    q: float,
    divs: dict[float, tuple[float, float]],
    steps: int,
) -> tuple[list[OptionQuote], dict[str, float]]:
    """Generate synthetic prices from a Kou jump-diffusion (single component)."""
    kou_params = {
        "sigma": 0.16,
        "lam": 0.9,
        "p": 0.35,
        "eta1": 12.0,  # > 1 for finite E[e^{X}] at u=-i
        "eta2": 6.0,
    }

    model = CompositeLevyCHF(
        float(S0),
        float(r),
        float(q),
        dict(divs),
        {"components": [{"type": "kou", "params": kou_params}]},
    )
    pr = COSPricer(model, N=512, L=10.0)

    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    maturities = [0.25, 0.5, 0.75]

    quotes: list[OptionQuote] = []
    for T in maturities:
        for K in strikes:
            for is_call in (False, True):
                px = float(
                    pr.american_price(
                        np.array([float(K)]),
                        float(T),
                        steps=int(steps),
                        is_call=bool(is_call),
                        use_softmax=True,
                        beta=100.0,
                    )[0]
                )
                quotes.append(OptionQuote(K=float(K), T=float(T), is_call=bool(is_call), price=px, weight=1.0))

    return quotes, {"q": float(q), **kou_params}


def _initial_perturbation_design(
    *,
    z0: np.ndarray,
    bounds: list[tuple[float, float]],
    step: np.ndarray,
) -> list[np.ndarray]:
    """Return [z0, z0 +/- step_i e_i] clipped to bounds."""
    z0 = np.asarray(z0, dtype=float).reshape((-1,))
    step = np.asarray(step, dtype=float).reshape((-1,))
    if z0.shape != step.shape:
        raise ValueError("z0/step dim mismatch")

    out = [z0.copy()]
    for i in range(z0.size):
        lo, hi = bounds[i]
        for sgn in (-1.0, 1.0):
            z = z0.copy()
            z[i] = float(np.clip(z[i] + sgn * step[i], lo, hi))
            out.append(z)
    return out


def _minimize_surrogate(
    model: HermiteRBF,
    *,
    bounds: list[tuple[float, float]],
    x0s: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    best_x = None
    best_f = np.inf

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = model.predict(x)
        return f, g

    for x0 in x0s:
        res = opt.minimize(
            lambda x: fun(x)[0],
            np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            jac=lambda x: fun(x)[1],
            bounds=bounds,
            options={"maxiter": 250},
        )
        if float(res.fun) < best_f:
            best_f = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)

    if best_x is None:
        raise RuntimeError("Surrogate minimization failed")
    return best_x, float(best_f)


def _unit_box_trust_region_bounds(*, center: np.ndarray, radius: float) -> list[tuple[float, float]]:
    """L-infinity trust region bounds inside [0,1]^d."""
    c = np.asarray(center, dtype=float).reshape((-1,))
    r = float(radius)
    if r <= 0.0:
        return [(0.0, 1.0) for _ in range(c.size)]
    out: list[tuple[float, float]] = []
    for i in range(c.size):
        lo = max(0.0, float(c[i] - r))
        hi = min(1.0, float(c[i] + r))
        out.append((lo, hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", type=str, default=None, help="CSV file with quote cloud")
    ap.add_argument("--synthetic", action="store_true", help="Use a built-in synthetic quote cloud")
    ap.add_argument(
        "--synthetic-scenario",
        type=str,
        default="merton_vg_self",
        choices=["merton_vg_self", "kou_to_merton", "kou_to_merton_vg"],
        help="Synthetic scenario",
    )
    ap.add_argument("--case", type=str, default="merton_vg_q", help="Model case (currently: merton_vg_q)")

    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--r", type=float, default=0.02)
    ap.add_argument("--div", action="append", default=[], help='Dividend spec "t:D:std" (repeatable)')

    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--beta", type=float, default=100.0)
    ap.add_argument("--sens-method", type=str, default="analytic")

    ap.add_argument("--eps-price", type=float, default=1e-8, help="epsilon for log(price+eps)")

    ap.add_argument("--sigma-floor", type=float, default=0.03)
    ap.add_argument("--sigma-penalty", type=float, default=1.0)
    ap.add_argument("--sigma-penalty-scale", type=float, default=0.01)
    
    ap.add_argument(
        "--european-init",
        action="store_true",
        help="Phase 1: calibrate to deep OTM options using fast European pricing, then refine with American",
    )
    ap.add_argument(
        "--european-otm-threshold",
        type=float,
        default=0.08,
        help="In European init phase, use options with normalized |log-moneyness|/(σ√T) > threshold (default=0.08)",
    )
    ap.add_argument(
        "--european-n-iter",
        type=int,
        default=15,
        help="Number of surrogate iterations in European init phase (default=15)",
    )

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-init", type=int, default=12, help="initial random evaluations")
    ap.add_argument("--add-z0-perturbations", action="store_true", help="add structured +/- perturbations around z0")
    ap.add_argument("--z0", type=float, nargs="+", default=None, help="initial guess in z-space (len=dim)")
    ap.add_argument("--z0-step", type=float, nargs="+", default=None, help="per-dim perturbation step in z-space (len=dim)")
    ap.add_argument(
        "--z0-step-log",
        type=float,
        default=0.35,
        help="default z-step for log-space parameters (multiplicative perturbations in physical space)",
    )
    ap.add_argument(
        "--z0-step-linear",
        type=float,
        default=0.06,
        help="default z-step for linear parameters (e.g., muJ, theta_free)",
    )
    ap.add_argument(
        "--z0-step-q",
        type=float,
        default=0.005,
        help="default z-step for q when using structured perturbations",
    )
    ap.add_argument("--n-iter", type=int, default=25, help="surrogate iterations")
    ap.add_argument("--n-restarts", type=int, default=12, help="multistart restarts for surrogate minimization")

    ap.add_argument(
        "--trust-radius",
        type=float,
        default=0.25,
        help="trust-region radius in unit-box coords (0 disables; smaller => more local steps)",
    )

    ap.add_argument(
        "--trust-radius-min",
        type=float,
        default=0.02,
        help="minimum trust-region radius when shrinking is enabled",
    )
    ap.add_argument(
        "--trust-shrink",
        type=float,
        default=0.85,
        help="multiply trust radius by this factor when shrinking (on rejections or no-improve)",
    )

    ap.add_argument(
        "--accept-mult",
        type=float,
        default=20.0,
        help="acceptance gate: require f_hat <= accept_mult * best + accept_abs (disabled if <=0)",
    )
    ap.add_argument(
        "--accept-abs",
        type=float,
        default=0.0,
        help="acceptance gate additive slack in f_hat <= accept_mult * best + accept_abs",
    )
    ap.add_argument(
        "--accept-max-tries",
        type=int,
        default=5,
        help="max attempts to find an acceptable surrogate proposal (shrinks trust region between tries)",
    )

    ap.add_argument(
        "--no-improve-patience",
        type=int,
        default=2,
        help="number of consecutive non-improving iterations before shrinking the trust radius",
    )
    ap.add_argument(
        "--fallback-step-frac",
        type=float,
        default=0.0,
        help="on no-improve/rejection, step this fraction of trust radius along local surrogate descent (unit-box coords; 0 disables)",
    )

    ap.add_argument("--rbf-reg", type=float, default=1e-10)
    ap.add_argument(
        "--rbf-reg-small-n-mult",
        type=float,
        default=1e6,
        help="multiply rbf-reg by this factor when the number of fit points is small",
    )
    ap.add_argument(
        "--rbf-reg-small-n-threshold",
        type=int,
        default=0,
        help="if #fit points <= threshold, apply the small-n regularization multiplier (0 => auto)",
    )
    ap.add_argument("--rbf-eps", type=float, default=None, help="override RBF eps (else auto from data)")
    ap.add_argument(
        "--rbf-kernel",
        type=str,
        default="gaussian",
        choices=["gaussian", "multiquadric"],
        help="RBF kernel type: 'gaussian' (exp) or 'multiquadric' (sqrt, linear asymptote)",
    )

    ap.add_argument(
        "--use-lbfgsb",
        action="store_true",
        help="Use L-BFGS-B optimizer directly instead of surrogate-based optimization",
    )
    ap.add_argument(
        "--lbfgsb-maxiter",
        type=int,
        default=500,
        help="Max iterations for L-BFGS-B per phase (default=500)",
    )

    ap.add_argument("--z-lb", type=float, nargs="+", default=None, help="lower bounds for z (len=dim)")
    ap.add_argument("--z-ub", type=float, nargs="+", default=None, help="upper bounds for z (len=dim)")

    ap.add_argument("--trace-csv", type=str, default=None, help="optional CSV path to write evaluation trace")

    ap.add_argument(
        "--iv-csv-out",
        type=str,
        default=None,
        help="optional CSV path to write target vs calibrated BS implied vols (from American prices)",
    )
    ap.add_argument(
        "--iv-plot-out",
        type=str,
        default=None,
        help="optional plot path (png) for target vs calibrated implied vols",
    )
    ap.add_argument(
        "--iv-plot-3d-out",
        type=str,
        default=None,
        help="optional 3D plot path (png) for target vs calibrated implied vols",
    )
    ap.add_argument(
        "--iv-plot-3d-html-out",
        type=str,
        default=None,
        help="optional interactive 3D plot path (html) for target vs calibrated implied vols (Plotly)",
    )
    ap.add_argument(
        "--iv-title",
        type=str,
        default=None,
        help="optional title for IV plot",
    )

    args = ap.parse_args()

    # Scenario-driven defaults
    if args.synthetic and args.synthetic_scenario == "kou_to_merton" and args.case == "merton_vg_q":
        args.case = "merton_q"

    tfm = ParamTransform(args.case)
    divs = parse_divs(args.div)

    if args.synthetic:
        if args.synthetic_scenario == "merton_vg_self":
            quotes, true_z = _make_synthetic_quotes(S0=args.S0, r=args.r, q=-0.005, divs=divs, steps=args.steps)
            print(f"Synthetic mode (merton_vg_self): {len(quotes)} quotes; true_z={true_z}")
        elif args.synthetic_scenario == "kou_to_merton_vg":
            quotes, kou_truth = _make_synthetic_quotes_kou(S0=args.S0, r=args.r, q=-0.005, divs=divs, steps=args.steps)
            true_z = None
            print(f"Synthetic mode (kou_to_merton_vg): {len(quotes)} quotes; kou_truth={kou_truth}")
        else:
            quotes, kou_truth = _make_synthetic_quotes_kou(S0=args.S0, r=args.r, q=-0.005, divs=divs, steps=args.steps)
            true_z = None
            print(f"Synthetic mode (kou_to_merton): {len(quotes)} quotes; kou_truth={kou_truth}")
    else:
        if args.quotes is None:
            raise ValueError("Provide --quotes or use --synthetic")
        quotes = load_quotes_csv(args.quotes)
        true_z = None
        print(f"Loaded {len(quotes)} quotes from {args.quotes}")

    dim = tfm.dim
    if args.z_lb is None:
        if tfm.case == "merton_q":
            z_lb = np.array([
                -0.10,   # q
                np.log(0.03),  # log sigma
                np.log(1e-4),  # log lam
                -0.5,    # muJ
                np.log(0.02),  # log sigmaJ
            ], dtype=float)
        else:
            z_lb = np.array([
                -0.10,   # q
                np.log(0.03),  # log sigma
                np.log(1e-4),  # log lam
                -0.5,    # muJ
                np.log(0.02),  # log sigmaJ
                -2.0,    # theta_free
                np.log(0.01),  # log vg_sigma
                np.log(1e-3),  # log nu
            ], dtype=float)
    else:
        z_lb = np.asarray(args.z_lb, dtype=float)
    if args.z_ub is None:
        if tfm.case == "merton_q":
            z_ub = np.array([
                0.10,     # q
                np.log(1.0),   # log sigma
                np.log(10.0),  # log lam
                0.5,      # muJ
                np.log(2.0),   # log sigmaJ
            ], dtype=float)
        else:
            z_ub = np.array([
                0.10,     # q
                np.log(1.0),   # log sigma
                np.log(10.0),  # log lam
                0.5,      # muJ
                np.log(2.0),   # log sigmaJ
                6.0,      # theta_free
                np.log(1.0),   # log vg_sigma
                np.log(5.0),   # log nu
            ], dtype=float)
    else:
        z_ub = np.asarray(args.z_ub, dtype=float)

    if z_lb.shape != (dim,) or z_ub.shape != (dim,):
        raise ValueError(f"Bounds must have len={dim} (got lb={z_lb.shape}, ub={z_ub.shape})")

    rng = np.random.default_rng(int(args.seed))

    tracer = TraceWriter(args.trace_csv, phys_names=tfm.names_phys, z_dim=tfm.dim)

    # Two-phase calibration setup
    if args.european_init:
        print(f"\n=== PHASE 1: European pricing on deep OTM (normalized |log-m|/(σ√T) > {args.european_otm_threshold}) ===")
        # Compute OTM count with normalized moneyness
        sigma_est = 0.2
        n_otm = sum(1 for q in quotes 
                    if abs(np.log(q.K / _forward_proxy(S0=args.S0, r=args.r, q=0.0, divs=divs, T=q.T)) / (sigma_est * np.sqrt(q.T))) > args.european_otm_threshold)
        
        if n_otm == 0:
            print(f"WARNING: No options qualify as deep OTM with threshold={args.european_otm_threshold}")
            print("Falling back to single-phase American calibration.")
            args.european_init = False
        else:
            print(f"Using {n_otm}/{len(quotes)} deep OTM options for initial calibration")
    
    if args.european_init:
        
        obj = CloudObjective(
            transform=tfm,
            quotes=quotes,
            S0=args.S0,
            r=args.r,
            divs=divs,
            N=args.N,
            L=args.L,
            steps=args.steps,
            beta=args.beta,
            eps_price=args.eps_price,
            sigma_floor=args.sigma_floor,
            sigma_penalty=args.sigma_penalty,
            sigma_penalty_scale=args.sigma_penalty_scale,
            sens_method=args.sens_method,
            use_european=True,
            otm_threshold=args.european_otm_threshold,
        )
        n_iter_phase1 = int(args.european_n_iter)
        n_iter_phase2 = int(args.n_iter)
    else:
        # Single-phase: American pricing on all options
        obj = CloudObjective(
            transform=tfm,
            quotes=quotes,
            S0=args.S0,
            r=args.r,
            divs=divs,
            N=args.N,
            L=args.L,
            steps=args.steps,
            beta=args.beta,
            eps_price=args.eps_price,
            sigma_floor=args.sigma_floor,
            sigma_penalty=args.sigma_penalty,
            sigma_penalty_scale=args.sigma_penalty_scale,
            sens_method=args.sens_method,
            use_european=False,
            otm_threshold=0.0,
        )
        n_iter_phase1 = int(args.n_iter)
        n_iter_phase2 = 0

    bounds = [(float(z_lb[i]), float(z_ub[i])) for i in range(dim)]

    def sample_z() -> np.ndarray:
        return z_lb + (z_ub - z_lb) * rng.random(dim)

    # Build / choose z0
    if args.z0 is not None:
        z0 = np.asarray(args.z0, dtype=float).reshape((-1,))
        if z0.shape != (dim,):
            raise ValueError(f"--z0 must have len={dim}")
    else:
        if tfm.case == "merton_q":
            z0 = np.array([
                0.0,
                np.log(0.10),
                np.log(0.5),
                -0.3,
                np.log(0.1),
            ], dtype=float)
        else:
            z0 = np.array([
                0.0,           # q
                np.log(0.05),  # Merton.sigma
                np.log(0.25),  # Merton.lam
                -0.2,          # Merton.muJ
                np.log(0.02),  # Merton.sigmaJ
                -0.2,          # VG.theta
                np.log(0.12),  # VG.sigma
                np.log(0.05),  # VG.nu
            ], dtype=float)
    z0 = np.clip(z0, z_lb, z_ub)

    # === L-BFGS-B PATH ===
    if args.use_lbfgsb:
        print("\n=== Using L-BFGS-B direct optimization ===")
        
        if args.european_init:
            print(f"\nPhase 1: L-BFGS-B on European objective")
            obj = CloudObjective(
                transform=tfm,
                quotes=quotes,
                S0=args.S0,
                r=args.r,
                divs=divs,
                N=args.N,
                L=args.L,
                steps=args.steps,
                beta=args.beta,
                eps_price=args.eps_price,
                sigma_floor=args.sigma_floor,
                sigma_penalty=args.sigma_penalty,
                sigma_penalty_scale=args.sigma_penalty_scale,
                sens_method=args.sens_method,
                use_european=True,
                otm_threshold=args.european_otm_threshold,
            )
            
            print(f"z0={z0}")
            phys0 = tfm.z_to_phys(z0)
            print("z0 physical params:")
            for k in tfm.names_phys:
                print(f"  {k:12s} = {phys0[k]: .6g}")
            
            # L-BFGS-B on European
            res_p1 = opt.minimize(
                lambda z: obj.loss_and_grad(z)[0],
                z0,
                method="L-BFGS-B",
                jac=lambda z: obj.loss_and_grad(z)[1],
                bounds=bounds,
                options={"maxiter": int(args.lbfgsb_maxiter), "disp": True},
            )
            z_best_p1 = np.asarray(res_p1.x, dtype=float)
            f_best_p1 = float(res_p1.fun)
            
            print(f"\nPhase 1 result: f={f_best_p1:.6e}")
            phys_p1 = tfm.z_to_phys(z_best_p1)
            for k in tfm.names_phys:
                print(f"  {k:12s} = {phys_p1[k]: .6g}")
            
            if n_iter_phase2 > 0:
                print(f"\nPhase 2: L-BFGS-B on American objective (warm-start from Phase 1)")
                obj_p2 = CloudObjective(
                    transform=tfm,
                    quotes=quotes,
                    S0=args.S0,
                    r=args.r,
                    divs=divs,
                    N=args.N,
                    L=args.L,
                    steps=args.steps,
                    beta=args.beta,
                    eps_price=args.eps_price,
                    sigma_floor=args.sigma_floor,
                    sigma_penalty=args.sigma_penalty,
                    sigma_penalty_scale=args.sigma_penalty_scale,
                    sens_method=args.sens_method,
                    use_european=False,
                    otm_threshold=0.0,
                )
                
                res_p2 = opt.minimize(
                    lambda z: obj_p2.loss_and_grad(z)[0],
                    z_best_p1,
                    method="L-BFGS-B",
                    jac=lambda z: obj_p2.loss_and_grad(z)[1],
                    bounds=bounds,
                    options={"maxiter": int(args.lbfgsb_maxiter), "disp": True},
                )
                best_z = np.asarray(res_p2.x, dtype=float)
            else:
                best_z = z_best_p1
        else:
            # Single-phase: L-BFGS-B on American
            print(f"\nL-BFGS-B on American objective")
            obj = CloudObjective(
                transform=tfm,
                quotes=quotes,
                S0=args.S0,
                r=args.r,
                divs=divs,
                N=args.N,
                L=args.L,
                steps=args.steps,
                beta=args.beta,
                eps_price=args.eps_price,
                sigma_floor=args.sigma_floor,
                sigma_penalty=args.sigma_penalty,
                sigma_penalty_scale=args.sigma_penalty_scale,
                sens_method=args.sens_method,
                use_european=False,
                otm_threshold=0.0,
            )
            
            print(f"z0={z0}")
            phys0 = tfm.z_to_phys(z0)
            print("z0 physical params:")
            for k in tfm.names_phys:
                print(f"  {k:12s} = {phys0[k]: .6g}")
            
            res = opt.minimize(
                lambda z: obj.loss_and_grad(z)[0],
                z0,
                method="L-BFGS-B",
                jac=lambda z: obj.loss_and_grad(z)[1],
                bounds=bounds,
                options={"maxiter": int(args.lbfgsb_maxiter), "disp": True},
            )
            best_z = np.asarray(res.x, dtype=float)
        
        best_phys = tfm.z_to_phys(best_z)
        print("\nBest z:")
        print(best_z)
        print("Best physical params:")
        for k in tfm.names_phys:
            print(f"  {k:12s} = {best_phys[k]: .6g}")
        
        tracer.close()
        
        # IV diagnostics (same as before)
        if (
            args.iv_csv_out is not None
            or args.iv_plot_out is not None
            or args.iv_plot_3d_out is not None
            or args.iv_plot_3d_html_out is not None
        ):
            try:
                if args.synthetic and args.synthetic_scenario in ("merton_vg_self", "kou_to_merton", "kou_to_merton_vg"):
                    q_target = -0.005
                else:
                    q_target = float(best_phys["q"])

                model_best = CompositeLevyCHF(
                    float(args.S0),
                    float(args.r),
                    float(best_phys["q"]),
                    dict(divs),
                    {"components": tfm.components_from_phys(best_phys)},
                )
                pr_best = COSPricer(model_best, N=int(args.N), L=float(args.L))

                px_fit = np.zeros(len(quotes), dtype=float)
                groups = _group_quotes(quotes)
                for (T, is_call), idxs in groups.items():
                    Ks = np.array([quotes[i].K for i in idxs], dtype=float)
                    px_T = pr_best.american_price(
                        Ks,
                        float(T),
                        steps=int(args.steps),
                        is_call=bool(is_call),
                        use_softmax=True,
                        beta=float(args.beta),
                    )
                    px_fit[idxs] = np.asarray(px_T, dtype=float)

                xs: list[float] = []
                iv_target: list[float] = []
                iv_fit: list[float] = []
                Ts: list[float] = []
                Ks: list[float] = []
                types: list[str] = []

                # Use American prices with target q for IV
                obj_iv = CloudObjective(
                    transform=tfm,
                    quotes=quotes,
                    S0=args.S0,
                    r=args.r,
                    divs=divs,
                    N=args.N,
                    L=args.L,
                    steps=args.steps,
                    beta=args.beta,
                    eps_price=args.eps_price,
                    sigma_floor=args.sigma_floor,
                    sigma_penalty=args.sigma_penalty,
                    sigma_penalty_scale=args.sigma_penalty_scale,
                    sens_method=args.sens_method,
                    use_european=False,
                    otm_threshold=0.0,
                )
                mkt_prices = obj_iv._mkt_prices

                for qte, px_mkt, px_mod in zip(quotes, mkt_prices.tolist(), px_fit.tolist()):
                    T = float(qte.T)
                    F = _forward_proxy(S0=float(args.S0), r=float(args.r), q=float(q_target), divs=divs, T=T)
                    x = float(np.log(float(qte.K) / F) / np.sqrt(T))
                    df = float(np.exp(-float(args.r) * T))

                    iv_mkt = _bs_implied_vol_forward(price=px_mkt, F=F, K=float(qte.K), df=df, T=T, is_call=bool(qte.is_call))
                    iv_mod = _bs_implied_vol_forward(price=px_mod, F=F, K=float(qte.K), df=df, T=T, is_call=bool(qte.is_call))

                    xs.append(x)
                    iv_target.append(iv_mkt)
                    iv_fit.append(iv_mod)
                    Ts.append(T)
                    Ks.append(float(qte.K))
                    types.append(_option_label(bool(qte.is_call)))

                if args.iv_csv_out is not None:
                    os.makedirs(os.path.dirname(args.iv_csv_out) or ".", exist_ok=True)
                    with open(args.iv_csv_out, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["T", "K", "type", "x_logmoneyness_sqrtT", "iv_target", "iv_fit", "price_target", "price_fit"])
                        for T, K, typ, x, ivt, ivf, pt, pf in zip(
                            Ts,
                            Ks,
                            types,
                            mkt_prices.tolist(),
                            iv_fit,
                            iv_fit,
                            mkt_prices.tolist(),
                            px_fit.tolist(),
                        ):
                            w.writerow([T, K, typ, x, ivt, ivf, pt, pf])

                if args.iv_plot_out is not None:
                    os.makedirs(os.path.dirname(args.iv_plot_out) or ".", exist_ok=True)
                    fig, ax = plt.subplots(figsize=(8.5, 5.0))
                    for T in sorted(set(Ts)):
                        idxs = [i for i, t in enumerate(Ts) if abs(float(t) - float(T)) < 1e-12]
                        if len(idxs) < 2:
                            continue
                        idxs_sorted = sorted(idxs, key=lambda i: xs[i])
                        x_line = [xs[i] for i in idxs_sorted]
                        y_line = [iv_target[i] for i in idxs_sorted]
                        ax.plot(x_line, y_line, color="red", alpha=0.5, linewidth=2.5)

                    ax.scatter(
                        xs,
                        iv_target,
                        s=40,
                        marker="o",
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.2,
                        label="Target",
                    )
                    ax.scatter(
                        xs,
                        iv_fit,
                        s=40,
                        marker="x",
                        color="blue",
                        linewidths=1.4,
                        label="Calibrated",
                    )

                    ax.set_xlabel(r"time-normalized log-moneyness $x = \ln(K/F)/\sqrt{T}$")
                    ax.set_ylabel("BS implied vol (from American price)")
                    if args.iv_title is not None:
                        ax.set_title(str(args.iv_title))
                    else:
                        ax.set_title(f"Targets vs fit ({tfm.case})")
                    ax.grid(True, alpha=0.3)
                    ax.legend(ncol=2, fontsize=9)
                    fig.tight_layout()
                    fig.savefig(args.iv_plot_out, dpi=180)
                    plt.close(fig)

                if args.iv_plot_3d_html_out is not None:
                    try:
                        import plotly.graph_objects as go
                    except Exception as e:
                        raise RuntimeError(
                            "Plotly is required for --iv-plot-3d-html-out. Install with: pip install plotly"
                        ) from e

                    hover = [
                        f"T={T:.4g}<br>K={K:.4g}<br>type={typ}<br>px_target={pt:.6g}<br>px_fit={pf:.6g}<br>iv_target={ivt:.6g}<br>iv_fit={ivf:.6g}"
                        for (T, K, typ, pt, pf, ivt, ivf) in zip(
                            Ts,
                            Ks,
                            types,
                            mkt_prices.tolist(),
                            px_fit.tolist(),
                            iv_target,
                            iv_fit,
                        )
                    ]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=Ts,
                            z=iv_target,
                            mode="markers",
                            name="Target",
                            marker=dict(symbol="circle-open", size=5, color="red", line=dict(width=2, color="red")),
                            text=hover,
                            hoverinfo="text",
                        )
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=Ts,
                            z=iv_fit,
                            mode="markers",
                            name="Calibrated",
                            marker=dict(symbol="x", size=5, color="blue"),
                            text=hover,
                            hoverinfo="text",
                        )
                    )

                    title = str(args.iv_title) if args.iv_title is not None else f"Targets vs fit ({tfm.case})"
                    fig.update_layout(
                        title=title,
                        scene=dict(
                            xaxis_title=r"x = ln(K/F)/sqrt(T)",
                            yaxis_title="T",
                            zaxis_title="BS implied vol (from American price)",
                            aspectmode="cube",
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                        margin=dict(l=0, r=0, b=0, t=40),
                    )

                    os.makedirs(os.path.dirname(args.iv_plot_3d_html_out) or ".", exist_ok=True)
                    fig.write_html(args.iv_plot_3d_html_out, include_plotlyjs="cdn")

                print("-")
                if args.iv_plot_out is not None:
                    print(f"Saved IV plot: {args.iv_plot_out}")
                if args.iv_plot_3d_html_out is not None:
                    print(f"Saved IV 3D HTML plot: {args.iv_plot_3d_html_out}")
                if args.iv_csv_out is not None:
                    print(f"Saved IV data: {args.iv_csv_out}")
            except Exception as e:
                print("-")
                print(f"IV diagnostics skipped: {type(e).__name__}: {e}")

        return 0

    # === SURROGATE PATH (original) ===
    X: list[np.ndarray] = []
    f_list: list[float] = []
    g_list: list[np.ndarray] = []

    best = (None, np.inf)

    trust_radius_cur = float(args.trust_radius)
    no_improve_streak = 0

    # Setup z0_step for structured perturbations
    if args.z0_step is not None:
        z0_step = np.asarray(args.z0_step, dtype=float).reshape((-1,))
        if z0_step.shape != (dim,):
            raise ValueError(f"--z0-step must have len={dim}")
    else:
        z0_step = np.full(dim, float(args.z0_step_linear), dtype=float)
        z0_step[0] = float(args.z0_step_q)
        if tfm.case == "merton_q":
            log_idxs = [1, 2, 4]
        else:
            log_idxs = [1, 2, 4, 6, 7]
        for i in log_idxs:
            if 0 <= i < dim:
                z0_step[i] = float(args.z0_step_log)
        if dim >= 4:
            z0_step[3] = float(args.z0_step_linear)
    z0_step = np.maximum(z0_step, 1e-12)

    print("Evaluating initial design...")
    print(f"Using case={tfm.case}, dim={dim}")
    print(f"z0={z0}")
    phys0 = tfm.z_to_phys(z0)
    print("z0 physical params:")
    for k in tfm.names_phys:
        print(f"  {k:12s} = {phys0[k]: .6g}")
    if args.trace_csv is not None:
        print(f"Tracing evaluations to CSV: {args.trace_csv}")

    if args.add_z0_perturbations:
        pts = _initial_perturbation_design(z0=z0, bounds=bounds, step=z0_step)
        for j, z in enumerate(pts):
            f, g = obj.loss_and_grad(z)
            X.append(z)
            f_list.append(float(f))
            g_list.append(np.asarray(g, dtype=float))
            if float(f) < best[1]:
                best = (z.copy(), float(f))
            phys = tfm.z_to_phys(z)
            print(f"  [z0-pert {j:02d}] f={float(f):.6e}  {_format_phys(phys=phys, names=tfm.names_phys)}")
            tracer.write(stage="init", it=0, tag=f"z0-pert-{j:02d}", f=float(f), z=z, phys=phys)

    for _ in range(int(args.n_init)):
        z = sample_z()
        f, g = obj.loss_and_grad(z)
        X.append(z)
        f_list.append(float(f))
        g_list.append(np.asarray(g, dtype=float))
        if float(f) < best[1]:
            best = (z.copy(), float(f))
        phys = tfm.z_to_phys(z)
        print(f"  [rand] f={float(f):.6e}  {_format_phys(phys=phys, names=tfm.names_phys)}")
        tracer.write(stage="init", it=0, tag="rand", f=float(f), z=z, phys=phys)

    for it in range(n_iter_phase1):
        X_arr = np.vstack(X)
        f_arr = np.asarray(f_list, dtype=float)
        g_arr = np.vstack(g_list)

        # Scale into a fixed unit box based on *global* bounds.
        # This avoids restricting the surrogate search to the current sample min/max,
        # which can cause certain params (e.g., muJ) to get stuck at an explored edge.
        X_min = z_lb
        span = np.maximum(z_ub - z_lb, 1e-12)
        Xn = (X_arr - X_min) / span

        eps = float(args.rbf_eps) if args.rbf_eps is not None else _default_eps_from_X(Xn)

        # With very few Hermite constraints, the system can be ill-conditioned and the surrogate can
        # develop extreme curvature. Use a stronger Tikhonov (L2) penalty early.
        n_fit = int(X_arr.shape[0])
        threshold = int(args.rbf_reg_small_n_threshold)
        if threshold <= 0:
            # Heuristic: Hermite constraints are strong; in practice you want many more than ~2*dim
            # points before trusting the interpolant. Use a conservative default.
            threshold = max(20, 4 * dim)

        reg_eff = float(args.rbf_reg)
        if n_fit <= threshold:
            mult = float(args.rbf_reg_small_n_mult)
            # Smooth boost: stronger when n_fit is far below threshold.
            reg_eff *= mult * (float(threshold) / max(float(n_fit), 1.0))

        if n_fit <= threshold and it == 0:
            print(f"RBF reg boost active: n_fit={n_fit} <= {threshold}, reg={reg_eff:.3e}")

        rbf = HermiteRBF(eps=eps, reg=reg_eff, kernel=args.rbf_kernel).fit(Xn, f_arr, g_arr * span[None, :])

        # multistart on surrogate (optionally trust-regioned around current best)
        if best[0] is not None:
            best_xn = (np.asarray(best[0], dtype=float) - X_min) / span
            best_xn = np.clip(best_xn, 0.0, 1.0)
        else:
            best_xn = None

        def _is_acceptable_hat(fhat: float, best_true: float) -> bool:
            if not np.isfinite(fhat):
                return False
            mult = float(args.accept_mult)
            if mult <= 0.0 or not np.isfinite(best_true):
                return True
            threshold = mult * float(best_true) + float(args.accept_abs)
            return fhat <= threshold

        def _fallback_step_from_best(*, local_bounds: list[tuple[float, float]]) -> tuple[np.ndarray, float] | None:
            """Return (x_fallback, f_hat_best) or None if fallback is disabled/unavailable."""
            step_frac = float(args.fallback_step_frac)
            if step_frac <= 0.0:
                return None
            if best_xn is None:
                return None
            # Use local surrogate descent direction at the incumbent best.
            f_best_hat, g_best_hat = rbf.predict(best_xn)
            g = np.asarray(g_best_hat, dtype=float).reshape((-1,))
            if not np.all(np.isfinite(g)):
                return None
            ng = float(np.linalg.norm(g))
            if ng <= 0.0:
                return None

            # Step length in unit box coordinates.
            tr = float(trust_radius_cur)
            if tr <= 0.0:
                tr = 0.10
            step = step_frac * tr
            d = (-g) / ng
            x = np.asarray(best_xn, dtype=float).copy() + step * d
            # Clip to chosen bounds (trust region) rather than global [0,1]
            for i, (lo, hi) in enumerate(local_bounds):
                x[i] = float(np.clip(x[i], lo, hi))
            return x, float(f_best_hat)

        # Propose by surrogate minimization; optionally reject obviously-bad f_hat and
        # shrink the trust region before trying again. If we still can't get an acceptable
        # proposal, fall back to a modest local step from the current best.
        x_star_n = None
        f_hat = np.inf
        tries = max(int(args.accept_max_tries), 1)
        for k in range(tries):
            if best_xn is not None and trust_radius_cur > 0.0:
                local_bounds = _unit_box_trust_region_bounds(center=best_xn, radius=trust_radius_cur)
            else:
                local_bounds = [(0.0, 1.0)] * dim

            x0s: list[np.ndarray] = []
            if best_xn is not None:
                x0s.append(best_xn)
            # Sample restarts inside the chosen bounds
            for _ in range(int(args.n_restarts)):
                x = np.array([rng.uniform(lo, hi) for (lo, hi) in local_bounds], dtype=float)
                x0s.append(x)

            x_try, f_try = _minimize_surrogate(rbf, bounds=local_bounds, x0s=x0s)
            if _is_acceptable_hat(float(f_try), float(best[1])):
                x_star_n = x_try
                f_hat = float(f_try)
                break

            # Reject and shrink trust region
            prev = trust_radius_cur
            if trust_radius_cur > 0.0:
                trust_radius_cur = max(float(args.trust_radius_min), trust_radius_cur * float(args.trust_shrink))
            print(
                f"iter {it+1:03d}: rejected proposal (try {k+1}/{tries}) f_hat={float(f_try):.6e} "
                f"vs best={float(best[1]):.6e}; trust_radius {prev:.3g}->{trust_radius_cur:.3g}"
            )

            # If we're out of tries, prefer a local fallback step rather than
            # evaluating a proposal we already believe is bad.
            if k == tries - 1:
                fb = _fallback_step_from_best(local_bounds=local_bounds)
                if fb is not None:
                    x_star_n, f_best_hat = fb
                    f_hat = float(f_best_hat)
                    print(f"iter {it+1:03d}: using fallback local step from incumbent best")
                else:
                    x_star_n = x_try
                    f_hat = float(f_try)

        if x_star_n is None:
            raise RuntimeError("Failed to propose a surrogate candidate")

        z_star = X_min + span * np.asarray(x_star_n, dtype=float)

        best_before = float(best[1])
        f_true, g_true = obj.loss_and_grad(z_star)
        X.append(z_star)
        f_list.append(float(f_true))
        g_list.append(np.asarray(g_true, dtype=float))

        improved = False
        if float(f_true) < best[1]:
            best = (z_star.copy(), float(f_true))
            improved = True

        phys_star = tfm.z_to_phys(z_star)
        print(
            f"iter {it+1:03d}: f_hat={float(f_hat):.6e}, f_true={float(f_true):.6e}, best={best[1]:.6e}  "
            + _format_phys(phys=phys_star, names=tfm.names_phys)
        )
        tracer.write(stage="iter", it=it + 1, tag="surrogate", f=float(f_true), z=z_star, phys=phys_star)

        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1

            # First try a modest local step from the incumbent best (often helps when
            # the surrogate is overconfident far away).
            if best_xn is not None:
                if best_xn is not None and trust_radius_cur > 0.0:
                    local_bounds_now = _unit_box_trust_region_bounds(center=best_xn, radius=trust_radius_cur)
                else:
                    local_bounds_now = [(0.0, 1.0)] * dim
                fb = _fallback_step_from_best(local_bounds=local_bounds_now)
                if fb is not None:
                    x_fb, f_best_hat = fb
                    z_fb = X_min + span * x_fb
                    f_fb, g_fb = obj.loss_and_grad(z_fb)
                    X.append(z_fb)
                    f_list.append(float(f_fb))
                    g_list.append(np.asarray(g_fb, dtype=float))
                    if float(f_fb) < best[1]:
                        best = (z_fb.copy(), float(f_fb))
                        no_improve_streak = 0
                    phys_fb = tfm.z_to_phys(z_fb)
                    print(
                        f"iter {it+1:03d}: fallback f_hat(best)={float(f_best_hat):.6e}, f_true={float(f_fb):.6e}, best={best[1]:.6e}  "
                        + _format_phys(phys=phys_fb, names=tfm.names_phys)
                    )
                    tracer.write(stage="iter", it=it + 1, tag="fallback", f=float(f_fb), z=z_fb, phys=phys_fb)

            # Only shrink after some patience.
            patience = max(int(args.no_improve_patience), 1)
            if no_improve_streak >= patience:
                if trust_radius_cur > 0.0 and float(args.trust_shrink) < 1.0:
                    prev = trust_radius_cur
                    trust_radius_cur = max(float(args.trust_radius_min), trust_radius_cur * float(args.trust_shrink))
                    if trust_radius_cur != prev:
                        print(
                            f"iter {it+1:03d}: no improvement streak={no_improve_streak} (best {best_before:.6e}->{best[1]:.6e}); "
                            f"trust_radius {prev:.3g}->{trust_radius_cur:.3g}"
                        )

    # === PHASE 2: Refine with American pricing on all options ===
    if args.european_init and n_iter_phase2 > 0:
        print(f"\n=== PHASE 2: American pricing on ALL {len(quotes)} options (no filtering) ===")
        print(f"Starting from Phase 1 best: f={best[1]:.6e}")
        
        # Switch objective to American pricing on ALL quotes
        obj = CloudObjective(
            transform=tfm,
            quotes=quotes,
            S0=args.S0,
            r=args.r,
            divs=divs,
            N=args.N,
            L=args.L,
            steps=args.steps,
            beta=args.beta,
            eps_price=args.eps_price,
            sigma_floor=args.sigma_floor,
            sigma_penalty=args.sigma_penalty,
            sigma_penalty_scale=args.sigma_penalty_scale,
            sens_method=args.sens_method,
            use_european=False,
            otm_threshold=0.0,  # No filtering - use ALL quotes
        )
        
        # Find best Phase 1 point for American objective (not just Phase 1 best)
        # Phase 1 optimized for European on OTM, which may be bad for American on all options
        print(f"Evaluating top Phase 1 candidates with American objective...")
        X_phase1 = np.vstack(X)
        f_phase1 = np.array(f_list, dtype=float)
        k_candidates = min(5, len(X))  # check top-5 Phase 1 points
        best_idxs = np.argsort(f_phase1)[:k_candidates]
        
        best_american = (None, np.inf)
        for j in best_idxs:
            z = X_phase1[j]
            f_am, _ = obj.loss_and_grad(z)
            print(f"  Phase1 rank {j+1}: European f={f_phase1[j]:.6e} -> American f={f_am:.6e}")
            if f_am < best_american[1]:
                best_american = (z.copy(), f_am)
        
        z0_phase2 = best_american[0]
        print(f"\nSelected Phase 2 starting point: American f={best_american[1]:.6e}")
        phys_phase2 = tfm.z_to_phys(z0_phase2)
        for k in tfm.names_phys:
            print(f"  {k:12s} = {phys_phase2[k]: .6g}")
        
        X = []
        f_list = []
        g_list = []
        best = (None, np.inf)
        
        # Run perturbation design around Phase 1 best point with smaller steps
        print(f"Running perturbation design around Phase 1 best point (smaller steps for American refinement)...")
        z0_step_phase2 = z0_step * 0.3  # Use only 30% of Phase 1 step size for refinement
        pts = _initial_perturbation_design(z0=z0_phase2, bounds=bounds, step=z0_step_phase2)
        for j, z in enumerate(pts):
            f, g = obj.loss_and_grad(z)
            X.append(z)
            f_list.append(float(f))
            g_list.append(np.asarray(g, dtype=float))
            if float(f) < best[1]:
                best = (z.copy(), float(f))
            phys = tfm.z_to_phys(z)
            print(f"  [P2-init {j:02d}] f={float(f):.6e}  {_format_phys(phys=phys, names=tfm.names_phys)}")
            tracer.write(stage="phase2-init", it=0, tag=f"warm-start-{j:02d}", f=float(f), z=z, phys=phys)
        
        print(f"Phase 2 best after warm-start: f={best[1]:.6e}")
        
        # Phase 2 trust radius should be tighter (much smaller steps for refinement)
        trust_radius_cur = max(0.05, float(args.trust_radius) * 0.3)
        no_improve_streak = 0
        
        # Continue optimization with American pricing
        for it in range(n_iter_phase2):
            X_arr = np.vstack(X)
            f_arr = np.asarray(f_list, dtype=float)
            g_arr = np.vstack(g_list)

            X_min = z_lb
            span = np.maximum(z_ub - z_lb, 1e-12)
            Xn = (X_arr - X_min) / span

            eps = float(args.rbf_eps) if args.rbf_eps is not None else _default_eps_from_X(Xn)

            n_fit = int(X_arr.shape[0])
            threshold = int(args.rbf_reg_small_n_threshold)
            if threshold <= 0:
                threshold = max(20, 4 * dim)

            reg_eff = float(args.rbf_reg)
            if n_fit <= threshold:
                mult = float(args.rbf_reg_small_n_mult)
                ratio = float(n_fit) / float(max(threshold, 1))
                reg_eff *= (1.0 + mult * (1.0 - ratio))
            
            # Phase 2: use much stronger regularization for American pricing (less trusting of surrogate)
            reg_eff *= 100.0

            gn = g_arr / span[None, :]
            rbf = HermiteRBF(eps=eps, reg=reg_eff, kernel=args.rbf_kernel).fit(Xn, f_arr, gn)

            best_xn = None
            if best[0] is not None:
                best_xn = (best[0] - X_min) / span

            def _is_acceptable_hat(fhat: float, best_true: float) -> bool:
                mult = float(args.accept_mult)
                if mult <= 0.0:
                    return True
                threshold = mult * float(best_true) + float(args.accept_abs)
                return fhat <= threshold

            def _fallback_step_from_best(*, local_bounds: list[tuple[float, float]]) -> tuple[np.ndarray, float] | None:
                step_frac = float(args.fallback_step_frac)
                if step_frac <= 0.0:
                    return None
                if best_xn is None:
                    return None
                f_best_hat, g_best_hat = rbf.predict(best_xn)
                g = np.asarray(g_best_hat, dtype=float).reshape((-1,))
                if not np.all(np.isfinite(g)):
                    return None
                ng = float(np.linalg.norm(g))
                if ng <= 0.0:
                    return None

                tr = float(trust_radius_cur)
                if tr <= 0.0:
                    tr = 0.10
                step = step_frac * tr
                d = (-g) / ng
                x = np.asarray(best_xn, dtype=float).copy() + step * d
                for i, (lo, hi) in enumerate(local_bounds):
                    x[i] = float(np.clip(x[i], lo, hi))
                return x, float(f_best_hat)

            x_star_n = None
            f_hat = np.inf
            tries = max(int(args.accept_max_tries), 1)
            for k in range(tries):
                if best_xn is not None and trust_radius_cur > 0.0:
                    local_bounds = _unit_box_trust_region_bounds(center=best_xn, radius=trust_radius_cur)
                else:
                    local_bounds = [(0.0, 1.0)] * dim

                x0s: list[np.ndarray] = []
                if best_xn is not None:
                    x0s.append(best_xn)
                for _ in range(int(args.n_restarts)):
                    x = np.array([rng.uniform(lo, hi) for (lo, hi) in local_bounds], dtype=float)
                    x0s.append(x)

                x_try, f_try = _minimize_surrogate(rbf, bounds=local_bounds, x0s=x0s)
                if _is_acceptable_hat(float(f_try), float(best[1])):
                    x_star_n = x_try
                    f_hat = float(f_try)
                    break

                prev = trust_radius_cur
                if trust_radius_cur > 0.0:
                    trust_radius_cur = max(float(args.trust_radius_min), trust_radius_cur * float(args.trust_shrink))
                print(
                    f"[P2] iter {it+1:03d}: rejected proposal (try {k+1}/{tries}) f_hat={float(f_try):.6e} "
                    f"vs best={float(best[1]):.6e}; trust_radius {prev:.3g}->{trust_radius_cur:.3g}"
                )

                if k == tries - 1:
                    fb = _fallback_step_from_best(local_bounds=local_bounds)
                    if fb is not None:
                        x_star_n, f_best_hat = fb
                        f_hat = float(f_best_hat)
                        print(f"[P2] iter {it+1:03d}: using fallback local step from incumbent best")
                    else:
                        x_star_n = x_try
                        f_hat = float(f_try)

            if x_star_n is None:
                raise RuntimeError("Failed to propose a surrogate candidate")

            z_star = X_min + span * np.asarray(x_star_n, dtype=float)

            best_before = float(best[1])
            f_true, g_true = obj.loss_and_grad(z_star)
            X.append(z_star)
            f_list.append(float(f_true))
            g_list.append(np.asarray(g_true, dtype=float))

            improved = False
            if float(f_true) < best[1]:
                best = (z_star.copy(), float(f_true))
                improved = True

            phys_star = tfm.z_to_phys(z_star)
            print(
                f"[P2] iter {it+1:03d}: f_hat={float(f_hat):.6e}, f_true={float(f_true):.6e}, best={best[1]:.6e}  "
                + _format_phys(phys=phys_star, names=tfm.names_phys)
            )
            tracer.write(stage="phase2", it=it + 1, tag="surrogate", f=float(f_true), z=z_star, phys=phys_star)

            if improved:
                no_improve_streak = 0
            else:
                no_improve_streak += 1

                if best_xn is not None:
                    if best_xn is not None and trust_radius_cur > 0.0:
                        local_bounds_now = _unit_box_trust_region_bounds(center=best_xn, radius=trust_radius_cur)
                    else:
                        local_bounds_now = [(0.0, 1.0)] * dim
                    fb = _fallback_step_from_best(local_bounds=local_bounds_now)
                    if fb is not None:
                        x_fb, f_best_hat = fb
                        z_fb = X_min + span * x_fb
                        f_fb, g_fb = obj.loss_and_grad(z_fb)
                        X.append(z_fb)
                        f_list.append(float(f_fb))
                        g_list.append(np.asarray(g_fb, dtype=float))
                        if float(f_fb) < best[1]:
                            best = (z_fb.copy(), float(f_fb))
                            no_improve_streak = 0
                        phys_fb = tfm.z_to_phys(z_fb)
                        print(
                            f"[P2] iter {it+1:03d}: fallback f_hat(best)={float(f_best_hat):.6e}, f_true={float(f_fb):.6e}, best={best[1]:.6e}  "
                            + _format_phys(phys=phys_fb, names=tfm.names_phys)
                        )
                        tracer.write(stage="phase2", it=it + 1, tag="fallback", f=float(f_fb), z=z_fb, phys=phys_fb)

                patience = max(int(args.no_improve_patience), 1)
                if no_improve_streak >= patience:
                    if trust_radius_cur > 0.0 and float(args.trust_shrink) < 1.0:
                        prev = trust_radius_cur
                        trust_radius_cur = max(float(args.trust_radius_min), trust_radius_cur * float(args.trust_shrink))
                        if trust_radius_cur != prev:
                            print(
                                f"[P2] iter {it+1:03d}: no improvement streak={no_improve_streak} (best {best_before:.6e}->{best[1]:.6e}); "
                                f"trust_radius {prev:.3g}->{trust_radius_cur:.3g}"
                            )

    best_z = best[0]
    if best_z is None:
        raise RuntimeError("No evaluations")

    best_phys = tfm.z_to_phys(best_z)
    print("\nBest z:")
    print(best_z)
    print("Best physical params:")
    for k in tfm.names_phys:
        print(f"  {k:12s} = {best_phys[k]: .6g}")

    # --- Target vs calibrated implied vol diagnostics (like calibrate_cross_model_kou_to_cgmy_american.py) ---
    if (
        args.iv_csv_out is not None
        or args.iv_plot_out is not None
        or args.iv_plot_3d_out is not None
        or args.iv_plot_3d_html_out is not None
    ):
        try:
            # Infer target q from synthetic scenario (if applicable) or use calibrated q as proxy
            if args.synthetic and args.synthetic_scenario in ("merton_vg_self", "kou_to_merton", "kou_to_merton_vg"):
                q_target = -0.005  # all synthetic scenarios use q=-0.005 for target
            else:
                q_target = float(best_phys["q"])  # for real data, use calibrated q as proxy

            # Calibrated model prices for all quotes
            model_best = CompositeLevyCHF(
                float(args.S0),
                float(args.r),
                float(best_phys["q"]),
                dict(divs),
                {"components": tfm.components_from_phys(best_phys)},
            )
            pr_best = COSPricer(model_best, N=int(args.N), L=float(args.L))

            px_fit = np.zeros(len(quotes), dtype=float)
            groups = _group_quotes(quotes)
            for (T, is_call), idxs in groups.items():
                Ks = np.array([quotes[i].K for i in idxs], dtype=float)
                px_T = pr_best.american_price(
                    Ks,
                    float(T),
                    steps=int(args.steps),
                    is_call=bool(is_call),
                    use_softmax=True,
                    beta=float(args.beta),
                )
                px_fit[idxs] = np.asarray(px_T, dtype=float)

            # Build IV diagnostics
            xs: list[float] = []
            iv_target: list[float] = []
            iv_fit: list[float] = []
            Ts: list[float] = []
            Ks: list[float] = []
            types: list[str] = []

            for qte, px_mkt, px_mod in zip(quotes, obj._mkt_prices.tolist(), px_fit.tolist()):
                T = float(qte.T)
                # Use target q for forward proxy so target IVs are invariant across different calibration runs.
                F = _forward_proxy(S0=float(args.S0), r=float(args.r), q=float(q_target), divs=divs, T=T)
                x = float(np.log(float(qte.K) / F) / np.sqrt(T))
                df = float(np.exp(-float(args.r) * T))

                iv_mkt = _bs_implied_vol_forward(price=px_mkt, F=F, K=float(qte.K), df=df, T=T, is_call=bool(qte.is_call))
                iv_mod = _bs_implied_vol_forward(price=px_mod, F=F, K=float(qte.K), df=df, T=T, is_call=bool(qte.is_call))

                xs.append(x)
                iv_target.append(iv_mkt)
                iv_fit.append(iv_mod)
                Ts.append(T)
                Ks.append(float(qte.K))
                types.append(_option_label(bool(qte.is_call)))

            if args.iv_csv_out is not None:
                os.makedirs(os.path.dirname(args.iv_csv_out) or ".", exist_ok=True)
                with open(args.iv_csv_out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["T", "K", "type", "x_logmoneyness_sqrtT", "iv_target", "iv_fit", "price_target", "price_fit"])
                    for T, K, typ, x, ivt, ivf, pt, pf in zip(
                        Ts,
                        Ks,
                        types,
                        xs,
                        iv_target,
                        iv_fit,
                        obj._mkt_prices.tolist(),
                        px_fit.tolist(),
                    ):
                        w.writerow([T, K, typ, x, ivt, ivf, pt, pf])

            if args.iv_plot_out is not None:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8.5, 5.0))
                for T in sorted(set(Ts)):
                    idxs = [i for i, t in enumerate(Ts) if abs(float(t) - float(T)) < 1e-12]
                    if len(idxs) < 2:
                        continue
                    idxs_sorted = sorted(idxs, key=lambda i: xs[i])
                    x_line = [xs[i] for i in idxs_sorted]
                    y_line = [iv_target[i] for i in idxs_sorted]
                    ax.plot(x_line, y_line, color="red", alpha=0.5, linewidth=2.5)

                ax.scatter(
                    xs,
                    iv_target,
                    s=40,
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    label="Target",
                )
                ax.scatter(
                    xs,
                    iv_fit,
                    s=40,
                    marker="x",
                    color="blue",
                    linewidths=1.4,
                    label="Calibrated",
                )

                ax.set_xlabel(r"time-normalized log-moneyness $x = \ln(K/F)/\sqrt{T}$")
                ax.set_ylabel("BS implied vol (from American price)")
                if args.iv_title is not None:
                    ax.set_title(str(args.iv_title))
                else:
                    ax.set_title(f"Targets vs fit ({tfm.case})")
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2, fontsize=9)

                os.makedirs(os.path.dirname(args.iv_plot_out) or ".", exist_ok=True)
                fig.tight_layout()
                fig.savefig(args.iv_plot_out, dpi=180)
                plt.close(fig)

            if args.iv_plot_3d_out is not None:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                fig = plt.figure(figsize=(8.8, 6.0))
                ax = fig.add_subplot(111, projection="3d")

                ax.scatter(
                    xs,
                    Ts,
                    iv_target,
                    s=40,
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    label="Target",
                )
                ax.scatter(
                    xs,
                    Ts,
                    iv_fit,
                    s=40,
                    marker="x",
                    color="blue",
                    linewidths=1.4,
                    label="Calibrated",
                )

                ax.set_xlabel(r"$x = \ln(K/F)/\sqrt{T}$")
                ax.set_ylabel("T")
                ax.set_zlabel("BS implied vol (from American price)")
                if args.iv_title is not None:
                    ax.set_title(str(args.iv_title))
                else:
                    ax.set_title(f"Targets vs fit ({tfm.case})")

                ax.legend(ncol=2, fontsize=9)
                os.makedirs(os.path.dirname(args.iv_plot_3d_out) or ".", exist_ok=True)
                fig.tight_layout()
                fig.savefig(args.iv_plot_3d_out, dpi=180)
                plt.close(fig)

            if args.iv_plot_3d_html_out is not None:
                # Interactive/rotatable 3D plot.
                try:
                    import plotly.graph_objects as go
                except Exception as e:
                    raise RuntimeError(
                        "Plotly is required for --iv-plot-3d-html-out. Install with: pip install plotly"
                    ) from e

                # Rich hover labels
                hover = [
                    f"T={T:.4g}<br>K={K:.4g}<br>type={typ}<br>px_target={pt:.6g}<br>px_fit={pf:.6g}<br>iv_target={ivt:.6g}<br>iv_fit={ivf:.6g}"
                    for (T, K, typ, pt, pf, ivt, ivf) in zip(
                        Ts,
                        Ks,
                        types,
                        obj._mkt_prices.tolist(),
                        px_fit.tolist(),
                        iv_target,
                        iv_fit,
                    )
                ]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=Ts,
                        z=iv_target,
                        mode="markers",
                        name="Target",
                        marker=dict(symbol="circle-open", size=5, color="red", line=dict(width=2, color="red")),
                        text=hover,
                        hoverinfo="text",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=Ts,
                        z=iv_fit,
                        mode="markers",
                        name="Calibrated",
                        marker=dict(symbol="x", size=5, color="blue"),
                        text=hover,
                        hoverinfo="text",
                    )
                )

                title = str(args.iv_title) if args.iv_title is not None else f"Targets vs fit ({tfm.case})"
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=r"x = ln(K/F)/sqrt(T)",
                        yaxis_title="T",
                        zaxis_title="BS implied vol (from American price)",
                        aspectmode="cube",
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                    margin=dict(l=0, r=0, b=0, t=40),
                )

                os.makedirs(os.path.dirname(args.iv_plot_3d_html_out) or ".", exist_ok=True)
                fig.write_html(args.iv_plot_3d_html_out, include_plotlyjs="cdn")

            print("-")
            if args.iv_plot_out is not None:
                print(f"Saved IV plot: {args.iv_plot_out}")
            if args.iv_plot_3d_out is not None:
                print(f"Saved IV 3D plot: {args.iv_plot_3d_out}")
            if args.iv_plot_3d_html_out is not None:
                print(f"Saved IV 3D HTML plot: {args.iv_plot_3d_html_out}")
            if args.iv_csv_out is not None:
                print(f"Saved IV data: {args.iv_csv_out}")
        except Exception as e:
            print("-")
            print(f"IV diagnostics skipped (csv/matplotlib issue): {type(e).__name__}: {e}")

    if true_z is not None:
        print("\nTruth (synthetic):")
        print(true_z)

    tracer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
