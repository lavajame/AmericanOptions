"""Volatility inversion and proxy helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.optimize as opt

from .base_cf import CharacteristicFunction
from .models.gbm import GBMCHF

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
        model = GBMCHF(S0, r, q, divs, {"sigma": vol})
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
    return GBMCHF(model.S0, model.r, model.q, model.divs, {"sigma": vol_eq})

