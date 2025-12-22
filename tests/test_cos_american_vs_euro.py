"""Tests comparing COS American pricing options (hard vs softmax) and European return."""

import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer


def test_american_returns_european_and_dominates():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {}
    vol = 0.25
    K = np.array([90.0, 100.0, 110.0])
    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(model, N=256, L=8.0)

    amer, euro_from_amer = pr.american_price(K, T, steps=60, beta=50.0, use_softmax=False, return_european=True)
    euro_direct = pr.european_price(K, T)
    # European returned equals standalone European
    assert np.allclose(euro_from_amer, euro_direct, rtol=1e-6, atol=1e-8)
    # American should not fall materially below European (allow small numerical slack)
    assert np.all(amer + 5e-4 >= euro_direct)


def test_softmax_matches_hard_max_for_large_beta():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {}
    vol = 0.20
    K = np.array([90.0, 100.0, 110.0])
    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(model, N=256, L=8.0)

    amer_hard = pr.american_price(K, T, steps=60, beta=50.0, use_softmax=False)
    amer_soft = pr.american_price(K, T, steps=60, beta=20.0, use_softmax=True)
    # Moderate beta keeps smoothness and should remain close to hard max
    assert np.allclose(amer_soft, amer_hard, rtol=2e-3, atol=5e-4)
