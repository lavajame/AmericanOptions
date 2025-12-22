"""Tests for equivalent_gbm helper across Lévy models."""

import numpy as np
from american_options import GBMCHF, MertonCHF, VGCHF, KouCHF, equivalent_gbm
from american_options.engine import COSPricer


def _prices(model, K, T):
    pr = COSPricer(model, N=256, L=8.0)
    return pr.european_price(K, T)


def test_equivalent_gbm_matches_merton_no_jumps():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {}
    vol = 0.30
    merton = MertonCHF(S0, r, q, divs, {"vol": vol, "lam": 0.0, "muJ": 0.0, "sigmaJ": 0.1})
    gbm_eq = equivalent_gbm(merton, T)
    assert isinstance(gbm_eq, GBMCHF)
    assert abs(gbm_eq.params["vol"] - vol) < 1e-12
    K = np.array([80.0, 100.0, 120.0])
    pm = _prices(merton, K, T)
    pg = _prices(gbm_eq, K, T)
    diff = pm - pg
    print("Merton vs GBM eq", pm, pg, diff)
    assert np.allclose(pm, pg, rtol=1e-4, atol=1e-6)


def test_equivalent_gbm_matches_vg_gaussian_limit():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {}
    vol = 0.25
    # theta chosen for Gaussian limit; small nu so jump effects are negligible
    theta = -0.5 * vol * vol
    vg = VGCHF(S0, r, q, divs, {"theta": theta, "sigma": vol, "nu": 1e-6})
    gbm_eq = equivalent_gbm(vg, T)
    K = np.array([90.0, 100.0, 110.0])
    pv = _prices(vg, K, T)
    pg = _prices(gbm_eq, K, T)
    diff = pv - pg
    print("VG vs GBM eq", pv, pg, diff)
    # allow small tolerance because nu is tiny but nonzero
    assert np.allclose(pv, pg, rtol=1e-3, atol=1e-4)


def test_equivalent_gbm_kou_small_intensity():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    divs = {}
    vol = 0.25
    lam = 0.01  # small jump intensity to stay close to GBM
    p = 0.5
    eta1, eta2 = 20.0, 25.0
    kou = KouCHF(S0, r, q, divs, {"vol": vol, "lam": lam, "p": p, "eta1": eta1, "eta2": eta2})
    gbm_eq = equivalent_gbm(kou, T)
    K = np.array([90.0, 100.0, 110.0])
    pk = _prices(kou, K, T)
    pg = _prices(gbm_eq, K, T)
    diff = pk - pg
    print("Kou vs GBM eq", pk, pg, diff, "vol_eq", gbm_eq.params["vol"])
    # should be close because jump intensity and jump sizes are tiny; allow loose tolerance
    assert np.allclose(pk, pg, rtol=2e-3, atol=5e-4)
