import math

import numpy as np

from american_options import GBMCHF
from american_options.engine import COSPricer


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bs_d1_d2(S0: float, K: np.ndarray, r: float, q: float, vol: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, dtype=float)
    sqrtT = math.sqrt(T)
    sig_sqrtT = vol * sqrtT
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    return d1, d2


def test_cos_digital_put_matches_bs_reasonably() -> None:
    # Digital payoffs are discontinuous => COS converges slower (Gibbs).
    # Use moderately high N and avoid extreme tails.
    S0 = 100.0
    r = 0.05
    q = 0.02
    vol = 0.25
    T = 1.0

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})
    pr = COSPricer(model, N=4096, L=12.0)

    B = np.array([70.0, 85.0, 100.0, 115.0, 130.0])

    cos_price = pr.digital_put_price(B, T)

    _, d2 = _bs_d1_d2(S0, B, r, q, vol, T)
    ref = math.exp(-r * T) * _norm_cdf(-d2)

    # Absolute tolerance: digital can be small or near discount factor
    # so a pure relative tolerance is brittle.
    np.testing.assert_allclose(cos_price, ref, rtol=2e-3, atol=2e-4)


def test_cos_asset_or_nothing_put_matches_bs_reasonably() -> None:
    S0 = 100.0
    r = 0.05
    q = 0.02
    vol = 0.25
    T = 1.0

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})
    pr = COSPricer(model, N=2048, L=12.0)

    B = np.array([70.0, 85.0, 100.0, 115.0, 130.0])

    cos_price = pr.asset_or_nothing_put_price(B, T)

    d1, _ = _bs_d1_d2(S0, B, r, q, vol, T)
    ref = S0 * math.exp(-q * T) * _norm_cdf(-d1)

    np.testing.assert_allclose(cos_price, ref, rtol=5e-4, atol=2e-4)


def test_spot_override_changes_prices_consistently() -> None:
    # Verify the optional `spot` override path is wired correctly.
    S0 = 100.0
    spot = 105.0
    r = 0.03
    q = 0.01
    vol = 0.2
    T = 0.5

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})
    pr = COSPricer(model, N=2048, L=12.0)

    B = np.array([90.0, 100.0, 110.0])

    dig = pr.digital_put_price(B, T, spot=spot)
    aon = pr.asset_or_nothing_put_price(B, T, spot=spot)

    _, d2 = _bs_d1_d2(spot, B, r, q, vol, T)
    ref_dig = math.exp(-r * T) * _norm_cdf(-d2)

    d1, _ = _bs_d1_d2(spot, B, r, q, vol, T)
    ref_aon = spot * math.exp(-q * T) * _norm_cdf(-d1)

    np.testing.assert_allclose(dig, ref_dig, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(aon, ref_aon, rtol=5e-4, atol=2e-4)
