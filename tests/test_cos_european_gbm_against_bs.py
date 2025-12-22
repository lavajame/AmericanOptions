import numpy as np
from scipy.stats import norm

from american_options import GBMCHF
from american_options.engine import COSPricer


def _bs_call(S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if vol <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    sig_sqrt = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrt
    d2 = d1 - sig_sqrt
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def test_cos_gbm_matches_cos_pdf_table_2_reference():
    # Parameters from COS.pdf Table 2 (also used in test_cos_gbm.py)
    S0 = 100.0
    r = 0.1
    q = 0.0
    divs = {}
    vol = 0.25
    T = 0.1
    K = np.array([80.0, 100.0, 120.0])

    # Reference values quoted in the workspace script test_cos_gbm.py
    refs = np.array([20.799226309, 3.659968453, 0.044577814])

    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pricer = COSPricer(model, N=256, L=10.0)
    prices = pricer.european_price(K, T)

    # These should match very tightly for GBM.
    assert np.allclose(prices, refs, rtol=1e-10, atol=2e-10)


def test_cos_gbm_matches_black_scholes_across_strikes():
    S0 = 100.0
    r = 0.02
    q = 0.0
    divs = {}
    vol = 0.25
    T = 1.0
    K = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 125.0, 150.0])

    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pricer = COSPricer(model, N=512, L=10.0)
    cos_prices = pricer.european_price(K, T)
    bs_prices = np.array([_bs_call(S0, float(k), r, q, vol, T) for k in K])

    assert np.allclose(cos_prices, bs_prices, rtol=1e-10, atol=2e-10)
