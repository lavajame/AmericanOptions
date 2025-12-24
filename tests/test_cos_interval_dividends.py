import math

import numpy as np

from american_options import GBMCHF
from american_options.engine import COSPricer


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bs_d2(S0: float, K: float, r: float, q: float, vol: float, dt: float) -> float:
    sig_sqrtT = vol * math.sqrt(dt)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * dt) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    return d2


def test_interval_digital_put_includes_dividend_in_window() -> None:
    # One deterministic proportional dividend at absolute time t_div.
    S0 = 100.0
    r = 0.03
    q = 0.01
    vol = 0.2
    t_div = 0.5
    m = 0.10

    # Choose cash dividend mean so internal conversion yields m_mean ~= m
    expected_pre = S0 * math.exp((r - q) * t_div)
    divs_cash = {t_div: (m * expected_pre, 0.0)}

    model = GBMCHF(S0=S0, r=r, q=q, divs=divs_cash, params={"vol": vol})
    pr = COSPricer(model, N=4096, L=12.0)

    spot_t0 = 100.0
    B = 100.0

    # Interval that DOES include the dividend (0.4, 0.55]
    t0 = 0.4
    t1 = 0.55
    dt = t1 - t0

    cos_price = float(pr.digital_put_price_interval(np.array([B]), t0, t1, spot=spot_t0)[0])

    # For GBM with deterministic proportional dividend, distribution is as if spot is multiplied by (1-m)
    spot_eff = spot_t0 * (1.0 - m)
    ref = math.exp(-r * dt) * float(_norm_cdf(-_bs_d2(spot_eff, B, r, q, vol, dt)))

    np.testing.assert_allclose(cos_price, ref, rtol=2e-3, atol=3e-4)

    # Interval that does NOT include dividend (0.4, 0.45]
    t0b = 0.4
    t1b = 0.45
    dtb = t1b - t0b
    cos_price_b = float(pr.digital_put_price_interval(np.array([B]), t0b, t1b, spot=spot_t0)[0])

    ref_b = math.exp(-r * dtb) * float(_norm_cdf(-_bs_d2(spot_t0, B, r, q, vol, dtb)))
    np.testing.assert_allclose(cos_price_b, ref_b, rtol=2e-3, atol=3e-4)
