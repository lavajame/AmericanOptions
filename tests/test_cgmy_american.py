import numpy as np
from american_options import CGMYCHF
from american_options.engine import COSPricer


def test_cgmy_american_equals_european_no_divs():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    K = np.array([90.0, 100.0, 110.0])
    params = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 0.5}
    model = CGMYCHF(S0, r, q, {}, params)
    pr = COSPricer(model, N=512, L=8.0)
    eu = pr.european_price(K, T)
    am = pr.american_price(K, T, steps=200, use_softmax=False)
    assert np.allclose(am, eu, rtol=1e-6, atol=1e-8)