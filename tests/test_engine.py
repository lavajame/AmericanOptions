"""Minimal tests for the engine to ensure it runs."""

import numpy as np
from american_options import GBMCHF, MertonCHF, KouCHF, VGCHF


def test_european_and_american_run():
    S0 = 100.0
    r = 0.02
    q = 0.01
    # Cash dividend at 6m: mean $3, std $0.5
    divs = {0.5: (3.0, 0.5)}
    params_gbm = {"vol": 0.25}
    gbm = GBMCHF(S0, r, q, divs, params_gbm)
    K = np.array([90.0, 100.0, 110.0])
    T = 1.0

    # use generic COS pricer via the model
    from american_options.engine import COSPricer
    pricer = COSPricer(gbm, N=256, L=8.0)
    e = pricer.european_price(K, T)
    a = pricer.american_price(K, T, steps=50)
    assert e.shape == (3,)
    assert a.shape == (3,)
    assert np.all(e >= 0)
    assert np.all(a >= 0)
    # American should be at least European (no early exercise premium for calls without dividends may be small)
    assert np.all(a + 1e-8 >= e)


if __name__ == "__main__":
    test_european_and_american_run()
    print("Basic tests passed")
