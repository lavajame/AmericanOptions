import numpy as np

from american_options import GBMCHF
from american_options.engine import COSPricer, cash_divs_to_proportional_divs


def _div_prod_from_cash(S0: float, r: float, q: float, divs_cash: dict) -> float:
    divs_prop = cash_divs_to_proportional_divs(float(S0), float(r), float(q), divs_cash)
    prod = 1.0
    for _t, (m, _s) in divs_prop.items():
        prod *= (1.0 - float(m))
    return float(prod)


def test_put_call_parity_implies_prepaid_forward_with_discrete_divs():
    """Parity check under the project-wide cash discrete dividend convention.

    With the engine's cash→proportional conversion, for remaining horizon tau and shifted
    cash dividends, parity should satisfy:
        C - P + K e^{-r tau} = S0 e^{-q tau} Π(1-m_mean)

    This is model-free given (r,q,divs_shift,tau) and does not depend on vol.
    """

    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    # Cash dividends in spot currency (mean, std)
    divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}
    vol = 0.25
    K = 100.0

    # Times chosen to straddle the ex-div dates (expected jumps) without using tau=0.
    times = np.array([0.01, 0.10, 0.24, 0.249, 0.251, 0.26, 0.50, 0.74, 0.749, 0.751, 0.76, 0.90], dtype=float)

    errs = []
    for t in times:
        tau = float(T - t)
        divs_shift = {dt - t: (Dm, Ds) for dt, (Dm, Ds) in divs.items() if dt > t + 1e-12}
        prod = _div_prod_from_cash(S0, r, q, divs_shift)

        prepaid_model = float(S0 * np.exp(-q * tau) * prod)

        model_t = GBMCHF(S0, r, q, divs_shift, {"vol": vol})
        pr = COSPricer(model_t, N=512, L=8.0)
        call = float(pr.european_price(np.array([K]), tau, is_call=True)[0])
        put = float(pr.european_price(np.array([K]), tau, is_call=False)[0])

        prepaid_implied = float(call - put + K * np.exp(-r * tau))
        errs.append(prepaid_implied - prepaid_model)

    errs = np.array(errs)
    # COS truncation/discretization introduces small numerical error; keep tight but not extreme.
    assert np.max(np.abs(errs)) < 5e-4
