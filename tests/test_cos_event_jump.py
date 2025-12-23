import numpy as np

from american_options import GBMCHF, DiscreteEventJump


def _bs_call_price(S0: float, K: float, r: float, q: float, T: float, vol: float) -> float:
    # Minimal Black-Scholes call (no dividends except continuous q)
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if vol <= 0.0:
        fwd = S0 * np.exp((r - q) * T)
        return np.exp(-r * T) * max(fwd - K, 0.0)

    from math import log, sqrt, exp

    d1 = (log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)

    # Standard normal CDF via erf
    from math import erf

    def n(x: float) -> float:
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    return S0 * exp(-q * T) * n(d1) - K * exp(-r * T) * n(d2)


def test_cos_european_with_discrete_event_matches_mixture_bs():
    S0 = 100.0
    K = 100.0
    r = 0.03
    q = 0.01
    T = 0.5
    vol = 0.2

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})

    # Event at t=0.4, with (p,u,d). We keep ensure_martingale=True so E[factor]=1.
    event = DiscreteEventJump(time=0.25, p=0.8, u=float(np.log(1.01)), d=float(np.log(0.85)), ensure_martingale=True)

    # COS European call with event
    cos_call = model.european_price(np.array([K]), T, is_call=True, event=event)[0]

    # Conditioning on the jump gives a mixture of Black-Scholes calls
    u_eff = event.u_eff
    d_eff = event.d_eff
    mix_call = (
        event.p * _bs_call_price(S0 * u_eff, K, r, q, T, vol)
        + (1.0 - event.p) * _bs_call_price(S0 * d_eff, K, r, q, T, vol)
    )

    assert np.isfinite(cos_call)
    assert abs(cos_call - mix_call) / max(1.0, abs(mix_call)) < 5e-3


def test_cos_event_increases_atm_call_value_vs_no_event():
    S0 = 100.0
    K = 100.0
    r = 0.02
    q = 0.0
    T = 1.0
    vol = 0.15

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})
    base = model.european_price(np.array([K]), T, is_call=True)[0]

    event = DiscreteEventJump(time=0.5, p=0.5, u=float(np.log(1.15)), d=float(np.log(0.90)), ensure_martingale=True)
    bumped = model.european_price(np.array([K]), T, is_call=True, event=event)[0]

    assert bumped >= base
