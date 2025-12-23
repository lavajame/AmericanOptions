"""Small runnable demo for the AmericanOptions engine."""

import numpy as np
from american_options import GBMCHF, MertonCHF, KouCHF, VGCHF, invert_vol_for_american_price
from american_options.engine import COSPricer
from american_options.events import DiscreteEventJump


def main() -> None:
    # Spot, rates, and dividend schedule
    S0 = 100.0
    r = 0.02
    q = 0.01
    divs = {0.5: (3.0, 0.5)}  # 6‑month cash dividend: mean $3.00, stdev $0.50

    # Model parameters
    params_gbm = {"vol": 0.25}
    params_merton = {"vol": 0.20, "lam": 0.10, "muJ": -0.05, "sigmaJ": 0.10}
    params_kou = {"vol": 0.20, "lam": 0.10, "p": 0.5, "eta1": 10, "eta2": 5}
    params_vg = {"theta": -0.14, "sigma": 0.20, "nu": 0.20}

    # Instantiate the characteristic‑function objects
    gbm = GBMCHF(S0, r, q, divs, params_gbm)
    merton = MertonCHF(S0, r, q, divs, params_merton)
    kou = KouCHF(S0, r, q, divs, params_kou)
    vg = VGCHF(S0, r, q, divs, params_vg)

    # Strike vector & maturity
    K = np.linspace(80, 120, 21)
    T = 1.0

    # European prices
    print("GBM European:", gbm.european_price(K, T))
    print("Merton European:", merton.european_price(K, T))
    print("Kou European:", kou.european_price(K, T))
    print("VG European:", vg.european_price(K, T))

    # American prices
    print("GBM American:", gbm.american_price(K, T))
    print("Merton American (approx):", merton.american_price(K, T))

    # Invert volatility from an American price
    amer_price = gbm.american_price(np.array([100.0]), T)[0]
    inv_vol = invert_vol_for_american_price(amer_price, S0, r, q, T, divs, 100.0)
    print("Inverted GBM volatility:", inv_vol)

    # --- Discrete event jump demo (scheduled binary multiplicative jump) ---
    # Example: at event_time, spot jumps by u w.p. p, otherwise by d.
    # With ensure_martingale=True, (u, d) are normalized so E[J]=1.
    event = DiscreteEventJump(time=0.30, p=0.60, u=1.10, d=0.92, ensure_martingale=True)

    # Use the COS pricer directly so we can pass event=...
    # (CharacteristicFunction.european_price / american_price also accept event=...)
    pricer = COSPricer(gbm, N=512, L=8.0)
    K_event = np.array([90.0, 100.0, 110.0])
    call_event = pricer.european_price(K_event, T, is_call=True, event=event)
    call_base = pricer.european_price(K_event, T, is_call=True, event=None)
    print("GBM European (no event):", call_base)
    print("GBM European (+ event):", call_event)


if __name__ == "__main__":
    main()
