"""Small runnable demo for the AmericanOptions engine."""

import numpy as np
from american_options import GBMCHF, MertonCHF, KouCHF, VGCHF, invert_vol_for_american_price


def main() -> None:
    # Spot, rates, and dividend schedule
    S0 = 100.0
    r = 0.02
    q = 0.01
    divs = {0.5: (0.03, 0.01)}  # 6‑month proportional dividend: mean 3%, std 1%

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


if __name__ == "__main__":
    main()
