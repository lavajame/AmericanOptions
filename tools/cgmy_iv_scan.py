from __future__ import annotations
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from american_options.models import CGMYCHF, MertonCHF
from american_options.engine import COSPricer
from plot_diagnostics import bs_implied_vol_call

def main():
    S0 = 100.0
    r = 0.01
    q = 0.015
    T = 0.125
    # Avoid a dividend exactly at expiry (that causes BS IV clipping). Move near-term div slightly earlier.
    # divs = {0.11: (0.01 * S0 * np.exp((r - q) * 0.12), 0.0), 0.35: (0.01 * S0 * np.exp((r - q) * 0.35), 0.0)}
    divs = {}
    # Flatter overall variance with a symmetric skew: smaller C, symmetric G=M, Y closer to 1
    params = {"C": 0.3, "G": 2.2, "M": 2.1, "Y": 1.3}

    # model = CGMYCHF(S0=S0, r=r, q=q, divs=divs, params=params)
    # Quick Merton symmetric-jump test: set muJ=0.0 but increase jump volatility/intensity
    # so the smile should be visible (symmetric around ATM).
    # Increase jump intensity and jump vol to make jump-driven smile visible
    model = MertonCHF(S0=S0, r=r, q=q, divs=divs, params={"sigma":0.05, "lam":0.5, "muJ":-0.05, "sigmaJ":0.05})
    pr = COSPricer(model, N=1024, L=10.0, M=2048)

    Ks = np.arange(90, 111, 1.0)
    prices = pr.european_price(Ks, T, is_call=True)

    ivs = []
    for K, p in zip(Ks, prices):
        iv = bs_implied_vol_call(price=float(p), S0=S0, K=float(K), r=r, q=q, T=T, divs=divs)
        ivs.append(iv)

    ivs = np.array(ivs, dtype=float)

    # Print table
    print("K, Price, IV")
    for K, p, iv in zip(Ks, prices, ivs):
        print(f"{K:.2f}, {float(p):.6f}, {iv:.6f}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(Ks, ivs, marker='o')
    plt.xlabel('Strike')
    plt.ylabel('Implied vol')
    plt.title(f'CGMY implied vols, T={T}')
    out = 'figs/cgmy_iv_T0125.png'
    plt.grid(True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
