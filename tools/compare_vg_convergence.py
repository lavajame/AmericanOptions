import numpy as np
from american_options import GBMCHF, VGCHF
from american_options.engine import COSPricer
from scipy.stats import norm


def bs_call(S, K, r, q, vol, T):
    sigma = vol
    if sigma * np.sqrt(T) <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def convergence_test():
    S0 = 100.0
    r = 0.02
    q = 0.0
    divs = {}
    vol = 0.25
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    T = 1.0

    bs_vals = np.array([bs_call(S0, k, r, q, vol, T) for k in K])

    nus = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    print("nu, max_abs_diff, diffs for strikes [80,90,100,110,120]")
    for nu in nus:
        # choose theta to match GBM in the nu -> 0 limit (theta = -0.5 * sigma^2)
        theta_match = -0.5 * vol * vol
        vg = VGCHF(S0, r, q, divs, {"theta": theta_match, "sigma": vol, "nu": nu})
        pr = COSPricer(vg, N=512, L=8.0)
        cos_vg = pr.european_price(K, T)
        diffs = cos_vg - bs_vals
        print(f"{nu:e}", np.max(np.abs(diffs)), np.round(diffs, 6), "theta", theta_match)

    print('\nDone')

if __name__ == '__main__':
    convergence_test()
