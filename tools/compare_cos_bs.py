import numpy as np
from american_options import GBMCHF, MertonCHF, NIGCHF, VGCHF, CGMYCHF
from american_options.engine import COSPricer
from scipy.stats import norm


def bs_call(S, K, r, q, vol, T):
    sigma = vol
    if sigma * np.sqrt(T) <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compare():
    S0 = 100.0
    r = 0.02
    q = 0.0  # no dividends for this comparison
    divs = {}
    vol = 0.25
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    T = 1.0

    # GBM
    gbm = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr_gbm = COSPricer(gbm, N=512, L=8.0)
    cos_gbm = pr_gbm.european_price(K, T)
    bs_gbm = np.array([bs_call(S0, k, r, q, vol, T) for k in K])

    # Merton with lam=0 reduces to GBM
    merton = MertonCHF(S0, r, q, divs, {"vol": vol, "lam": 0.0, "muJ": 0.0, "sigmaJ": 0.1})
    pr_merton = COSPricer(merton, N=512, L=8.0)
    cos_merton = pr_merton.european_price(K, T)

    # VG with theta=0 and small nu approximates GBM with sigma
    vg = VGCHF(S0, r, q, divs, {"theta": 0.0, "sigma": vol, "nu": 1e-6})
    pr_vg = COSPricer(vg, N=512, L=8.0)
    cos_vg = pr_vg.european_price(K, T)

    # CGMY: Approximate BS by using large M, G and matching C to target variance
    # For Y=0.5, M=G=1e5, C ~ 1115077.57 matches vol=0.25
    cgmy = CGMYCHF(S0, r, q, divs, {"C": 1115077.57, "G": 100000.0, "M": 100000.0, "Y": 0.5})
    pr_cgmy = COSPricer(cgmy, N=512, L=8.0)
    cos_cgmy = pr_cgmy.european_price(K, T)

    # NIG: Approximate BS by using beta=0 and large alpha; choose delta so Var matches.
    alpha = 100.0
    delta = (vol ** 2) * alpha  # for beta=0, var_rate = delta/alpha
    nig = NIGCHF(S0, r, q, divs, {"alpha": alpha, "beta": 0.0, "delta": delta, "mu": 0.0})
    pr_nig = COSPricer(nig, N=512, L=8.0)
    cos_nig = pr_nig.european_price(K, T)

    print("Strike", "BS(GBM)", "COS(GBM)", "Diff", "COS(Merton)", "Diff", "COS(VG)", "Diff", "COS(CGMY)", "Diff(CGMY)", "COS(NIG)", "Diff(NIG)")
    for i, k in enumerate(K):
        print(f"{k:6.1f}", f"{bs_gbm[i]:10.6f}", f"{cos_gbm[i]:10.6f}", f"{(cos_gbm[i]-bs_gbm[i]):9.3e}",
              f"{cos_merton[i]:10.6f}", f"{(cos_merton[i]-bs_gbm[i]):9.3e}",
              f"{cos_vg[i]:10.6f}", f"{(cos_vg[i]-bs_gbm[i]):9.3e}",
              f"{cos_cgmy[i]:10.6f}", f"{(cos_cgmy[i]-bs_gbm[i]):9.3e}",
              f"{cos_nig[i]:10.6f}", f"{(cos_nig[i]-bs_gbm[i]):9.3e}")

    print("\nMax abs diffs (vs BS):")
    print("GBM COS vs BS:   ", np.max(np.abs(cos_gbm - bs_gbm)))
    print("Merton COS vs BS:", np.max(np.abs(cos_merton - bs_gbm)))
    print("VG COS vs BS:    ", np.max(np.abs(cos_vg - bs_gbm)))
    print("CGMY COS vs BS:  ", np.max(np.abs(cos_cgmy - bs_gbm)))
    print("NIG COS vs BS:   ", np.max(np.abs(cos_nig - bs_gbm)))


if __name__ == '__main__':
    compare()
