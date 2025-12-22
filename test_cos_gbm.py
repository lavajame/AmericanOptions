import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer

# Parameters from Table 2 in COS.pdf
S0 = 100.0
r = 0.1
q = 0.0
divs = {}
vol = 0.25
T = 0.1
K = np.array([80.0, 100.0, 120.0])

# Reference values from PDF
refs = [20.799226309, 3.659968453, 0.044577814]

# GBM model
gbm = GBMCHF(S0, r, q, divs, {"vol": vol})

# Test with different N
# for N in [64, 128, 256, 512]:
for N_ in [2, 3, 4, 5, 6, 7]:
    N = 2**N_
    pricer = COSPricer(gbm, N=N, L=10.0)
    prices = pricer.european_price(K, T)
    print(f"N={N}")
    for i, k in enumerate(K):
        diff = prices[i] - refs[i]
        print(f"K={k}: COS={prices[i]:.9f}, Ref={refs[i]:.9f}, Diff={diff:.2e}")
    print()