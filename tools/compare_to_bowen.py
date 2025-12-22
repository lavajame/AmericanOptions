import os
import sys
import numpy as np
import pandas as pd
# ensure workspace root is on sys.path for local package imports
sys.path.insert(0, os.getcwd())
from american_options import CGMYCHF
from american_options.engine import COSPricer

results = []
S0 = 100.0
r = 0.1
Ks = [110.0]

for Y in [1.5, 1.98]:
    params = {"C": 1.0, "G": 5.0, "M": 5.0, "Y": Y}
    # paper uses q=0.05 for American prices in Table 4/5
    for q in [0.05]:
        model = CGMYCHF(S0, r, q, {}, params)
        # safety truncation to avoid huge COS grids for Y near 2
        model.params['safety_trunc_var'] = 4.0
        # Choose N per paper notes: use larger N for Y close to 2
        N = 1024 if Y >= 1.98 else 512
        pr = COSPricer(model, N=N, L=8.0)
        for K in Ks:
            Karr = np.array([K])
            # European price
            eu = pr.european_price(Karr, 1.0)[0]
            # Richardson extrapolation per Eq (27): base M -> use M,2M,4M,8M to compute vAM
            for baseM in [8, 16, 32]:
                Ms = [baseM, 2*baseM, 4*baseM, 8*baseM]
                vals = []
                for m in Ms:
                    # compute Bermudan approx via COSPricer. Use hard max (use_softmax=False).
                    v = pr.american_price(Karr, 1.0, steps=m, use_softmax=False)[0]
                    vals.append(v)
                vM, v2M, v4M, v8M = vals
                vAM = (1.0/21.0)*(64.0*v8M - 56.0*v4M + 14.0*v2M - 1.0*vM)
                results.append({"Y": Y, "q": q, "K": K, "baseM": baseM, "N": N,
                                "eu": eu, "vM": vM, "v2M": v2M, "v4M": v4M, "v8M": v8M, "vAM": vAM})
                print(f"Y={Y}, baseM={baseM}, eu={eu:.6f}, vAM={vAM:.6f}")


# save to CSV
import os
os.makedirs('figs', exist_ok=True)
df = pd.DataFrame(results)
df.to_csv('figs/cgmy_bowen_comparison.csv', index=False)
print('Saved figs/cgmy_bowen_comparison.csv')
