from american_options import GBMCHF
from american_options.engine import COSPricer
import numpy as np

S0, r, q, T = 100.0, 0.02, 0.0, 1.0
divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}
K = np.array([100.0])
vol = 0.25

for t in [0.24,0.245,0.249,0.25,0.251,0.255,0.745,0.746,0.747,0.748,0.749,0.75,0.751]:
    tau = T - t
    divs_shift = {dt - t: (m,s) for dt,(m,s) in divs.items() if dt > t + 1e-12}
    model_t = GBMCHF(S0, r, q, divs_shift, {"vol": vol})
    pr = COSPricer(model_t, N=256, L=8.0)
    try:
        amer, euro = pr.american_price(K, tau, steps=80, beta=20.0, use_softmax=True, return_european=True)
        print(f"t={t:.3f} tau={tau:.3f} divs_shift={divs_shift} -> amer={amer[0]:.6f} euro_rb={euro[0]:.6f}")
    except Exception as e:
        print(f"t={t:.3f} ERROR: {e}")
