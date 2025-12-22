"""Quick diagnostic: sweep q values and print FDM vs COS (standalone, rollback, american).

Run this directly from the repo root with: python tests/test_cos_vs_fdm_q_sweep.py
"""

import os, sys
import numpy as np

# Ensure repo root is on sys.path for direct execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from plot_diagnostics import FDMPricer

def run_q_sweep(q_vals=(0.0, 0.01, 0.02)):
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    vol = 0.2
    divs = {0.5: (5.0, 0.0)}
    NS = 400
    NT = NS
    N = 512
    L = 10.0
    steps = NT

    print('q, type, standalone_euro, rollback_euro, american')
    for q in q_vals:
        print('\nq =', q)
        # FDM
        fdm = FDMPricer(S0, r, q, vol, divs, NS=NS, NT=NT)
        fdm_euro_call = fdm.price(K, T, american=False, is_call=True)
        fdm_amer_call = fdm.price(K, T, american=True, is_call=True)
        fdm_euro_put = fdm.price(K, T, american=False, is_call=False)
        fdm_amer_put = fdm.price(K, T, american=True, is_call=False)

        # COS
        model = GBMCHF(S0, r, q, divs, {"vol": vol})
        pr = COSPricer(model, N=N, L=L)
        stand_call = pr.european_price(np.array([K]), T, is_call=True)[0]
        stand_put = pr.european_price(np.array([K]), T, is_call=False)[0]
        am_call, euro_rb_call = pr.american_price(np.array([K]), T, steps=steps, is_call=True, return_european=True)
        am_put, euro_rb_put = pr.american_price(np.array([K]), T, steps=steps, is_call=False, return_european=True)

        print('call:', f'{stand_call:.12f}', f'{euro_rb_call[0]:.12f}', f'{am_call[0]:.12f}')
        print('put :', f'{stand_put:.12f}', f'{euro_rb_put[0]:.12f}', f'{am_put[0]:.12f}')


if __name__ == '__main__':
    run_q_sweep()
