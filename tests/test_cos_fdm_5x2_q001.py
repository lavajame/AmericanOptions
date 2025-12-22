"""Print a 5x2 table (Call / Put) for q=0.01 showing FDM and COS values.

Rows:
 - FDM Euro
 - COS Euro (standalone)
 - COS Euro (rollback)
 - FDM Amer
 - COS Amer

Run: python tests/test_cos_fdm_5x2_q001.py
"""
import os, sys
import numpy as np

# Ensure repo root is on sys.path for direct execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from plot_diagnostics import FDMPricer

def main():
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.03
    q = 0.01
    # Toggle discrete dividends on/off here.
    divs = {0.5: (5.0, 0.0)}
    # divs = {}
    vol = 0.05
    NS = 500
    NT = NS
    N = 512
    L = 10.0
    steps = NT

    # FDM
    fdm = FDMPricer(S0, r, q, vol, divs, NS=NS, NT=NT)
    fdm_euro_call = fdm.price(K, T, american=False, is_call=True)
    fdm_amer_call = fdm.price(K, T, american=True, is_call=True)
    fdm_euro_put = fdm.price(K, T, american=False, is_call=False)
    fdm_amer_put = fdm.price(K, T, american=True, is_call=False)

    # COS
    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(model, N=N, L=L)
    cos_euro_call = pr.european_price(np.array([K]), T, is_call=True)[0]
    cos_euro_put_standalone = pr.european_price(np.array([K]), T, is_call=False)[0]
    cos_amer_call, cos_euro_rb_call = pr.american_price(np.array([K]), T, steps=steps, is_call=True, return_european=True)
    cos_amer_put, cos_euro_rb_put = pr.american_price(np.array([K]), T, steps=steps, is_call=False, return_european=True)

    rows = [
        ("FDM Euro", fdm_euro_call, fdm_euro_put),
        ("COS Euro (standalone)", cos_euro_call, cos_euro_put_standalone),
        ("COS Euro (rollback)", cos_euro_rb_call[0], cos_euro_rb_put[0]),
        ("FDM Amer", fdm_amer_call, fdm_amer_put),
        ("COS Amer", cos_amer_call[0], cos_amer_put[0]),
    ]

    print(f"{'Row':<28}{'Call':>18}{'Put':>18}")
    for name, call_v, put_v in rows:
        print(f"{name:<28}{call_v:18.8f}{put_v:18.8f}")

if __name__ == '__main__':
    main()
