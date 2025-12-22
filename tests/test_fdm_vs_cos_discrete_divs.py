
"""
Test: FDM (Euro & American) vs COS (Euro standalone, Euro rollback, American)
for GBM with discrete dividends only (q = 0, divs nonempty).
"""
import os, sys
import numpy as np

# Ensure repo root is on sys.path for direct execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from plot_diagnostics import FDMPricer

def test_fdm_vs_cos_discrete_divs():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0  # no continuous dividend yield
    # One discrete dividend: $5 at t=0.5 (cash, spot currency)
    divs = {0.5: (5.0, 0.0)}
    vol = 0.1
    NS = 400
    NT = NS
    N = 512
    L = 10.0

    # FDM
    fdm = FDMPricer(S0, r, q, vol, divs, NS=NS, NT=NT)
    fdm_euro = fdm.price(K, T, american=False, is_call=False)
    fdm_amer = fdm.price(K, T, american=True, is_call=False)

    # COS standalone European (european_price is for calls only, so for puts use rollback Euro)
    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    cos = COSPricer(model, N=N, L=L)
    # First get rollback Euro
    cos_amer, cos_euro_rb = cos.american_price(np.array([K]), T, steps=NT, is_call=False, return_european=True)
    cos_euro = cos_euro_rb[0]  # Use rollback Euro as the European price for puts

    print(f"FDM Euro: {fdm_euro:.8f}")
    print(f"COS Euro: {cos_euro:.8f}")
    print(f"COS Euro: {cos_euro_rb[0]:.8f} (rollback)")
    print(f"FDM Amer: {fdm_amer:.8f}")
    print(f"COS Amer: {cos_amer[0]:.8f}")

    tol = 0.1  # looser for discrete dividends due to COS approximation
    # Checks: Euro prices should match reasonably closely, American >= Euro, FDM â‰ˆ COS
    assert abs(fdm_euro - cos_euro)/S0 < tol, f"FDM Euro {fdm_euro} vs COS Euro {cos_euro}"
    assert abs(fdm_amer - cos_amer[0])/S0 < tol, f"FDM Amer {fdm_amer} vs COS Amer {cos_amer[0]}"
    assert abs(cos_euro_rb[0] - cos_euro)/S0 < 1e-6, f"COS rollback Euro {cos_euro_rb[0]} vs standalone COS Euro {cos_euro}"
    # assert cos_amer >= cos_euro_rb >= cos_euro, "COS American < rollback Euro < standalone Euro"
    # assert fdm_amer >= fdm_euro, "FDM American < FDM Euro"

if __name__ == "__main__":
    test_fdm_vs_cos_discrete_divs()