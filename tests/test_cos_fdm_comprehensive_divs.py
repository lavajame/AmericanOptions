"""Comprehensive 5x2 comparison across 4 dividend scenarios.

Scenarios:
  1. No dividends (q=0)
  2. Continuous divs only (q=0.01, discrete divs={})
    3. Discrete divs only (q=0, discrete divs={0.5:(5.0,0.0)})
    4. Both dividend types (q=0.01, discrete divs={0.5:(5.0,0.0)})

For each scenario, prints a 5x2 table (Call / Put):
  - FDM Euro
  - COS Euro (standalone)
  - COS Euro (rollback)
  - FDM Amer
  - COS Amer

Run: python tests/test_cos_fdm_comprehensive_divs.py
"""
import os, sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from plot_diagnostics import FDMPricer


def run_scenario(name, q, divs):
    """Run pricing for one dividend scenario."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    vol = 0.1
    NS = 1000
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
    cos_amer_call, cos_euro_rb_call = pr.american_price(
        np.array([K]), T, steps=steps, is_call=True, return_european=True
    )
    cos_amer_put, cos_euro_rb_put = pr.american_price(
        np.array([K]), T, steps=steps, is_call=False, return_european=True
    )

    rows = [
        ("FDM Euro", fdm_euro_call, fdm_euro_put),
        ("COS Euro (standalone)", cos_euro_call, cos_euro_put_standalone),
        ("COS Euro (rollback)", cos_euro_rb_call[0], cos_euro_rb_put[0]),
        ("FDM Amer", fdm_amer_call, fdm_amer_put),
        ("COS Amer", cos_amer_call[0], cos_amer_put[0]),
    ]

    print(f"\n{name}")
    print(f"{'Row':<28}{'Call':>18}{'Put':>18}")
    for row_name, call_v, put_v in rows:
        print(f"{row_name:<28}{call_v:18.8f}{put_v:18.8f}")


def main():
    # Scenario 1: No dividends
    run_scenario("1. No Dividends (q=0, divs={})", q=0.0, divs={})

    # Scenario 2: Continuous divs only
    run_scenario("2. Continuous Divs Only (q=0.01, divs={})", q=0.01, divs={})

    # Scenario 3: Discrete divs only
    run_scenario("3. Discrete Divs Only (q=0, divs={0.5:(5.0,0.0)})", q=0.0, divs={0.5: (5.0, 0.0)})

    # Scenario 4: Both dividend types
    run_scenario("4. Both Divs (q=0.01, divs={0.5:(5.0,0.0)})", q=0.01, divs={0.5: (5.0, 0.0)})


if __name__ == "__main__":
    main()
