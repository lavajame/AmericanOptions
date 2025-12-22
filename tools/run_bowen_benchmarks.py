"""Run Bowen (Zhang & Oosterlee) Section 5 numerical examples.

Source text: bowen_extracted.txt (Section 5)

This runner implements only examples in Section 5 that include *reference values*:
- Fig. 3: GBM Bermudan call, reference 53.355758
- Fig. 4: CGMY Bermudan call (Y=0.5), reference 23.574835
- Fig. 5: CGMY Bermudan call (Y=1.98), reference 99.053582
- Table 4: CGMY American call (Y=1.5), reference 44.0934
- Table 5: CGMY American call (Y=1.98), reference 99.1739

Notes
-----
The extract explicitly states (Section 5 header): r=0.1, q=0.02, S0=100, K=110,
T=1, M=10. Figures 3–5 override some parameters (q/T/M/sigma/Y) but do not
repeat S0/K in each caption. We run the “as-extracted” S0/K, and also print a
diagnostic “swapped S0/K” variant for Figs 3–5 to help resolve OCR ambiguity.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Ensure repo root is on sys.path when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF, CGMYCHF  # noqa: E402


@dataclass(frozen=True)
class BowenCase:
    name: str
    value: float
    ref: float


def richardson_4point(pricer: COSPricer, K: float, T: float, base_steps: int) -> float:
    """4-point Richardson extrapolation using M,2M,4M,8M Bermudan prices."""
    v1 = pricer.american_price(np.array([K]), T, steps=base_steps)[0]
    v2 = pricer.american_price(np.array([K]), T, steps=2 * base_steps)[0]
    v4 = pricer.american_price(np.array([K]), T, steps=4 * base_steps)[0]
    v8 = pricer.american_price(np.array([K]), T, steps=8 * base_steps)[0]
    return float((64.0 * v8 - 56.0 * v4 + 14.0 * v2 - v1) / 21.0)


def print_case(label: str, value: float, ref: Optional[float] = None) -> None:
    if ref is None:
        print(f"{label}: {value:.6f}")
        return
    print(f"{label}: {value:.6f} (ref {ref:.6f}, diff {value - ref:+.6f})")


def bermudan_call_gbm(S0: float, K: float, r: float, q: float, sigma: float, T: float, steps: int, N: int, L: float) -> float:
    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": float(sigma)})
    pricer = COSPricer(model, N=N, L=L)
    return float(pricer.american_price(np.array([K]), T, steps=steps, is_call=True)[0])


def bermudan_call_cgmy(S0: float, K: float, r: float, q: float, C: float, G: float, M: float, Y: float, T: float, steps: int, N: int, L: float) -> float:
    model = CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(C), "G": float(G), "M": float(M), "Y": float(Y)})
    pricer = COSPricer(model, N=N, L=L)
    return float(pricer.american_price(np.array([K]), T, steps=steps, is_call=True)[0])


def american_call_cgmy_richardson(S0: float, K: float, r: float, q: float, C: float, G: float, M: float, Y: float, T: float, base_steps: int, N: int, L: float) -> float:
    model = CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(C), "G": float(G), "M": float(M), "Y": float(Y)})
    pricer = COSPricer(model, N=N, L=L)
    return richardson_4point(pricer, K=K, T=T, base_steps=base_steps)


def main() -> int:
    # As-extracted Section 5 header values
    S0 = 100.0
    K = 110.0
    r = 0.1

    print("Bowen Section 5 benchmarks (bowen_extracted.txt)")
    print("-")

    # Fig. 3 caption: GBM model, r=0.1,q=0.02,sigma=0.2,T=10,M=50,L∈[10,30], ref 53.355758
    fig3_ref = 53.355758
    for L in (10.0, 20.0, 30.0):
        fig3_val = bermudan_call_gbm(S0=S0, K=K, r=r, q=0.02, sigma=0.2, T=10.0, steps=50, N=512, L=L)
        print_case(f"Fig3 GBM Bermudan Call (as-extracted S0=100,K=110, L={L:g})", fig3_val, fig3_ref)
        fig3_val_swap = bermudan_call_gbm(S0=K, K=S0, r=r, q=0.02, sigma=0.2, T=10.0, steps=50, N=512, L=L)
        print_case(f"Fig3 GBM Bermudan Call (swapped S0/K, L={L:g})", fig3_val_swap, fig3_ref)

    # Fig. 4 caption: CGMY, q=0.02,Y=0.5,M=24,L∈[8,10], “other parameters as previous experiments”, ref 23.574835
    fig4_ref = 23.574835
    for L in (8.0, 9.0, 10.0):
        fig4_val = bermudan_call_cgmy(S0=S0, K=K, r=r, q=0.02, C=1.0, G=5.0, M=5.0, Y=0.5, T=1.0, steps=24, N=512, L=L)
        print_case(f"Fig4 CGMY Bermudan Call Y=0.5 (as-extracted T=1, L={L:g})", fig4_val, fig4_ref)
        fig4_val_swap = bermudan_call_cgmy(S0=K, K=S0, r=r, q=0.02, C=1.0, G=5.0, M=5.0, Y=0.5, T=1.0, steps=24, N=512, L=L)
        print_case(f"Fig4 CGMY Bermudan Call Y=0.5 (swapped S0/K, T=1, L={L:g})", fig4_val_swap, fig4_ref)

    # Fig. 5 caption: CGMY, q=0.05,Y=1.98,M=10,L∈[8,10], ref 99.053582
    fig5_ref = 99.053582
    for L in (8.0, 9.0, 10.0):
        fig5_val = bermudan_call_cgmy(S0=S0, K=K, r=r, q=0.05, C=1.0, G=5.0, M=5.0, Y=1.98, T=1.0, steps=10, N=512, L=L)
        print_case(f"Fig5 CGMY Bermudan Call Y=1.98 (as-extracted T=1, L={L:g})", fig5_val, fig5_ref)
        fig5_val_swap = bermudan_call_cgmy(S0=K, K=S0, r=r, q=0.05, C=1.0, G=5.0, M=5.0, Y=1.98, T=1.0, steps=10, N=512, L=L)
        print_case(f"Fig5 CGMY Bermudan Call Y=1.98 (swapped S0/K, T=1, L={L:g})", fig5_val_swap, fig5_ref)

    # Table 4: American call by Richardson, CGMY Y=1.5,q=0.05, ref 44.0934
    t4_ref = 44.0934
    t4_val = american_call_cgmy_richardson(S0=S0, K=K, r=r, q=0.05, C=1.0, G=5.0, M=5.0, Y=1.5, T=1.0, base_steps=8, N=512, L=10.0)
    print_case("Table4 CGMY American Call Y=1.5 (Richardson base M=8)", t4_val, t4_ref)

    # Table 5: American call by Richardson, CGMY Y=1.98,q=0.05, ref 99.1739
    t5_ref = 99.1739
    t5_val = american_call_cgmy_richardson(S0=S0, K=K, r=r, q=0.05, C=1.0, G=5.0, M=5.0, Y=1.98, T=1.0, base_steps=8, N=1024, L=10.0)
    print_case("Table5 CGMY American Call Y=1.98 (Richardson base M=8)", t5_val, t5_ref)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
