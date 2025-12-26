"""Run Fang & Oosterlee (2009) numerical examples against this repo's COS engine.

This is based on the extracted paper text in:
- fang_oosterlee_numerical_examples.txt

Supported (implemented here)
----------------------------
- Table 1: Bermudan *put* options with 10 exercise dates
  - Test 1: BS (GBM)
  - Test 2: CGMY

- Table 2 / Table 3: American *put* under CGMY via 4-point Richardson extrapolation
  (Eq. (76) in the paper, using Bermudan prices with M = 2^d exercise dates).

Not supported in this codebase (skipped)
---------------------------------------
- Barrier options (Tables 4–7): requires a discrete barrier pricer.
- NIG tables (Tables 4–7): the NIG model is implemented, but barrier pricers are not.

Notes
-----
- The paper uses L=8 for truncation in later numerical experiments.
- For the American put (Test 3), the paper uses N=512 for COS.

This script is meant as a regression/benchmark runner, not a perfect reproduction of
CPU timings in the paper.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Ensure repo root is on sys.path when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF, CGMYCHF  # noqa: E402


@dataclass(frozen=True)
class Result:
    label: str
    value: float
    ref: float | None = None


def print_result(r: Result) -> None:
    if r.ref is None:
        print(f"{r.label}: {r.value:.12f}")
    else:
        print(f"{r.label}: {r.value:.12f} (ref {r.ref:.12f}, diff {r.value - r.ref:+.12e})")


def richardson_4point(pricer: COSPricer, K: float, T: float, base_steps: int, *, is_call: bool) -> float:
    """Eq. (76): 4-point Richardson extrapolation from Bermudan ladder.

    v_AM(d) = (1/21) * (64 v(2^{d+3}) - 56 v(2^{d+2}) + 14 v(2^{d+1}) - v(2^d))

    Here `base_steps` corresponds to 2^d.
    """

    v1 = pricer.american_price(np.array([K]), T, steps=base_steps, is_call=is_call)[0]
    v2 = pricer.american_price(np.array([K]), T, steps=2 * base_steps, is_call=is_call)[0]
    v4 = pricer.american_price(np.array([K]), T, steps=4 * base_steps, is_call=is_call)[0]
    v8 = pricer.american_price(np.array([K]), T, steps=8 * base_steps, is_call=is_call)[0]

    return float((64.0 * v8 - 56.0 * v4 + 14.0 * v2 - v1) / 21.0)


def bermudan_put_bsm(*, S0: float, K: float, T: float, r: float, q: float, sigma: float, steps: int, N: int, L: float) -> float:
    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"sigma": float(sigma)})
    pricer = COSPricer(model, N=N, L=L)
    return float(pricer.american_price(np.array([K]), T, steps=steps, is_call=False)[0])


def bermudan_put_cgmy(*, S0: float, K: float, T: float, r: float, q: float, C: float, G: float, M: float, Y: float, steps: int, N: int, L: float) -> float:
    model = CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(C), "G": float(G), "M": float(M), "Y": float(Y)})
    pricer = COSPricer(model, N=N, L=L)
    return float(pricer.american_price(np.array([K]), T, steps=steps, is_call=False)[0])


def american_put_cgmy_richardson(*, S0: float, K: float, T: float, r: float, q: float, C: float, G: float, M: float, Y: float, d: int, N: int, L: float) -> float:
    """American put under CGMY using Richardson, matching Table 3 style."""

    model = CGMYCHF(S0=S0, r=r, q=q, divs={}, params={"C": float(C), "G": float(G), "M": float(M), "Y": float(Y)})
    pricer = COSPricer(model, N=N, L=L)
    base_steps = 2 ** int(d)
    return richardson_4point(pricer, K=K, T=T, base_steps=base_steps, is_call=False)


def main() -> int:
    print("Fang & Oosterlee (2009) benchmarks")
    print("Source: fang_oosterlee_numerical_examples.txt")
    print("-")

    # ------------------------------------------------------------------
    # Table 1: Bermudan put options with 10 exercise dates
    # ------------------------------------------------------------------
    # Table 1 (OCR-extracted):
    # Test 1: BS, S0=100, K=110, T=1, r=0.10, sigma=0.2
    # Test 2: CGMY, S0=100, K=80,  T=1, r=0.10, C=1,G=5,M=5,Y=1.5
    # Text below Table 1 gives reference values:
    #   BS Bermudan put (10 exercise dates): 10.479520123
    #   CGMY Bermudan put (10 exercise dates): 28.829781986
    L = 8.0
    N = 512

    t1_bs_ref = 10.479520123
    t1_bs = bermudan_put_bsm(S0=100.0, K=110.0, T=1.0, r=0.10, q=0.0, sigma=0.2, steps=10, N=N, L=L)
    print_result(Result("Table1 Test1 BS Bermudan Put (M=10)", t1_bs, t1_bs_ref))

    t1_cgmy_ref = 28.829781986
    t1_cgmy = bermudan_put_cgmy(S0=100.0, K=80.0, T=1.0, r=0.10, q=0.0, C=1.0, G=5.0, M=5.0, Y=1.5, steps=10, N=N, L=L)
    print_result(Result("Table1 Test2 CGMY Bermudan Put (M=10, Y=1.5)", t1_cgmy, t1_cgmy_ref))

    print("-")

    # ------------------------------------------------------------------
    # Table 2 / 3: American put under CGMY via Richardson extrapolation
    # ------------------------------------------------------------------
    # Table 2 (OCR-extracted): Test 3 uses S0=1, K=1, T=1, r=0.1, q=0, sigma=0,
    # C=1,G=5,M=5,Y=0.5 with reference value V(0)=0.112152.
    # Table 3 reports errors for d=0..3.
    ref_am = 0.112152

    print("Table2/Test3 CGMY American Put via Richardson (Eq. 76)")
    for d in (0, 1, 2, 3):
        val = american_put_cgmy_richardson(S0=1.0, K=1.0, T=1.0, r=0.10, q=0.0, C=1.0, G=5.0, M=5.0, Y=0.5, d=d, N=N, L=L)
        print_result(Result(f"Table3 d={d} (base M=2^{d})", val, ref_am))

    print("-")
    print("Skipped: Barrier-option tables (Tables 4–7) – barrier pricers not implemented in this repo")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
