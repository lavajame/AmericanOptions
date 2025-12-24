import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# ensure workspace root is on sys.path for local package imports
sys.path.insert(0, os.getcwd())

from american_options import GBMCHF, cash_divs_to_proportional_divs  # noqa: E402
from american_options.engine import COSPricer  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    S0: float
    K: float
    T: float
    r: float
    q: float
    vol: float
    divs_cash: dict[float, tuple[float, float]]


def _price_case(case: Case, *, N: int, L: float, steps_rb: int, steps_boundary: int) -> dict:
    divs = cash_divs_to_proportional_divs(case.S0, case.r, case.q, case.divs_cash)
    model = GBMCHF(case.S0, case.r, case.q, divs, {"vol": case.vol})
    pr = COSPricer(model, N=N, L=L)

    Karr = np.array([case.K], dtype=float)

    t0 = time.perf_counter()
    euro_put = float(pr.european_price(Karr, case.T, is_call=False)[0])
    t1 = time.perf_counter()

    amer_bowen = float(pr.american_price(Karr, case.T, steps=steps_rb, is_call=False, use_softmax=False)[0])
    t2 = time.perf_counter()

    t_grid, boundary = pr.solve_american_put_boundary_eep(case.K, case.T, steps=steps_boundary)
    amer_eep, euro_eep, eep = pr.american_put_price_eep_from_boundary(case.K, case.T, t_grid, boundary)
    t3 = time.perf_counter()

    return {
        "name": case.name,
        "euro_put": euro_put,
        "amer_bowen": amer_bowen,
        "amer_eep": float(amer_eep),
        "eep_premium": float(eep),
        "euro_put_eep": float(euro_eep),
        "abs_diff": float(abs(amer_bowen - amer_eep)),
        "ms_euro": 1e3 * (t1 - t0),
        "ms_bowen": 1e3 * (t2 - t1),
        "ms_eep_total": 1e3 * (t3 - t2),
    }


def main() -> None:
    # Keep defaults modest so it runs quickly on a laptop.
    N = int(os.environ.get("N", "1024"))
    L = float(os.environ.get("L", "10.0"))
    steps_rb = int(os.environ.get("STEPS_RB", "120"))
    steps_boundary = int(os.environ.get("STEPS_BND", "80"))

    cases = [
        Case(
            name="GBM no dividends",
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            q=0.05,
            vol=0.25,
            divs_cash={},
        ),
        Case(
            name="GBM with discrete cash div",
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            q=0.02,
            vol=0.25,
            # Cash dividend at t=0.5: mean=1.0, std=0.0
            divs_cash={0.5: (1.0, 0.0)},
        ),
    ]

    print("Bowen-style COS rollback vs Andersen EEP (American put)")
    print(f"Settings: N={N}, L={L}, STEPS_RB={steps_rb}, STEPS_BND={steps_boundary}\n")

    for case in cases:
        out = _price_case(case, N=N, L=L, steps_rb=steps_rb, steps_boundary=steps_boundary)
        print(f"== {out['name']} ==")
        print(f"European put (COS):            {out['euro_put']:.10f}   ({out['ms_euro']:.1f} ms)")
        print(f"American put (Bowen rollback): {out['amer_bowen']:.10f}   ({out['ms_bowen']:.1f} ms)")
        print(f"American put (Andersen EEP):   {out['amer_eep']:.10f}   ({out['ms_eep_total']:.1f} ms)")
        print(f"  EEP premium:                 {out['eep_premium']:.10f}")
        print(f"  |diff| vs Bowen rollback:    {out['abs_diff']:.10f}\n")


if __name__ == "__main__":
    main()
