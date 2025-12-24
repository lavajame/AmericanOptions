import math

import numpy as np

from american_options import GBMCHF
from american_options.engine import COSPricer


def test_boundary_solver_smoke_matches_rollback_gbm() -> None:
    # Moderate settings: solver is O(n^2) per time node; keep it small for tests.
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.02
    vol = 0.25
    T = 1.0

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})

    pr_eep = COSPricer(model, N=512, L=10.0)
    t_grid, boundary = pr_eep.solve_american_put_boundary_eep(K, T, steps=30, bisect_tol=1e-7)

    # Boundary should be within [0, K] and non-decreasing in time.
    assert float(boundary.min()) >= 0.0
    assert float(boundary.max()) <= K + 1e-8
    assert np.all(np.diff(boundary) >= -1e-10)

    amer_eep, euro_eep, eep = pr_eep.american_put_price_eep_from_boundary(K, T, t_grid, boundary, spot=S0)
    assert amer_eep >= euro_eep
    assert eep >= -1e-10

    # Compare to existing COS rollback American PUT price.
    amer_rb = float(pr_eep.american_price(np.array([K]), T, steps=60, is_call=False, use_softmax=False)[0])

    # Loose tolerance: first-pass boundary solver + coarse grids.
    assert abs(amer_eep - amer_rb) < 3e-2


def test_boundary_solver_sets_prediv_zero_when_requested() -> None:
    S0 = 100.0
    K = 100.0
    r = 0.03
    q = 0.01
    vol = 0.2
    T = 1.0

    t_div = 0.5
    m = 0.10
    expected_pre = S0 * math.exp((r - q) * t_div)
    divs_cash = {t_div: (m * expected_pre, 0.0)}

    model = GBMCHF(S0=S0, r=r, q=q, divs=divs_cash, params={"vol": vol})
    pr = COSPricer(model, N=256, L=10.0)

    # Force grid to include the dividend time; solver will insert a pre-div node at (t_div - eps).
    t_grid = np.array([0.0, 0.25, t_div, 0.75, 1.0], dtype=float)
    t_out, boundary = pr.solve_american_put_boundary_eep(K, T, t_grid=t_grid, enforce_prediv_zero=True, prediv_epsilon=1e-6)

    # At the inserted pre-div node, boundary should be forced to 0.
    idx = int(np.argmin(np.abs(t_out - (t_div - 1e-6))))
    assert abs(float(t_out[idx]) - (t_div - 1e-6)) <= 1e-12
    assert abs(float(boundary[idx]) - 0.0) <= 1e-14
