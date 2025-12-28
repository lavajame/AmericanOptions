import numpy as np

from american_options.engine import COSPricer, GBMCHF


def test_rollback_boundary_extraction_put_gbm_smoke() -> None:
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.02
    vol = 0.25
    T = 1.0

    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": vol})
    pr = COSPricer(model, N=256, L=10.0, M=768)

    price, (t_grid, boundary) = pr.american_price(
        np.array([K], dtype=float),
        T,
        steps=40,
        is_call=False,
        use_softmax=False,
        return_boundary=True,
    )

    assert np.isfinite(float(price[0]))

    t_grid = np.asarray(t_grid, dtype=float)
    boundary = np.asarray(boundary, dtype=float)

    assert t_grid.ndim == 1
    assert boundary.shape == (t_grid.shape[0], 1)

    # Boundary is defined on decision times t in [0,T), so include 0 and exclude T.
    assert abs(float(t_grid[0]) - 0.0) <= 1e-14
    assert float(t_grid[-1]) < T + 1e-14

    b = boundary[:, 0]
    assert np.all((np.isnan(b)) | (b >= 0.0))
    assert np.nanmax(b) <= K + 5e-2

    # For a vanilla GBM put with no discrete dividends, the boundary should be (approximately)
    # non-decreasing in time; allow tiny numerical wiggles.
    mask = np.isfinite(b)
    if int(np.sum(mask)) >= 3:
        assert np.all(np.diff(b[mask]) >= -2e-2)

    # Heuristic next-exercise-node estimate should return a valid node.
    t_star, b_star, dp_star = pr.estimate_next_exercise_node_from_boundary(
        t_grid,
        boundary,
        is_call=False,
        strike_index=0,
        spot=S0,
    )
    assert np.any(np.isclose(t_grid, float(t_star), atol=0.0, rtol=0.0))
    assert (np.isnan(b_star)) or (b_star >= 0.0)
    assert dp_star >= 0.0
