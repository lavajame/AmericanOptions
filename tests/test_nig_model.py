import numpy as np

from american_options.engine import COSPricer, CompositeLevyCHF, NIGCHF


def test_nig_martingale_no_dividends_matches_forward():
    S0, r, q = 100.0, 0.03, 0.01
    T = 0.7
    divs = {}
    params = {"alpha": 15.0, "beta": -5.0, "delta": 0.6, "mu": 0.0}

    model = NIGCHF(S0, r, q, divs, params)
    expected_forward = S0 * np.exp((r - q) * T)

    phi_minus_i = model.char_func(np.array([-1j]), T)[0]
    assert abs(phi_minus_i - expected_forward) / expected_forward < 1e-12


def test_nig_char_func_unit_mass_at_u0():
    model = NIGCHF(100.0, 0.02, 0.0, {}, {"alpha": 12.0, "beta": -3.0, "delta": 0.9, "mu": 0.1})
    phi0 = model.increment_char(np.array([0.0]), 0.5)[0]
    assert abs(phi0 - 1.0) < 1e-14


def test_nig_cumulants_match_char_func_derivatives_small_u():
    # Check c1 and c2 via numerical derivatives of log phi for ln S_T.
    S0, r, q = 100.0, 0.01, 0.0
    T = 0.4
    model = NIGCHF(S0, r, q, {}, {"alpha": 15.0, "beta": -4.0, "delta": 0.55, "mu": 0.02})

    c1, c2, _c4 = model.cumulants(T)

    # log phi(u) ~ i u c1 - 0.5 u^2 c2 + ...
    h = 1e-5
    u = np.array([h, -h], dtype=complex)
    phi = model.char_func(u, T)
    logphi = np.log(phi)

    # First derivative at 0: d/du logphi|0 = i*c1
    d1 = (logphi[0] - logphi[1]) / (2.0 * h)
    c1_num = (d1 / (1j)).real

    # Second derivative at 0: d2/du2 logphi|0 = -c2
    # Use symmetric second difference around 0.
    phi0 = model.char_func(np.array([0.0], dtype=complex), T)[0]
    logphi0 = np.log(phi0)
    d2 = (logphi[0] - 2.0 * logphi0 + logphi[1]) / (h ** 2)
    c2_num = (-d2).real

    assert abs(c1_num - c1) / max(1.0, abs(c1)) < 5e-6
    assert abs(c2_num - c2) / max(1.0, abs(c2)) < 5e-6


def test_nig_composite_single_component_matches_nig_char_func():
    S0, r, q = 100.0, 0.03, 0.01
    T = 0.5
    divs = {}
    nig_params = {"alpha": 14.0, "beta": -4.0, "delta": 0.7, "mu": 0.0}

    base = NIGCHF(S0, r, q, divs, nig_params)
    comp = CompositeLevyCHF(S0, r, q, divs, {"components": [{"type": "nig", "params": nig_params}]})

    u = np.linspace(-40.0, 40.0, 81).astype(float)
    phi_base = base.char_func(u, T)
    phi_comp = comp.char_func(u, T)

    assert np.max(np.abs(phi_base - phi_comp)) < 5e-11


def test_nig_prices_are_finite_and_monotone_in_strike_for_calls():
    S0, r, q = 100.0, 0.02, 0.0
    T = 1.0
    divs = {}
    params = {"alpha": 15.0, "beta": -5.0, "delta": 0.6, "mu": 0.0}

    model = NIGCHF(S0, r, q, divs, params)
    pricer = COSPricer(model, N=1024, L=10.0)

    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = pricer.european_price(Ks, T, is_call=True)

    assert np.all(np.isfinite(prices))
    # Call price should be non-increasing in strike.
    assert np.all(np.diff(prices) <= 1e-10)
