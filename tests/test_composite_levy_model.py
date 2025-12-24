import numpy as np

from american_options.engine import (
    CompositeLevyCHF,
    MertonCHF,
    VGCHF,
)


def test_composite_single_component_matches_merton_char_func():
    S0, r, q = 100.0, 0.03, 0.01
    T = 0.7
    divs = {}
    merton_params = {"vol": 0.18, "lam": 0.9, "muJ": -0.08, "sigmaJ": 0.22}

    base = MertonCHF(S0, r, q, divs, merton_params)
    comp = CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {"components": [{"type": "merton", "params": merton_params}]},
    )

    u = np.linspace(-50.0, 50.0, 101).astype(float)
    phi_base = base.char_func(u, T)
    phi_comp = comp.char_func(u, T)

    # Allow tiny numerical differences (complex exponent paths differ).
    assert np.max(np.abs(phi_base - phi_comp)) < 5e-11


def test_composite_single_component_matches_vg_char_func():
    S0, r, q = 100.0, 0.03, 0.01
    T = 0.7
    divs = {}
    vg_params = {"theta": -0.12, "sigma": 0.18, "nu": 0.25}

    base = VGCHF(S0, r, q, divs, vg_params)
    comp = CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {"components": [("vg", vg_params)]},
    )

    u = np.linspace(-50.0, 50.0, 101).astype(float)
    phi_base = base.char_func(u, T)
    phi_comp = comp.char_func(u, T)

    assert np.max(np.abs(phi_base - phi_comp)) < 5e-11


def test_composite_merton_plus_vg_martingale_and_variance_additivity():
    S0, r, q = 100.0, 0.03, 0.01
    T = 0.5
    divs = {}

    merton_params = {"vol": 0.12, "lam": 0.6, "muJ": -0.10, "sigmaJ": 0.25}
    vg_params = {"theta": -0.10, "sigma": 0.15, "nu": 0.30}

    merton = MertonCHF(S0, r, q, divs, merton_params)
    vg = VGCHF(S0, r, q, divs, vg_params)

    comp = CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {"components": [{"type": "merton", "params": merton_params}, {"type": "vg", "params": vg_params}]},
    )

    # Martingale: E[S_T] = S0 * exp((r-q)T) when no dividends.
    expected_forward = S0 * np.exp((r - q) * T)
    phi_minus_i = comp.char_func(np.array([-1j]), T)[0]
    assert abs(phi_minus_i - expected_forward) / expected_forward < 1e-12

    # Variance additivity (excluding dividend variance, consistent with existing _var2 usage).
    v_comp = comp._var2(T)
    v_sum = merton._var2(T) + vg._var2(T)
    assert abs(v_comp - v_sum) / max(1.0, v_sum) < 1e-12
