import numpy as np

from american_options.engine import CompositeLevyCHF, GBMCHF, MertonCHF, VGCHF


def _fd_price(model, K, T, *, pname: str, h: float, american: bool, use_softmax: bool):
    base = float(model.params[pname])
    model.params[pname] = base + h
    p_up = float((model.american_price(np.array([K]), T, steps=80, use_softmax=use_softmax)[0] if american else model.european_price(np.array([K]), T)[0]))
    model.params[pname] = base - h
    p_dn = float((model.american_price(np.array([K]), T, steps=80, use_softmax=use_softmax)[0] if american else model.european_price(np.array([K]), T)[0]))
    model.params[pname] = base
    return (p_up - p_dn) / (2.0 * h)


def test_gbm_european_price_vol_sensitivity_matches_fd():
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.5
    K = 105.0
    divs = {}
    model = GBMCHF(S0, r, q, divs, {"sigma": 0.2})

    price, sens = model.european_price(np.array([K]), T, return_sensitivities=True)
    d_analytic = float(sens["sigma"][0])

    h = 1e-5
    d_fd = _fd_price(model, K, T, pname="sigma", h=h, american=False, use_softmax=False)

    # COS truncation window is frozen; allow a small tolerance.
    assert abs(d_analytic - d_fd) / max(1.0, abs(d_fd)) < 5e-3


def test_vg_european_price_theta_sensitivity_matches_fd():
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.25
    K = 95.0
    divs = {}
    model = VGCHF(S0, r, q, divs, {"theta": -0.15, "sigma": 0.18, "nu": 0.25})

    price, sens = model.european_price(np.array([K]), T, return_sensitivities=True)
    d_analytic = float(sens["theta"][0])

    h = 1e-5
    d_fd = _fd_price(model, K, T, pname="theta", h=h, american=False, use_softmax=False)

    assert abs(d_analytic - d_fd) / max(1.0, abs(d_fd)) < 2e-2


def test_gbm_american_softmax_vol_sensitivity_runs_and_is_reasonable():
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.5
    K = 100.0
    divs = {}
    model = GBMCHF(S0, r, q, divs, {"sigma": 0.2})

    price, sens = model.american_price(
        np.array([K]),
        T,
        steps=80,
        use_softmax=True,
        beta=25.0,
        return_sensitivities=True,
    )
    d_analytic = float(sens["sigma"][0])

    h = 2e-5
    d_fd = _fd_price(model, K, T, pname="sigma", h=h, american=True, use_softmax=True)

    assert np.isfinite(d_analytic)
    assert abs(d_analytic - d_fd) / max(1.0, abs(d_fd)) < 5e-2


def test_merton_increment_char_grad_shapes():
    S0, r, q = 100.0, 0.02, 0.0
    model = MertonCHF(S0, r, q, {}, {"sigma": 0.15, "lam": 0.8, "muJ": -0.05, "sigmaJ": 0.12})
    u = np.linspace(0.0, 50.0, 64)
    phi, grad = model.increment_char_and_grad(u, 0.3)
    assert phi.shape == (64,)
    for k in ["sigma", "lam", "muJ", "sigmaJ"]:
        assert k in grad
        assert grad[k].shape == (64,)


def test_composite_merton_vg_european_sensitivity_matches_fd():
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.35
    K = 102.0
    divs = {}
    model = CompositeLevyCHF(
        S0,
        r,
        q,
        divs,
        {
            "components": [
                {"type": "merton", "params": {"vol": 0.10, "lam": 0.6, "muJ": -0.04, "sigmaJ": 0.18}},
                {"type": "vg", "params": {"theta": -0.12, "sigma": 0.16, "nu": 0.30}},
            ]
        },
    )

    price, sens = model.european_price(np.array([K]), T, return_sensitivities=True)
    d_analytic = float(sens["VG.theta"][0])

    # FD bump on nested param
    h = 1e-5
    base = float(model._components[1]["params"]["theta"])
    model._components[1]["params"]["theta"] = base + h
    p_up = float(model.european_price(np.array([K]), T)[0])
    model._components[1]["params"]["theta"] = base - h
    p_dn = float(model.european_price(np.array([K]), T)[0])
    model._components[1]["params"]["theta"] = base
    d_fd = (p_up - p_dn) / (2.0 * h)

    assert np.isfinite(d_analytic)
    assert abs(d_analytic - d_fd) / max(1.0, abs(d_fd)) < 3e-2
