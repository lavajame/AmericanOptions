import copy

import numpy as np

from american_options.engine import COSPricer, CompositeLevyCHF


def _build_cgmy_vg_model(*, S0: float, r: float, q: float, cgmy_params: dict, vg_params: dict) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        {},
        {
            "components": [
                {"type": "cgmy", "params": dict(cgmy_params)},
                {"type": "vg", "params": dict(vg_params)},
            ]
        },
    )


def _bumped_params(base_cgmy: dict, base_vg: dict, name: str, bump: float) -> tuple[dict, dict]:
    cgmy = copy.deepcopy(base_cgmy)
    vg = copy.deepcopy(base_vg)

    prefix, key = name.split(".", 1)
    if prefix.upper() == "CGMY":
        cgmy[key] = float(cgmy[key]) + float(bump)
    elif prefix.upper() == "VG":
        vg[key] = float(vg[key]) + float(bump)
    else:
        raise ValueError(f"Unexpected composite param prefix: {prefix}")

    return cgmy, vg


def test_cgmy_plus_vg_increment_cf_sensitivities_match_fd():
    S0, r, q = 100.0, 0.02, 0.0
    cgmy_params = {"C": 0.03, "G": 1.5, "M": 10.0, "Y": 1.3}
    vg_params = {"theta": -0.30, "sigma": 0.15, "nu": 0.25}

    model = _build_cgmy_vg_model(S0=S0, r=r, q=q, cgmy_params=cgmy_params, vg_params=vg_params)

    dt = 0.4
    u = np.linspace(-15.0, 15.0, 71)

    params = model.param_names() + ["q"]

    phi_a, grad_a = model.increment_char_and_grad(u, dt, method="analytic", params=params)
    phi_fd, grad_fd = model.increment_char_and_grad(u, dt, method="fd", params=params, rel_step=1e-6)

    assert np.max(np.abs(phi_a - phi_fd)) < 1e-10

    for p in params:
        ga = grad_a[p]
        gf = grad_fd[p]
        denom = np.maximum(1.0, np.maximum(np.abs(ga), np.abs(gf)))
        rel = np.max(np.abs(ga - gf) / denom)
        assert rel < 3e-4


def test_cgmy_plus_vg_european_price_sensitivities_match_naive_bump():
    # Compare the pricer's sensitivity propagation (via analytic char_func_and_grad)
    # against a simple central-difference bump on the full European price.
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.75
    K = np.array([100.0])

    cgmy_params = {"C": 0.03, "G": 1.5, "M": 10.0, "Y": 1.3}
    vg_params = {"theta": -0.30, "sigma": 0.15, "nu": 0.25}

    base_model = _build_cgmy_vg_model(S0=S0, r=r, q=q, cgmy_params=cgmy_params, vg_params=vg_params)
    pricer = COSPricer(base_model, N=2048, L=12.0)

    sens_params = base_model.param_names() + ["q"]

    price0, sens = pricer.european_price(
        K,
        T,
        is_call=True,
        return_sensitivities=True,
        sens_method="analytic",
        sens_params=sens_params,
    )
    price0 = float(price0[0])

    rel_step = 1e-4

    for name in sens_params:
        if name == "q":
            base_val = float(q)
        else:
            prefix, key = name.split(".", 1)
            base_val = float(cgmy_params[key] if prefix.upper() == "CGMY" else vg_params[key])

        h = rel_step * max(1.0, abs(base_val))
        if prefix.upper() in {"CGMY"} and key in {"C", "G", "M"}:
            h = min(h, 0.25 * max(base_val, 1e-12))
        if prefix.upper() == "VG" and key in {"sigma", "nu"}:
            h = min(h, 0.25 * max(base_val, 1e-12))

        if name == "q":
            model_p = _build_cgmy_vg_model(S0=S0, r=r, q=q + h, cgmy_params=cgmy_params, vg_params=vg_params)
            model_m = _build_cgmy_vg_model(S0=S0, r=r, q=q - h, cgmy_params=cgmy_params, vg_params=vg_params)
        else:
            cgmy_p, vg_p = _bumped_params(cgmy_params, vg_params, name, +h)
            cgmy_m, vg_m = _bumped_params(cgmy_params, vg_params, name, -h)

            model_p = _build_cgmy_vg_model(S0=S0, r=r, q=q, cgmy_params=cgmy_p, vg_params=vg_p)
            model_m = _build_cgmy_vg_model(S0=S0, r=r, q=q, cgmy_params=cgmy_m, vg_params=vg_m)

        pr_p = COSPricer(model_p, N=2048, L=12.0)
        pr_m = COSPricer(model_m, N=2048, L=12.0)

        vp = float(pr_p.european_price(K, T, is_call=True)[0])
        vm = float(pr_m.european_price(K, T, is_call=True)[0])
        fd = (vp - vm) / (2.0 * h)

        ana = float(sens[name][0])

        assert np.isfinite(ana)
        assert np.isfinite(fd)

        # Tolerances: COS truncation + numerical cancellation in bumps.
        scale = max(1.0, abs(fd), abs(ana), abs(price0))
        assert abs(ana - fd) / scale < 5e-4
