import copy

import numpy as np

from american_options.engine import COSPricer, CompositeLevyCHF


def _build_merton_vg_model(*, S0: float, r: float, q: float, merton_params: dict, vg_params: dict) -> CompositeLevyCHF:
    return CompositeLevyCHF(
        S0,
        r,
        q,
        {},
        {
            "components": [
                {"type": "merton", "params": dict(merton_params)},
                {"type": "vg", "params": dict(vg_params)},
            ]
        },
    )


def _bumped_params(base_merton: dict, base_vg: dict, name: str, bump: float) -> tuple[dict, dict]:
    merton = copy.deepcopy(base_merton)
    vg = copy.deepcopy(base_vg)

    prefix, key = name.split(".", 1)
    if prefix.upper() == "MERTON":
        merton[key] = float(merton[key]) + float(bump)
    elif prefix.upper() == "VG":
        vg[key] = float(vg[key]) + float(bump)
    else:
        raise ValueError(f"Unexpected composite param prefix: {prefix}")

    return merton, vg


def test_merton_plus_vg_american_put_price_sensitivities_match_naive_bump():
    # Compare the pricer's American sensitivity propagation (via increment_char_and_grad)
    # against a simple central-difference bump on the full American price.
    #
    # Note: American rollback uses truncation/grid contexts that depend on model parameters.
    # We compare end-to-end analytic sensitivities against naive bumps on the full American
    # price, accepting slightly looser tolerances due to discretization/truncation effects.
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.8
    K = np.array([90.0, 100.0, 110.0])

    merton_params = {"sigma": 0.16, "lam": 0.8, "muJ": -0.08, "sigmaJ": 0.20}
    vg_params = {"theta": -0.20, "sigma": 0.14, "nu": 0.20}

    base_model = _build_merton_vg_model(S0=S0, r=r, q=q, merton_params=merton_params, vg_params=vg_params)

    pricer = COSPricer(base_model, N=1024, L=12.0)
    steps = 40
    beta = 100.0

    sens_params = base_model.param_names() + ["q"]

    price0, sens = pricer.american_price(
        K,
        T,
        steps=steps,
        is_call=False,
        use_softmax=True,
        beta=beta,
        return_sensitivities=True,
        sens_method="analytic",
        sens_params=sens_params,
    )
    price0 = np.asarray(price0, dtype=float)

    rel_step = 1e-4

    for name in sens_params:
        if name == "q":
            base_val = float(q)
        else:
            prefix, key = name.split(".", 1)
            base_val = float(merton_params[key] if prefix.upper() == "MERTON" else vg_params[key])

        h = rel_step * max(1.0, abs(base_val))
        if key in {"lam", "sigmaJ", "sigma", "nu"}:
            h = min(h, 0.25 * max(base_val, 1e-12))

        if name == "q":
            model_p = _build_merton_vg_model(S0=S0, r=r, q=q + h, merton_params=merton_params, vg_params=vg_params)
            model_m = _build_merton_vg_model(S0=S0, r=r, q=q - h, merton_params=merton_params, vg_params=vg_params)
        else:
            m_p, v_p = _bumped_params(merton_params, vg_params, name, +h)
            m_m, v_m = _bumped_params(merton_params, vg_params, name, -h)

            model_p = _build_merton_vg_model(S0=S0, r=r, q=q, merton_params=m_p, vg_params=v_p)
            model_m = _build_merton_vg_model(S0=S0, r=r, q=q, merton_params=m_m, vg_params=v_m)

        pr_p = COSPricer(model_p, N=pricer.N, L=pricer.L)
        pr_m = COSPricer(model_m, N=pricer.N, L=pricer.L)

        vp = np.asarray(
            pr_p.american_price(K, T, steps=steps, is_call=False, use_softmax=True, beta=beta),
            dtype=float,
        )
        vm = np.asarray(
            pr_m.american_price(K, T, steps=steps, is_call=False, use_softmax=True, beta=beta),
            dtype=float,
        )
        fd = (vp - vm) / (2.0 * h)

        ana = np.asarray(sens[name], dtype=float)

        assert np.all(np.isfinite(ana))
        assert np.all(np.isfinite(fd))

        # American rollback introduces additional numerical error; keep tolerances looser.
        denom = np.maximum(1.0, np.maximum(np.abs(ana), np.abs(fd)))
        rel = float(np.max(np.abs(ana - fd) / denom))
        assert rel < 2e-2, f"{name}: rel={rel:.3e} ana={ana} fd={fd}"
