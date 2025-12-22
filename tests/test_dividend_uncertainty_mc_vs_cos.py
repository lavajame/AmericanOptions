import numpy as np

from american_options import GBMCHF
from american_options.engine import cash_divs_to_proportional_divs
from american_options.engine import COSPricer


def _mc_gbm_euro_uncertain_divs(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    divs: dict,
    is_call: bool,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)

    # Project-wide convention: `divs` are cash dividends. Convert to internal proportional form.
    divs_prop = cash_divs_to_proportional_divs(float(S0), float(r), float(q), divs)

    div_items = [(float(t), float(m), float(std)) for t, (m, std) in divs_prop.items() if 0.0 < float(t) <= float(T)]
    div_items.sort(key=lambda z: z[0])

    times = [0.0] + [t for (t, _, _) in div_items] + [float(T)]
    dts = np.diff(np.array(times, dtype=float))

    S = np.full(int(n_paths), float(S0), dtype=float)
    drift = float(r) - float(q) - 0.5 * float(vol) ** 2

    for idx, dt in enumerate(dts, start=1):
        if dt > 0.0:
            Z = rng.standard_normal(S.shape[0])
            S *= np.exp(drift * dt + float(vol) * np.sqrt(dt) * Z)

        if idx < len(times) - 1:
            _, m, std = div_items[idx - 1]
            Zd = rng.standard_normal(S.shape[0])
            lnD = np.log(max(1.0 - m, 1e-12)) - 0.5 * (std ** 2) + std * Zd
            S *= np.exp(lnD)

    if is_call:
        payoff = np.maximum(S - float(K), 0.0)
    else:
        payoff = np.maximum(float(K) - S, 0.0)

    disc = np.exp(-float(r) * float(T))
    price = float(disc * payoff.mean())
    se = float(disc * payoff.std(ddof=1) / np.sqrt(float(n_paths)))
    return price, se


def test_mc_matches_cos_for_uncertain_dividends_reasonably():
    # Keep this test reasonably fast + stable.
    S0 = 100.0
    r = 0.02
    q = 0.0
    T = 1.0
    vol = 0.20
    # Cash dividends in spot currency (mean, std)
    divs = {0.25: (2.0, 0.5), 0.75: (2.0, 0.5)}

    # A few representative strikes
    strikes = [80.0, 100.0, 120.0]

    model = GBMCHF(S0, r, q, divs, {"vol": vol})
    pr = COSPricer(model, N=512, L=10.0)

    for K in strikes:
        cos_price = float(pr.european_price(np.array([K]), T, is_call=True)[0])
        mc_price, mc_se = _mc_gbm_euro_uncertain_divs(
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            vol=vol,
            divs=divs,
            is_call=True,
            n_paths=40_000,
            seed=123,
        )

        # Require COS to fall within a few MC standard errors + small absolute slack.
        tol = 4.0 * mc_se + 5e-3
        assert abs(mc_price - cos_price) <= tol
