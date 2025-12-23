"""Generate a linked 1x2 interactive 3D implied-vol surface HTML with a discrete event.

Outputs (default, depends on --model):
  - figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html
  - figs/event_iv_surfaces_gbm_vs_merton_linked_3d.html

Notes
-----
- Uses COS European pricing (no dividends).
- Inverts to Black–Scholes implied vol (call).
- Uses Plotly.js via CDN.
- Syncs 3D camera between the two subplots: rotate either, both rotate.
"""

from __future__ import annotations

import argparse
import json
import os
from math import erf, exp, log, sqrt

import numpy as np
from scipy.optimize import root_scalar

from american_options import DiscreteEventJump, GBMCHF, MertonCHF, VGCHF
from american_options.engine import COSPricer


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs_call_price(*, S0: float, K: float, r: float, q: float, T: float, vol: float) -> float:
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if vol <= 0.0:
        fwd = S0 * exp((r - q) * T)
        return exp(-r * T) * max(fwd - K, 0.0)

    sig_sqrt = vol * sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrt
    d2 = d1 - sig_sqrt
    return S0 * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def _implied_vol_call(*, target_price: float, S0: float, K: float, r: float, q: float, T: float) -> float:
    if not np.isfinite(target_price) or T <= 0.0:
        return float("nan")

    disc = exp(-r * T)
    fwd = S0 * exp((r - q) * T)
    intrinsic = disc * max(fwd - K, 0.0)
    upper = disc * fwd  # loose but safe

    if target_price < intrinsic - 1e-10 or target_price > upper + 1e-8:
        return float("nan")

    def f(sig: float) -> float:
        return _bs_call_price(S0=S0, K=K, r=r, q=q, T=T, vol=float(sig)) - float(target_price)

    lo, hi = 1e-6, 2.0
    flo, fhi = f(lo), f(hi)
    while np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) == np.sign(fhi) and hi < 8.0:
        hi = min(8.0, hi * 1.5)
        fhi = f(hi)

    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return float("nan")
    if np.sign(flo) == np.sign(fhi):
        return float("nan")

    sol = root_scalar(f, bracket=(lo, hi), method="brentq", xtol=1e-10)
    return float(sol.root) if sol.converged else float("nan")


def _iv_surface_from_cos(
    *,
    model,
    strikes: np.ndarray,
    maturities: np.ndarray,
    event: DiscreteEventJump | None,
    N: int,
    L: float,
) -> np.ndarray:
    pricer = COSPricer(model, N=N, L=L)
    iv = np.full((len(strikes), len(maturities)), np.nan, dtype=float)

    for j, T in enumerate(maturities):
        prices = pricer.european_price(strikes, float(T), is_call=True, event=event)
        for i, K in enumerate(strikes):
            iv[i, j] = _implied_vol_call(
                target_price=float(prices[i]),
                S0=float(model.S0),
                K=float(K),
                r=float(model.r),
                q=float(model.q),
                T=float(T),
            )

    return iv


def _write_html(*, out_path: str, payload: dict) -> None:
    # Keep JSON reasonably compact (arrays still dominate).
    data_json = json.dumps(payload, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Linked 3D IV Surfaces (GBM vs Complex)</title>
  <script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; }}
    .wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px; }}
    .plot {{ width: 100%; height: 86vh; border: 1px solid #e5e7eb; }}
    .hdr {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; background: #fafafa; }}
    .sub {{ color: #555; font-size: 13px; }}
    @media (max-width: 1100px) {{ .wrap {{ grid-template-columns: 1fr; }} .plot {{ height: 70vh; }} }}
  </style>
</head>
<body>
  <div class=\"hdr\">
    <div><strong>Linked 3D implied-vol surfaces</strong> (rotate either plot; the other follows)</div>
    <div class=\"sub\" id=\"meta\"></div>
  </div>
  <div class=\"wrap\">
    <div id=\"plot_gbm\" class=\"plot\"></div>
    <div id=\"plot_complex\" class=\"plot\"></div>
  </div>

<script>
const payload = {data_json};
const K = payload.strikes;
const T = payload.maturities;

// Plotly surface expects z as 2D array indexed [y][x] if we pass x=T, y=K.
// Our iv arrays are shaped (len(K), len(T)), which matches y=K, x=T.
function makeSurface(divId, title, zEvent, zBase) {{
  const camera0 = {{
    eye: {{ x: -1.8, y: -1.6, z: 0.9 }},
    center: {{ x: 0.0, y: 0.0, z: 0.0 }},
    up: {{ x: 0.0, y: 0.0, z: 1.0 }}
  }};

  const traceBase = {{
    type: 'surface',
    x: T,
    y: K,
    z: zBase,
    colorscale: 'Viridis',
    opacity: 0.5,
    showscale: false,
    contours: {{
      x: {{ show: true, color: 'rgba(255,255,255,0.25)', width: 1 }},
      y: {{ show: true, color: 'rgba(255,255,255,0.25)', width: 1 }},
      z: {{ show: true, color: 'rgba(0,0,0,0.25)', width: 1 }}
    }},
    hovertemplate: 'T=%{{x:.3f}}<br>K=%{{y:.2f}}<br>IV (no event)=%{{z:.4f}}<extra></extra>'
  }};

  const traceEvent = {{
    type: 'surface',
    x: T,
    y: K,
    z: zEvent,
    colorscale: 'Magma',
    opacity: 0.7,
    showscale: false,
    contours: {{
      x: {{ show: true, color: 'rgba(255,255,255,0.25)', width: 1 }},
      y: {{ show: true, color: 'rgba(255,255,255,0.25)', width: 1 }},
      z: {{ show: true, color: 'rgba(0,0,0,0.25)', width: 1 }}
    }},
    hovertemplate: 'T=%{{x:.3f}}<br>K=%{{y:.2f}}<br>IV (event)=%{{z:.4f}}<extra></extra>'
  }};

  const layout = {{
    title: {{ text: title, x: 0.5 }},
    margin: {{ l: 0, r: 0, b: 0, t: 45 }},
    scene: {{
      xaxis: {{ title: 'Maturity T' }},
      yaxis: {{ title: 'Strike K', autorange: 'reversed' }},
      zaxis: {{ title: 'Implied vol' }},
      camera: camera0,
      aspectmode: 'cube'
    }}
  }};

  const config = {{ responsive: true }};
  Plotly.newPlot(divId, [traceBase, traceEvent], layout, config);
}}

makeSurface('plot_gbm', 'GBM (flat vol) + event', payload.iv_gbm_event, payload.iv_gbm_base);
makeSurface('plot_complex', payload.complex_title, payload.iv_complex_event, payload.iv_complex_base);

document.getElementById('meta').textContent =
  `Event at t=${{payload.event.time.toFixed(2)}} | p=${{payload.event.p.toFixed(3)}} | u=${{payload.event.u.toFixed(4)}} | d=${{payload.event.d.toFixed(4)}} | martingale_norm=${{payload.event.ensure_martingale}}`;

// Link camera rotations between the two 3D scenes.
const divA = document.getElementById('plot_gbm');
const divB = document.getElementById('plot_complex');
let syncing = false;

function extractCamera(evt) {{
  // Keys may be 'scene.camera' or nested in evt['scene.camera']
  if (evt && evt['scene.camera']) return evt['scene.camera'];
  // Sometimes relayout emits {{'scene.camera.eye.x': ..., ...}}.
  // We only sync when full camera object is present to avoid partial updates.
  return null;
}}

function syncCamera(sourceDiv, targetDiv) {{
  sourceDiv.on('plotly_relayout', (evt) => {{
    const cam = extractCamera(evt);
    if (!cam) return;
    if (syncing) return;
    syncing = true;
    Plotly.relayout(targetDiv, {{ 'scene.camera': cam }}).then(() => {{ syncing = false; }});
  }});
}}

syncCamera(divA, divB);
syncCamera(divB, divA);
</script>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate linked 3D implied-vol surfaces (GBM vs VG/Merton) with a scheduled discrete event jump."
    )
    parser.add_argument(
        "--model",
        choices=("vg", "merton"),
        default="vg",
        help="Which model to use for the right-hand surface.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. If omitted, uses a default under figs/ based on --model.",
    )
    args = parser.parse_args(argv)

    os.makedirs("figs", exist_ok=True)

    # Common market inputs
    S0 = 100.0
    r = 0.02
    q = 0.0

    event_time = 0.05
    sigma = 0.15
    vg_theta = -0.4
    vg_nu = 0.02
    merton_lam = 2.5
    merton_muJ = -0.05
    merton_sigmaJ = 0.02

    # Choose which non-GBM model to plot on the right.
    # "VG" gives a non-flat implied vol; "MERTON" gives jump-diffusion.
    complex_kind = args.model.upper()
    
    T = 0.25
    k_lim = np.exp(2.0 * sigma * np.sqrt(T))

    # Grid (strike x maturity)
    strikes = np.linspace(S0 / k_lim, S0 * k_lim, 41, dtype=float)
    maturities = np.linspace(0.02, T, 101, dtype=float)

    # Event definition (same for both models)
    u = 1.04
    event = DiscreteEventJump(time=event_time, p=0.5, u=u, d=1/u, ensure_martingale=True)

    # Model 1: GBM (flat base vol)
    gbm = GBMCHF(S0=S0, r=r, q=q, divs={}, params={"vol": sigma})

    # Model 2: choose VG or Merton
    if complex_kind.upper() == "VG":
      # Match instantaneous variance per unit time: Var[X_t]/t = sigma_vg^2 + theta^2 * nu
      sigma_vg_sq = float(sigma) ** 2 - (float(vg_theta) ** 2) * float(vg_nu)
      sigma_vg = float(np.sqrt(max(1e-12, sigma_vg_sq)))
      complex_model = VGCHF(
        S0=S0,
        r=r,
        q=q,
        divs={},
        params={"sigma": sigma_vg, "theta": float(vg_theta), "nu": float(vg_nu)},
      )
      complex_title = "VG (non-flat IV) + event"
      print(f"vg sigma for target atm vol {sigma:.4f} is {complex_model.params['sigma']:.6f}")
    elif complex_kind.upper() == "MERTON":
      # Match instantaneous variance per unit time: Var[X_t]/t = vol^2 + lam*(muJ^2 + sigmaJ^2)
      vol_sq = float(sigma) ** 2 - float(merton_lam) * (float(merton_muJ) ** 2 + float(merton_sigmaJ) ** 2)
      vol = float(np.sqrt(max(1e-12, vol_sq)))
      complex_model = MertonCHF(
        S0=S0,
        r=r,
        q=q,
        divs={},
        params={"vol": vol, "lam": float(merton_lam), "muJ": float(merton_muJ), "sigmaJ": float(merton_sigmaJ)},
      )
      complex_title = "Merton (JD) + event"
      print(f"merton vol for target atm vol {sigma:.4f} is {complex_model.params['vol']:.6f}")
    else:
      raise ValueError(f"Unknown complex_kind: {complex_kind}")

    # COS settings
    N_gbm, L_gbm = 2**12, 8.0
    N_complex, L_complex = 2**12, 15.0

    iv_gbm = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=event, N=N_gbm, L=L_gbm)
    iv_complex = _iv_surface_from_cos(model=complex_model, strikes=strikes, maturities=maturities, event=event, N=N_complex, L=L_complex)

    iv_gbm_base = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=None, N=N_gbm, L=L_gbm)
    iv_complex_base = _iv_surface_from_cos(model=complex_model, strikes=strikes, maturities=maturities, event=None, N=N_complex, L=L_complex)

    payload = {
        "strikes": strikes.tolist(),
        "maturities": maturities.tolist(),
      "iv_gbm_event": iv_gbm.tolist(),
      "iv_gbm_base": iv_gbm_base.tolist(),
      "iv_complex_event": iv_complex.tolist(),
      "iv_complex_base": iv_complex_base.tolist(),
      "complex_title": complex_title,
        "event": {
            "time": float(event.time),
            "p": float(event.p),
            "u": float(event.u),
            "d": float(event.d),
            "ensure_martingale": bool(event.ensure_martingale),
        },
    }

    out_path = args.out
    if out_path is None:
      out_path = (
        "figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html"
        if complex_kind.upper() == "VG"
        else "figs/event_iv_surfaces_gbm_vs_merton_linked_3d.html"
      )
    _write_html(out_path=out_path, payload=payload)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
