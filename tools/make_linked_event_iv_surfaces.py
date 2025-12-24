"""Generate a linked 1x2 interactive 3D implied-vol surface HTML with a discrete event.

Outputs (default, depends on --model):
  - figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html
  - figs/event_iv_surfaces_gbm_vs_merton_linked_3d.html
  - figs/event_iv_surfaces_gbm_vs_kouvg_linked_3d.html

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
import sys
from math import erf, exp, log, sqrt

import numpy as np
from scipy.optimize import root_scalar

# Allow running as: `python tools/make_linked_event_iv_surfaces.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from american_options import CompositeLevyCHF, DiscreteEventJump, GBMCHF, MertonCHF, VGCHF
from american_options.engine import COSPricer


def _round_array(a: np.ndarray, ndp: int) -> np.ndarray:
  # Keep NaNs as NaN; round finite values only.
  if ndp is None:
    return a
  a = np.asarray(a, dtype=float)
  out = a.copy()
  mask = np.isfinite(out)
  out[mask] = np.round(out[mask], ndp)
  return out


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

makeSurface('plot_gbm', payload.gbm_title || 'GBM + event', payload.iv_gbm_event, payload.iv_gbm_base);
makeSurface('plot_complex', payload.complex_title, payload.iv_complex_event, payload.iv_complex_base);

document.getElementById('meta').textContent =
  `Event (log-jump) at t=${{payload.event.time.toFixed(3)}} | p=${{payload.event.p.toFixed(3)}} | u=${{payload.event.u.toFixed(4)}} (factor=${{Math.exp(payload.event.u).toFixed(4)}}) | d=${{payload.event.d.toFixed(4)}} (factor=${{Math.exp(payload.event.d).toFixed(4)}}) | martingale_norm=${{payload.event.ensure_martingale}}`;

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
      choices=("vg", "merton", "kouvg"),
        default="vg",
        help="Which model to use for the right-hand surface.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. If omitted, uses a default under figs/ based on --model.",
    )
    parser.add_argument(
      "--payload-ndp",
      type=int,
      default=4,
      help="Round payload numeric arrays to this many decimal places to reduce HTML size.",
    )
    args = parser.parse_args(argv)

    os.makedirs("figs", exist_ok=True)

    # Common market inputs
    S0 = 100.0
    r = 0.02
    q = 0.0

    event_time = 0.05
    sigma = 0.20
    vg_theta = -0.4
    vg_nu = 0.02
    merton_lam = 2.5
    merton_muJ = -0.05
    merton_sigmaJ = 0.02
    # Kou+VG composite: choose a Kou jump component + a VG component
    # and split instantaneous variance 50/50.
    # These defaults are chosen to produce a more equity-like left skew (less symmetric wings),
    # similar in spirit to `plot_diagnostics.py`'s Kou+VG example.
    kou_lam = 0.6
    kou_p = 0.25
    kou_eta1 = 30.0
    kou_eta2 = 8.0
    kou_vg_theta = -0.30
    kou_vg_nu = 0.10

    # Choose which non-GBM model to plot on the right.
    # "VG" gives a non-flat implied vol; "MERTON" gives jump-diffusion; "KOUVG" is a composite.
    complex_kind = args.model.upper()
    
    T = 0.25
    k_lim = np.exp(2.0 * sigma * np.sqrt(T))

    # Grid (strike x maturity)
    strikes = np.linspace(S0 / k_lim, S0 * k_lim, 41, dtype=float)
    maturities = np.linspace(0.02, T, 101, dtype=float)

    # Event definition (same for both models): specified in log-jump space.
    # If J=u then factor is exp(u); if J=d then factor is exp(d).
    u_factor = 1.04
    d_factor = 1.0 / u_factor
    event = DiscreteEventJump(
        time=event_time,
        p=0.5,
        u=float(np.log(u_factor)),
        d=float(np.log(d_factor)),
        ensure_martingale=True,
    )

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
        complex_title = (
            f"VG sigma={complex_model.params['sigma']:.4f}, theta={vg_theta:.4f}, nu={vg_nu:.4f}"
            f"<br>event t={event_time:.3f}, p={event.p:.3f}, u={event.u:.4f}, d={event.d:.4f}"
            f" (factors {np.exp(event.u):.4f}/{np.exp(event.d):.4f}), martingale_norm={event.ensure_martingale}"
        )
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
            params={
                "vol": vol,
                "lam": float(merton_lam),
                "muJ": float(merton_muJ),
                "sigmaJ": float(merton_sigmaJ),
            },
        )
        complex_title = (
            f"Merton vol={complex_model.params['vol']:.4f}, lam={merton_lam:.3f}, muJ={merton_muJ:.4f}, sigmaJ={merton_sigmaJ:.4f}"
            f"<br>event t={event_time:.3f}, p={event.p:.3f}, u={event.u:.4f}, d={event.d:.4f}"
            f" (factors {np.exp(event.u):.4f}/{np.exp(event.d):.4f}), martingale_norm={event.ensure_martingale}"
        )
        print(f"merton vol for target atm vol {sigma:.4f} is {complex_model.params['vol']:.6f}")
    elif complex_kind.upper() == "KOUVG":
      # Split instantaneous variance (per unit time) across components:
      # target total: sigma^2
      # Kou: Var/t = vol_kou^2 + lam * E[Y^2] where Y is double-exponential log-jump
      # VG:  Var/t = sigma_vg^2 + theta^2 * nu
      half_var = 0.5 * (float(sigma) ** 2)

      EY2 = 2.0 * float(kou_p) * (1.0 / (float(kou_eta1) ** 2)) + 2.0 * (1.0 - float(kou_p)) * (1.0 / (float(kou_eta2) ** 2))
      kou_jump_var = float(kou_lam) * float(EY2)
      kou_vol_sq = half_var - kou_jump_var
      kou_vol = float(np.sqrt(max(1e-12, kou_vol_sq)))

      vg_sigma_sq = half_var - (float(kou_vg_theta) ** 2) * float(kou_vg_nu)
      vg_sigma = float(np.sqrt(max(1e-12, vg_sigma_sq)))

      complex_model = CompositeLevyCHF(
        S0=S0,
        r=r,
        q=q,
        divs={},
        params={
          "components": [
            {
              "type": "kou",
              "params": {
                "vol": kou_vol,
                "lam": float(kou_lam),
                "p": float(kou_p),
                "eta1": float(kou_eta1),
                "eta2": float(kou_eta2),
              },
            },
            {
              "type": "vg",
              "params": {
                "sigma": vg_sigma,
                "theta": float(kou_vg_theta),
                "nu": float(kou_vg_nu),
              },
            },
          ]
        },
      )
      complex_title = (
        f"Kou+VG (50/50 var split)"
        f"<br>Kou vol={kou_vol:.4f}, lam={kou_lam:.3f}, p={kou_p:.3f}, eta1={kou_eta1:.2f}, eta2={kou_eta2:.2f}"
        f"<br>VG sigma={vg_sigma:.4f}, theta={kou_vg_theta:.4f}, nu={kou_vg_nu:.4f}"
        f"<br>event t={event_time:.3f}, p={event.p:.3f}, u={event.u:.4f}, d={event.d:.4f}"
        f" (factors {np.exp(event.u):.4f}/{np.exp(event.d):.4f}), martingale_norm={event.ensure_martingale}"
      )
    else:
        raise ValueError(f"Unknown complex_kind: {complex_kind}")

    # COS settings
    N_gbm, L_gbm = 2**12, 8.0
    N_complex, L_complex = 2**12, 15.0

    iv_gbm = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=event, N=N_gbm, L=L_gbm)
    iv_complex = _iv_surface_from_cos(model=complex_model, strikes=strikes, maturities=maturities, event=event, N=N_complex, L=L_complex)

    iv_gbm_base = _iv_surface_from_cos(model=gbm, strikes=strikes, maturities=maturities, event=None, N=N_gbm, L=L_gbm)
    iv_complex_base = _iv_surface_from_cos(model=complex_model, strikes=strikes, maturities=maturities, event=None, N=N_complex, L=L_complex)

    ndp = int(args.payload_ndp) if args.payload_ndp is not None else None
    strikes_out = _round_array(strikes, ndp)
    maturities_out = _round_array(maturities, ndp)
    iv_gbm_out = _round_array(iv_gbm, ndp)
    iv_complex_out = _round_array(iv_complex, ndp)
    iv_gbm_base_out = _round_array(iv_gbm_base, ndp)
    iv_complex_base_out = _round_array(iv_complex_base, ndp)

    gbm_title = (
      f"GBM sigma={sigma:.4f}"
      f"<br>event t={event_time:.3f}, p={event.p:.3f}, u={event.u:.4f}, d={event.d:.4f}"
      f" (factors {np.exp(event.u):.4f}/{np.exp(event.d):.4f}), martingale_norm={event.ensure_martingale}"
    )

    payload = {
      "strikes": strikes_out.tolist(),
      "maturities": maturities_out.tolist(),
      "iv_gbm_event": iv_gbm_out.tolist(),
      "iv_gbm_base": iv_gbm_base_out.tolist(),
      "iv_complex_event": iv_complex_out.tolist(),
      "iv_complex_base": iv_complex_base_out.tolist(),
      "gbm_title": gbm_title,
      "complex_title": complex_title,
        "event": {
        "time": float(round(float(event.time), ndp)) if ndp is not None else float(event.time),
        "p": float(round(float(event.p), ndp)) if ndp is not None else float(event.p),
        "u": float(round(float(event.u), ndp)) if ndp is not None else float(event.u),
        "d": float(round(float(event.d), ndp)) if ndp is not None else float(event.d),
            "ensure_martingale": bool(event.ensure_martingale),
        },
    }

    out_path = args.out
    if out_path is None:
      if complex_kind.upper() == "VG":
        out_path = "figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html"
      elif complex_kind.upper() == "MERTON":
        out_path = "figs/event_iv_surfaces_gbm_vs_merton_linked_3d.html"
      elif complex_kind.upper() == "KOUVG":
        out_path = "figs/event_iv_surfaces_gbm_vs_kouvg_linked_3d.html"
      else:
        raise ValueError(f"Unknown complex_kind: {complex_kind}")
    _write_html(out_path=out_path, payload=payload)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
