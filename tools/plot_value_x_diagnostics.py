"""Diagnostic plot: how quickly the American value function varies across x_grid.

Produces a 2x2 figure:
  - Top row: value/intrinsic/continuation vs S
  - Bottom row: |dV/dx| and |d^2V/dx^2| vs S (finite differences on x_grid)

Example:
  python tools/plot_value_x_diagnostics.py --model Merton --divs "0.1:10" --M 128 --steps 128 --L 5 --snap-t 0.1 --out tools/value_x_diag.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from american_options.models import MertonCHF


def parse_divs(s: str) -> dict:
    out: Dict[float, Tuple[float, float]] = {}
    if not s:
        return out
    for part in s.split(";"):
        p = part.strip()
        if not p:
            continue
        if ":" in p:
            a, b = p.split(":", 1)
            try:
                t = float(a)
                D = float(b)
                out[float(t)] = (float(D), 0.0)
            except Exception:
                continue
    return out


def make_model(name: str, *, S0: float, r: float, q: float, divs_cash: dict, params: dict):
    n = name.strip().lower()
    if n == "gbm":
        vol = float(params.get("vol", 0.2))
        return GBMCHF(S0=S0, r=r, q=q, divs=divs_cash, params={"vol": vol})
    if n == "merton":
        return MertonCHF(S0=S0, r=r, q=q, divs=divs_cash, params=params)
    raise ValueError("Model must be GBM or Merton for this diagnostic.")


def _finite_diffs_on_uniform_x(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = float(x[1] - x[0])
    # central differences (nan at edges)
    dy = np.full_like(y, np.nan, dtype=float)
    d2y = np.full_like(y, np.nan, dtype=float)
    if len(x) >= 3:
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
        d2y[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dx * dx)
    return dy, d2y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="tools/value_x_diagnostics.png")
    ap.add_argument("--model", type=str, default="Merton")
    ap.add_argument("--model-params", type=str, default="{}")
    ap.add_argument("--divs", type=str, default="")
    ap.add_argument("--is-call", action="store_true", help="Plot call space (default).")
    ap.add_argument("--is-put", action="store_true", help="Plot put space.")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--snap-t", type=float, default=0.0, help="Snapshot rollback time t to capture (e.g. 0, 0.1).")
    ap.add_argument("--s-min", type=float, default=None, help="Optional S-min for the derivative plots (focus window).")
    ap.add_argument("--s-max", type=float, default=None, help="Optional S-max for the derivative plots (focus window).")

    args = ap.parse_args()

    try:
        model_params = json.loads(args.model_params)
    except Exception:
        model_params = {}

    # Sensible defaults for Merton if user didn't supply params
    if (not model_params) and args.model.strip().lower() == "merton":
        model_params = {"sigma": 0.03, "lam": 0.5, "muJ": -0.05, "sigmaJ": 0.03}

    divs_cash = parse_divs(args.divs)

    # baseline contract (keep consistent with other tool scripts)
    S0 = 100.0
    K = 100.0
    r = 0.01
    q = 0.02
    T = 0.5

    is_call = True
    if args.is_put:
        is_call = False
    if args.is_call:
        is_call = True

    model = make_model(args.model, S0=S0, r=r, q=q, divs_cash=divs_cash, params=model_params)
    pr = COSPricer(model, N=int(args.N), L=float(args.L), M=int(args.M))

    # Request a snapshot from the rollback.
    out = pr.american_price(
        np.array([K], dtype=float),
        float(T),
        steps=int(args.steps),
        is_call=bool(is_call),
        use_softmax=False,
        return_grid_snapshot=True,
        snapshot_time=float(args.snap_t),
    )

    # american_price returns (prices, grid_snapshot) in this call
    if not isinstance(out, tuple) or len(out) < 2:
        raise RuntimeError("Expected (prices, grid_snapshot) from american_price")

    grid = out[-1]
    if grid is None:
        raise RuntimeError("No grid snapshot captured. Try a snap-t that matches a rollback node (e.g. 0.0 or a dividend time).")

    t_snap = float(grid["t"])
    x = np.asarray(grid["x_grid"], dtype=float)
    S = np.asarray(grid["S_grid"], dtype=float)
    V = np.asarray(grid["value"], dtype=float)
    I = np.asarray(grid["intrinsic"], dtype=float)
    C = np.asarray(grid["continuation"], dtype=float)

    # Replace non-finite with NaN so diffs/summaries don’t explode.
    V = np.where(np.isfinite(V), V, np.nan)
    I = np.where(np.isfinite(I), I, np.nan)
    C = np.where(np.isfinite(C), C, np.nan)

    dVdx, d2Vdx2 = _finite_diffs_on_uniform_x(x, V)
    dIdx, d2Idx = _finite_diffs_on_uniform_x(x, I)
    dCdx, d2Cdx2 = _finite_diffs_on_uniform_x(x, C)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)

    ax = axes[0, 0]
    ax.plot(S, V, lw=2.0, label="Value V")
    ax.plot(S, C, lw=1.5, label="Continuation")
    ax.plot(S, I, lw=1.5, label="Intrinsic")
    ax.set_title(f"Value / continuation / intrinsic at t={t_snap:g}")
    ax.set_xlabel("S")
    ax.set_ylabel("Payoff-space value")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    ax = axes[0, 1]
    ax.plot(x, V, lw=2.0, label="V(x)")
    ax.plot(x, C, lw=1.5, label="C(x)")
    ax.plot(x, I, lw=1.5, label="I(x)")
    ax.set_title("Same curves vs x = ln S")
    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 0]
    # Focus window for derivatives (defaults to a central region around spot)
    s_min = args.s_min
    s_max = args.s_max
    if s_min is None:
        s_min = 0.25 * S0
    if s_max is None:
        s_max = 4.0 * S0
    focus = (S >= float(s_min)) & (S <= float(s_max))

    ax.plot(S[focus], np.abs(dVdx)[focus], lw=2.0, label="|dV/dx|")
    ax.plot(S[focus], np.abs(dCdx)[focus], lw=1.5, label="|dC/dx|")
    ax.plot(S[focus], np.abs(dIdx)[focus], lw=1.5, label="|dI/dx|")
    ax.set_title("First derivative magnitude")
    ax.set_xlabel("S")
    ax.set_ylabel("|d(·)/dx|")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 1]
    ax.plot(S[focus], np.abs(d2Vdx2)[focus], lw=2.0, label="|d²V/dx²|")
    ax.plot(S[focus], np.abs(d2Cdx2)[focus], lw=1.5, label="|d²C/dx²|")
    ax.plot(S[focus], np.abs(d2Idx)[focus], lw=1.5, label="|d²I/dx²|")
    ax.set_title("Second derivative magnitude")
    ax.set_xlabel("S")
    ax.set_ylabel("|d²(·)/dx²|")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"American value sensitivity on x_grid (M={int(args.M)}, steps={int(args.steps)}, L={float(args.L)})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    print(f"Wrote {args.out}")

    # Small numeric summary to quantify how 'fast' it changes.
    print(f"finite V points: {int(np.sum(np.isfinite(V)))}/{int(V.size)}")
    if np.any(np.isfinite(V)):
        print(f"V range (finite): [{float(np.nanmin(V)):.6g}, {float(np.nanmax(V)):.6g}]")
    if np.any(focus):
        d1 = np.abs(dVdx)[focus]
        d2 = np.abs(d2Vdx2)[focus]
        if np.any(np.isfinite(d1)):
            print(f"focus S in [{float(s_min):g}, {float(s_max):g}]: max |dV/dx| = {float(np.nanmax(d1)):.6g}")
        if np.any(np.isfinite(d2)):
            print(f"focus S in [{float(s_min):g}, {float(s_max):g}]: max |d2V/dx2| = {float(np.nanmax(d2)):.6g}")


if __name__ == "__main__":
    main()
