"""Sweep spatial grid size M for fixed COS settings (N, L) and report stability.

This is a diagnostic helper to choose a reasonable M.

Example:
  python tools/sweep_M_convergence.py --model Merton --divs "0.1:10" --N 128 --L 8 --steps 128 --snap-t 0.1 --M-list "128,192,256,384,512,768,1024"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np

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


def boundary_at_time(boundary_curve, t_query: float) -> float | None:
    if boundary_curve is None:
        return None
    t_bnd, bnd_mat = boundary_curve
    t_bnd = np.asarray(t_bnd, dtype=float)
    bnd = np.asarray(bnd_mat, dtype=float)[:, 0]
    # exact match preferred
    for i, ti in enumerate(t_bnd):
        if abs(float(ti) - float(t_query)) <= 1e-12:
            return float(bnd[i])
    # nearest
    j = int(np.argmin(np.abs(t_bnd - float(t_query))))
    return float(bnd[j])


def summarize_snapshot(grid_snapshot) -> tuple[float, float, int]:
    if grid_snapshot is None:
        return (float("nan"), float("nan"), 0)
    V = np.asarray(grid_snapshot.get("value"), dtype=float)
    finite = np.isfinite(V)
    n_fin = int(np.sum(finite))
    if n_fin == 0:
        return (float("nan"), float("nan"), 0)
    return (float(np.nanmin(V)), float(np.nanmax(V)), n_fin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Merton")
    ap.add_argument("--model-params", type=str, default="{}")
    ap.add_argument("--divs", type=str, default="")
    ap.add_argument("--is-call", action="store_true")
    ap.add_argument("--is-put", action="store_true")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--snap-t", type=float, default=0.1)
    ap.add_argument("--M-list", type=str, default="128,192,256,384,512")
    args = ap.parse_args()

    try:
        model_params = json.loads(args.model_params)
    except Exception:
        model_params = {}

    if (not model_params) and args.model.strip().lower() == "merton":
        model_params = {"sigma": 0.03, "lam": 0.5, "muJ": -0.05, "sigmaJ": 0.03}

    divs_cash = parse_divs(args.divs)

    # contract
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

    # parse M list
    Ms = []
    for part in str(args.M_list).split(","):
        p = part.strip()
        if not p:
            continue
        Ms.append(int(float(p)))
    Ms = sorted(set(Ms))

    # pick baseline as max M
    M_base = max(Ms)

    def run_for_M(M: int):
        model = make_model(args.model, S0=S0, r=r, q=q, divs_cash=divs_cash, params=model_params)
        pr = COSPricer(model, N=int(args.N), L=float(args.L), M=int(M))
        out = pr.american_price(
            np.array([K], dtype=float),
            float(T),
            steps=int(args.steps),
            is_call=bool(is_call),
            use_softmax=False,
            return_boundary=True,
            return_grid_snapshot=True,
            snapshot_time=float(args.snap_t),
        )
        if not isinstance(out, tuple):
            raise RuntimeError("Expected tuple output")
        prices = np.asarray(out[0], dtype=float)
        boundary_curve = out[-2]
        grid_snapshot = out[-1]
        b_snap = boundary_at_time(boundary_curve, float(args.snap_t))
        vmin, vmax, nfin = summarize_snapshot(grid_snapshot)
        return float(prices[0]), b_snap, float(grid_snapshot["t"]) if grid_snapshot else float("nan"), vmin, vmax, nfin

    base_price, base_b, base_t, base_vmin, base_vmax, base_nfin = run_for_M(M_base)

    print(f"Fixed N={int(args.N)} L={float(args.L)} steps={int(args.steps)} snap_t={float(args.snap_t)} ({'call' if is_call else 'put'})")
    print(f"Divs: {args.divs or '(none)'}")
    print(f"Baseline M={M_base}: price={base_price:.10g}  b@{float(args.snap_t):g}={base_b if base_b is not None else float('nan'):.10g}  snap_node_t={base_t:.10g}  Vmax={base_vmax:.6g}")
    print("\nM, price, |Δprice|, b@snap, |Δb|, snap_node_t, Vmax")

    for M in Ms:
        price, b_snap, t_node, vmin, vmax, nfin = run_for_M(M)
        dp = abs(price - base_price)
        db = abs((b_snap - base_b)) if (b_snap is not None and base_b is not None) else float("nan")

        flag = ""
        if not math.isfinite(vmax) or vmax > 1e6:
            flag = "  !!Vmax"
        print(f"{M}, {price:.10g}, {dp:.3g}, {b_snap if b_snap is not None else float('nan'):.10g}, {db:.3g}, {t_node:.10g}, {vmax:.6g}{flag}")


if __name__ == "__main__":
    main()
