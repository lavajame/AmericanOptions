"""Sweep softmax `beta` values and plot EEP vs beta alongside hard-max baseline.

Saves: `tools/softmax_beta_sweep.csv` and `tools/softmax_beta_sweep.png`.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import csv
import os
import time

import matplotlib.pyplot as plt

from american_options import GBMCHF
import american_options.engine as engine
from american_options.engine import COSPricer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--vol", type=float, default=0.25)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--L", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n-betas", type=int, default=40, help="Number of log-spaced betas between min and max")
    p.add_argument("--beta-min", type=float, default=1.0)
    p.add_argument("--beta-max", type=float, default=1e4)
    p.add_argument("--out-csv", type=str, default="tools/softmax_beta_sweep.csv")
    p.add_argument("--out-png", type=str, default="tools/softmax_beta_sweep.png")
    return p.parse_args()


def main():
    args = parse_args()

    S = args.S
    K = args.K
    r = args.r
    q = args.q
    vol = args.vol
    T = args.T

    model = GBMCHF(S0=S, r=r, q=q, divs={}, params={"vol": vol})

    # European reference (use default pricer)
    tmp_pricer = COSPricer(model, N=args.N, L=args.L)
    euro_arr = tmp_pricer.european_price(np.array([K]), T, is_call=False)
    european = float(euro_arr[0])

    # Hard max baseline (use temporary pricer)
    t0 = time.time()
    hard_arr = tmp_pricer.american_price(np.array([K]), T, steps=int(args.steps), is_call=False, use_softmax=False)
    hard_amer = float(hard_arr[0])
    hard_eep = hard_amer - european
    print(f"European={european:.12f}, Hard American={hard_amer:.12f}, Hard EEP={hard_eep:.12f} (computed in {time.time()-t0:.2f}s)")

    # Log-spaced beta sweep
    betas = np.logspace(math.log10(args.beta_min), math.log10(args.beta_max), args.n_betas)

    methods = [
        ("logsumexp_pair", engine.softmax_pair),
        ("sqrt", engine.softmax_sqrt_ab),
    ]

    results_by_method = {name: [] for name, _ in methods}

    out_dir = os.path.dirname(args.out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        header = ["beta"]
        for name, _ in methods:
            header += [f"american_{name}", f"eep_{name}", f"eepdiff_{name}", f"time_{name}"]
        writer.writerow(header)

        # For each method create a pricer and sweep betas
        for name, fn in methods:
            engine.SOFTMAX_FN = fn
            print(f"Using softmax method: {name}")
            pricer = COSPricer(model, N=args.N, L=args.L)
            for i, b in enumerate(betas, start=1):
                t1 = time.time()
                arr = pricer.american_price(np.array([K]), T, steps=int(args.steps), is_call=False, beta=float(b), use_softmax=True)
                amer = float(arr[0])
                eep = amer - european
                dt = time.time() - t1
                eepdiff = abs(eep - hard_eep)
                results_by_method[name].append((b, amer, eep, eepdiff, dt))
                print(f"[{name}] [{i}/{len(betas)}] beta={b:.6g}: American={amer:.12f}, EEP={eep:.12f}, |diff|={eepdiff:.3g}, dt={dt:.2f}s")

        # Write rows merging methods by beta
        for idx, b in enumerate(betas):
            row = [f"{b:.8g}"]
            for name, _ in methods:
                _, amer, eep, eepdiff, dt = results_by_method[name][idx]
                row += [f"{amer:.12f}", f"{eep:.12f}", f"{eepdiff:.12e}", f"{dt:.4f}"]
            writer.writerow(row)

    # Plot comparison
    betas_arr = betas
    # Plot absolute difference to hard-max EEP and use log y-axis
    plt.figure(figsize=(8, 5))
    for name, _ in methods:
        eeps_diff = np.array([r[3] for r in results_by_method[name]])
        # clip to a tiny positive value to avoid zeros on log scale
        eeps_diff_clipped = np.maximum(eeps_diff, 1e-16)
        plt.plot(betas_arr, eeps_diff_clipped, marker="o", label=f"{name} |EEP - hard|")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("beta (softmax)")
    plt.ylabel("Absolute EEP error vs hard-max (log scale)")
    plt.title(f"|EEP - Hard| vs softmax beta (S=K={S}, r={r}, q={q}, vol={vol}, T={T})")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved CSV -> {args.out_csv}")
    print(f"Saved plot -> {args.out_png}")


if __name__ == "__main__":
    main()
