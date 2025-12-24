"""Compare EEP (American - European) for GBM using COSPricer.

Usage:
  python tools/compare_external_input.py [options]

Example:
  python tools/compare_external_input.py --S 55 --K 50 --r 0.05 --q 0.02 --vol 0.2 --T 1.0 --L 10 --N 1024 --steps 200
"""
from __future__ import annotations

import argparse
import numpy as np

from american_options import GBMCHF
from american_options.engine import COSPricer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=float, default=55.0)
    p.add_argument("--K", type=float, default=50.0)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--q", type=float, default=0.02)
    p.add_argument("--vol", type=float, default=0.2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--L", type=float, default=10.0, help="COS truncation L")
    p.add_argument("--N", type=int, default=512, help="COS series terms N")
    p.add_argument("--steps", type=int, default=100, help="time steps for American rollback")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--use-softmax", action="store_true")
    p.add_argument("--sweep-betas", type=str, default=None, help="Comma-separated betas to sweep (enables sweep mode)")
    return p.parse_args()


def main():
    args = parse_args()

    S = float(args.S)
    K = float(args.K)
    r = float(args.r)
    q = float(args.q)
    vol = float(args.vol)
    T = float(args.T)

    # Build GBM model
    model = GBMCHF(S0=S, r=r, q=q, divs={}, params={"vol": vol})

    pricer = COSPricer(model, N=args.N, L=args.L)

    # European reference (single call)
    euro_arr = pricer.european_price(np.array([K]), T, is_call=False)
    european = float(euro_arr[0])

    # Sweep mode: run for a list of betas (softmax) and include hard max baseline
    if args.sweep_betas is not None:
        beta_list = [float(x) for x in args.sweep_betas.split(",") if x.strip()]
        print(f"Parameters: S={S}, K={K}, r={r}, q={q}, vol={vol}, T={T}")
        print(f"COS settings: N={args.N}, L={args.L}, steps={args.steps}")
        print(f"European Price (ref): {european:.12f}\n")

        # Hard max baseline
        out_hard = pricer.american_price(np.array([K]), T, steps=int(args.steps), is_call=False, use_softmax=False)
        amer_hard = float(out_hard[0]) if hasattr(out_hard, "__len__") else float(out_hard)
        print(f"Hard max (use_softmax=False): American={amer_hard:.12f}, EEP={amer_hard - european:.12f}")

        # Softmax sweep
        for b in beta_list:
            out_soft = pricer.american_price(np.array([K]), T, steps=int(args.steps), is_call=False, beta=float(b), use_softmax=True)
            amer_soft = float(out_soft[0]) if hasattr(out_soft, "__len__") else float(out_soft)
            print(f"Softmax beta={b:>6}: American={amer_soft:.12f}, EEP={amer_soft - european:.12f}")
        return

    # Single-point run (existing behaviour)
    out = pricer.american_price(np.array([K]), T, steps=int(args.steps), is_call=False, beta=float(args.beta), use_softmax=bool(args.use_softmax), return_european=True)

    if isinstance(out, tuple) and len(out) >= 2:
        amer_arr, euro_arr = out[0], out[1]
    else:
        # Fallback if API returns a single array (shouldn't happen with return_european=True)
        amer_arr = out
        euro_arr = euro_arr

    american = float(amer_arr[0])
    eep = american - european

    print(f"Parameters: S={S}, K={K}, r={r}, q={q}, vol={vol}, T={T}")
    print(f"COS settings: N={args.N}, L={args.L}, steps={args.steps}, beta={args.beta}, use_softmax={args.use_softmax}")
    print(f"European Price: {european:.12f}")
    print(f"American Price: {american:.12f}")
    print(f"EEP (American - European): {eep:.12f}")


if __name__ == "__main__":
    main()
