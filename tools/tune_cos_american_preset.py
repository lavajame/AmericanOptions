"""Tune a COS-American accuracy preset.

Goal: find a low-cost (N, steps) that achieves max abs error <= tol = 1e-4 * S0
across a 100-option chain (mixed calls/puts), for representative scenarios.

We measure error against a higher-resolution COS-American reference (same model).

Examples
--------
    python tools/tune_cos_american_preset.py --model gbm
    python tools/tune_cos_american_preset.py --model merton --Tmax 0.5
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
import argparse

import numpy as np


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from american_options import GBMCHF, MertonCHF, DiscreteEventJump  # noqa: E402
from american_options.engine import COSPricer, forward_price  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    S0: float = 100.0
    r: float = 0.01
    q: float = 0.00
    vol: float = 0.20
    T: float = 0.50
    # cash divs
    divs: dict[float, tuple[float, float]] = None

    def __post_init__(self):
        if self.divs is None:
            object.__setattr__(
                self,
                "divs",
                {
                    0.25: (0.8, 0.0),
                },
            )


def _build_chain(sc: Scenario, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    F0 = float(forward_price(sc.S0, sc.r, sc.q, sc.T, dict(sc.divs)))
    width = float(sc.vol) * float(np.sqrt(sc.T))
    K_lo = F0 * float(np.exp(-2.5 * width))
    K_hi = F0 * float(np.exp(+2.5 * width))
    K = np.linspace(K_lo, K_hi, n, dtype=float)
    is_call = np.zeros_like(K, dtype=bool)
    is_call[::2] = True
    return K, is_call


def _make_model(model_name: str, sc: Scenario):
    model_name = str(model_name).lower().strip()
    if model_name == "gbm":
        return GBMCHF(sc.S0, sc.r, sc.q, sc.divs, {"vol": sc.vol})
    if model_name == "merton":
        # Equity-like Merton parameters (log-jumps):
        # - moderate jump intensity
        # - negative mean jump (crash skew)
        # - sizeable jump vol
        params = {
            "vol": sc.vol,
            "lam": 0.6,
            "muJ": -0.10,
            "sigmaJ": 0.25,
        }
        return MertonCHF(sc.S0, sc.r, sc.q, sc.divs, params)
    raise ValueError(f"Unsupported model: {model_name}")


def _price(model_name: str, sc: Scenario, K: np.ndarray, is_call: np.ndarray, *, N: int, steps: int, L: float, event) -> np.ndarray:
    model = _make_model(model_name, sc)
    pricer = COSPricer(model, N=int(N), L=float(L))
    # Price puts and calls as two batches (faster during tuning than mixed recursion).
    out = np.empty_like(K, dtype=float)
    call_mask = np.asarray(is_call, dtype=bool)
    put_mask = ~call_mask
    if np.any(put_mask):
        out[put_mask] = np.asarray(
            pricer.american_price(K[put_mask], sc.T, steps=int(steps), is_call=False, use_softmax=False, event=event),
            dtype=float,
        )
    if np.any(call_mask):
        out[call_mask] = np.asarray(
            pricer.american_price(K[call_mask], sc.T, steps=int(steps), is_call=True, use_softmax=False, event=event),
            dtype=float,
        )
    return out


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return float(time.perf_counter() - t0), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbm", "merton"], default="merton")
    ap.add_argument("--Tmax", type=float, default=0.5, help="Max maturity (years) to cover")
    ap.add_argument("--nT", type=int, default=3, help="Number of maturities to test up to Tmax")
    ap.add_argument("--nK", type=int, default=100, help="Number of strikes in the chain")
    ap.add_argument("--refN", type=int, default=1536, help="Reference COS N")
    ap.add_argument("--refSteps", type=int, default=250, help="Reference time steps")
    args = ap.parse_args()

    sc0 = Scenario(T=float(args.Tmax))

    tol = 1e-4 * sc0.S0
    L = 10.0

    # Scheduled event (kept inside (0, Tmin] by scaling with maturity).
    base_event = DiscreteEventJump(time=0.30, p=0.5, u=0.10, d=-0.08, ensure_martingale=True)

    # Reference settings (high-ish but not insane)
    ref_N = int(args.refN)
    ref_steps = int(args.refSteps)

    Ts = np.linspace(float(args.Tmax) / float(args.nT), float(args.Tmax), int(args.nT), dtype=float)
    Ts = np.maximum(Ts, 1e-6)

    print("Model:", args.model)
    print("Base scenario (for div schedule):", sc0)
    print("Maturities:", Ts.tolist())
    print(f"Target tolerance: max_abs_err <= {tol:.6g} (= 1e-4*S0)")

    # Precompute (K, is_call) per maturity to keep moneyness range meaningful.
    chains = []
    for T in Ts:
        sc = Scenario(T=float(T))
        K, is_call = _build_chain(sc, n=int(args.nK))
        chains.append((sc, K, is_call))

    def scaled_event_for_T(T: float):
        # Keep event time inside (0,T] to ensure coverage.
        t_evt = float(min(max(0.5 * T, 1e-6), T))
        return DiscreteEventJump(time=t_evt, p=base_event.p, u=base_event.u, d=base_event.d, ensure_martingale=True)

    # Reference prices (no-event and with-event) for each maturity
    print(f"Computing reference: N={ref_N}, steps={ref_steps} ...")
    ref_no = []
    ref_ev = []
    for sc, K, is_call in chains:
        print(f"  [ref] T={sc.T:.6g} ...", flush=True)
        _, px_no = _time(lambda: _price(args.model, sc, K, is_call, N=ref_N, steps=ref_steps, L=L, event=None))
        _, px_ev = _time(lambda: _price(args.model, sc, K, is_call, N=ref_N, steps=ref_steps, L=L, event=scaled_event_for_T(sc.T)))
        ref_no.append(px_no)
        ref_ev.append(px_ev)

    candidates_N = [96, 128, 192, 256, 384, 512]
    candidates_steps = [25, 35, 50, 75, 100, 125, 150]

    def scan(label: str, ref_list: list[np.ndarray], ev_kind: str):
        best = None
        for steps in candidates_steps:
            for N in candidates_N:
                dt0 = 0.0
                worst_err = 0.0
                for (sc, K, is_call), ref in zip(chains, ref_list):
                    ev = None
                    if ev_kind == "event":
                        ev = scaled_event_for_T(sc.T)
                    dt, px = _time(lambda: _price(args.model, sc, K, is_call, N=N, steps=steps, L=L, event=ev))
                    dt0 += dt
                    worst_err = max(worst_err, float(np.max(np.abs(px - ref))))
                dt0 /= float(len(chains))
                err = worst_err
                ok = err <= tol
                print(f"{label} N={N:4d} steps={steps:3d}  avg_time_per_T={dt0:.4f}s  worst_max_abs_err={err:.6g}  {'OK' if ok else ''}")
                if ok:
                    best = (N, steps, dt0, err)
                    return best
        return best

    print("\n--- Tuning: no event ---")
    best_no = scan("no_event", ref_no, "none")

    print("\n--- Tuning: with event ---")
    best_ev = scan("with_event", ref_ev, "event")

    print("\nSummary:")
    print("best_no_event:", best_no)
    print("best_with_event:", best_ev)

    if best_no and best_ev:
        N_preset = max(best_no[0], best_ev[0])
        steps_preset = max(best_no[1], best_ev[1])
        print(f"\nSuggested preset (covers both): N={N_preset}, steps={steps_preset}, L={L}")


if __name__ == "__main__":
    main()
