import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# ensure workspace root is on sys.path for local package imports
sys.path.insert(0, os.getcwd())

from american_options import GBMCHF, MertonCHF, cash_divs_to_proportional_divs  # noqa: E402
from american_options.engine import COSPricer  # noqa: E402


@dataclass(frozen=True)
class Problem:
    name: str
    model: str  # "gbm" | "merton"
    S0: float
    K: float
    T: float
    r: float
    q: float
    vol: float
    divs_cash: dict[float, tuple[float, float]]
    merton_lam: float = 0.0
    merton_muJ: float = 0.0
    merton_sigmaJ: float = 0.1


def _parse_divs_cash(spec: str) -> dict[float, tuple[float, float]]:
    """Parse dividends from a compact string.

    Format:
      "t:Dmean:Dstd,t:Dmean:Dstd,..."  (comma-separated)
    Example:
      "0.25:1.0:0,0.5:1.0:0,0.75:1.0:0"
    """
    spec = (spec or "").strip()
    if not spec:
        return {}
    out: dict[float, tuple[float, float]] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        fields = [f.strip() for f in p.split(":")]
        if len(fields) != 3:
            raise ValueError(f"Invalid dividend spec chunk: {p!r}")
        t, dm, ds = float(fields[0]), float(fields[1]), float(fields[2])
        out[float(t)] = (float(dm), float(ds))
    return out


def _make_pricers(prob: Problem, *, N: int, L: float, N_REF: int, L_REF: float) -> tuple[COSPricer, COSPricer]:
    divs_prop = cash_divs_to_proportional_divs(prob.S0, prob.r, prob.q, prob.divs_cash)
    model_kind = prob.model.strip().lower()
    if model_kind == "gbm":
        model = GBMCHF(prob.S0, prob.r, prob.q, divs_prop, {"vol": prob.vol})
    elif model_kind == "merton":
        model = MertonCHF(
            prob.S0,
            prob.r,
            prob.q,
            divs_prop,
            {"vol": prob.vol, "lam": prob.merton_lam, "muJ": prob.merton_muJ, "sigmaJ": prob.merton_sigmaJ},
        )
    else:
        raise ValueError(f"Unknown model kind: {prob.model!r}")

    return COSPricer(model, N=N, L=L), COSPricer(model, N=N_REF, L=L_REF)


def _median_wall_time(
    fn,
    *,
    repeats: int,
    warmups: int = 1,
    clear_american_ctx_cache: bool = False,
) -> tuple[float, float]:
    # Important benchmarking note:
    # COSPricer.american_price uses a shared class-level context cache keyed by (steps, N, M, ...).
    # If you run Richardson sweeps in increasing baseM, later runs will be artificially faster
    # because earlier runs already populated the cache for overlapping step counts.
    if clear_american_ctx_cache:
        COSPricer._AMERICAN_CTX_CACHE.clear()

    for _ in range(int(max(0, warmups))):
        if clear_american_ctx_cache:
            COSPricer._AMERICAN_CTX_CACHE.clear()
        fn()

    times = []
    out = None
    for _ in range(int(max(1, repeats))):
        if clear_american_ctx_cache:
            COSPricer._AMERICAN_CTX_CACHE.clear()
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(out), float(np.median(times))


def _price_manyM(pr: COSPricer, K: float, T: float, M: int, repeats: int) -> tuple[float, float]:
    Karr = np.array([K], dtype=float)

    def _run() -> float:
        return float(pr.american_price(Karr, T, steps=int(M), is_call=False, use_softmax=False)[0])

    return _median_wall_time(
        _run,
        repeats=repeats,
        warmups=_WARMUPS,
        clear_american_ctx_cache=_CLEAR_AMERICAN_CTX_CACHE,
    )


def _price_richardson(pr: COSPricer, K: float, T: float, baseM: int, repeats: int) -> tuple[float, float]:
    # vAM = (1/21)*(64*v8M - 56*v4M + 14*v2M - vM)
    Karr = np.array([K], dtype=float)

    def _run() -> float:
        Ms = [int(baseM), int(2 * baseM), int(4 * baseM), int(8 * baseM)]
        vals = [float(pr.american_price(Karr, T, steps=m, is_call=False, use_softmax=False)[0]) for m in Ms]
        vM, v2M, v4M, v8M = vals
        return float((1.0 / 21.0) * (64.0 * v8M - 56.0 * v4M + 14.0 * v2M - 1.0 * vM))

    return _median_wall_time(
        _run,
        repeats=repeats,
        warmups=_WARMUPS,
        clear_american_ctx_cache=_CLEAR_AMERICAN_CTX_CACHE,
    )


def main() -> None:
    # ---- Config via env vars (kept simple for quick iteration) ----
    # Use BENCH_* names to avoid collisions with other scripts.
    N = int(os.environ.get("BENCH_N", os.environ.get("N", "1024")))
    L = float(os.environ.get("BENCH_L", os.environ.get("L", "10.0")))
    repeats = int(os.environ.get("BENCH_REPEATS", os.environ.get("REPEATS", "1")))

    global _CLEAR_AMERICAN_CTX_CACHE, _WARMUPS
    _CLEAR_AMERICAN_CTX_CACHE = bool(int(os.environ.get("BENCH_CLEAR_AMERICAN_CTX_CACHE", "0")))
    _WARMUPS = int(os.environ.get("BENCH_WARMUPS", "1"))

    # Baseline settings (usually higher)
    N_REF = int(os.environ.get("BENCH_N_REF", os.environ.get("N_REF", str(max(N, 2048)))))
    L_REF = float(os.environ.get("BENCH_L_REF", os.environ.get("L_REF", str(L))))
    M_REF = int(os.environ.get("BENCH_M_REF", os.environ.get("M_REF", "2048")))

    # Grid of M values
    manyM_list = [int(x) for x in os.environ.get("BENCH_MANYM", os.environ.get("MANYM", "8,16,32,64,128,256,512")).split(",") if x.strip()]
    baseM_list = [int(x) for x in os.environ.get("BENCH_BASEM", os.environ.get("BASEM", "2,4,8,16,32,64")).split(",") if x.strip()]

    # Dividend schedule (cash): default to quarterly dividends.
    div_spec = os.environ.get("BENCH_DIVS", os.environ.get("DIVS", "0.25:1.0:0,0.5:1.0:0,0.75:1.0:0"))
    divs_cash = _parse_divs_cash(div_spec)

    # Shared option params.
    S0 = float(os.environ.get("S0", "100"))
    K = float(os.environ.get("K", "100"))
    T = float(os.environ.get("T", "1"))
    r = float(os.environ.get("R", "0.05"))
    q = float(os.environ.get("Q", "0.02"))
    vol = float(os.environ.get("VOL", "0.25"))

    # Merton jump params (reasonable defaults).
    m_lam = float(os.environ.get("MERTON_LAM", "0.2"))
    m_muJ = float(os.environ.get("MERTON_MUJ", "-0.1"))
    m_sigmaJ = float(os.environ.get("MERTON_SIGMAJ", "0.2"))

    problems = [
        Problem(
            name=os.environ.get("NAME_GBM", "GBM put (many dividends)"),
            model="gbm",
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            vol=vol,
            divs_cash=divs_cash,
        ),
        Problem(
            name=os.environ.get("NAME_MERTON", "Merton put (many dividends)"),
            model="merton",
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            vol=vol,
            divs_cash=divs_cash,
            merton_lam=m_lam,
            merton_muJ=m_muJ,
            merton_sigmaJ=m_sigmaJ,
        ),
    ]

    print("Benchmark: Bowen rollback many-M vs Richardson (discrete dividends)")
    print(f"Dividends (cash): {divs_cash}")
    print(f"Grid: N={N}, L={L}, repeats={repeats}")
    print(f"Timing: warmups={_WARMUPS}, clear_american_ctx_cache={_CLEAR_AMERICAN_CTX_CACHE}")
    print(f"Baseline: N_REF={N_REF}, L_REF={L_REF}, M_REF={M_REF}\n")

    rows = []

    for prob in problems:
        pr, pr_ref = _make_pricers(prob, N=N, L=L, N_REF=N_REF, L_REF=L_REF)
        print(f"== {prob.name} ==")
        if prob.model.lower() == "merton":
            print(f"Merton params: lam={prob.merton_lam}, muJ={prob.merton_muJ}, sigmaJ={prob.merton_sigmaJ}")
        print(f"Params: S0={prob.S0}, K={prob.K}, T={prob.T}, r={prob.r}, q={prob.q}, vol={prob.vol}")

        ref_price, ref_s = _price_manyM(pr_ref, prob.K, prob.T, M_REF, repeats=1)
        print(f"Reference (many-M): price={ref_price:.12f}  (wall={1e3*ref_s:.1f} ms)\n")

        # ---- many-M sweep ----
        for M in manyM_list:
            price, wall = _price_manyM(pr, prob.K, prob.T, M, repeats=repeats)
            err = abs(price - ref_price)
            rows.append(
                {
                    "scenario": prob.name,
                    "model": prob.model,
                    "method": "manyM",
                    "M": int(M),
                    "baseM": "",
                    "price": float(price),
                    "abs_error": float(err),
                    "wall_ms": float(1e3 * wall),
                }
            )
            print(f"manyM  M={M:4d}  price={price:.12f}  abs_err={err:.3e}  wall={1e3*wall:.1f} ms")

        print("")

        # ---- Richardson sweep ----
        for baseM in baseM_list:
            price, wall = _price_richardson(pr, prob.K, prob.T, baseM, repeats=repeats)
            err = abs(price - ref_price)
            rows.append(
                {
                    "scenario": prob.name,
                    "model": prob.model,
                    "method": "richardson",
                    "M": int(8 * baseM),
                    "baseM": int(baseM),
                    "price": float(price),
                    "abs_error": float(err),
                    "wall_ms": float(1e3 * wall),
                }
            )
            print(
                f"rich   baseM={baseM:4d} (uses M={8*baseM:4d})  price={price:.12f}  abs_err={err:.3e}  wall={1e3*wall:.1f} ms"
            )
        print("")

    # ---- Save CSV + plot ----
    os.makedirs("figs", exist_ok=True)
    csv_path = os.path.join("figs", "richardson_vs_manyM_discrete_divs.csv")

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    except Exception:
        import csv

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        def _sel(scenario: str, method: str) -> list[dict]:
            return [r for r in rows if r["scenario"] == scenario and r["method"] == method]

        scenarios = list(dict.fromkeys([r["scenario"] for r in rows]))
        if len(scenarios) == 0:
            raise RuntimeError("No results to plot")

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        markers = {"manyM": "o", "richardson": "s"}
        linestyles = {"manyM": "-", "richardson": "--"}

        for scenario in scenarios:
            for method in ("manyM", "richardson"):
                pts = _sel(scenario, method)
                ax.plot(
                    [p["wall_ms"] for p in pts],
                    [p["abs_error"] for p in pts],
                    marker=markers[method],
                    linestyle=linestyles[method],
                    linewidth=1.5,
                    label=f"{scenario} — {method}",
                )

        ax.set_yscale("log")
        ax.set_xlabel("Wall time (ms)")
        ax.set_ylabel("Absolute error vs reference")
        ax.set_title("Discrete dividends: accuracy vs wall time (many-M vs Richardson)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.legend()

        out_png = os.path.join("figs", "richardson_vs_manyM_discrete_divs.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"\nSaved: {csv_path}")
        print(f"Plot not generated (matplotlib issue): {e}")


if __name__ == "__main__":
    main()
