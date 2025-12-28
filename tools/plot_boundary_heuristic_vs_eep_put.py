"""Compare rollback-implied exercise boundary vs Andersen-EEP boundary (puts).

Produces a 1x2 figure:
- [A] continuous dividend yield q, no discrete dividends
- [B] discrete cash dividend at a known time, q=0

For each panel we plot:
- COS rollback-implied boundary (from intrinsic vs continuation on rollback grid)
- Andersen-EEP boundary (solve_american_put_boundary_eep)
- Heuristic "most likely next exercise node" markers for both curves

Usage
-----
C:/workspace/AmericanOptions/.venv/Scripts/python.exe tools/plot_boundary_heuristic_vs_eep_put.py --out figs/boundary_heuristic_vs_eep_put.png

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running as `python tools/<script>.py` without installing the package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class Case:
    name: str
    S0: float
    K: float
    r: float
    q: float
    T: float
    vol: float
    divs_cash: dict[float, tuple[float, float]]


def _compute_all(case: Case, *, N: int, L: float, M: int, steps_rb: int, steps_eep: int) -> dict:
    from american_options.engine import COSPricer, GBMCHF

    model = GBMCHF(S0=case.S0, r=case.r, q=case.q, divs=case.divs_cash, params={"vol": case.vol})
    pr = COSPricer(model, N=int(N), L=float(L), M=int(M))

    # Rollback-implied boundary.
    price_rb, (t_rb, bnd_rb_mat) = pr.american_price(
        np.array([case.K], dtype=float),
        float(case.T),
        steps=int(steps_rb),
        is_call=False,
        use_softmax=False,
        return_boundary=True,
    )
    t_rb = np.asarray(t_rb, dtype=float)
    bnd_rb = np.asarray(bnd_rb_mat, dtype=float)[:, 0]

    t_star_rb, b_star_rb, dp_star_rb = pr.estimate_next_exercise_node_from_boundary(
        t_rb,
        bnd_rb_mat,
        is_call=False,
        strike_index=0,
        spot=case.S0,
    )

    # Andersen-EEP boundary.
    t_eep, bnd_eep = pr.solve_american_put_boundary_eep(
        float(case.K),
        float(case.T),
        steps=int(steps_eep),
        bisect_tol=1e-8,
        enforce_prediv_zero=True,
        prediv_epsilon=1e-6,
    )
    t_eep = np.asarray(t_eep, dtype=float)
    bnd_eep = np.asarray(bnd_eep, dtype=float)

    t_star_eep, b_star_eep, dp_star_eep = pr.estimate_next_exercise_node_from_boundary(
        t_eep,
        bnd_eep,
        is_call=False,
        spot=case.S0,
    )

    # Sanity price from EEP boundary (optional, for annotation/debug)
    amer_eep, euro_eep, eep = pr.american_put_price_eep_from_boundary(
        float(case.K),
        float(case.T),
        t_eep,
        bnd_eep,
        spot=case.S0,
    )

    return {
        "case": case,
        "pricer": pr,
        "price_rb": float(np.asarray(price_rb, dtype=float)[0]),
        "t_rb": t_rb,
        "bnd_rb": bnd_rb,
        "t_star_rb": float(t_star_rb),
        "b_star_rb": float(b_star_rb),
        "dp_star_rb": float(dp_star_rb),
        "t_eep": t_eep,
        "bnd_eep": bnd_eep,
        "t_star_eep": float(t_star_eep),
        "b_star_eep": float(b_star_eep),
        "dp_star_eep": float(dp_star_eep),
        "amer_eep": float(amer_eep),
        "euro_eep": float(euro_eep),
        "eep": float(eep),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figs/boundary_heuristic_vs_eep_put.png")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--steps_rb", type=int, default=120)
    parser.add_argument("--steps_eep", type=int, default=80)
    args = parser.parse_args()

    # Shared baseline params
    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    vol = 0.25

    # [A] Continuous yield (no discrete dividends)
    case_a = Case(
        name="Continuous yield",
        S0=S0,
        K=K,
        r=r,
        q=0.02,
        T=T,
        vol=vol,
        divs_cash={},
    )

    # [B] Single discrete cash dividend (q=0), sized as 10% of expected pre-div forward.
    t_div = 0.5
    m = 0.10
    expected_pre = S0 * math.exp((r - 0.0) * t_div)
    D = m * expected_pre
    case_b = Case(
        name="Discrete cash dividend",
        S0=S0,
        K=K,
        r=r,
        q=0.0,
        T=T,
        vol=vol,
        divs_cash={float(t_div): (float(D), 0.0)},
    )

    res_a = _compute_all(case_a, N=args.N, L=args.L, M=args.M, steps_rb=args.steps_rb, steps_eep=args.steps_eep)
    res_b = _compute_all(case_b, N=args.N, L=args.L, M=args.M, steps_rb=args.steps_rb, steps_eep=args.steps_eep)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
        for ax, res, tag in zip(axes, [res_a, res_b], ["[A]", "[B]"]):
            case = res["case"]

            ax.plot(res["t_rb"], res["bnd_rb"], lw=1.8, label="Rollback implied boundary")
            ax.plot(res["t_eep"], res["bnd_eep"], lw=1.8, ls="--", label="Andersen-EEP boundary")

            ax.scatter(
                [res["t_star_rb"]],
                [res["b_star_rb"]],
                s=45,
                marker="o",
                label=f"Heuristic node (rollback)  dp={res['dp_star_rb']:.3g}",
                zorder=5,
            )
            ax.scatter(
                [res["t_star_eep"]],
                [res["b_star_eep"]],
                s=55,
                marker="x",
                label=f"Heuristic node (EEP)       dp={res['dp_star_eep']:.3g}",
                zorder=6,
            )

            ax.set_title(
                f"{tag} {case.name}\nS0={case.S0:g}, K={case.K:g}, r={case.r:g}, q={case.q:g}, vol={case.vol:g}, T={case.T:g}"
            )
            ax.set_xlabel("t")
            ax.grid(True, ls="--", lw=0.5, alpha=0.7)

            # Mark discrete dividend time if present
            if case.divs_cash:
                for td in sorted(case.divs_cash.keys()):
                    ax.axvline(float(td), color="k", lw=0.8, alpha=0.25)

        axes[0].set_ylabel("Boundary B(t) (spot level)")
        axes[1].legend(loc="best", fontsize=8)

        fig.suptitle("American PUT boundary: rollback implied vs Andersen-EEP + marginal-probability heuristic", y=1.02)
        plt.tight_layout()
        plt.savefig(args.out, dpi=160)
        plt.close(fig)
        print(f"Wrote {args.out}")

    except Exception as e:
        print(f"Plot not generated (matplotlib issue): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
