"""Plot skew/curvature persistence charts from `figs/skew_persistence_by_model.csv`.

Produces a 1x2 matplotlib figure:
- Left: ATM skew vs maturity
- Right: ATM curvature vs maturity

Usage:
  python tools/plot_skew_persistence.py

By default it reads `figs/skew_persistence_by_model.csv` and writes:
  figs/skew_persistence_skew_curv.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

# Allow running as: `python tools/plot_skew_persistence.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot ATM skew and curvature vs maturity from a CSV.")
    p.add_argument("--in", dest="in_path", default="figs/skew_persistence_by_model.csv")
    p.add_argument("--out", dest="out_path", default="figs/skew_persistence_skew_curv.png")
    p.add_argument("--exclude", type=str, default="GBM", help="Comma-separated model names to exclude")
    p.add_argument("--logx", action="store_true", help="Use log-scale for maturity axis")
    args = p.parse_args(argv)

    rows = _read_csv(str(args.in_path))
    if not rows:
        raise RuntimeError(f"No rows found in {args.in_path}")

    exclude = {s.strip() for s in str(args.exclude).split(",") if s.strip()}

    # Group by model
    by_model: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    # (T, skew, curv)
    for r in rows:
        model = str(r["model"]).strip()
        if model in exclude:
            continue
        T = float(r["T"])
        skew = float(r["skew_atm"])
        curv = float(r["curv_atm"])
        if np.isfinite(T) and np.isfinite(skew) and np.isfinite(curv):
            by_model[model].append((T, skew, curv))

    if not by_model:
        raise RuntimeError("No model data to plot (check --exclude)")

    # Sort each model by maturity
    for model in list(by_model.keys()):
        by_model[model].sort(key=lambda t: t[0])

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    import matplotlib.pyplot as plt

    fig, (ax_skew, ax_curv) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)

    # Plot style: consistent colors per model (matplotlib default cycle)
    for model, pts in sorted(by_model.items(), key=lambda kv: kv[0]):
        T = np.array([p[0] for p in pts], dtype=float)
        skew = np.array([p[1] for p in pts], dtype=float)
        curv = np.array([p[2] for p in pts], dtype=float)

        ax_skew.plot(T, skew, marker="o", linewidth=1.7, markersize=4, label=model)
        ax_curv.plot(T, curv, marker="o", linewidth=1.7, markersize=4, label=model)

    if bool(args.logx):
        ax_skew.set_xscale("log")
        ax_curv.set_xscale("log")

    ax_skew.axhline(0.0, color="0.3", linewidth=1.0, alpha=0.5)
    ax_curv.axhline(0.0, color="0.3", linewidth=1.0, alpha=0.5)

    ax_skew.set_title("ATM skew persistence")
    ax_curv.set_title("ATM curvature persistence")

    ax_skew.set_xlabel("Maturity T (years)")
    ax_curv.set_xlabel("Maturity T (years)")

    ax_skew.set_ylabel("d(IV)/d ln(K/F) at k=0")
    ax_curv.set_ylabel("d²(IV)/d ln(K/F)² at k=0")

    ax_skew.grid(True, alpha=0.25)
    ax_curv.grid(True, alpha=0.25)

    # One legend for both panels (right side)
    handles, labels = ax_skew.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)

    fig.suptitle("Skew / Smile Persistence by Model", y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))

    fig.savefig(str(args.out_path), dpi=160)
    plt.close(fig)

    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()
