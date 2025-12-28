import argparse
import csv
import math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scenario", default="B_call")
    ap.add_argument("--t", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["scenario"] == args.scenario:
                rows.append(row)

    if not rows:
        raise SystemExit(f"No rows for scenario={args.scenario}")

    r0 = rows[0]
    print(
        args.scenario,
        "t_star_prob=",
        r0.get("t_star_prob"),
        "t_star_bs=",
        r0.get("t_star_bs_jump"),
    )

    target = float(args.t)
    for row in rows:
        if abs(float(row["t"]) - target) < 1e-12:
            print(
                "t=",
                target,
                "b=",
                row["b"],
                "p_marg=",
                row["p_marg"],
                "p_bs=",
                row["p_bs"],
            )
            break

    rows_sorted = sorted(rows, key=lambda rr: float(rr["t"]))
    idx = min(range(len(rows_sorted)), key=lambda i: abs(float(rows_sorted[i]["t"]) - target))
    lo = max(0, idx - int(args.window))
    hi = min(len(rows_sorted), idx + int(args.window) + 1)

    print("--- around t ---")
    for i in range(lo, hi):
        t = float(rows_sorted[i]["t"])
        b = float(rows_sorted[i]["b"])
        p = float(rows_sorted[i]["p_marg"])
        lp = -math.inf if p <= 0.0 else math.log(p)
        if i > 0:
            pprev = float(rows_sorted[i - 1]["p_marg"])
            lpp = -math.inf if pprev <= 0.0 else math.log(pprev)
            dln = lp - lpp
        else:
            dln = float("nan")
        print(f"i={i:4d} t={t:.12g} b={b:.6g} p={p:.3e} ln(p)={lp:.6g} dln={dln:.6g}")

    pmax = max(float(rr["p_marg"]) for rr in rows)
    print("max p_marg=", pmax)


if __name__ == "__main__":
    main()
