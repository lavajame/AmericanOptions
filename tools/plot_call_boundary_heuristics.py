"""Call early-exercise boundary inspection (COS rollback) with cheap vs marginal-probability heuristics.

Produces a 1x2 matplotlib figure:
- [A] continuous dividend yield q (no discrete dividends)
- [B] discrete cash dividend (q=0)

For each panel we compute the rollback-implied call boundary B(t) and overlay markers for:
1) Marginal-probability heuristic (extra work):
   - p_i ≈ P(S_{t_i} >= B(t_i)) using COS digitals
   - choose node with largest positive increment Δp_i
2) Free heuristic (boundary-only):
   - choose earliest node where B(t_i) <= S0 (immediate/near exercise at spot)
   - else choose node minimizing (B(t_i) - S0) over B>S0
3) Free heuristic (rollback continuation-at-spot):
   - using return_continuation_trajectory, find earliest node where intrinsic(S0) >= cont(S0,t)
   - fallback to node minimizing |intrinsic(S0) - cont(S0,t)|

Usage
-----
C:/workspace/AmericanOptions/.venv/Scripts/python.exe tools/plot_call_boundary_heuristics.py --out figs/call_boundary_heuristics.png

"""

from __future__ import annotations

import argparse
import math
import os
    vol = 0.05

    # [A] Continuous yield: use q>0 to make early exercise relevant for calls.
    case_a = Case(
        name="Continuous yield",
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        vol=vol,
        divs_cash={},
    )

    # [B] Discrete cash dividend, q=0.
    t_div = 0.125
    m = 0.01
    expected_pre = S0 * math.exp((r - q) * t_div)
    D = m * expected_pre
    # Add a second discrete cash dividend at t=0.35
    t_div2 = 0.35
    m2 = 0.01
    expected_pre2 = S0 * math.exp((r - q) * t_div2)
    D2 = m2 * expected_pre2
    case_b = Case(
        name="Discrete cash dividend (two)",
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        vol=vol,
        divs_cash={float(t_div): (float(D), 0.0), float(t_div2): (float(D2), 0.0)},
    )

    # run both continuous and discrete cases under MertonCHF (symmetric jumps)
    from american_options.models import MertonCHF
    merton_params = {"sigma": 0.03, "lam": 0.5, "muJ": -0.04, "sigmaJ": 0.02}
    print(f"Effective Merton volatility: {float(np.sqrt(vol**2 + merton_params['lam'] * (merton_params['muJ']**2 + merton_params['sigmaJ']**2))):.6f}")
    res_a_merton = _compute_case(case_a, N=args.N, L=args.L, M=args.M, steps=args.steps, model_cls=MertonCHF, model_params=merton_params)
    res_b_merton = _compute_case(case_b, N=args.N, L=args.L, M=args.M, steps=args.steps, model_cls=MertonCHF, model_params=merton_params)

    # Print concise benchmark / selection summary to console
    def _print_summary(res: dict) -> None:
        case = res["case"]
        print("---")
        print(f"Case: {case.name}")
        print(f"Selected (COS marginal): t={res['t_star_prob']:.6f}, B={res['b_star_prob']:.6f}")
        print(f"Selected (BS approx):   t={res['t_star_bs']:.6f}, B={res['b_star_bs']:.6f}")
        if "time_marginal_avg" in res:
            print(f"Avg time (COS marginal scan): {res['time_marginal_avg']:.6f}s")
        print(f"Avg time (BS scan):           {res.get('time_bs_avg', res['time_bs']):.6f}s")
        # If this case has discrete dividends, show probabilities at those times
        if case.divs_cash:
            divs = sorted([float(t) for t in case.divs_cash.keys()])
            t_eval = res['t_eval']
            b_eval = res['b_eval']
            p_marg = res['p_marg']
            p_bs = res['p_bs']
            print("Dividend times and marginal probabilities:")
            for dv in divs:
                # find nearest index
                j = int(np.argmin(np.abs(t_eval - float(dv))))
                print(f"  t={t_eval[j]:.6f}: B={b_eval[j]:.6f}, p_marg={p_marg[j]:.6f}, p_bs={p_bs[j]:.6f}")
            # Additional diagnostics: compare cumulant-based log-mean vs forward-based log-mean
            try:
                from american_options.dividends import forward_price
                for dv in divs:
                    j = int(np.argmin(np.abs(t_eval - float(dv))))
                    ti = float(t_eval[j])
                    B = float(b_eval[j])
                    c1, c2, _ = res['pricer'].model.cumulants(float(ti))
                    fwd = forward_price(res['pricer'].model.S0, res['pricer'].model.r, res['pricer'].model.q, float(ti), res['pricer'].model.divs)
                    ln_fwd = float(np.log(max(fwd, 1e-300)))
                    # implied log-mean from cumulants (c1) vs implied from forward assuming log-normal: ln(fwd) - 0.5*c2
                    ln_from_cumulant = float(c1)
                    ln_from_forward = float(ln_fwd - 0.5 * float(c2))
                    # alternative BS probability anchored at forward (mu_alt)
                    mu_alt = ln_from_forward
                    d_alt = (mu_alt - float(np.log(max(B, 1e-300)))) / float(np.sqrt(max(c2, 1e-300))) if c2 > 0.0 else (1.0 if res['case'].S0 >= B else 0.0)
                    p_bs_alt = float(0.5 * (1.0 + math.erf(d_alt / math.sqrt(2.0)))) if c2 > 0.0 else (1.0 if res['case'].S0 >= B else 0.0)
                    # compute forward-anchored sigma used in BS proxy (sqrt(c2/T))
                    sigma_forward = float(np.sqrt(max(c2, 0.0) / float(max(ti, 1e-300)))) if c2 > 0.0 else 0.0
                    # compute BS prob using forward-anchored sigma (what we now plot)
                    denom_f = sigma_forward * math.sqrt(float(ti)) if sigma_forward > 0.0 else 1e-300
                    mu_forward = ln_fwd - 0.5 * (sigma_forward ** 2) * float(ti)
                    d_f = (mu_forward - float(np.log(max(B, 1e-300)))) / float(max(denom_f, 1e-300))
                    p_bs_forward = float(0.5 * (1.0 + math.erf(d_f / math.sqrt(2.0))))
                    # also compute BS prob using cumulant-anchored mu=c1 and sigma from c2 (conventional)
                    sigma_cumulant = float(np.sqrt(max(c2, 0.0) / float(max(ti, 1e-300)))) if c2 > 0.0 else 0.0
                    denom_c = sigma_cumulant * math.sqrt(float(ti)) if sigma_cumulant > 0.0 else 1e-300
                    d_c = (float(c1) - float(np.log(max(B, 1e-300)))) / float(max(denom_c, 1e-300))
                    p_bs_cumulant = float(0.5 * (1.0 + math.erf(d_c / math.sqrt(2.0))))
                    print(
                        f"  diag t={ti:.6f}: c1={c1:.6f}, c2={c2:.6f}, fwd={fwd:.6f}, sigma_forward={sigma_forward:.6f}, B={B:.6f}, p_marg={p_marg[j]:.6f}, p_bs_forward={p_bs_forward:.6f}, p_bs_cumulant={p_bs_cumulant:.6f}"
                    )
            except Exception:
                pass

    # compute running-max based selection for both results
    def _compute_running_choice(res: dict, running_thresh: float) -> None:
        try:
            p_marg = np.asarray(res.get("p_marg", []), dtype=float)
            t_eval = np.asarray(res.get("t_eval", []), dtype=float)
            b_eval = np.asarray(res.get("b_eval", []), dtype=float)
            if p_marg.size == 0:
                return
            global_max = float(np.max(p_marg))
            # running max (non-decreasing)
            running_max = np.maximum.accumulate(p_marg)
            # earliest index where p_marg reaches threshold * global_max
            thresh_val = running_thresh * global_max
            idxs = np.where(p_marg >= thresh_val)[0]
            if idxs.size > 0:
                idx = int(idxs[0])
            else:
                # fallback to global maximum index
                idx = int(np.argmax(p_marg))
            res["p_marg_running"] = running_max.tolist()
            res["running_thresh"] = float(running_thresh)
            res["t_star_running"] = float(t_eval[idx])
            res["b_star_running"] = float(b_eval[idx])
            res["p_star_running"] = float(p_marg[idx])
        except Exception:
            return

    _compute_running_choice(res_a_merton, args.running_thresh)
    print("--- Merton discrete case ---")
    _print_summary(res_b_merton)
    _compute_running_choice(res_b_merton, args.running_thresh)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), sharey=True)
        for ax, res, tag in zip(axes, [res_a_merton, res_b_merton], ["[A]", "[B]"]):
            case = res["case"]

            # Plot boundary more prominently
            ax.plot(res["t"], res["b"], lw=2.0, label="Rollback implied boundary", alpha=0.9, color="#2b83ba")

            # Markers (only plot when boundary value is finite)
            def _safe_scatter(ax, tx, bx, **kwargs):
                if tx is None or bx is None:
                    return None
                try:
                    txf = float(tx)
                    bxf = float(bx)
                except Exception:
                    return None
                if not np.isfinite(bxf) or not np.isfinite(txf):
                    return None
                return ax.scatter([txf], [bxf], **kwargs)

            _safe_scatter(
                ax,
                res["t_star_prob"],
                res["b_star_prob"],
                s=80,
                marker="o",
                zorder=6,
                facecolors="none",
                edgecolors="#b35806",
                linewidths=1.6,
                label=f"Marginal-prob node (Δp={res['dp_star']:.3g})",
            )

            

            # BS-approx marker (cheap)
            _safe_scatter(
                ax,
                res["t_star_bs"],
                res["b_star_bs"],
                s=100,
                marker="^",
                zorder=8,
                facecolors="none",
                edgecolors="#7b3294",
                linewidths=1.6,
                label=f"BS-approx node",
            )

            

            # Annotate each visible marker with callout lines and label with time/level and timing cost
            # avoid overlapping annotations by offsetting when markers coincide
            annotation_counts: dict[tuple[int, int], int] = {}
            def _annotate_marker(ax, tx, bx, text, color="#000000"):
                try:
                    txf = float(tx)
                    bxf = float(bx)
                except Exception:
                    return
                if not (np.isfinite(txf) and np.isfinite(bxf)):
                    return
                # key by rounded coords to group coincident markers
                key = (int(round(txf, 6) * 1e6), int(round(bxf, 6) * 1e6))
                idx = annotation_counts.get(key, 0)
                annotation_counts[key] = idx + 1
                # base offsets
                dx = 0.03 * (case.T or 1.0)
                dy = 0.03 * case.K
                # apply staggered offset when multiple annotations share the same point
                # alternate above/below and spread horizontally to avoid stacked boxes
                sign = 1 if (idx % 2 == 0) else -1
                offset_y = (idx // 2 + 1) * dy * sign
                offset_x = ((idx % 3) - 1) * dx * (1 + (idx // 3) * 0.25)
                # short horizontal and vertical guide lines
                ax.hlines(bxf, txf - 0.02 * (case.T or 1.0), txf + 0.02 * (case.T or 1.0), colors=color, linewidth=0.8, alpha=0.9)
                ax.vlines(txf, bxf - 0.02 * case.K, bxf + 0.02 * case.K, colors=color, linewidth=0.8, alpha=0.9)
                # annotation text placed offset to avoid overlap
                ax.annotate(
                    text,
                    xy=(txf, bxf),
                    xytext=(txf + dx + offset_x, bxf + offset_y),
                    fontsize=8,
                    color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.7),
                )

            # Marginal-prob annotation
            _annotate_marker(
                ax,
                res["t_star_prob"],
                res["b_star_prob"],
                f"marginal\nt={res['t_star_prob']:.3f}\nB={res['b_star_prob']:.3f}\n{res['time_marginal']:.3f}s",
                color="#b35806",
            )
            

            # BS-approx annotation
            _annotate_marker(
                ax,
                res["t_star_bs"],
                res["b_star_bs"],
                f"BS\nt={res['t_star_bs']:.3f}\nB={res['b_star_bs']:.3f}\n{res['time_bs']:.6f}s",
                color="#7b3294",
            )

            # Running-max selection annotation (earliest time reaching threshold*global_max)
            if res.get("t_star_running", None) is not None:
                _safe_scatter(
                    ax,
                    res["t_star_running"],
                    res["b_star_running"],
                    s=90,
                    marker="s",
                    zorder=7,
                    facecolors="none",
                    edgecolors="#1a9850",
                    linewidths=1.6,
                    label=f"Running-max node (th={res.get('running_thresh', 0.0):.2f})",
                )
                _annotate_marker(
                    ax,
                    res.get("t_star_running"),
                    res.get("b_star_running"),
                    f"run\nt={res.get('t_star_running'):.3f}\nB={res.get('b_star_running'):.3f}\np={res.get('p_star_running'):.3f}",
                    color="#1a9850",
                )

                # Secondary y-axis: plot marginal and BS probabilities through time
                try:
                    t_series = np.asarray(res.get("t_eval", []), dtype=float)
                    p_marg_series = np.asarray(res.get("p_marg", []), dtype=float)
                    p_bs_series = np.asarray(res.get("p_bs", []), dtype=float)
                    ax2 = ax.twinx()
                    if t_series.size and p_marg_series.size:
                        ax2.plot(t_series, p_marg_series, color="#b35806", lw=1.5, label="p_marg")
                    if t_series.size and p_bs_series.size:
                        ax2.plot(t_series, p_bs_series, color="#7b3294", lw=1.2, ls="--", label="p_bs")
                    if res.get("p_marg_running", None) is not None:
                        p_run = np.asarray(res.get("p_marg_running", []), dtype=float)
                        if p_run.size:
                            ax2.plot(t_series, p_run, color="#1a9850", lw=1.2, ls=":", label="p_marg_running")
                    ax2.set_ylabel("Probability")
                    ax2.set_ylim(0.0, 1.0)
                    ax2.grid(False)
                    # combine legends from primary and secondary axes
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
                except Exception:
                    pass

            

            ax.set_title(
                f"{tag} Call boundary: {case.name}\nS0={case.S0:g}, K={case.K:g}, r={case.r:g}, q={case.q:g}, vol={case.vol:g}, T={case.T:g}"
            )
            ax.set_xlabel("t")
            ax.grid(True, ls="--", lw=0.5, alpha=0.7)

            # Mark discrete dividend time(s)
            if case.divs_cash:
                for td in sorted(case.divs_cash.keys()):
                    ax.axvline(float(td), color="k", lw=0.9, alpha=0.25)

        axes[0].set_ylabel("Boundary B(t) (spot level)")
        fig.suptitle("American CALL: rollback implied boundary + 'free' vs marginal-probability next-node heuristics", y=1.02)
        plt.tight_layout()
        plt.savefig(args.out, dpi=160)
        plt.close(fig)
        print(f"Wrote {args.out}")

    except Exception as e:
        print(f"Plot not generated (matplotlib issue): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
