"""Clean 2x2 plotting: top row boundaries for CALLs with optimal markers; bottom row boundaries for PUTs.

Usage examples:
    python tools/plot_call_boundary_heuristics_clean.py --out figs/out.png --model Merton --model-params '{"sigma":0.03, "lam":0.5, "muJ":0.0, "sigmaJ":0.2}' --divs "0.125:1.0;0.35:1.0"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ensure repo root on path so we can import package modules
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from american_options.engine import COSPricer, GBMCHF
from american_options.models import MertonCHF
from american_options.dividends import forward_price


def parse_divs(s: str) -> dict:
    """Parse semicolon separated `t:D` pairs into dict{t: (D, 0.0)}."""
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


def make_model_cls(name: str):
    n = name.strip().lower()
    if n == "merton":
        return MertonCHF
    if n == "gbm":
        return None
    if n == "cgmy":
        try:
            from american_options.models import CGMYCHF

            return CGMYCHF
        except Exception:
            return None
    return None


def compute_case(
    case,
    N,
    L,
    M,
    steps,
    is_call=True,
    model_cls=None,
    model_params=None,
    *,
    compute_marginal: bool = True,
    convex_p_min: float = 1e-7,
    convex_standout_mult: float = 10.0,
):
    """Compute rollback boundary and identify the "most likely next" early-ex node.

    Selection rule (for both COS marginal and BS proxy):
      - Compute p(t) at the rollback-implied boundary B(t)
      - Consider points with sufficiently large probability (p >= convex_p_min)
      - Pick the earliest index achieving the maximum positive increment Δp
        where Δp_i := p(t_i) - p(t_{i-1})
      - If there is no positive Δp among eligible points, fallback to the final timepoint

    Boundary evaluation at inserted dividend times is stepwise (left-/right-limit)
    to avoid interpolating across discontinuities.
    """
    if model_cls is None:
        model = GBMCHF(
            S0=case['S0'],
            r=case['r'],
            q=case['q'],
            divs=case['divs_cash'],
            params={'vol': case['vol']},
        )
    else:
        model = model_cls(
            S0=case['S0'],
            r=case['r'],
            q=case['q'],
            divs=case['divs_cash'],
            params=(model_params or {}),
        )
    pr = COSPricer(model, N=int(N), L=float(L), M=int(M))

    price, traj, cont_traj, (t_grid, bnd_mat) = pr.american_price(
        np.array([case['K']], dtype=float),
        float(case['T']),
        steps=int(steps),
        is_call=bool(is_call),
        use_softmax=False,
        return_trajectory=True,
        return_continuation_trajectory=True,
        return_boundary=True,
    )

    t = np.asarray(t_grid, dtype=float)
    b = np.asarray(bnd_mat, dtype=float)[:, 0]
    div_times = sorted([float(tt) for tt in (case['divs_cash'].keys() if case['divs_cash'] else [])])
    t_eval = np.asarray(sorted(set(list(t.tolist()) + div_times)), dtype=float)

    # When evaluating digitals / proxy probabilities, avoid using timestamps that
    # coincide exactly with dividend event times by shifting the *time* by a tiny
    # epsilon while keeping the boundary level B(t) fixed at t.
    #
    # Convention at dividend times:
    #   - CALLs: evaluate at the left-limit (t - eps) since optimal call exercise is
    #            just BEFORE a cash dividend.
    #   - PUTs:  evaluate at the right-limit (t + eps) to stay consistent with the
    #            post-div boundary sampling used for puts and to avoid the event-time
    #            discontinuity.
    diffs = np.diff(np.unique(t_eval))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    min_dt = float(np.min(diffs)) if diffs.size else 1e-6
    eps_t = float(min(1e-6, 0.01 * min_dt))

    def _t_for_prob(ti: float, *, is_call_local: bool, atol: float = 1e-12) -> float:
        """Time used for probability evaluation at boundary level B(ti).

        For dividend timestamps:
          - calls: left-limit (ti - eps_t)
          - puts:  right-limit (ti + eps_t)
        """
        ti = float(ti)
        if ti <= 0.0:
            return 0.0
        if div_times:
            for dv in div_times:
                if math.isclose(ti, float(dv), abs_tol=atol):
                    if is_call_local:
                        return float(min(max(ti - eps_t, 0.0), float(case['T'])))
                    return float(min(max(ti + eps_t, 0.0), float(case['T'])))
        return float(min(ti, float(case['T'])))

    def _boundary_eval_with_div_jumps(
        t_src: np.ndarray,
        b_src: np.ndarray,
        t_query: np.ndarray,
        div_ts: list[float],
        *,
        is_call_local: bool,
        atol: float = 1e-12,
    ) -> np.ndarray:
        """Evaluate boundary on the rollback grid, handling dividend timestamps robustly.

        For non-dividend times: left-limit (largest k with t_src[k] <= t).

        For dividend times (t close to an element of div_ts):
                    - calls: if t_src contains t exactly, use that node; otherwise use strictly
                                     pre-div boundary (largest k with t_src[k] < t)
          - puts:  use strictly post-div boundary (smallest k with t_src[k] > t)

        This avoids ambiguity if the rollback grid contains the dividend time itself.
        """
        t_src = np.asarray(t_src, dtype=float)
        b_src = np.asarray(b_src, dtype=float)
        tq = np.asarray(t_query, dtype=float)
        idx = np.searchsorted(t_src, tq, side='right') - 1  # default left-limit

        if div_ts:
            div_arr = np.asarray(div_ts, dtype=float)
            is_div = np.zeros_like(tq, dtype=bool)
            for dv in div_arr:
                is_div |= np.isclose(tq, dv, atol=atol)

            if np.any(is_div):
                tq_div = tq[is_div]
                if is_call_local:
                    # Prefer the exact node if present; else strictly pre-div.
                    left = np.searchsorted(t_src, tq_div, side='left')
                    idx_div = left - 1
                    # Exact match: left is the first index where t_src[idx] >= tq.
                    # If that equals tq (within atol), use it.
                    exact = (left < len(t_src)) & np.isclose(t_src[np.clip(left, 0, len(t_src) - 1)], tq_div, atol=atol)
                    if np.any(exact):
                        idx_div = np.where(exact, left, idx_div)
                else:
                    # strictly post-div
                    idx_div = np.searchsorted(t_src, tq_div, side='right')
                idx[is_div] = idx_div

        idx = np.clip(idx, 0, len(t_src) - 1)
        return b_src[idx]

    # Use a single boundary definition everywhere: dividend-aware stepwise boundary
    # evaluated on the analysis grid t_eval.
    b_eval = _boundary_eval_with_div_jumps(t, b, t_eval, div_times, is_call_local=bool(is_call))

    # handle case where no finite boundary exists
    has_boundary = np.any(np.isfinite(b))

    # Marginal probabilities via COS digitals at the boundary level.
    # This is the expensive part; allow skipping via compute_marginal=False.
    p_marg = np.full_like(t_eval, np.nan, dtype=float)
    if compute_marginal:
        p_marg = np.zeros_like(t_eval, dtype=float)
        for i, ti in enumerate(t_eval):
            if ti <= 0.0:
                # degenerate
                p_marg[i] = 1.0 if (case['S0'] >= b_eval[i] if is_call else case['S0'] <= b_eval[i]) else 0.0
                continue
            t_prob = _t_for_prob(float(ti), is_call_local=bool(is_call))

            # p_marg at b_eval
            price_disc = float(pr.digital_put_price(np.array([float(b_eval[i])]), float(t_prob), spot=case['S0'])[0])
            prob_put = price_disc / max(np.exp(-float(pr.model.r) * float(t_prob)), 1e-300)
            prob_put = float(min(max(prob_put, 0.0), 1.0))
            p_marg[i] = (1.0 - prob_put) if is_call else prob_put

    def _select_max_positive_increment_else_final(
        t_grid: np.ndarray,
        p: np.ndarray,
        *,
        p_min: float = 1e-7,
    ) -> int | None:
        """Pick earliest index with maximum positive increment Δp.

        Eligible i are those with p[i] >= p_min and finite.
        Uses Δp_i = p[i] - p[i-1] for i>=1.
        If no positive Δp among eligible indices, return the final index.
        """
        _ = np.asarray(t_grid, dtype=float)  # only for length/shape validation
        p_arr = np.asarray(p, dtype=float)
        n = int(len(p_arr))
        if n <= 0:
            return None
        if n == 1:
            return 0

        p_min = float(p_min)
        best_i: int | None = None
        best_dp = -np.inf
        for i in range(1, n):
            if (not np.isfinite(p_arr[i])) or float(p_arr[i]) < p_min:
                continue
            if not np.isfinite(p_arr[i - 1]):
                continue
            dp = float(p_arr[i] - p_arr[i - 1])
            if dp <= 0.0:
                continue
            if (best_i is None) or (dp > best_dp + 1e-18):
                best_i = int(i)
                best_dp = dp
        return int(n - 1) if best_i is None else int(best_i)

    t_star_prob = None
    b_star_prob = None
    if compute_marginal and p_marg.size:
        i_star = _select_max_positive_increment_else_final(
            t_eval,
            p_marg,
            p_min=float(convex_p_min),
        )
        if i_star is None:
            i_star = 0
        t_star_prob = float(t_eval[i_star])
        b_star_prob = float(b_eval[i_star])

    # BS approx probabilities (forward-anchored using cumulant c2)
    p_bs = np.zeros_like(t_eval, dtype=float)
    for i, ti in enumerate(t_eval):
        if ti <= 0.0:
            p_bs[i] = 1.0 if (case['S0'] >= b_eval[i] if is_call else case['S0'] <= b_eval[i]) else 0.0
            continue
        t_prob = _t_for_prob(float(ti), is_call_local=bool(is_call))
        try:
            c1, c2, _ = pr.model.cumulants(float(t_prob))
            c2 = float(max(c2, 0.0))
        except Exception:
            c2 = float(max(pr.model._var2(float(t_prob)), 0.0))
        # forward_price expects the external *cash dividend* convention. `pr.model.divs`
        # is the internal proportional-dividend dict, so use the case's cash-div schedule.
        fwd = float(forward_price(pr.model.S0, pr.model.r, pr.model.q, float(t_prob), case['divs_cash']))
        ln_fwd = float(np.log(max(fwd, 1e-300)))
        B = float(max(float(b_eval[i]), 1e-300))
        sigma = float(np.sqrt(c2 / float(max(t_prob, 1e-300))))
        if sigma <= 0.0 or c2 <= 0.0:
            p_bs[i] = 1.0 if (case['S0'] >= b_eval[i] if is_call else case['S0'] <= b_eval[i]) else 0.0
            continue
        mu = ln_fwd - 0.5 * (sigma ** 2) * float(t_prob)
        denom = sigma * math.sqrt(float(t_prob))
        d_val = (mu - float(np.log(B))) / float(max(denom, 1e-300))
        cdf = float(0.5 * (1.0 + math.erf(d_val / math.sqrt(2.0))))
        # For the forward-anchored lognormal proxy:
        #   cdf = Phi((mu - ln(B)) / (sigma*sqrt(t))) = Phi(d)
        # This equals P(S_t >= B) for a lognormal with mu = ln(fwd) - 0.5*sigma^2*t.
        # Therefore for calls use `cdf`, for puts use `1 - cdf`.
        p_bs[i] = cdf if is_call else (1.0 - cdf)

    # BS proxy selection: max positive increment Δp of p_bs for p>=p_min, else final point
    t_star_bs_jump = None
    b_star_bs_jump = None
    p_star_bs_jump = None
    if p_bs.size:
        idx_bs = _select_max_positive_increment_else_final(
            t_eval,
            p_bs,
            p_min=float(convex_p_min),
        )
        if idx_bs is not None:
            t_star_bs_jump = float(t_eval[idx_bs])
            b_star_bs_jump = float(b_eval[idx_bs])
            p_star_bs_jump = float(p_bs[idx_bs])

    return {
        'case': case,
        'pricer': pr,
        't': t,
        'b': b,
        't_eval': t_eval,
        'b_eval': b_eval,
        'p_marg': p_marg,
        'p_bs': p_bs,
        't_star_prob': t_star_prob,
        'b_star_prob': b_star_prob,
        't_star_bs_jump': t_star_bs_jump,
        'b_star_bs_jump': b_star_bs_jump,
        'p_star_bs_jump': p_star_bs_jump,
        'has_boundary': has_boundary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='figs/call_boundary_heuristics_merton_2x2_clean.png')
    parser.add_argument(
        '--with-marginal',
        action='store_true',
        help='Also compute COS-digital marginal probabilities (p_marg) and the corresponding marker (slower).',
    )
    parser.add_argument(
        '--cheap',
        action='store_true',
        help='(deprecated) BS-only is now the default; use --with-marginal to enable p_marg.',
    )
    parser.add_argument(
        '--csv-out',
        type=str,
        default='',
        help='Optional CSV output path. Writes B_call (discrete dividend call) time series including BS proxy probability p_bs(t).',
    )
    parser.add_argument('--dpi', type=int, default=160, help='DPI for output PNG (default: 160).')
    parser.add_argument('--N', type=int, default=2**8)
    parser.add_argument('--L', type=float, default=10.0)
    parser.add_argument('--M', type=int, default=2**7)
    parser.add_argument('--steps', type=int, default=2**7)
    parser.add_argument('--model', type=str, default='Merton', help='Model: GBM, Merton, CGMY')
    parser.add_argument('--model-params', type=str, default='{}', help='JSON string of model params')
    parser.add_argument('--divs', type=str, default='', help='Semicolon-separated discrete divs t:D (e.g. "0.125:1.0;0.35:1.0")')
    parser.add_argument('--conv-p-min', type=float, default=1e-7, help='Min probability threshold for convexity selection (default: 1e-7)')
    parser.add_argument('--conv-standout-mult', type=float, default=10.0, help='Standout multiplier vs median |convexity| (default: 10)')
    args = parser.parse_args()

    # Back-compat: --cheap is now redundant. If both are set, fail fast.
    if bool(args.cheap) and bool(args.with_marginal):
        raise SystemExit('Flags conflict: use either --with-marginal (slower) or omit it for default BS-only.')

    compute_marginal = bool(args.with_marginal)

    try:
        model_params = json.loads(args.model_params)
    except Exception:
        model_params = {}

    # Provide sensible defaults for Merton if user didn't supply params
    if (not model_params) and args.model.strip().lower() == 'merton':
        model_params = {'sigma': 0.03, 'lam': 0.5, 'muJ': -0.05, 'sigmaJ': 0.03}

    print(f"Effective implied volatility for model {args.model} with params {model_params}: {np.sqrt(model_params.get('sigma', 0.2) ** 2 + model_params.get('lam', 0.0) * (model_params.get('muJ', 0.0) ** 2 + model_params.get('sigmaJ', 0.0) ** 2)):.2%}")

    divs_input = parse_divs(args.divs)

    # Baseline
    S0 = 100.0
    K = 100.0
    r = 0.03
    q = 0.025
    T = 0.5

    # case A: continuous yield (unless user provides divs)
    case_a = {'name': 'Continuous yield', 'S0': S0, 'K': K, 'r': r, 'q': q, 'T': T, 'divs_cash': divs_input if divs_input else {}}

    # case B: discrete dividends (user-provided or defaults)
    # default two cash dividends chosen to match prior experiments
    if divs_input:
        divs_b = divs_input
    else:
        divs_b = {0.1: (1.0, 0.0), 0.25: (1.0, 0.0), 0.35: (1.0, 0.0)}
    case_b = {'name': 'Discrete cash dividend (two)', 'S0': S0, 'K': K, 'r': r, 'q': q, 'T': T, 'divs_cash': divs_b}

    model_cls = make_model_cls(args.model)

    # compute both calls and puts for both cases
    res_a_call = compute_case(
        case_a,
        N=args.N,
        L=args.L,
        M=args.M,
        steps=args.steps,
        is_call=True,
        model_cls=model_cls,
        model_params=model_params,
        compute_marginal=compute_marginal,
        convex_p_min=args.conv_p_min,
        convex_standout_mult=args.conv_standout_mult,
    )
    res_b_call = compute_case(
        case_b,
        N=args.N,
        L=args.L,
        M=args.M,
        steps=args.steps,
        is_call=True,
        model_cls=model_cls,
        model_params=model_params,
        compute_marginal=compute_marginal,
        convex_p_min=args.conv_p_min,
        convex_standout_mult=args.conv_standout_mult,
    )
    res_a_put = compute_case(
        case_a,
        N=args.N,
        L=args.L,
        M=args.M,
        steps=args.steps,
        is_call=False,
        model_cls=model_cls,
        model_params=model_params,
        compute_marginal=compute_marginal,
        convex_p_min=args.conv_p_min,
        convex_standout_mult=args.conv_standout_mult,
    )
    res_b_put = compute_case(
        case_b,
        N=args.N,
        L=args.L,
        M=args.M,
        steps=args.steps,
        is_call=False,
        model_cls=model_cls,
        model_params=model_params,
        compute_marginal=compute_marginal,
        convex_p_min=args.conv_p_min,
        convex_standout_mult=args.conv_standout_mult,
    )

    # Plot 2x2: top row CALLs, bottom row PUTs (cube aspect: square subplots)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 10.0), sharex='col')
    for i, (res_call, res_put, tag) in enumerate(zip([res_a_call, res_b_call], [res_a_put, res_b_put], ['[A]', '[B]'])):
        case = res_call['case']
        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        # Top: CALL boundary and selections
        res = res_call
        ax_top.plot(res['t_eval'], res['b_eval'], lw=2.0, color='#2b83ba', label='Rollback implied boundary')
        if res.get('has_boundary', False):
            # Orange circle: COS selection (skipped in --cheap)
            if res.get('t_star_prob', None) is not None:
                ax_top.scatter([res['t_star_prob']], [res['b_star_prob']], facecolors='none', edgecolors='#b35806', marker='o', s=80, label=f"Most likely (COS Δp)")
            # Red X: BS proxy selection (Δp)
            if res.get('t_star_bs_jump', None) is not None:
                ax_top.scatter([res['t_star_bs_jump']], [res['b_star_bs_jump']], marker='x', color='red', s=100, label='Most likely (BS Δp)')
        else:
            ax_top.text(0.5, 0.5, 'No early-exercise boundary', transform=ax_top.transAxes, ha='center', va='center')

        ax_top.set_title(f"{tag} {case['name']}")
        ax_top.set_xlim(0.0, T)
        ax_top.set_ylabel('Boundary B(t)')
        ax_top.grid(True, ls='--', lw=0.5, alpha=0.7)
        if case['divs_cash']:
            for td in sorted(case['divs_cash'].keys()):
                ax_top.axvline(float(td), color='k', lw=0.9, alpha=0.25)

        # Bottom: PUT boundary and selections
        res_put = res_put
        if res_put.get('has_boundary', False):
            ax_bot.plot(res_put['t_eval'], res_put['b_eval'], lw=2.0, color='#3288bd', label='Rollback put boundary')
            if res_put.get('t_star_prob', None) is not None:
                ax_bot.scatter([res_put['t_star_prob']], [res_put['b_star_prob']], facecolors='none', edgecolors='#b35806', marker='o', s=80, label=f"Most likely (COS Δp)")
            if res_put.get('t_star_bs_jump', None) is not None:
                ax_bot.scatter([res_put['t_star_bs_jump']], [res_put['b_star_bs_jump']], marker='x', color='red', s=100, label='Most likely (BS Δp)')
        else:
            ax_bot.text(0.5, 0.5, 'No early-exercise boundary', transform=ax_bot.transAxes, ha='center', va='center')

        ax_bot.set_xlim(0.0, T)
        ax_bot.set_xlabel('t')
        ax_bot.set_ylabel('Boundary / Selection')
        ax_bot.grid(True, ls='--', lw=0.5, alpha=0.7)

        # legends
        h_top, l_top = ax_top.get_legend_handles_labels()
        if h_top:
            ax_top.legend(h_top, l_top, loc='best', fontsize=8)
        h_bot, l_bot = ax_bot.get_legend_handles_labels()
        if h_bot:
            ax_bot.legend(h_bot, l_bot, loc='best', fontsize=8)

    fig.suptitle('American CALL (top) and PUT (bottom): rollback boundaries and selected nodes')
    plt.tight_layout()
    plt.savefig(args.out, dpi=int(args.dpi))
    print(f'Wrote {args.out}')

    # Optional CSV dump for B_call: BS proxy probability through time.
    if args.csv_out:
        out_csv = os.path.abspath(args.csv_out)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        scenario = 'B_call'
        res = res_b_call
        t_eval = np.asarray(res.get('t_eval', []), dtype=float)
        b_eval = np.asarray(res.get('b_eval', []), dtype=float)
        p_bs = np.asarray(res.get('p_bs', []), dtype=float)
        p_marg = np.asarray(res.get('p_marg', []), dtype=float)

        fieldnames = [
            'scenario',
            't',
            'b',
            'p_bs',
            'p_marg',
            't_star_prob',
            'b_star_prob',
            't_star_bs_jump',
            'b_star_bs_jump',
            'p_star_bs_jump',
        ]
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(int(len(t_eval))):
                w.writerow(
                    {
                        'scenario': scenario,
                        't': float(t_eval[i]),
                        'b': float(b_eval[i]) if i < len(b_eval) else float('nan'),
                        'p_bs': float(p_bs[i]) if i < len(p_bs) else float('nan'),
                        'p_marg': float(p_marg[i]) if i < len(p_marg) else float('nan'),
                        't_star_prob': res.get('t_star_prob', None),
                        'b_star_prob': res.get('b_star_prob', None),
                        't_star_bs_jump': res.get('t_star_bs_jump', None),
                        'b_star_bs_jump': res.get('b_star_bs_jump', None),
                        'p_star_bs_jump': res.get('p_star_bs_jump', None),
                    }
                )
        print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
