#!/usr/bin/env python
"""
Calibrate Lévy models to American option clouds using L-BFGS-B optimizer.
Two-phase approach: 
  Phase 1: European pricing on deep OTM options (warm-start)
  Phase 2: American pricing on all options (refinement)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import optimize as opt

# Ensure repo root is on sys.path when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.engine import COSPricer, CompositeLevyCHF  # noqa: E402
from tools.model_specs import CompositeModelSpec, MODEL_REGISTRY, GENERATION_MODELS  # noqa: E402


def _option_label(is_call: bool) -> str:
    return "call" if is_call else "put"


def _pv_cash_divs(*, divs: dict[float, tuple[float, float]], r: float, T: float) -> float:
    pv = 0.0
    for t, (div_yield, std) in divs.items():
        if 0 < t <= T:
            pv += div_yield * np.exp(-r * t)
    return float(pv)


def _forward_proxy(*, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], T: float) -> float:
    """Forward price with continuous dividend yield q and discrete dividend payments."""
    pv_div = _pv_cash_divs(divs=divs, r=r, T=T)
    return float((S0 * np.exp((r - q) * T)) - pv_div)


def _bs_price_forward(*, F: float, K: float, df: float, sigma: float, T: float, is_call: bool) -> float:
    """BS price on forward F with discount factor df."""
    if sigma <= 0.0 or T <= 0.0:
        if is_call:
            return float(max(0.0, df * (F - K)))
        else:
            return float(max(0.0, df * (K - F)))
    d = np.log(F / K) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    from scipy.stats import norm
    if is_call:
        return float(df * (F * norm.cdf(d) - K * norm.cdf(d - sigma * np.sqrt(T))))
    else:
        return float(df * (K * norm.cdf(-d + sigma * np.sqrt(T)) - F * norm.cdf(-d)))


def _bs_implied_vol_forward(*, price: float, F: float, K: float, df: float, T: float, is_call: bool) -> float:
    """Invert BS price to get implied vol on forward."""
    if df <= 0.0 or T <= 0.0 or price < 0.0:
        return np.nan
    intrinsic = float(max(0.0, df * (F - K))) if is_call else float(max(0.0, df * (K - F)))
    if price < intrinsic * 0.9999:
        return np.nan
    
    def objective(sigma: float) -> float:
        return _bs_price_forward(F=F, K=K, df=df, sigma=sigma, T=T, is_call=is_call) - price
    
    try:
        result = opt.brentq(objective, 1e-6, 5.0)
        return float(result)
    except ValueError:
        return np.nan


@dataclass(frozen=True)
class OptionQuote:
    T: float
    K: float
    is_call: bool
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


def parse_divs(specs: Iterable[str]) -> dict[float, tuple[float, float]]:
    """Parse dividend specs: 't:D:std' -> {t: (div_yield, std)}."""
    divs: dict[float, tuple[float, float]] = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid div spec: {spec} (expected t:D:std)")
        t, D, std = float(parts[0]), float(parts[1]), float(parts[2])
        divs[t] = (D, std)
    return divs


class ParamTransform:
    """Transform between z-space (optimization variables) and physical parameters using modular specs."""

    def __init__(self, case: str):
        self.case = str(case).lower().strip()
        # Parse case like "merton_q", "vg_q", "merton_vg_q", "cgmy_vg_q", etc.
        parts = self.case.split("_")
        if parts[-1] != "q":
            raise ValueError(f"Case must end with '_q': {self.case}")
        
        # Extract component names (everything before the final "_q")
        component_names = parts[:-1]
        if not component_names:
            raise ValueError(f"Case must specify at least one model component: {self.case}")
        
        # Build composite spec from component names
        self.spec = CompositeModelSpec(component_names)
        self.names_phys = self.spec.param_names
        self.dim = self.spec.dim

    def z_to_phys(self, z: np.ndarray) -> dict[str, float]:
        """Convert z (optimization space) to physical parameters."""
        return self.spec.z_to_phys(z)

    def components_from_phys(self, phys: dict[str, float]) -> list[dict]:
        """Convert physical params to component list for CompositeLevyCHF."""
        return self.spec.to_component_list(phys)


class CloudObjective:
    """Loss function for option cloud calibration."""

    def __init__(
        self,
        *,
        transform: ParamTransform,
        quotes: list[OptionQuote],
        S0: float,
        r: float,
        divs: dict[float, tuple[float, float]],
        N: int,
        L: float,
        steps: int,
        beta: float,
        is_european: bool = False,
        otm_threshold: float = 0.0,
    ):
        self.transform = transform
        self.quotes = quotes
        self.S0 = float(S0)
        self.r = float(r)
        self.divs = dict(divs)
        self.N = int(N)
        self.L = float(L)
        self.steps = int(steps)
        self.beta = float(beta)
        self.is_european = bool(is_european)
        self.otm_threshold = float(otm_threshold)
        
        # Market prices
        self._mkt_prices = np.array([q.mid for q in quotes], dtype=float)
        
        # Compute OTM mask if filtering
        if is_european and otm_threshold > 0.0:
            sigma_est = 0.2
            moneyness = np.array([
                abs(np.log(q.K / _forward_proxy(S0=S0, r=r, q=0.0, divs=divs, T=q.T)) 
                    / (sigma_est * np.sqrt(q.T)))
                for q in quotes
            ], dtype=float)
            self._otm_mask = moneyness > otm_threshold
            
            # Find and print diagnostics
            n_active = int(np.sum(self._otm_mask))
            print(f"  OTM filter: {n_active}/{len(quotes)} quotes active (threshold={otm_threshold:.4f})")
            
            # Find option closest to threshold
            distances = np.abs(moneyness - otm_threshold)
            closest_idx = int(np.argmin(distances))
            closest_q = quotes[closest_idx]
            closest_dist = distances[closest_idx]
            print(f"  Closest to boundary: T={closest_q.T:.3f}, K={closest_q.K:.1f}, distance={closest_dist:.4f}")
        else:
            self._otm_mask = np.ones(len(quotes), dtype=bool)

    def loss_and_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute loss and gradient."""
        import copy
        z = np.asarray(z, dtype=float).reshape((-1,))
        phys = self.transform.z_to_phys(z)
        
        # Build components dict with deep copy to avoid pricer mutation
        components_dict = {"components": copy.deepcopy(self.transform.components_from_phys(phys))}
        
        # Group quotes by (T, is_call)
        groups: dict[tuple[float, bool], list[int]] = {}
        for i, q in enumerate(self.quotes):
            key = (q.T, q.is_call)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Compute prices and sensitivities
        model_px = np.zeros(len(self.quotes), dtype=float)
        grad_dict: dict[str, np.ndarray] = {name: np.zeros(len(self.quotes), dtype=float) for name in self.transform.names_phys}
        
        # Request sensitivities from pricer
        sens_params_eff = list(self.transform.names_phys)
        for (T, is_call), idxs in groups.items():
            Ks = np.array([self.quotes[i].K for i in idxs], dtype=float)
            
            # Create FRESH model and pricer EVERY time with fresh dict copy
            fresh_components = copy.deepcopy(components_dict)
            fresh_model = CompositeLevyCHF(self.S0, self.r, phys["q"], self.divs, fresh_components)
            pr = COSPricer(fresh_model, N=self.N, L=self.L)
            
            if self.is_european:
                px, sens = pr.european_price(
                    Ks,
                    float(T),
                    is_call=is_call,
                    return_sensitivities=True,
                    sens_params=sens_params_eff,
                )
            else:
                px, sens = pr.american_price(
                    Ks, float(T),
                    steps=self.steps,
                    is_call=is_call,
                    use_softmax=True,
                    beta=self.beta,
                    return_sensitivities=True,
                    sens_params=sens_params_eff,
                )
            
            model_px[idxs] = np.asarray(px, dtype=float)
            
            # Store sensitivities returned by pricer
            for name in sens_params_eff:
                if name in sens:
                    grad_dict[name][idxs] = np.asarray(sens[name], dtype=float)
        
        # Apply OTM mask and compute loss
        mkt_px = self._mkt_prices.copy()
        mkt_px[~self._otm_mask] = np.inf
        model_px_masked = model_px.copy()
        model_px_masked[~self._otm_mask] = np.inf
        
        # Log-space loss
        eps = 1e-8
        log_model = np.log(np.clip(model_px_masked, eps, np.inf))
        log_mkt = np.log(np.clip(mkt_px, eps, np.inf))
        valid = np.isfinite(log_model) & np.isfinite(log_mkt)
        
        r = log_model[valid] - log_mkt[valid]
        f = 0.5 * float(np.sum(r * r))
        
        # Gradient via chain rule
        inv = 1.0 / (np.clip(model_px_masked, eps, np.inf))
        g_phys = {}
        for name in self.transform.names_phys:
            g_vals = (inv * grad_dict[name])[valid]
            g_phys[name] = float(np.sum((log_model[valid] - log_mkt[valid]) * g_vals))
        
        # Chain from z to phys (Jacobian using modular grad_factors)
        g_z = np.zeros(self.transform.dim, dtype=float)
        factors = self.transform.spec.grad_factors(phys)
        for i, (name, factor) in enumerate(zip(self.transform.names_phys, factors)):
            g_z[i] = g_phys[name] * factor
        
        return f, g_z


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate Lévy models to American option clouds with L-BFGS-B")
    ap.add_argument("--quotes", type=str, default=None, help="CSV file with option quotes")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic quotes")
    
    # Generate all calibration case choices dynamically from MODEL_REGISTRY
    all_cases = [f"{name}_q" for name in MODEL_REGISTRY.keys()]
    # Add all two-model combinations
    model_names = list(MODEL_REGISTRY.keys())
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            all_cases.extend([f"{name1}_{name2}_q", f"{name2}_{name1}_q"])
    
    ap.add_argument("--synthetic-model", type=str, default="kou_vg", 
                    choices=list(GENERATION_MODELS.keys()), 
                    help="Model to generate synthetic quotes")
    ap.add_argument("--case", type=str, default="merton_q", 
                    choices=all_cases,
                    help="Model case to calibrate (use <model>_q or <model1>_<model2>_q)")
    
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--r", type=float, default=0.02)
    ap.add_argument("--div", action="append", default=[], help='Dividend spec "t:D:std"')
    
    # European phase
    ap.add_argument("--european-init", action="store_true", help="Use European phase for warm-start")
    ap.add_argument("--european-otm-threshold", type=float, default=0.08)
    ap.add_argument("--european-N", type=int, default=256, help="COS N for European phase")
    ap.add_argument("--european-L", type=float, default=10.0, help="COS L for European phase")
    ap.add_argument("--european-maxiter", type=int, default=500)
    
    # American phase
    ap.add_argument("--american-N", type=int, default=512, help="COS N for American phase")
    ap.add_argument("--american-L", type=float, default=10.0, help="COS L for American phase")
    ap.add_argument("--american-M", type=int, default=None, help="Grid points for American phase (optional)")
    ap.add_argument("--american-steps", type=int, default=40, help="Binomial steps for American phase")
    ap.add_argument("--american-beta", type=float, default=100.0, help="Softmax beta")
    ap.add_argument("--american-maxiter", type=int, default=500)
    
    ap.add_argument("--z0", type=float, nargs="+", default=None)
    ap.add_argument("--z-lb", type=float, nargs="+", default=None)
    ap.add_argument("--z-ub", type=float, nargs="+", default=None)
    
    ap.add_argument("--iv-csv-out", type=str, default=None)
    ap.add_argument("--iv-plot-3d-html-out", type=str, default=None)
    
    args = ap.parse_args()
    
    tfm = ParamTransform(args.case)
    divs = parse_divs(args.div)
    
    # Load or generate quotes
    if args.synthetic:
        from american_options.engine import CompositeLevyCHF, COSPricer
        
        print(f"Generating synthetic quotes using: {args.synthetic_model.upper()}")
        
        # Get components from GENERATION_MODELS
        if args.synthetic_model not in GENERATION_MODELS:
            raise ValueError(f"Unknown synthetic model: {args.synthetic_model}")
        
        components = GENERATION_MODELS[args.synthetic_model]["components"]
        
        model = CompositeLevyCHF(
            args.S0,
            args.r,
            -0.005,
            divs,
            {"components": components},
        )
        pr = COSPricer(model, N=512, L=10.0)
        
        quotes = []
        for T in [0.0416, 0.0833, 0.25,0.5, 0.75]:
            for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
                for is_call in [False, True]:
                    px = pr.american_price(np.array([K]), T, steps=40, is_call=is_call, use_softmax=True, beta=100.0)[0]
                    quotes.append(OptionQuote(T=T, K=K, is_call=is_call, bid=float(px) * 0.99, ask=float(px) * 1.01))
    else:
        if args.quotes is None:
            raise ValueError("Provide --quotes or use --synthetic")
        quotes = []
        with open(args.quotes) as f:
            reader = csv.DictReader(f)
            for row in reader:
                quotes.append(OptionQuote(
                    T=float(row["T"]),
                    K=float(row["K"]),
                    is_call=row["type"].lower() == "call",
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                ))
    
    print(f"Loaded {len(quotes)} quotes")
    
    # Setup bounds using modular system
    if args.z_lb is None or args.z_ub is None:
        z_lb_auto, z_ub_auto = tfm.spec.get_bounds()
        if args.z_lb is None:
            z_lb = z_lb_auto
        else:
            z_lb = np.array(args.z_lb, dtype=float)
        if args.z_ub is None:
            z_ub = z_ub_auto
        else:
            z_ub = np.array(args.z_ub, dtype=float)
    else:
        z_lb = np.array(args.z_lb, dtype=float)
        z_ub = np.array(args.z_ub, dtype=float)
    
    bounds = [(float(z_lb[i]), float(z_ub[i])) for i in range(tfm.dim)]
    
    # Initial guess using modular system
    if args.z0 is None:
        z0 = tfm.spec.get_z0()
    else:
        z0 = np.array(args.z0, dtype=float)
    
    z0 = np.clip(z0, z_lb, z_ub)
    
    best_z = z0.copy()
    
    # Phase 1: European
    if args.european_init:
        print(f"\n=== Phase 1: European (N={args.european_N}, L={args.european_L}, OTM threshold={args.european_otm_threshold}) ===")
        obj = CloudObjective(
            transform=tfm,
            quotes=quotes,
            S0=args.S0,
            r=args.r,
            divs=divs,
            N=args.european_N,
            L=args.european_L,
            steps=1,
            beta=1.0,
            is_european=True,
            otm_threshold=args.european_otm_threshold,
        )
        
        # Callback for progress output
        iter_count_eur = [0]
        def callback_eur(z):
            iter_count_eur[0] += 1
            if iter_count_eur[0] % 10 == 0 or iter_count_eur[0] == 1:
                phys = tfm.z_to_phys(z)
                f = obj.loss_and_grad(z)[0]
                param_str = ", ".join([f"{name}={phys[name]:.3f}" for name in tfm.names_phys])
                print(f"  Iter {iter_count_eur[0]:3d}: f={f:.4e}  [{param_str}]")
        
        res = opt.minimize(
            lambda z: obj.loss_and_grad(z)[0],
            z0,
            method="SLSQP",
            jac=lambda z: obj.loss_and_grad(z)[1],
            bounds=bounds,
            options={"maxiter": args.european_maxiter, "disp": True},
            callback=callback_eur,
        )
        best_z = np.asarray(res.x, dtype=float)
        print(f"Phase 1 result: f={res.fun:.6e}")
        print(f"  Optimization success: {res.success}")
        print(f"  Message: {res.message}")
        print(f"  Iterations: {res.nit}")
        print(f"  Function evals: {res.nfev}")
        phys = tfm.z_to_phys(best_z)
        for name in tfm.names_phys:
            print(f"  {name:15s} = {phys[name]: .6g}")
    
    # Phase 2: American
    print(f"\n=== Phase 2: American (N={args.american_N}, L={args.american_L}, steps={args.american_steps}) ===")
    obj = CloudObjective(
        transform=tfm,
        quotes=quotes,
        S0=args.S0,
        r=args.r,
        divs=divs,
        N=args.american_N,
        L=args.american_L,
        steps=args.american_steps,
        beta=args.american_beta,
        is_european=False,
        otm_threshold=0.0,
    )
    
    # Callback for progress output
    iter_count = [0]
    def callback(z):
        iter_count[0] += 1
        if iter_count[0] % 10 == 0 or iter_count[0] == 1:
            phys = tfm.z_to_phys(z)
            f = obj.loss_and_grad(z)[0]
            param_str = ", ".join([f"{name}={phys[name]:.3f}" for name in tfm.names_phys])
            print(f"  Iter {iter_count[0]:3d}: f={f:.4e}  [{param_str}]")
    
    res = opt.minimize(
        lambda z: obj.loss_and_grad(z)[0],
        best_z,
        method="SLSQP",
        jac=lambda z: obj.loss_and_grad(z)[1],
        bounds=bounds,
        options={"maxiter": args.american_maxiter, "disp": True},
        callback=callback,
    )
    best_z = np.asarray(res.x, dtype=float)
    
    best_phys = tfm.z_to_phys(best_z)
    print(f"\nFinal result: f={res.fun:.4e}")
    for name in tfm.names_phys:
        print(f"  {name:15s} = {best_phys[name]:.3f}")
    
    # Auto-generate HTML filename if not specified
    if args.iv_plot_3d_html_out is None and args.synthetic:
        args.iv_plot_3d_html_out = f"figs/iv_fit_{args.synthetic_model}_to_{args.case}.html"
    
    # IV diagnostics
    if args.iv_csv_out is not None or args.iv_plot_3d_html_out is not None:
        print("\nGenerating IV diagnostics...")
        model = CompositeLevyCHF(
            args.S0,
            args.r,
            best_phys["q"],
            divs,
            {"components": tfm.components_from_phys(best_phys)},
        )
        pr = COSPricer(model, N=args.american_N, L=args.american_L)
        
        groups: dict[tuple[float, bool], list[int]] = {}
        for i, q in enumerate(quotes):
            key = (q.T, q.is_call)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        model_px = np.zeros(len(quotes), dtype=float)
        for (T, is_call), idxs in groups.items():
            Ks = np.array([quotes[i].K for i in idxs], dtype=float)
            px = pr.american_price(Ks, float(T), steps=args.american_steps, is_call=is_call, use_softmax=True, beta=args.american_beta)
            model_px[idxs] = np.asarray(px, dtype=float)
        
        # IV inversion
        q_target = -0.005 if args.synthetic else best_phys["q"]
        xs, ivs_mkt, ivs_fit, Ts, Ks, types = [], [], [], [], [], []
        
        for q, px_mkt, px_fit in zip(quotes, obj._mkt_prices, model_px):
            F = _forward_proxy(S0=args.S0, r=args.r, q=q_target, divs=divs, T=q.T)
            x = np.log(q.K / F) / np.sqrt(q.T)
            df = np.exp(-args.r * q.T)
            
            iv_mkt = _bs_implied_vol_forward(price=px_mkt, F=F, K=q.K, df=df, T=q.T, is_call=q.is_call)
            iv_fit = _bs_implied_vol_forward(price=px_fit, F=F, K=q.K, df=df, T=q.T, is_call=q.is_call)
            
            xs.append(x)
            ivs_mkt.append(iv_mkt)
            ivs_fit.append(iv_fit)
            Ts.append(q.T)
            Ks.append(q.K)
            types.append(_option_label(q.is_call))
        
        if args.iv_csv_out is not None:
            os.makedirs(os.path.dirname(args.iv_csv_out) or ".", exist_ok=True)
            with open(args.iv_csv_out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["T", "K", "type", "x", "iv_target", "iv_fit"])
                for T, K, typ, x, iv_mkt, iv_fit in zip(Ts, Ks, types, xs, ivs_mkt, ivs_fit):
                    w.writerow([T, K, typ, x, iv_mkt, iv_fit])
            print(f"Saved IV CSV: {args.iv_csv_out}")
        
        if args.iv_plot_3d_html_out is not None:
            try:
                import plotly.graph_objects as go
            except ImportError:
                print("Plotly not installed, skipping 3D plot")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=xs, y=Ts, z=ivs_mkt, mode="markers", name="Target",
                    customdata=np.column_stack((Ks, types)),
                    hovertemplate="<b>Target</b><br>T=%{y:.3f}<br>K=%{customdata[0]:.1f}<br>Type=%{customdata[1]}<br>x=%{x:.3f}<br>IV=%{z:.4f}<extra></extra>",
                    marker=dict(size=10, color="red", symbol="circle-open", opacity=0.4,
                               line=dict(color="red", width=1))
                ))
                fig.add_trace(go.Scatter3d(
                    x=xs, y=Ts, z=ivs_fit, mode="markers", name="Fit",
                    customdata=np.column_stack((Ks, types)),
                    hovertemplate="<b>Fit</b><br>T=%{y:.3f}<br>K=%{customdata[0]:.1f}<br>Type=%{customdata[1]}<br>x=%{x:.3f}<br>IV=%{z:.4f}<extra></extra>",
                    marker=dict(size=10, color="blue", symbol="cross", opacity=0.4, line=dict(width=0.5))
                ))
                fig.update_layout(
                    scene=dict(xaxis_title="log-moneyness", yaxis_title="T", zaxis_title="IV"),
                    title=f"IV Fit ({tfm.case})",
                )
                os.makedirs(os.path.dirname(args.iv_plot_3d_html_out) or ".", exist_ok=True)
                fig.write_html(args.iv_plot_3d_html_out, include_plotlyjs="cdn")
                print(f"Saved 3D plot: {args.iv_plot_3d_html_out}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
