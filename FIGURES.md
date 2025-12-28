# Figures and Outputs

This repo generates most of its diagnostics into [figs/](figs/), now organized into logical subfolders:

```
figs/
├── boundaries/                  # Early-exercise boundary analysis
├── calibration_3d/              # 3D interactive IV fit plots (Plotly/HTML)
├── calibration_surfaces/        # IV surfaces, slices, and 2D comparisons
├── american_pricing/            # American option value & continuation diagnostics
├── levy_models/                 # Lévy model comparisons (MC vs COS)
└── model_comparison/            # Skew persistence, collinearity, parameters
```

## Quick view

If you just want to refresh everything:

- Run the main diagnostics: `python plot_diagnostics.py`
- Run COS-vs-FDM profiling: `python tools/profile_cos_vs_fdm_time_complexity.py`
- Run Lévy skew persistence analysis: `python tools/analyze_skew_persistence.py`
- Run event IV surface plots:
  - Static: `python tools/plot_event_iv_surfaces.py`
  - Interactive (linked 3D): `python tools/make_linked_event_iv_surfaces.py --model vg` (or `--model merton`)
- **NEW**: Run calibration suite:
  - Display mode: `python tools/run_calibration.py`
  - Auto-run: `python tools/run_calibration_auto.py`
  - Interactive: `python -c "from calibration_explorer import scenario_basic; scenario_basic().run_all()"`

## Key modeling features

- **Dividend uncertainty**: All uncertain cash dividends are modeled using an inverse Gaussian (IG) process, preserving positivity while allowing intuitive cash-space mean/stdev specification.
- **CGMY skew persistence**: CGMY's asymmetric tempering (G < M) produces the most persistent skew and smile across maturities, especially on the call wing, making it ideal for mixing short-dated and long-dated calibrations.
- **Linked parameters in combo models**: Merton+VG and Kou+VG models share sigma parameter across components, reducing dimension (merton_vg: 7 parameters, kou_vg: 8 parameters).
- **3D interactive IV plots**: Calibration output generated as rotatable Plotly HTML plots showing target vs fitted IV surfaces.

## Calibration suite (NEW - Dec 2025)

### 3D Interactive IV Fit Plots ([figs/calibration_3d/](figs/calibration_3d/))

All generated via `calibrate_cloud_lbfgsb.py` with `--iv-plot-3d-html-out` flag:

- [figs/calibration_3d/iv_fit_merton_vg_to_merton_q.html](figs/calibration_3d/iv_fit_merton_vg_to_merton_q.html) — Merton+VG generating → Merton+q calibration
- [figs/calibration_3d/iv_fit_merton_vg_to_kou_q.html](figs/calibration_3d/iv_fit_merton_vg_to_kou_q.html) — Merton+VG generating → Kou+q calibration (cross-model)
- [figs/calibration_3d/iv_fit_cgmy_vg_to_merton_q.html](figs/calibration_3d/iv_fit_cgmy_vg_to_merton_q.html) — CGMY+VG → Merton+q (infinite-activity model)
- [figs/calibration_3d/iv_fit_kou_vg_to_cgmy_q.html](figs/calibration_3d/iv_fit_kou_vg_to_cgmy_q.html) — Kou+VG → CGMY+q (jump vs infinite-activity)

**Features**: Red hollow circles (target prices), blue crosses (fitted prices), interactive rotation/zoom, hover data showing strike, maturity, moneyness, IV.

### Static PNG figures

### COS / dividends / American diagnostics (from [plot_diagnostics.py](plot_diagnostics.py))

- COS vs Black–Scholes sanity check: [figs/levy_models/cos_vs_bs_gbm.png](figs/levy_models/cos_vs_bs_gbm.png)

  ![COS vs BS (GBM)](figs/levy_models/cos_vs_bs_gbm.png)

  What it shows: European call prices under GBM with no dividends, COS vs the analytic Black–Scholes price (plus the residual).

- American rollback: hard max vs softmax: [figs/american_pricing/american_hard_vs_soft.png](figs/american_pricing/american_hard_vs_soft.png)

  ![American hard vs soft rollback](figs/american_pricing/american_hard_vs_soft.png)

  What it shows: the continuation/intrinsic comparison under two rollback variants (hard max vs softmax smoothing).

- Lévy models vs equivalent GBM proxy: [figs/levy_models/levy_vs_equiv_gbm.png](figs/levy_models/levy_vs_equiv_gbm.png)

  ![Levy vs equivalent GBM](figs/levy_models/levy_vs_equiv_gbm.png)

  What it shows: European prices under jump/Lévy models (Merton/Kou/VG) compared against a variance-matched GBM proxy.

- Dividend continuation (European vs American continuation at $t=0$): [figs/american_pricing/american_dividend_continuation.png](figs/american_pricing/american_dividend_continuation.png)

  ![American dividend continuation](figs/american_pricing/american_dividend_continuation.png)

  What it shows: how the dividend-aware continuation behaves at time 0, comparing European vs American continuation components.

- Through-time comparison (COS European/American vs FDM): [figs/american_pricing/american_through_time.png](figs/american_pricing/american_through_time.png)

  ![American through time](figs/american_pricing/american_through_time.png)

  What it shows: a through-time snapshot of values/continuation for COS (European + American rollback) alongside an FDM reference.

- Continuation value through time (with CSV):
  - Plot: [figs/american_pricing/continuation_through_time.png](figs/american_pricing/continuation_through_time.png)
  - Data: [figs/american_pricing/continuation_through_time.csv](figs/american_pricing/continuation_through_time.csv)

  ![Continuation through time](figs/american_pricing/continuation_through_time.png)

  What it shows: a time series view of the continuation value (and related quantities) that can be used for debugging rollback behavior.

- Forward / parity through time (with CSV):
  - Plot: [figs/american_pricing/forward_parity_through_time.png](figs/american_pricing/forward_parity_through_time.png)
  - Data: [figs/american_pricing/forward_parity_through_time.csv](figs/american_pricing/forward_parity_through_time.csv)

  ![Forward parity through time](figs/american_pricing/forward_parity_through_time.png)

  What it shows: forward/parity consistency checks over time in the presence of discrete dividends.

- Dividend uncertainty: CGMY COS vs Monte Carlo (with CSV):
  - Plot: [figs/model_comparison/cgmy_mc_vs_cos_div_uncertainty.png](figs/model_comparison/cgmy_mc_vs_cos_div_uncertainty.png)
  - Data: [figs/model_comparison/cgmy_mc_vs_cos_div_uncertainty.csv](figs/model_comparison/cgmy_mc_vs_cos_div_uncertainty.csv)

  ![CGMY MC vs COS under dividend uncertainty](figs/model_comparison/cgmy_mc_vs_cos_div_uncertainty.png)

  What it shows: compares COS pricing against a CGMY Monte Carlo cross-check under uncertain cash dividends (modeled via inverse Gaussian). Uses equity-like CGMY parameterization (G=2.5, M=8.0, Y=0.7) with negative skew. Dividend scenarios: $2.0 mean with stdev sweep ($0.00, $0.75, $1.50, $3.00). Validates that COS and MC agree under the IG dividend factor model across all uncertainty levels.

- Richardson extrapolation for discrete dividends (with CSV):
  - Plot: [figs/levy_models/richardson_vs_manyM_discrete_divs.png](figs/levy_models/richardson_vs_manyM_discrete_divs.png)
  - Data: [figs/levy_models/richardson_vs_manyM_discrete_divs.csv](figs/levy_models/richardson_vs_manyM_discrete_divs.csv)

  ![Richardson extrapolation discrete dividends](figs/levy_models/richardson_vs_manyM_discrete_divs.png)

  What it shows: compares Richardson extrapolation convergence against brute-force refinement (many timesteps) for American options with discrete dividends under GBM.

### Dividend call early-ex identification heuristic (from [tools/plot_call_boundary_heuristics_clean.py](tools/plot_call_boundary_heuristics_clean.py))

- Merton 2x2 boundary + “most likely” node markers (high DPI):
  - Plot: [figs/boundaries/call_boundary_heuristics_merton_2x2_clean_bsjump_hi.png](figs/boundaries/call_boundary_heuristics_merton_2x2_clean_bsjump_hi.png)
  - Data (B_call BS probability-through-time): [figs/boundaries/b_call_prob_through_time.csv](figs/boundaries/b_call_prob_through_time.csv)

  ![Dividend call early-ex identification (Merton)](figs/boundaries/call_boundary_heuristics_merton_2x2_clean_bsjump_hi.png)

  What it shows: rollback-implied American exercise boundaries for CALLs (top row) and PUTs (bottom row) under two scenarios: (A) continuous yield and (B) discrete cash dividends. The default “most likely” marker uses a forward-anchored BS/lognormal proxy probability-through-time. Use `--with-marginal` to additionally compute the COS-digital marginal-probability scan (slower).

### COS vs FDM time/accuracy tradeoff (from [tools/profile_cos_vs_fdm_time_complexity.py](tools/profile_cos_vs_fdm_time_complexity.py))

- COS vs FDM scalability: [figs/levy_models/cos_vs_fdm_time_complexity.png](figs/levy_models/cos_vs_fdm_time_complexity.png)

  ![COS vs FDM time complexity](figs/levy_models/cos_vs_fdm_time_complexity.png)

  What it shows: for both European and American cases, compares runtime vs error for COS (sweeping $N$) and FDM (sweeping grid sizes).

### Lévy model skew persistence and calibration (from [tools/analyze_skew_persistence.py](tools/analyze_skew_persistence.py))

- Skew and curvature persistence after hybrid calibration:
  - Plot: [figs/model_comparison/skew_persistence_skew_curv_calib_cumulants_from_avgshape_T025_hybrid.png](figs/model_comparison/skew_persistence_skew_curv_calib_cumulants_from_avgshape_T025_hybrid.png)
  - Data: [figs/model_comparison/skew_persistence_by_model_calib_cumulants_from_avgshape_T025_hybrid.csv](figs/model_comparison/skew_persistence_by_model_calib_cumulants_from_avgshape_T025_hybrid.csv)

  ![Skew persistence hybrid calibration](figs/model_comparison/skew_persistence_skew_curv_calib_cumulants_from_avgshape_T025_hybrid.png)

  What it shows: after matching VG/Merton/Kou/CGMY models at T=0.25 using a hybrid objective (cumulants + IV-slice penalty), compares how ATM skew and curvature evolve across maturities (0.05Y to 1Y). VG shows the sharpest/most extreme smile decay; Kou and Merton are similar and moderate; CGMY sits in between but with the most persistent smile across maturities, especially on the call wing.

- Calibrated IV slices (z-normalized moneyness):
  - Plot: [figs/calibration_surfaces/iv_slices_calib_cumulants_from_avgshape_T025_hybrid_z_T005_T025_T100.png](figs/calibration_surfaces/iv_slices_calib_cumulants_from_avgshape_T025_hybrid_z_T005_T025_T100.png)

  ![IV slices z-normalized](figs/calibration_surfaces/iv_slices_calib_cumulants_from_avgshape_T025_hybrid_z_T005_T025_T100.png)

  What it shows: implied vol slices at T=0.05, T=0.25, T=1.0 for the hybrid-calibrated models, plotted vs normalized moneyness z = ln(K/F)/(σ√T). Shows how each model's smile shape evolves with maturity after ensuring fair starting conditions at T=0.25.

- Calibration parameter outputs:
  - Hybrid calibration: [figs/model_comparison/calib_params_cumulants_from_avgshape_T025_hybrid.json](figs/model_comparison/calib_params_cumulants_from_avgshape_T025_hybrid.json)
  - Other variants available in `figs/model_comparison/calib_params_*.json`

### Event implied-vol surfaces (static)

- Event IV surfaces (GBM+event vs VG+event): [figs/model_comparison/event_iv_surfaces_gbm_vs_vg.png](figs/model_comparison/event_iv_surfaces_gbm_vs_vg.png)

  ![Event IV surfaces GBM vs VG](figs/model_comparison/event_iv_surfaces_gbm_vs_vg.png)

- Event IV surfaces (GBM+event vs Merton+event): [figs/model_comparison/event_iv_surfaces_gbm_vs_merton.png](figs/model_comparison/event_iv_surfaces_gbm_vs_merton.png)

  ![Event IV surfaces GBM vs Merton](figs/model_comparison/event_iv_surfaces_gbm_vs_merton.png)

- Event IV surfaces (GBM+event vs Kou+VG composite+event): [figs/model_comparison/event_iv_surfaces_gbm_vs_kouvg.png](figs/model_comparison/event_iv_surfaces_gbm_vs_kouvg.png)

  ![Event IV surfaces GBM vs Kou+VG](figs/model_comparison/event_iv_surfaces_gbm_vs_kouvg.png)

  Generated by: [tools/plot_event_iv_surfaces.py](tools/plot_event_iv_surfaces.py)
  
  What it shows: 2D static contour plots showing how implied vol surfaces differ between GBM (flat vol baseline) and various Lévy models when a discrete event jump (at t=0.05, martingale-normalized) is embedded in the characteristic function. Filled contours = with event, black lines = no event. All three comparisons use the same event parameters. These provide quick static snapshots; see the interactive 3D versions below for full exploration.

## Interactive HTML figures (Plotly)

These are real interactive Plotly plots (pan/zoom/rotate), not images:

### Calibration 3D scatter plots (NEW - Dec 2025)

**Synthetic calibration with SLSQP optimizer**:
- [figs/calibration_3d/iv_fit_merton_vg_to_merton_q.html](figs/calibration_3d/iv_fit_merton_vg_to_merton_q.html) — Self-calibration benchmark
- [figs/calibration_3d/iv_fit_merton_vg_to_kou_q.html](figs/calibration_3d/iv_fit_merton_vg_to_kou_q.html) — Cross-model fit (Merton+VG → Kou)
- [figs/calibration_3d/iv_fit_cgmy_vg_to_merton_q.html](figs/calibration_3d/iv_fit_cgmy_vg_to_merton_q.html) — Infinite-activity model (CGMY+VG → Merton)
- [figs/calibration_3d/iv_fit_kou_vg_to_cgmy_q.html](figs/calibration_3d/iv_fit_kou_vg_to_cgmy_q.html) — Jump vs infinite-activity comparison

All generated via synthetic calibration pipeline with SLSQP optimizer, sensitivities via COS-American analytic gradients. **Linked sigma parameters** in combo models reduce dimension (merton_vg: 7 params, kou_vg: 8 params).

### Event IV surfaces (3D interactive)

- Linked 3D surfaces (GBM vs VG): [figs/calibration_3d/event_iv_surfaces_gbm_vs_vg_linked_3d.html](figs/calibration_3d/event_iv_surfaces_gbm_vs_vg_linked_3d.html)
- Linked 3D surfaces (GBM vs Merton): [figs/calibration_3d/event_iv_surfaces_gbm_vs_merton_linked_3d.html](figs/calibration_3d/event_iv_surfaces_gbm_vs_merton_linked_3d.html)
- Linked 3D surfaces (GBM vs Kou+VG composite): [figs/calibration_3d/event_iv_surfaces_gbm_vs_kouvg_linked_3d.html](figs/calibration_3d/event_iv_surfaces_gbm_vs_kouvg_linked_3d.html)

Generated by: [tools/make_linked_event_iv_surfaces.py](tools/make_linked_event_iv_surfaces.py)

**Recommended**: The VG and CGMY models show the most persistent and realistic equity-like skew/smile behavior. The Kou+VG composite blends finite-activity and infinite-activity jump components for intermediate characteristics.

### How to open HTML interactives

GitHub (and VS Code’s default text editor view) will often show you the HTML source.
To view it as an interactive plot, use one of these:

1) Local browser (recommended)

- Start a local server from the repo root:
  - `python -m http.server 8000`
- Then open (example):
  - `http://localhost:8000/figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html`

2) VS Code

- In the Explorer, right-click the `.html` file and choose “Open in Default Browser” (or “Open With…” if prompted).

3) From a GitHub link (if you want click-to-preview)

- Use an HTML preview proxy such as https://htmlpreview.github.io/ (paste the GitHub URL to the HTML file)
  - Note: availability/CSP behavior can vary by browser.

## Notes

- Some dev environments create local copies like `figs/* - Copy.png`. These are not part of the tracked artifact set.
