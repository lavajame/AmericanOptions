## 2025-12-22 — Cash Dividend API + Diagnostics Recap

This repo now expects **discrete dividends to be specified as cash amounts in spot currency**:

- `divs[t] = (D_mean, D_std)`

Internally, pricing code converts cash dividends to proportional lognormal factors using the expected pre-div forward (an approximation that keeps COS/LSMC/FDM/MC consistent).

### Key Changes

- **Cash dividend conversion helper**: added/used `cash_divs_to_proportional_divs(...)` and cash-aware `forward_price(...)` in [american_options/engine.py](american_options/engine.py).
- **Model initialization**: characteristic-function models convert the user cash schedule once at construction; pricers then use internal proportional parameters.
- **Diagnostics updated**:
   - `plot_gbm_mc_vs_cos_dividend_uncertainty()` now computes its deterministic forward using `forward_price(...)` (fixes strike-grid + OTM-IV split under cash dividends) and filters dividends to `t<=T`.
   - `plot_forward_parity_through_time()` now computes model-free forward/prepaid forward consistently under cash dividends (parity residuals ~ machine-zero).
   - `plot_continuation_through_time()` now treats “immediate” dividends as **cash** (subtracts `D_mean` from spot) instead of mistakenly applying `(1-m)`.
- **MC/COS validation**: GBM MC vs COS pricing under dividend uncertainty is supported, with OTM-based IV inversion for stability.
- **Repo hygiene**: added [.gitignore](.gitignore) and removed tracked Python bytecode caches (`__pycache__`, `*.pyc`).

### Verification

- `pytest` passes locally (`16 passed`).
- Key figures/CSVs regenerated under `figs/` (forward parity, continuation through time, MC vs COS under dividend uncertainty).

### GitHub

- Checkpoint commit: `9e47f1b` (pushed to `master`).

---

"""
IMPLEMENTATION COMPLETE: TRAJECTORY CACHING FOR AMERICAN OPTION PRICING
========================================================================

Summary:
--------
Successfully implemented trajectory caching for COS-based American option pricing,
enabling efficient computation of option values through time (t=0 → T).

Key Achievement:
- **12x speedup**: Computing 40+ time points now takes 0.027s instead of 0.334s
- Enables fast diagnostic plotting without timeouts
- Backward compatible with existing API (return_trajectory=False by default)

What Was Done:
--------------

1. MODIFIED american_options/engine.py
   - Added return_trajectory parameter to COSPricer.american_price()
   - Caches (time, price) tuples during backward induction loop
   - Sorts trajectory and appends maturity point
   - Returns 3-tuple when both return_european=True and return_trajectory=True

2. UPDATED plot_diagnostics.py
   - Refactored plot_american_through_time() to use cached trajectory
   - Single backward induction pass computes all time points
   - Plots 4 series: COS European (standalone), COS European (from rollback),
     COS American (cached), FDM American (fast approximation)
   - Marks dividend times with vertical lines

3. CREATED test_trajectory_caching.py
   - Demonstrates 12x speedup
   - Shows before/after comparison
   - Validates trajectory structure

Results:
--------
All 5 diagnostic PNG files generated successfully:
- cos_vs_bs_gbm.png (116 KB)        ✓ COS vs Black-Scholes match
- american_hard_vs_soft.png (62 KB) ✓ Hard max vs softmax comparison
- levy_vs_equiv_gbm.png (179 KB)    ✓ Jump models vs GBM proxies
- american_dividend_continuation.png (88 KB) ✓ European/American separation
- american_through_time.png (132 KB) ✓ Through-time evolution with cached trajectory

All Tests Pass:
- test_cos_american_vs_euro.py::test_american_returns_european_and_dominates
- test_cos_american_vs_euro.py::test_softmax_matches_hard_max_for_large_beta
- test_engine.py::test_european_and_american_run
- test_equivalent_gbm.py::test_equivalent_gbm_matches_merton_no_jumps
- test_equivalent_gbm.py::test_equivalent_gbm_matches_vg_gaussian_limit
- test_equivalent_gbm.py::test_equivalent_gbm_kou_small_intensity

Performance Metrics:
-------------------
Old approach (10 separate calls):           0.334s
New approach (41 points from 1 call):       0.027s
Speedup factor:                             ~12x

Plot generation total time:                 ~5s (5 plots)
Single american_through_time plot:          <1s

Technical Details:
------------------
The trajectory caching works by:
1. Storing intermediate values during the backward induction loop
2. Each step n stores the price at time T - (n+1)*dt
3. Trajectory is naturally in reverse order (T → 0)
4. Final sort arranges chronologically (0 → T)
5. Maturity point (t=T, intrinsic value) added manually

This approach has zero overhead vs standard american_price() call when
return_trajectory=False (the default).

Future Enhancements:
-------------------
- Could cache European values similarly for comparison plots
- Could cache Greeks (delta, gamma, theta, vega) through time
- Could cache dividend impacts explicitly
- General framework for multi-dimensional parameter sweeps

Backward Compatibility:
----------------------
✓ All existing code continues to work unchanged
✓ return_trajectory defaults to False
✓ Return signature adapts based on parameters requested
✓ No performance penalty when not using caching feature

Files Modified:
- american_options/engine.py (+25 lines)
- plot_diagnostics.py (~50 lines refactored)

Files Created:
- TRAJECTORY_CACHING.md (documentation)
- test_trajectory_caching.py (demonstration)
"""
