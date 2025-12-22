"""
TRAJECTORY CACHING IMPLEMENTATION SUMMARY
=========================================

Problem: Computing option prices at multiple time points required repeatedly calling
american_price() for each tau, which recomputed the entire backward induction each time.
This was extremely slow for plotting through-time diagnostics.

Solution: Added trajectory caching to COSPricer.american_price():
- New parameter: return_trajectory=bool
- Returns cached (time, price) tuples from all intermediate backward induction steps
- Enables computing 40+ time points in a single pass vs 40+ separate calls

Implementation Details:
- Modified american_price() signature to accept return_trajectory parameter
- Cache list stores (current_time, american_at_time) tuples during each backward step
- Sorting applied to trajectory cache at end (backward induction processes time in reverse)
- Maturity point (t=T, intrinsic value) appended manually to complete trajectory
- Works seamlessly with existing return_european parameter

Performance Impact:
- Old approach (10 time points): 0.334s (10 separate calls)
- New approach (41 time points): 0.027s (1 call with trajectory caching)
- Speedup: ~12x faster for comparable temporal resolution

Usage Example:
    pricer = COSPricer(model, N=128, L=8.0)
    
    # Get American prices AND full trajectory through time
    american_prices, european_prices, trajectory = pricer.american_price(
        K=np.array([100.0]),
        T=1.0,
        steps=40,
        return_european=True,
        return_trajectory=True
    )
    
    # trajectory is list of (t, price) tuples: [(0.0, 12.34), (0.025, 12.10), ...]
    times = np.array([t for t, _ in trajectory])
    values = np.array([v for _, v in trajectory])

Applications:
- plot_american_through_time(): Uses trajectory caching to efficiently plot COS vs FDM
  Now completes in ~0.2s instead of timing out
- Future: Greeks computation could cache sensitivities similarly
- General: Any multi-time computation benefits from single backward induction pass

Testing:
- All existing tests pass (6/6)
- Backward compatibility maintained (return_trajectory=False is default)
- Tested with N=128, steps=40 on both no-dividend and dividend cases

Files Modified:
- american_options/engine.py: Added trajectory caching to american_price()
- plot_diagnostics.py: Updated plot_american_through_time() to use cached trajectory
- test_trajectory_caching.py: Demo script showing ~12x speedup
"""
