"""Quick test demonstrating trajectory caching efficiency."""

import time
import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer

# Setup
K = np.array([100.0])
model = GBMCHF(S0=100, r=0.05, q=0.02, divs={}, params={"vol": 0.2})
pricer = COSPricer(model, N=128, L=8.0)

# Method 1: Multiple calls (old way - slow)
print("Method 1: Multiple separate american_price() calls")
t0 = time.time()
prices_old = []
for tau in np.linspace(1.0, 0.1, 10):
    p = pricer.american_price(K, tau)
    prices_old.append(p[0])
t1 = time.time()
tau1 = t1-t0
print(f"  Time: {tau1:.3f}s for 10 time points")

# Method 2: Single call with trajectory caching (new way - fast)
print("\nMethod 2: Single american_price() call with return_trajectory=True")
t0 = time.time()
prices, euro, trajectory = pricer.american_price(
    K, T=1.0, steps=40, return_european=True, return_trajectory=True
)
t1 = time.time()
tau2 = t1-t0
print(f"  Time: {tau2:.3f}s for 41 trajectory points (40 steps + maturity)")

# Compare
print(f"\nSpeedup: ~{tau1/tau2:.1f}x faster when computing full trajectory at once")
print(f"Trajectory points available: {len(trajectory)}")
print(f"First 3 points (time, value):")
for i in range(min(3, len(trajectory))):
    t, v = trajectory[i]
    print(f"  t={t:.4f}: value={v:.4f}")
