"""Validate dividend behavior in COS American pricing."""

import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer

# Setup with dividends at exact times
divs = {0.25: (0.02, 0.0), 0.75: (0.02, 0.0)}  # 2% proportional dividends
model = GBMCHF(S0=100, r=0.05, q=0.02, divs=divs, params={'vol': 0.2})
pricer = COSPricer(model, N=128, L=8.0)

K = np.array([100.0])
amer, euro, traj = pricer.american_price(K, T=1.0, steps=80, return_european=True, return_trajectory=True)

print("DIVIDEND BEHAVIOR VALIDATION")
print("=" * 60)
print(f"Strike K={K[0]}, Spot S0={model.S0}")
print(f"Dividends: 2% at t=0.25 and t=0.75\n")

# Check behavior at dividend dates
print("Prices BEFORE dividend ex-dates:")
print("-" * 60)
for t, p in traj:
    if abs(t - 0.25) < 0.04 and t < 0.25:
        print(f"  t={t:.4f} (before 1st div): American={p:.4f}")
    if abs(t - 0.75) < 0.04 and t < 0.75:
        print(f"  t={t:.4f} (before 2nd div): American={p:.4f}")

print("\nPrices AT and AFTER dividend ex-dates:")
print("-" * 60)
for t, p in traj:
    if abs(t - 0.25) < 0.04 and t >= 0.25:
        print(f"  t={t:.4f} (at/after 1st div): American={p:.4f}")
    if abs(t - 0.75) < 0.04 and t >= 0.75:
        print(f"  t={t:.4f} (at/after 2nd div): American={p:.4f}")

# Find prices just before and after dividend
print("\nDividend Impact (change in American price):")
print("-" * 60)
before_25 = None
after_25 = None
before_75 = None
after_75 = None

for t, p in traj:
    if abs(t - 0.25) < 0.005 and t < 0.25:
        before_25 = (t, p)
    if abs(t - 0.25) < 0.005 and t > 0.25:
        after_25 = (t, p)
    if abs(t - 0.75) < 0.005 and t < 0.75:
        before_75 = (t, p)
    if abs(t - 0.75) < 0.005 and t > 0.75:
        after_75 = (t, p)

if before_25 and after_25:
    drop_25 = before_25[1] - after_25[1]
    print(f"  1st dividend (t≈0.25): {before_25[1]:.4f} → {after_25[1]:.4f} (drop={drop_25:.4f})")
    print(f"    ✓ CORRECT: Price decreased" if drop_25 > 0 else "    ✗ ERROR: Price increased")

if before_75 and after_75:
    drop_75 = before_75[1] - after_75[1]
    print(f"  2nd dividend (t≈0.75): {before_75[1]:.4f} → {after_75[1]:.4f} (drop={drop_75:.4f})")
    print(f"    ✓ CORRECT: Price decreased" if drop_75 > 0 else "    ✗ ERROR: Price increased")
