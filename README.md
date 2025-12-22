# AmericanOptions

A Python research/pricing toolkit centered on **COS (Fourier-cosine) option pricing** with:

- European and American pricing (rollback)
- Multiple models via characteristic functions (GBM, Merton, Kou, VG, CGMY)
- Discrete dividends (with optional uncertainty)
- Diagnostics that compare COS vs references (Black–Scholes, FDM, Monte Carlo)

This repo is designed for reproducible numerical experiments: most key comparisons write **PNGs/CSVs to `figs/`** and the test suite provides regression protection.

## What We Built / Changed

### 1) Project-wide dividend convention (cash dividends)
Throughout the codebase, **discrete dividends are specified as cash amounts in spot currency**:

```python
# divs[t] = (cash_mean, cash_std)
divs = {
  0.25: (2.0, 0.5),
  0.75: (2.0, 0.5),
}
```

Internally, the pricers convert cash dividends to proportional multiplicative factors using the **expected pre-div forward level** (an approximation that keeps COS/LSMC/FDM/MC consistent).

Key helpers live in:
- [american_options/engine.py](american_options/engine.py): `cash_divs_to_proportional_divs(...)`, `forward_price(...)`

### 2) COS American rollback stability under dividends
The COS American rollback logic and diagnostics were hardened for discrete dividends:
- Dividend-aware time handling (avoid τ=0 grid pitfalls)
- Stable continuation/rollback diagnostics and CSV export

### 3) Diagnostics and validation plots
The main diagnostics live in:
- [plot_diagnostics.py](plot_diagnostics.py)

Notable outputs:
- COS vs BS sanity check: `figs/cos_vs_bs_gbm.png`
- American hard-max vs softmax rollback: `figs/american_hard_vs_soft.png`
- Jump models vs equivalent GBM: `figs/levy_vs_equiv_gbm.png`
- Continuation through time + references: `figs/continuation_through_time.png` + CSV
- Forward/parity through time: `figs/forward_parity_through_time.png` + CSV
- GBM MC vs COS under dividend uncertainty + implied-vol inversion: `figs/gbm_mc_vs_cos_div_uncertainty.png` + CSV

The dividend-uncertainty implied-vol diagnostics use **OTM options** for inversion stability (puts on low strikes, calls on high strikes), split using a deterministic forward.

### 4) Monte Carlo cross-check for dividend uncertainty
A vectorized GBM Monte Carlo was added to validate COS pricing under the same dividend-uncertainty model used by the engine.

### 5) Repo hygiene
- `.gitignore` added so Python bytecode caches (`__pycache__`, `*.pyc`) are not tracked.

## Repository Layout

- `american_options/`
  - `engine.py`: characteristic functions + COS pricing + dividend handling + forward helper
  - `lsmc.py`: LSMC pricer (diagnostics / alternative reference)
- `tests/`: pytest suite
- `tools/`: helper scripts
- `figs/`: generated PNG/CSV diagnostics
- `plot_diagnostics.py`: produces most plots/CSVs

## Quickstart

### Install
Create a venv and install dependencies (typical stack: numpy, scipy, matplotlib, pandas, pytest).

If you already have an environment, you can just run tests.

### Run tests

```bash
pytest -q
```

### Generate diagnostics

```bash
python plot_diagnostics.py
```

This writes plots/CSVs into `figs/`.

### Run demo

```bash
python run_demo.py
```

## How to Use the Pricing Engine

A minimal COS European call example under GBM with cash dividends:

```python
import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer

S0, r, q, T = 100.0, 0.02, 0.0, 1.0
vol = 0.25
K = np.array([100.0])

# Cash dividends: (mean, std) in spot currency
# (std=0 means deterministic cash dividend)
divs = {0.25: (2.0, 0.0), 0.75: (2.0, 0.0)}

model = GBMCHF(S0, r, q, divs, {"vol": vol})
pricer = COSPricer(model, N=512, L=8.0)

call = pricer.european_price(K, T, is_call=True)[0]
put = pricer.european_price(K, T, is_call=False)[0]
print(call, put)
```

American pricing (rollback) is available through `COSPricer.american_price(...)`.

## Notes / Caveats

- Cash dividends are approximated internally as proportional drops relative to an expected pre-div forward. This is deliberate for consistency across COS/MC/FDM/LSMC within this repo.
- Some figures in `figs/` are generated artifacts; re-run `python plot_diagnostics.py` if you want to refresh them.

## Additional Documentation

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [TRAJECTORY_CACHING.md](TRAJECTORY_CACHING.md)

## License

No license file is provided in this repository yet. Add a `LICENSE` if you want to publish it under a specific license.
