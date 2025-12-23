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

### 6) Discrete event jumps (scheduled binary multiplicative jump)
This repo supports a simple **scheduled, independent, binary event jump** at a known time $t_e$:

$$S_{t_e+} = S_{t_e-}\times J,\quad J=\begin{cases}u & \text{w.p. } p\\ d & \text{w.p. } (1-p)\end{cases}$$

Represent it with `DiscreteEventJump(time, p, u, d, ensure_martingale=True)` from:
- [american_options/events.py](american_options/events.py)

If `ensure_martingale=True` (default), the jump factors are internally normalized so the event has mean 1 under the pricing measure:

$$M = \mathbb{E}[J]=pu+(1-p)d,\qquad u_{eff}=u/M,\ d_{eff}=d/M.$$

You can pass the event into COS pricing as an optional argument:
- `COSPricer.european_price(..., event=event)`
- `COSPricer.american_price(..., event=event)`

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

## Discrete event jump usage

European pricing with an event jump (no dividends shown here):

```python
import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer
from american_options.events import DiscreteEventJump

S0, r, q, T = 100.0, 0.02, 0.0, 1.0
K = np.array([80.0, 100.0, 120.0])

model = GBMCHF(S0, r, q, divs={}, params={"vol": 0.25})
pricer = COSPricer(model, N=512, L=8.0)

event = DiscreteEventJump(time=0.30, p=0.60, u=1.10, d=0.92, ensure_martingale=True)

call = pricer.european_price(K, T, is_call=True, event=event)
put = pricer.european_price(K, T, is_call=False, event=event)
print(call, put)
```

American pricing uses the same `event=...` parameter and applies the mixture mapping at the event boundary during rollback.

## IV surface plots for event impact

Static 1x2 implied-vol surface (GBM+event vs VG+event), European COS, no dividends:

```bash
python tools/plot_event_iv_surfaces.py
```

Output:
- `figs/event_iv_surfaces_gbm_vs_vg.png`

Interactive linked 3D wireframes (rotate either plot; the other follows):

```bash
# Generates VG as the second model
python tools/make_linked_event_iv_surfaces.py --model vg

# Generates Merton as the second model
python tools/make_linked_event_iv_surfaces.py --model merton
```

Outputs:
- `figs/event_iv_surfaces_gbm_vs_vg_linked_3d.html`
- `figs/event_iv_surfaces_gbm_vs_merton_linked_3d.html`

Tip: open the HTML via a local HTTP server (some browsers restrict local file access):

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/figs/`.

## Notes / Caveats

- Cash dividends are approximated internally as proportional drops relative to an expected pre-div forward. This is deliberate for consistency across COS/MC/FDM/LSMC within this repo.
- Some figures in `figs/` are generated artifacts; re-run `python plot_diagnostics.py` if you want to refresh them.

## Additional Documentation

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [TRAJECTORY_CACHING.md](TRAJECTORY_CACHING.md)

## License

No license file is provided in this repository yet. Add a `LICENSE` if you want to publish it under a specific license.
