# Calibration Suite Documentation

Two clean Python scripts for testing Lévy option pricing models against synthetic data.

## Quick Start

### Option 1: View Commands Only (No CLI Hacking)

```bash
python tools/run_calibration.py
```

This displays:
- Configuration summary (true model, parameters, dividend schedule)
- Dimension reduction with linked parameters
- Ready-to-copy calibration commands for each test model

### Option 2: Run Everything Automatically

```bash
python tools/run_calibration_auto.py
```

This:
1. Defines your true/generating model
2. Generates synthetic option prices
3. Runs calibrations for all test models in sequence
4. Saves results to JSON with timestamps

Results are saved to: `calibration_results/results_YYYYMMDD_HHMMSS.json`

---

## Configuration

Edit the configuration section at the top of your chosen script:

### True Model Setup

```python
TRUE_MODEL_CONFIG = {
    "type": "merton_vg",  # merton|kou|vg|cgmy|merton_vg|kou_vg|cgmy_vg
    "q": 0.05,            # Dividend yield
    "components": [
        {
            "type": "merton",
            "params": {
                "sigma": 0.15,
                "lam": 0.5,
                "muJ": -0.2,
                "sigmaJ": 0.1,
            }
        },
        {
            "type": "vg",
            "params": {
                "theta": -0.2,
                "sigma": 0.15,  # Linked to Merton.sigma
                "nu": 0.05,
            }
        },
    ]
}
```

### Dividend Schedule

List of `(time_fraction, amount)` tuples where `time_fraction ∈ [0, 1]`:

```python
DIVIDEND_SCHEDULE = [
    (0.25, 1.0),   # $1.00 at t=0.25
    (0.5, 1.0),    # $1.00 at t=0.50
    (0.75, 2.0),   # $2.00 at t=0.75
]
```

### Test Models

Which models to calibrate to the synthetic data:

```python
TEST_MODELS = [
    "merton",      # Mono-component
    "vg",
    "kou",
    "merton_vg",   # Combo models (with linked sigma)
    "kou_vg",
    "cgmy_vg",
]
```

### Calibration Parameters

**European Phase** (warm-start with OTM quotes):
```python
EUROPEAN_INIT_PARAMS = {
    "N": 256,        # Fourier terms
    "L": 10.0,       # Integration boundary
    "max_iter": 50,  # SLSQP iterations
}
```

**American Phase** (refinement with full set):
```python
AMERICAN_REFINEMENT_PARAMS = {
    "N": 128,
    "L": 8.0,
    "steps": 50,      # Time stepping for American
    "beta": 100.0,    # Early exercise penalty
    "max_iter": 50,   # SLSQP iterations
}
```

---

## Key Features

### 1. Linked Sigma Parameters

**merton_vg** and **kou_vg** automatically link their two sigma parameters:
- Reduces dimension (fewer variables to optimize)
- Avoids degeneracy (both sigmas contribute to same effect)
- Mathematically sound (both enter diffusion component)

Example:
```
merton_vg without linking:  8 parameters
merton_vg with linking:     7 parameters  (VG.sigma → Merton.sigma)
```

### 2. Parameter Specifications

All Lévy models defined in `tools/model_specs.py`:

| Model | Parameters | Dimension |
|-------|-----------|-----------|
| Merton | σ, λ, μ_J, σ_J | 4 |
| Kou | σ, λ, p, η₁, η₂ | 5 |
| VG | θ, σ, ν | 3 |
| CGMY | C, G, M, Y | 4 |
| merton_vg | (linked σ) | 7 |
| kou_vg | (linked σ) | 8 |

Plus `q` (dividend yield) in composite models.

### 3. Two-Phase Calibration

**Phase 1 (European)**:
- Uses OTM quotes (faster, more stable)
- Warm-starts the optimizer
- Reduces initial loss significantly

**Phase 2 (American)**:
- Uses full option set (ITM + ATM + OTM)
- Refines parameters
- Accounts for early exercise

### 4. Persistent Configuration

All runs saved to JSON with:
- Timestamp
- True model specification
- Dividend schedule
- Calibration parameters
- Results for all models tested

Load with: `json.load(open("calibration_results/results_*.json"))`

---

## Example Workflows

### Test Which Model Fits Best

```python
TRUE_MODEL_CONFIG["type"] = "cgmy_vg"
TRUE_MODEL_CONFIG["components"] = [...]

TEST_MODELS = [
    "merton",
    "kou", 
    "vg",
    "cgmy",
    "merton_vg",
    "kou_vg", 
    "cgmy_vg",  # Should fit perfectly
]

# Run: python tools/run_calibration_auto.py
```

Results tell you which models can approximate your data.

### Test Sensitivity to Dividend Schedule

```python
# Test 1: No dividends
DIVIDEND_SCHEDULE = []

# Test 2: Few large dividends
DIVIDEND_SCHEDULE = [(0.5, 5.0)]

# Test 3: Many small dividends
DIVIDEND_SCHEDULE = [(t/20, 0.5) for t in range(1, 20)]

# Run each configuration
```

### Vary True Model Parameters

```python
# Test low volatility
TRUE_MODEL_CONFIG["components"][0]["params"]["sigma"] = 0.05

# Test high jump intensity
TRUE_MODEL_CONFIG["components"][0]["params"]["lam"] = 2.0

# Test fat tails
TRUE_MODEL_CONFIG["components"][1]["params"]["nu"] = 0.2
```

---

## Understanding Results

Each calibration produces:

```
[1/4] MERTON
Dimension: 5
...
✓ Calibration completed successfully
  Phase 1 result: f=7.330005e-03
```

**Interpretation**:
- **Dimension 5**: q + 4 Merton parameters
- **Phase 1 loss**: How well OTM quotes are fit
- **Phase 2 loss**: Final fit after refinement (usually much better)

Lower loss = better fit

---

## Advanced Usage

### Manual Command Execution

From `run_calibration.py` output, copy any command:

```bash
python tools/calibrate_cloud_lbfgsb.py \
  --synthetic \
  --synthetic-model merton_vg \
  --case kou_q \
  --div 0.25:0.5:0.0 --div 0.5:1.0:0.0 \
  --european-init --european-N 256 --european-L 10.0 \
  --american-N 128 --american-L 8.0 --american-steps 50
```

### Inspect Results JSON

```python
import json

with open("calibration_results/results_20251228_164218.json") as f:
    results = json.load(f)

# View true model
print(results["true_model"])

# View which models succeeded
for model, res in results["results"].items():
    print(f"{model}: {res['status']}")
```

### Batch Testing Different Scenarios

Create variants of the script:

```python
# run_calibration_vg_only.py
TRUE_MODEL_CONFIG["type"] = "vg"
TEST_MODELS = ["vg"]

# run_calibration_gamma_jump.py
TRUE_MODEL_CONFIG["components"][0]["params"]["lam"] = 3.0
TEST_MODELS = ["merton", "kou", "merton_vg", "kou_vg"]
```

---

## Troubleshooting

### Calibration Times Out
Reduce `max_iter` in phase params or `N` (Fourier terms).

### Poor Fit Despite Multiple Models
- Increase `american_steps` (more time accuracy)
- Check dividend schedule is realistic
- Verify true model parameters are reasonable

### Models Not Converging
- Decrease `L` boundary (tighter integration)
- Increase `max_iter` in SLSQP
- Check initial guess in `model_specs.py`

---

## Scripts Reference

### `run_calibration.py`
- Displays configuration and commands
- Does NOT run calibrations
- Good for understanding what will happen
- Quick CPU time (~1 second)

### `run_calibration_auto.py`
- Actually runs all calibrations
- Saves results to JSON
- Takes minutes (depends on `max_iter`)
- Can handle timeouts and errors gracefully

---

## See Also

- [model_specs.py](tools/model_specs.py) — Parameter specifications
- [calibrate_cloud_lbfgsb.py](tools/calibrate_cloud_lbfgsb.py) — Core calibration logic
- [GRADIENT_VERIFICATION.md](GRADIENT_VERIFICATION.md) — Analytical gradient proofs
