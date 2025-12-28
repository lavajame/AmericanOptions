# Calibration Suite - Quick Reference

Clean, persistent scripts for testing Lévy option pricing models. No CLI hacking required.

## Three Ways to Run

### 1. **Show Configuration Only** (30 seconds)
```bash
python tools/run_calibration.py
```
Displays what will happen without running anything. Good for understanding the setup.

### 2. **Run Everything** (5-10 minutes)
```bash
python tools/run_calibration_auto.py
```
Generates synthetic data and calibrates all test models automatically. Saves results to JSON.

### 3. **Interactive Exploration** (Real-time)
```python
from tools.calibration_explorer import scenario_basic, CalibrationSetup

# Use a preset
scenario_basic().run_all()

# Or create custom
setup = CalibrationSetup("My Test")
setup.set_true_model("kou", q=0.05)
setup.add_test_model("kou_vg")
setup.run_all()
```

---

## Edit Configuration In Code

All scripts have a clear configuration section at the top:

```python
TRUE_MODEL_CONFIG = {
    "type": "merton_vg",
    "q": 0.05,
    "components": [...]
}

DIVIDEND_SCHEDULE = [
    (0.25, 1.0),
    (0.5, 1.0),
    (0.75, 2.0),
]

TEST_MODELS = ["merton", "vg", "kou", "merton_vg"]
```

Change these, save, and run. That's it.

---

## Results Storage

All calibration results saved to: `calibration_results/results_*.json`

```json
{
    "timestamp": "20251228_164218",
    "true_model": {...},
    "test_models": ["merton", "vg", "kou", "merton_vg"],
    "results": {
        "merton": {"status": "success", "dimension": 5},
        "vg": {"status": "success", "dimension": 4},
        ...
    }
}
```

Persistent and queryable.

---

## Key Improvements Over CLI

| CLI Approach | Script Approach |
|--------------|-----------------|
| One-off commands | Persistent configuration |
| Easy to typo | Validated Python |
| Hard to compare | JSON results saved |
| Parameter tracking | Full audit trail |
| Manual iterations | Batch automation |

---

## Common Scenarios

### Test Robustness to Dividends
```python
TRUE_MODEL_CONFIG["type"] = "merton_vg"

for div_schedule in [[], [(0.5, 2.0)], [(0.25, 1.0), (0.75, 3.0)]]:
    DIVIDEND_SCHEDULE = div_schedule
    # Run script
```

### Compare Mono vs Combo Models
```python
TEST_MODELS = [
    "merton",
    "vg", 
    "kou",
    "cgmy",
    "merton_vg",
    "kou_vg",
    "cgmy_vg",
]
```

### Test Model Misspecification
```python
TRUE_MODEL_CONFIG["type"] = "cgmy_vg"  # True: CGMY+VG

TEST_MODELS = [
    "merton",         # Underspecified
    "merton_vg",      # Misspecified
    "cgmy_vg",        # Correct
]
```

---

## File Organization

```
AmericanOptions/
├── tools/
│   ├── run_calibration.py           # Show config & commands
│   ├── run_calibration_auto.py       # Run all auto
│   ├── calibration_explorer.py       # Interactive class
│   ├── model_specs.py                # Parameter definitions
│   └── calibrate_cloud_lbfgsb.py     # Core optimizer
├── calibration_results/
│   ├── results_20251228_164218.json  # Timestamped results
│   ├── results_20251228_164128.json
│   └── ...
└── CALIBRATION_SUITE.md              # Full documentation
```

---

## Model Information

### Mono Models
- **Merton**: Finite activity jumps (σ, λ, μ_J, σ_J)
- **Kou**: Asymmetric jump distribution (σ, λ, p, η₁, η₂)
- **VG**: Variance Gamma, infinite activity (θ, σ, ν)
- **CGMY**: Tempered stable (C, G, M, Y)

### Combo Models (with linked sigma)
- **merton_vg**: 7 parameters (VG.sigma → Merton.sigma)
- **kou_vg**: 8 parameters (VG.sigma → Kou.sigma)
- **cgmy_vg**: 9 parameters (no linking, no sigma in CGMY)

Plus `q` (dividend yield) in all composite models.

---

## Example: Full Workflow

```python
# 1. Edit tools/run_calibration_auto.py

TRUE_MODEL_CONFIG = {
    "type": "kou_vg",
    "q": 0.075,
    "components": [
        {
            "type": "kou",
            "params": {
                "sigma": 0.12,
                "lam": 1.0,
                "p": 0.4,
                "eta1": 15.0,
                "eta2": 8.0,
            }
        },
        {
            "type": "vg",
            "params": {
                "theta": -0.15,
                "sigma": 0.12,
                "nu": 0.08,
            }
        },
    ]
}

DIVIDEND_SCHEDULE = [
    (0.3, 1.5),
    (0.6, 1.5),
    (0.9, 1.5),
]

TEST_MODELS = [
    "kou",
    "vg",
    "merton_vg",
    "kou_vg",  # Should fit best
]

# 2. Run
# python tools/run_calibration_auto.py

# 3. Inspect results
# cat calibration_results/results_*.json
```

---

## See Also

- [CALIBRATION_SUITE.md](CALIBRATION_SUITE.md) — Detailed documentation
- [GRADIENT_VERIFICATION.md](GRADIENT_VERIFICATION.md) — Gradient math
- [model_specs.py](tools/model_specs.py) — Parameter specs
