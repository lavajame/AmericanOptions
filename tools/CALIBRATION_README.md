# Clean Calibration Suite

Instead of CLI hacks, use these well-organized Python scripts to test Lévy models against synthetic data.

## What You Get

Three complementary ways to run calibrations:

| Script | Purpose | Time | Best For |
|--------|---------|------|----------|
| `run_calibration.py` | Show config & commands | 1 sec | Understanding setup |
| `run_calibration_auto.py` | Auto-run all calibrations | 5-10 min | Production runs |
| `calibration_explorer.py` | Interactive class-based API | Real-time | Jupyter/notebooks |

## Quick Start

```bash
# Option 1: See what will happen
python tools/run_calibration.py

# Option 2: Run everything
python tools/run_calibration_auto.py

# Option 3: Example/demo
python tools/example_basic_calibration.py
```

All results automatically saved to: `calibration_results/results_*.json`

## Edit Configuration

In any script, modify the config section at the top:

```python
# True model that generates synthetic prices
TRUE_MODEL_CONFIG = {
    "type": "merton_vg",
    "q": 0.05,
    "components": [...]
}

# Dividend schedule
DIVIDEND_SCHEDULE = [
    (0.25, 1.0),
    (0.5, 1.0),
    (0.75, 2.0),
]

# Models to test
TEST_MODELS = ["merton", "vg", "kou", "merton_vg"]
```

Save and run. No CLI typing.

## Key Features

- ✅ Persistent configuration (not lost between runs)
- ✅ Linked sigma parameters in merton_vg and kou_vg
- ✅ JSON results with timestamps
- ✅ Two-phase calibration (European + American)
- ✅ Clean, readable output
- ✅ Full audit trail

## Model Information

**Mono Models:**
- Merton: Finite jumps (σ, λ, μ_J, σ_J)
- Kou: Asymmetric jumps (σ, λ, p, η₁, η₂)
- VG: Variance Gamma (θ, σ, ν)
- CGMY: Tempered stable (C, G, M, Y)

**Combo Models (sigma linked):**
- merton_vg: 7 params (VG.σ → Merton.σ)
- kou_vg: 8 params (VG.σ → Kou.σ)
- cgmy_vg: 9 params (no linking)

Plus `q` (dividend yield) added in all composite models.

## Example Workflows

### Test Different Dividend Schedules
```python
scenarios = {
    "no divs": [],
    "one div": [(0.5, 2.0)],
    "many divs": [(t/20, 0.5) for t in range(1, 20)],
}

for name, divs in scenarios.items():
    DIVIDEND_SCHEDULE = divs
    # Run calibration
```

### Test Model Misspecification
```python
TRUE_MODEL_CONFIG["type"] = "cgmy_vg"  # True model

TEST_MODELS = [
    "merton",          # Underspecified
    "vg",              # Partially correct
    "merton_vg",       # Misspecified
    "cgmy_vg",         # Correct model
]
```

### Batch Testing Different Parameters
```python
for jump_intensity in [0.1, 0.5, 1.0, 2.0]:
    TRUE_MODEL_CONFIG["components"][0]["params"]["lam"] = jump_intensity
    # Run calibration
```

## File Structure

```
tools/
├── run_calibration.py            # Config display mode
├── run_calibration_auto.py        # Auto-run mode  
├── calibration_explorer.py        # Interactive API
├── example_basic_calibration.py   # Copy-paste example
├── model_specs.py                 # Parameter definitions
└── calibrate_cloud_lbfgsb.py      # Core optimizer

calibration_results/
├── results_20251228_164218.json   # Timestamped results
├── results_20251228_164128.json
└── ...

CALIBRATION_SUITE.md              # Detailed docs
CALIBRATION_QUICK_START.md        # Quick ref
```

## Interactive Usage

```python
from calibration_explorer import CalibrationSetup, scenario_basic

# Use preset scenario
scenario_basic().run_all()

# Or build custom setup
setup = CalibrationSetup("My Experiment")
setup.set_true_model("kou_vg", q=0.075)
setup.add_test_model("kou")
setup.add_test_model("kou_vg")
setup.set_divs([(0.3, 2.0), (0.7, 2.0)])
setup.run_all()

# Or just print commands for review
setup.print_commands()
```

## Documentation

- [CALIBRATION_SUITE.md](../CALIBRATION_SUITE.md) — Full reference
- [CALIBRATION_QUICK_START.md](../CALIBRATION_QUICK_START.md) — Quick start guide
- [model_specs.py](model_specs.py) — Parameter specifications
- [GRADIENT_VERIFICATION.md](../GRADIENT_VERIFICATION.md) — Gradient proofs

## Comparison with CLI

### Before (CLI Hacky):
```bash
python calibrate_cloud_lbfgsb.py --synthetic \
  --synthetic-model merton_vg --case kou_q \
  --div 0.25:0.5:0.0 --div 0.5:1.0:0.0 \
  --european-init --european-N 256 --european-L 10.0 \
  --american-N 128 --american-L 8.0 \
  ...
```
❌ Hard to read  
❌ Easy to typo  
❌ Results not saved  
❌ Can't compare runs  

### After (Python Scripts):
```python
config = CalibrationSetup("My Test")
config.set_true_model("merton_vg", q=0.05)
config.add_test_model("kou_vg")
config.run_all()
```
✅ Clear intent  
✅ Type-safe  
✅ JSON results saved  
✅ Easy to compare/analyze  

## Next Steps

1. **Get started**: `python tools/run_calibration_auto.py`
2. **Customize**: Edit configuration section in any script
3. **Analyze**: Open `calibration_results/results_*.json`
4. **Explore**: Use `calibration_explorer.py` for interactive work

---

Questions? See [CALIBRATION_SUITE.md](../CALIBRATION_SUITE.md) for detailed docs.
