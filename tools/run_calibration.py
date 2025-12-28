#!/usr/bin/env python3
"""
Clean calibration script with persistent configuration.

Define a true model and dividend schedule, then calibrate different mono/combo
models to the synthetic option prices it generates.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from model_specs import CompositeModelSpec


# ============================================================================
# CONFIGURATION
# ============================================================================

# True/Generating Model Configuration
TRUE_MODEL_CONFIG = {
    "type": "merton_vg",  # Options: "merton", "kou", "vg", "cgmy", "merton_vg", "kou_vg", "cgmy_vg"
    "q": 0.05,  # Dividend yield
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
                "sigma": 0.15,  # Linked to Merton.sigma in merton_vg
                "nu": 0.05,
            }
        },
    ]
}

# Dividend Schedule: list of (spot_date_fraction, div_amount) tuples
# For example: [(0.25, 1.0), (0.5, 1.0), (0.75, 2.0)] means divs at T=0.25, 0.5, 0.75
DIVIDEND_SCHEDULE = [
    (0.25, 1.0),
    (0.5, 1.0),
    (0.75, 2.0),
]

# Market parameters (for European phase initialization)
S0 = 100.0
r = 0.02

# Calibration parameters
EUROPEAN_INIT_PARAMS = {
    "N": 256,  # Number of Fourier terms
    "L": 10.0,  # Boundary for characteristic function
    "max_iter": 50,  # Max iterations in SLSQP
}

AMERICAN_REFINEMENT_PARAMS = {
    "N": 128,
    "L": 8.0,
    "steps": 50,  # Time steps for American pricing
    "beta": 100.0,  # Penalty parameter for early exercise
    "max_iter": 50,
}

# Models to test (calibrate to the synthetic data)
# Options: "merton", "kou", "vg", "cgmy", "merton_vg", "kou_vg", "cgmy_vg"
TEST_MODELS = [
    "merton",
    "vg",
    "kou",
    "merton_vg",
]

# Output configuration
OUTPUT_DIR = Path(__file__).parent.parent / "calibration_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_divs_for_cli(div_schedule):
    """Convert dividend schedule list to CLI format string."""
    parts = []
    for spot_frac, amount in div_schedule:
        # spot_frac is fraction of [0, 1], assume max T=2 so multiply by 2
        t_value = spot_frac * 2.0
        parts.append(f"{spot_frac}:{t_value}:0.0")
    return parts


def build_component_list(config):
    """Build component list for CompositeLevyCHF from config dict."""
    components = []
    for comp in config.get("components", []):
        components.append({
            "type": comp["type"],
            "params": comp["params"]
        })
    return components


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * len(title))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print_header("CALIBRATION SUITE - SYNTHETIC DATA GENERATION & MODEL TESTING")
    
    # ========================================================================
    # PHASE 1: Display Configuration
    # ========================================================================
    print_section("Configuration Summary")
    
    print(f"\nTrue Model: {TRUE_MODEL_CONFIG['type'].upper()}")
    print(f"Dividend Yield (q): {TRUE_MODEL_CONFIG['q']:.4f}")
    print(f"Market: S0={S0}, r={r:.4f}")
    
    print(f"\nTrue Model Parameters:")
    for comp in TRUE_MODEL_CONFIG["components"]:
        print(f"  {comp['type'].upper()}:")
        for param, value in comp["params"].items():
            print(f"    {param}: {value}")
    
    print(f"\nDividend Schedule:")
    for t_frac, amount in DIVIDEND_SCHEDULE:
        print(f"  t={t_frac}: ${amount:.2f}")
    
    print(f"\nEuropean Phase (warm-start):")
    for key, val in EUROPEAN_INIT_PARAMS.items():
        print(f"  {key}: {val}")
    
    print(f"\nAmerican Phase (refinement):")
    for key, val in AMERICAN_REFINEMENT_PARAMS.items():
        print(f"  {key}: {val}")
    
    print(f"\nTest Models:")
    for model in TEST_MODELS:
        print(f"  - {model}")
    
    # ========================================================================
    # PHASE 2: Generate Synthetic Data
    # ========================================================================
    print_header("PHASE 1: SYNTHETIC DATA GENERATION")
    
    print(f"Generating option prices from {TRUE_MODEL_CONFIG['type'].upper()} model...")
    
    # Build components for true model
    true_components = build_component_list(TRUE_MODEL_CONFIG)
    true_q = TRUE_MODEL_CONFIG["q"]
    
    # Build CLI command for synthetic data generation
    div_args = []
    for t_frac, amount in DIVIDEND_SCHEDULE:
        div_args.extend(["--div", f"{t_frac}:{t_frac*2.0}:0.0"])
    
    print(f"\n  Components: {', '.join(c['type'] for c in true_components)}")
    print(f"  Q parameter: {true_q}")
    print(f"  Dividend events: {len(DIVIDEND_SCHEDULE)}")
    
    print(f"\n✓ Configuration ready for calibration")
    
    # ========================================================================
    # PHASE 3: Calibrate Test Models
    # ========================================================================
    print_header("PHASE 2: MODEL CALIBRATION")
    
    results = {}
    
    for test_model in TEST_MODELS:
        print_section(f"Testing: {test_model.upper()}")
        
        # Create spec for test model
        if "_" in test_model:
            comp_names = test_model.split("_")
            spec = CompositeModelSpec(comp_names)
        else:
            spec = CompositeModelSpec([test_model])
        
        print(f"Parameters in test model: {spec.dim}")
        print(f"  Names: {spec.param_names}")
        
        # Build command
        cmd_parts = [
            "calibrate_cloud_lbfgsb.py",
            "--synthetic",
            f"--synthetic-model {TRUE_MODEL_CONFIG['type']}",
            f"--case {test_model}_q",
        ]
        
        # Add divs
        for t_frac, amount in DIVIDEND_SCHEDULE:
            cmd_parts.append(f"--div {t_frac}:{t_frac*2.0}:0.0")
        
        # Add european params
        cmd_parts.extend([
            "--european-init",
            f"--european-N {EUROPEAN_INIT_PARAMS['N']}",
            f"--european-L {EUROPEAN_INIT_PARAMS['L']}",
            f"--european-maxiter {EUROPEAN_INIT_PARAMS['max_iter']}",
        ])
        
        # Add american params
        cmd_parts.extend([
            f"--american-N {AMERICAN_REFINEMENT_PARAMS['N']}",
            f"--american-L {AMERICAN_REFINEMENT_PARAMS['L']}",
            f"--american-steps {AMERICAN_REFINEMENT_PARAMS['steps']}",
            f"--american-beta {AMERICAN_REFINEMENT_PARAMS['beta']}",
            f"--american-maxiter {AMERICAN_REFINEMENT_PARAMS['max_iter']}",
        ])
        
        # Add 3D HTML plot with clear model names
        html_out = f"--iv-plot-3d-html-out figs/iv_fit_{TRUE_MODEL_CONFIG['type']}_to_{test_model}.html"
        cmd_parts.append(html_out)
        
        print(f"\n  To run this calibration manually:")
        print(f"    python {' '.join(cmd_parts)}\n")
        
        results[test_model] = {
            "dimension": spec.dim,
            "parameters": spec.param_names,
            "command": " ".join(cmd_parts),
            "status": "ready",
        }
    
    # ========================================================================
    # PHASE 4: Summary and Instructions
    # ========================================================================
    print_header("EXECUTION SUMMARY")
    
    print(f"\nConfiguration has been set up for {len(TEST_MODELS)} test model(s):")
    for i, model in enumerate(TEST_MODELS, 1):
        print(f"  {i}. {model}")
    
    print(f"\nTo run calibrations:")
    print(f"  Option A: Run each model manually using the commands above")
    print(f"  Option B: Modify TEST_MODELS and run this script again")
    print(f"  Option C: Use CLI directly:")
    print(f"    python calibrate_cloud_lbfgsb.py --synthetic \\")
    print(f"      --synthetic-model {TRUE_MODEL_CONFIG['type']} \\")
    print(f"      --case <test_model>_q \\")
    print(f"      --div 0.25:0.5:0.0 --div 0.5:1.0:0.0 --div 0.75:1.5:0.0 \\")
    print(f"      --european-init --european-N 256 --european-L 10.0 \\")
    print(f"      --american-N 128 --american-L 8.0 --american-steps 50")
    
    # Save configuration for reference
    config_output = OUTPUT_DIR / f"config_{TIMESTAMP}.json"
    config_data = {
        "timestamp": TIMESTAMP,
        "true_model": TRUE_MODEL_CONFIG,
        "dividend_schedule": DIVIDEND_SCHEDULE,
        "market": {"S0": S0, "r": r},
        "european_params": EUROPEAN_INIT_PARAMS,
        "american_params": AMERICAN_REFINEMENT_PARAMS,
        "test_models": TEST_MODELS,
        "results": results,
    }
    
    with open(config_output, "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {config_output}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
