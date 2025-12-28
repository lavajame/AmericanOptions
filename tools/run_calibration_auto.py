#!/usr/bin/env python3
"""
Automated calibration runner with configuration persistence.

This script:
1. Defines a true/generating model with parameters and dividend schedule
2. Generates synthetic option prices
3. Calibrates different mono/combo models to the synthetic data
4. Reports results in a clean, organized manner
"""

import numpy as np
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from model_specs import CompositeModelSpec


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
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
# where spot_date_fraction is in [0, 1]
DIVIDEND_SCHEDULE = [
    (0.25, 1.0),
    (0.5, 1.0),
    (0.75, 2.0),
]

# Market parameters
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
TEST_MODELS = [
    "merton",
    "vg",
    "kou",
    "merton_vg",
]

# Run settings
RUN_CALIBRATIONS = True  # Set to False to just show commands
VERBOSE = True  # Print detailed output

# Output configuration
OUTPUT_DIR = Path(__file__).parent.parent / "calibration_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * len(title))


def print_param_dict(d, indent=2):
    """Pretty print a parameter dictionary."""
    for key, value in d.items():
        if isinstance(value, float):
            print(" " * indent + f"{key}: {value:.6f}")
        else:
            print(" " * indent + f"{key}: {value}")


def build_calibration_command(true_model, test_model, divs, eur_params, am_params):
    """Build the calibration command."""
    # Build base command
    cmd = [
        "python",
        "tools/calibrate_cloud_lbfgsb.py",
        "--synthetic",
        f"--synthetic-model {true_model}",
        f"--case {test_model}_q",
    ]
    
    # Add dividend schedule
    for t_frac, amount in divs:
        cmd.append(f"--div {t_frac}:{t_frac*2.0}:0.0")
    
    # Add European phase params
    cmd.append("--european-init")
    cmd.append(f"--european-N {eur_params['N']}")
    cmd.append(f"--european-L {eur_params['L']}")
    cmd.append(f"--european-maxiter {eur_params['max_iter']}")
    
    # Add American phase params
    cmd.append(f"--american-N {am_params['N']}")
    cmd.append(f"--american-L {am_params['L']}")
    cmd.append(f"--american-steps {am_params['steps']}")
    cmd.append(f"--american-beta {am_params['beta']}")
    cmd.append(f"--american-maxiter {am_params['max_iter']}")
    
    # Add 3D HTML plot output with model names
    html_out = f"figs/iv_fit_{true_model}_to_{test_model}.html"
    cmd.append(f"--iv-plot-3d-html-out {html_out}")
    
    return " ".join(cmd)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print_header("CALIBRATION SUITE - AUTOMATED SYNTHETIC DATA TESTING")
    
    # ========================================================================
    # Configuration Summary
    # ========================================================================
    print_section("Configuration")
    
    print(f"\nTrue Model: {TRUE_MODEL_CONFIG['type'].upper()}")
    print(f"Q (dividend yield): {TRUE_MODEL_CONFIG['q']:.4f}")
    print(f"Market: S0={S0}, r={r:.4f}\n")
    
    print("True Model Parameters:")
    for comp in TRUE_MODEL_CONFIG["components"]:
        print(f"  {comp['type'].upper()}:")
        print_param_dict(comp["params"], indent=4)
    
    print(f"\nDividend Schedule ({len(DIVIDEND_SCHEDULE)} events):")
    for t_frac, amount in DIVIDEND_SCHEDULE:
        print(f"  t={t_frac:.2f}: ${amount:.2f}")
    
    print(f"\nEuropean Phase (warm-start):")
    print_param_dict(EUROPEAN_INIT_PARAMS, indent=2)
    
    print(f"\nAmerican Phase (refinement):")
    print_param_dict(AMERICAN_REFINEMENT_PARAMS, indent=2)
    
    print(f"\nTest Models ({len(TEST_MODELS)}):")
    for model in TEST_MODELS:
        if "_" in model:
            comp_names = model.split("_")
            spec = CompositeModelSpec(comp_names)
        else:
            spec = CompositeModelSpec([model])
        print(f"  - {model:12s} (dim={spec.dim})")
    
    # ========================================================================
    # Run Calibrations
    # ========================================================================
    if not RUN_CALIBRATIONS:
        print_header("COMMANDS TO RUN")
        for test_model in TEST_MODELS:
            print_section(f"{test_model.upper()}")
            cmd = build_calibration_command(
                TRUE_MODEL_CONFIG["type"],
                test_model,
                DIVIDEND_SCHEDULE,
                EUROPEAN_INIT_PARAMS,
                AMERICAN_REFINEMENT_PARAMS
            )
            print(f"\n{cmd}\n")
        print()
        return
    
    print_header("RUNNING CALIBRATIONS")
    
    results = {}
    
    for i, test_model in enumerate(TEST_MODELS, 1):
        print_section(f"[{i}/{len(TEST_MODELS)}] {test_model.upper()}")
        
        # Get model dimension
        if "_" in test_model:
            comp_names = test_model.split("_")
            spec = CompositeModelSpec(comp_names)
        else:
            spec = CompositeModelSpec([test_model])
        
        print(f"Dimension: {spec.dim}")
        
        # Build command
        cmd = build_calibration_command(
            TRUE_MODEL_CONFIG["type"],
            test_model,
            DIVIDEND_SCHEDULE,
            EUROPEAN_INIT_PARAMS,
            AMERICAN_REFINEMENT_PARAMS
        )
        
        if VERBOSE:
            print(f"\nCommand:\n  {cmd}\n")
        
        # Run calibration
        print("Running calibration...")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print("[OK] Calibration completed successfully")
                results[test_model] = {
                    "status": "success",
                    "dimension": spec.dim,
                    "stdout_length": len(result.stdout),
                }
                
                # Extract key results from stdout (if available)
                if "Phase 1 result" in result.stdout:
                    for line in result.stdout.split("\n"):
                        if "Phase" in line and "result" in line:
                            print(f"  {line.strip()}")
            else:
                print(f"[FAIL] Calibration failed with return code {result.returncode}")
                print(f"\nError output:\n{result.stderr[-500:]}")  # Last 500 chars
                results[test_model] = {
                    "status": "failed",
                    "dimension": spec.dim,
                    "error": result.stderr[-200:],
                }
        
        except subprocess.TimeoutExpired:
            print("[TIMEOUT] Calibration timed out after 10 minutes")
            results[test_model] = {
                "status": "timeout",
                "dimension": spec.dim,
            }
        except Exception as e:
            print(f"[ERROR] Calibration error: {e}")
            results[test_model] = {
                "status": "error",
                "dimension": spec.dim,
                "error": str(e),
            }
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_header("SUMMARY")
    
    print(f"\nResults ({len(results)} models tested):\n")
    for model, res in results.items():
        status_icon = "[OK]" if res["status"] == "success" else "[X]"
        print(f"  {status_icon} {model:12s} - {res['status']:10s} (dim={res['dimension']})")
    
    # Save results
    config_output = OUTPUT_DIR / f"results_{TIMESTAMP}.json"
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
    
    print(f"\n[OK] Results saved to: {config_output}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
