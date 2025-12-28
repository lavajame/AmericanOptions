#!/usr/bin/env python3
"""
Generate all mono- and combo-model calibration demonstration plots.

This script runs calibrations for all supported model combinations
using merton_vg as the generating/true model, producing clean 3D HTML
plots with smooth IV curves for publication.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# All models to test
MODELS = [
    ("merton", 5),
    ("vg", 4),
    ("kou", 6),
    ("cgmy", 5),
    ("nig", 4),
    ("merton_vg", 7),
    ("kou_vg", 8),
    ("cgmy_vg", 8),
    ("kou_nig", 10),
]

TRUE_MODEL = "merton_vg"
OUTPUT_DIR = Path(__file__).parent.parent / "figs" / "calibration_3d"
RESULTS_DIR = Path(__file__).parent.parent / "calibration_results"

def run_calibration(test_model: str) -> bool:
    """Run one calibration and return success status."""
    html_out = OUTPUT_DIR / f"iv_fit_{TRUE_MODEL}_to_{test_model}.html"
    
    cmd = [
        "python",
        str(Path(__file__).parent / "calibrate_cloud_lbfgsb.py"),
        "--synthetic",
        f"--synthetic-model {TRUE_MODEL}",
        f"--case {test_model}_q",
        "--european-init",
        "--european-N", "256",
        "--european-L", "10.0",
        "--american-N", "128",
        "--american-L", "8.0",
        "--american-steps", "50",
        "--american-beta", "100.0",
        f"--iv-plot-3d-html-out", str(html_out),
    ]
    
    print(f"\n{'='*70}")
    print(f"  {test_model.upper():20s} (dim={dict(MODELS)[test_model]})")
    print(f"{'='*70}")
    print(f"  Output: {html_out.name}")
    print()
    
    result = subprocess.run(
        " ".join(cmd),
        shell=True,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Extract final loss from output
        for line in result.stdout.split("\n"):
            if line.startswith("Final result:"):
                print(f"  ✓ {line.strip()}")
                break
        return True
    else:
        print(f"  ✗ FAILED")
        if "Invalid composite params" in result.stderr or "ValueError" in result.stderr:
            print(f"  (parameter bounds or numerical issue)")
        return False


def main():
    print("\n" + "="*70)
    print("  CALIBRATION DEMONSTRATION SUITE")
    print("="*70)
    print(f"\nGenerating {len(MODELS)} model calibrations")
    print(f"  True model: {TRUE_MODEL.upper()}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    for model_name, dim in MODELS:
        results[model_name] = run_calibration(model_name)
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}\n")
    
    successful = [m for m, success in results.items() if success]
    failed = [m for m, success in results.items() if not success]
    
    print(f"  Successful: {len(successful)}/{len(MODELS)}")
    for model in successful:
        print(f"    ✓ {model}")
    
    if failed:
        print(f"\n  Failed: {len(failed)}/{len(MODELS)}")
        for model in failed:
            print(f"    ✗ {model}")
    
    # List output files
    html_files = sorted(OUTPUT_DIR.glob("iv_fit_merton_vg_to_*.html"))
    print(f"\n  Output files ({len(html_files)} HTML plots):")
    for f in html_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:6.1f} KB)")
    
    print(f"\n{'='*70}\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
