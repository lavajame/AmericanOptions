#!/usr/bin/env python
"""Quick calibration: merton_vg → kou_nig composite."""

import sys
import subprocess
import json
import tempfile
from pathlib import Path

def main():
    """Generate synthetic IV surface from merton_vg, calibrate to kou_nig."""
    
    # Output path
    output_html = Path(__file__).parent.parent / "figs" / "calibration_3d" / "iv_fit_merton_vg_to_kou_nig_q.html"
    output_json = Path(__file__).parent.parent / "calibration_results" / "calibration_kou_nig_q.json"
    
    # Ensure directories exist
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Calibration command
    cmd = [
        "python", 
        str(Path(__file__).parent / "calibrate_cloud_lbfgsb.py"),
        "--synthetic",
        "--synthetic-model", "merton_vg",
        "--case", "kou_nig_q",
        "--european-init",
        "--iv-csv-out", str(Path(__file__).parent.parent / "calibration_results" / "iv_merton_vg_to_kou_nig_q.csv"),
        "--iv-plot-3d-html-out", str(output_html),
    ]
    
    print(f"Running calibration: merton_vg → kou_nig")
    print(f"  Output: {output_html}")
    print(f"  Results: {output_json}")
    print()
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    
    if result.returncode == 0:
        print("\n✓ Calibration complete!")
        print(f"  HTML plot: {output_html}")
        print(f"  JSON results: {output_json}")
        if output_json.exists():
            with open(output_json) as f:
                results = json.load(f)
                print(f"  Optimization status: {results.get('optimization', {}).get('success', 'unknown')}")
                print(f"  Final MSE: {results.get('optimization', {}).get('mse_final', 'unknown'):.6f}")
    else:
        print(f"\n✗ Calibration failed with exit code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
