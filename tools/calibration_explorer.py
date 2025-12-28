#!/usr/bin/env python3
"""
Interactive calibration explorer - reusable configuration snippets.

This module provides easy-to-modify configuration templates and functions
for running specific calibration scenarios.
"""

import json
from pathlib import Path
from datetime import datetime
import subprocess


class CalibrationSetup:
    """Reusable calibration configuration."""
    
    def __init__(self, name="default"):
        self.name = name
        self.true_model = "merton_vg"
        self.q = 0.05
        self.divs = [(0.25, 1.0), (0.5, 1.0), (0.75, 2.0)]
        self.test_models = ["merton", "vg", "merton_vg"]
        self.european_N = 256
        self.european_L = 10.0
        self.american_N = 128
        self.american_L = 8.0
        self.results = {}
    
    def add_test_model(self, model_name):
        """Add a test model."""
        if model_name not in self.test_models:
            self.test_models.append(model_name)
        return self
    
    def set_true_model(self, model_name, q=None):
        """Set the true/generating model."""
        self.true_model = model_name
        if q is not None:
            self.q = q
        return self
    
    def set_divs(self, divs):
        """Set dividend schedule."""
        self.divs = divs
        return self
    
    def set_european_params(self, N=256, L=10.0):
        """Set European phase parameters."""
        self.european_N = N
        self.european_L = L
        return self
    
    def set_american_params(self, N=128, L=8.0):
        """Set American phase parameters."""
        self.american_N = N
        self.american_L = L
        return self
    
    def build_cmd(self, test_model):
        """Build calibration command for a test model."""
        cmd = [
            "python",
            "tools/calibrate_cloud_lbfgsb.py",
            "--synthetic",
            f"--synthetic-model {self.true_model}",
            f"--case {test_model}_q",
        ]
        
        for t_frac, amount in self.divs:
            cmd.append(f"--div {t_frac}:{t_frac*2.0}:0.0")
        
        cmd.extend([
            "--european-init",
            f"--european-N {self.european_N}",
            f"--european-L {self.european_L}",
            f"--european-maxiter 50",
            f"--american-N {self.american_N}",
            f"--american-L {self.american_L}",
            f"--american-steps 50",
            f"--american-beta 100.0",
            f"--american-maxiter 50",
            f"--iv-plot-3d-html-out figs/iv_fit_{self.true_model}_to_{test_model}.html",
        ])
        
        return " ".join(cmd)
    
    def show_config(self):
        """Display configuration."""
        print(f"\n{'='*70}")
        print(f"  Configuration: {self.name}")
        print(f"{'='*70}\n")
        print(f"True Model:      {self.true_model}")
        print(f"Dividend Yield:  {self.q:.4f}")
        print(f"Dividend Events: {len(self.divs)}")
        print(f"Test Models:     {', '.join(self.test_models)}")
        print(f"European Phase:  N={self.european_N}, L={self.european_L}")
        print(f"American Phase:  N={self.american_N}, L={self.american_L}")
        print()
    
    def run_all(self):
        """Run all calibrations."""
        self.show_config()
        print("Running calibrations...\n")
        
        for i, model in enumerate(self.test_models, 1):
            print(f"[{i}/{len(self.test_models)}] {model}...", end=" ", flush=True)
            cmd = self.build_cmd(model)
            
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=Path(__file__).parent.parent,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                if result.returncode == 0:
                    print("[OK]")
                    self.results[model] = "success"
                else:
                    print("[FAIL]")
                    self.results[model] = "failed"
            except subprocess.TimeoutExpired:
                print("[TIMEOUT]")
                self.results[model] = "timeout"
            except Exception as e:
                print(f"[ERROR] {e}")
                self.results[model] = str(e)
        
        print(f"\nSummary:")
        for model, status in self.results.items():
            icon = "[OK]" if status == "success" else "[X]"
            print(f"  {icon} {model}")
        print()
    
    def print_commands(self):
        """Print all commands without running them."""
        self.show_config()
        print("Commands to run:\n")
        
        for model in self.test_models:
            cmd = self.build_cmd(model)
            print(f"# {model}")
            print(f"{cmd}\n")


# ============================================================================
# PRESET SCENARIOS
# ============================================================================

def scenario_basic():
    """Basic test: Merton+VG generating data."""
    return (CalibrationSetup("Basic")
            .set_true_model("merton_vg", q=0.05)
            .add_test_model("merton")
            .add_test_model("vg")
            .add_test_model("merton_vg"))


def scenario_jump_models():
    """Comparing jump models: Kou vs Merton."""
    return (CalibrationSetup("Jump Models")
            .set_true_model("kou", q=0.0)
            .add_test_model("merton")
            .add_test_model("kou")
            .add_test_model("merton_vg")
            .add_test_model("kou_vg"))


def scenario_infinite_activity():
    """CGMY (infinite activity)."""
    return (CalibrationSetup("Infinite Activity")
            .set_true_model("cgmy_vg", q=0.05)
            .add_test_model("vg")
            .add_test_model("cgmy")
            .add_test_model("cgmy_vg"))


def scenario_no_dividends():
    """No dividend case."""
    return (CalibrationSetup("No Dividends")
            .set_true_model("merton_vg", q=0.0)
            .set_divs([])
            .add_test_model("merton")
            .add_test_model("vg")
            .add_test_model("merton_vg"))


def scenario_high_div_yield():
    """High dividend yield case."""
    return (CalibrationSetup("High Dividends")
            .set_true_model("merton_vg", q=0.10)
            .set_divs([(t/10, 2.0) for t in range(1, 10)])
            .add_test_model("merton")
            .add_test_model("merton_vg")
            .add_test_model("kou_vg"))


def scenario_all_models():
    """Test all combinations."""
    setup = CalibrationSetup("All Models")
    setup.set_true_model("merton_vg", q=0.05)
    for model in ["merton", "kou", "vg", "cgmy", "merton_vg", "kou_vg", "cgmy_vg"]:
        setup.add_test_model(model)
    return setup


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 Calibration Explorer                          ║
    ║         Use these configurations in your scripts:             ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Available scenarios:
      - scenario_basic()
      - scenario_jump_models()
      - scenario_infinite_activity()
      - scenario_no_dividends()
      - scenario_high_div_yield()
      - scenario_all_models()
    
    Or create custom:
      setup = CalibrationSetup("My Test")
      setup.set_true_model("cgmy_vg", q=0.075)
      setup.add_test_model("vg")
      setup.add_test_model("kou")
      setup.run_all()
    
    Methods:
      .show_config()     -- Display setup
      .print_commands()  -- Show commands (no execution)
      .run_all()         -- Run all calibrations
    
    Examples:
    """)
    
    # Example 1: Show a scenario
    print("1. Show basic scenario:")
    print("-" * 70)
    scenario_basic().show_config()
    
    # Example 2: Print commands for review
    print("2. Print commands (no execution):")
    print("-" * 70)
    scenario_jump_models().print_commands()
