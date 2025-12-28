#!/usr/bin/env python3
"""
COPY-PASTE READY: Minimal example to test all this works.

Just run: python tools/example_basic_calibration.py
"""

from calibration_explorer import (
    CalibrationSetup,
    scenario_basic,
    scenario_jump_models,
    scenario_all_models,
)


def main():
    print("\n" + "="*70)
    print("  CALIBRATION SUITE - MINIMAL EXAMPLE")
    print("="*70 + "\n")
    
    # ====================================================================
    # EXAMPLE 1: Show configuration (no execution)
    # ====================================================================
    print("\n[EXAMPLE 1] Display configuration for basic scenario\n")
    print("-" * 70)
    scenario_basic().print_commands()
    
    # ====================================================================
    # EXAMPLE 2: Create and run a custom scenario
    # ====================================================================
    print("\n[EXAMPLE 2] Custom scenario: Kou model")
    print("-" * 70)
    
    custom = CalibrationSetup("Kou Test")
    custom.set_true_model("kou", q=0.03)
    custom.add_test_model("merton")
    custom.add_test_model("kou")
    custom.add_test_model("kou_vg")
    custom.set_divs([(0.5, 2.0)])
    
    custom.show_config()
    print("\nTo run this scenario, uncomment below or call:")
    print("  custom.run_all()\n")
    # custom.run_all()
    
    # ====================================================================
    # EXAMPLE 3: Batch testing
    # ====================================================================
    print("\n[EXAMPLE 3] Batch scenarios (print commands only)")
    print("-" * 70)
    
    scenarios = {
        "Basic": scenario_basic(),
        "Jump Models": scenario_jump_models(),
        "All Models": scenario_all_models(),
    }
    
    for name, scenario in scenarios.items():
        print(f"\n{name}:")
        print(f"  True: {scenario.true_model}")
        print(f"  Test: {', '.join(scenario.test_models)}")
        print(f"  Divs: {len(scenario.divs)} events")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("  TO RUN FULL CALIBRATION")
    print("="*70)
    print("""
    Option A: Auto-run (calls all calibrations)
    -----------
    from tools.calibration_explorer import scenario_basic
    scenario_basic().run_all()
    
    Option B: Print commands first (review, then copy-paste)
    -----------
    from tools.calibration_explorer import scenario_jump_models
    scenario_jump_models().print_commands()
    
    Option C: Use the main scripts
    -----------
    python tools/run_calibration_auto.py        # Auto-run all
    python tools/run_calibration.py             # Show config only
    
    Option D: Customize with your own parameters
    -----------
    Edit tools/run_calibration_auto.py:
    - TRUE_MODEL_CONFIG
    - DIVIDEND_SCHEDULE
    - TEST_MODELS
    - Calibration parameters
    
    Then run: python tools/run_calibration_auto.py
    """)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
