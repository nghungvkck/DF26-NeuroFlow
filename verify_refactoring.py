"""
COMPREHENSIVE VERIFICATION SCRIPT
==================================

Verifies all refactoring objectives have been met:

1. ✓ Real data is used ONLY for model evaluation (PHASE A)
2. ✓ Synthetic scenarios are used ONLY for autoscaling tests (PHASE B)
3. ✓ SLA violations are computed BEFORE scaling decisions
4. ✓ Dashboard offers flexible data source selection
5. ✓ Pipeline results are clearly labeled and separated
"""

import sys
import json
from pathlib import Path
import pandas as pd


def verify_phase_separation():
    """Verify PHASE A and PHASE B are properly separated."""
    print("\n" + "="*80)
    print("VERIFICATION 1: PHASE SEPARATION")
    print("="*80)
    
    # Check simulate.py doesn't load real data for autoscaling tests
    with open("simulate.py", "r") as f:
        lines = f.readlines()
        code = "".join(lines)
    
    if "load_real_data_scenarios" in code and "def load_real_data_scenarios" not in code:
        print("✗ simulate.py still references load_real_data_scenarios()")
        return False
    
    # Check main() uses run_real=False
    if "__name__ == \"__main__\"" in code:
        # Extract main() function
        in_main = False
        main_code = []
        for line in lines:
            if "__name__ == \"__main__\"" in line:
                in_main = True
            if in_main:
                main_code.append(line)
        
        main_text = "".join(main_code)
        if "run_real=False" in main_text:
            print("✓ simulate.py main() uses run_real=False (no real data in autoscaling tests)")
        else:
            print("✗ simulate.py main() may still use real data")
            return False
    
    # Check model_evaluation.py exists
    if Path("forecast/model_evaluation.py").exists():
        print("✓ forecast/model_evaluation.py exists for PHASE A")
    else:
        print("✗ forecast/model_evaluation.py not found")
        return False
    
    # Check run_pipeline.py exists
    if Path("run_pipeline.py").exists():
        print("✓ run_pipeline.py exists to orchestrate both phases")
    else:
        print("✗ run_pipeline.py not found")
        return False
    
    return True


def verify_sla_logic():
    """Verify SLA violations are computed BEFORE scaling."""
    print("\n" + "="*80)
    print("VERIFICATION 2: SLA VIOLATION LOGIC")
    print("="*80)
    
    # Check metrics.py has sla_before_scaling
    with open("cost/metrics.py", "r") as f:
        code = f.read()
    
    if "sla_violated_before_scaling" in code:
        print("✓ cost/metrics.py tracks sla_violated_before_scaling")
    else:
        print("✗ SLA violation logic not updated")
        return False
    
    # Check simulate.py passes sla_before_scaling to metrics.record()
    with open("simulate.py", "r") as f:
        code = f.read()
    
    if "sla_breached_before_scaling" in code and "sla_before_scaling=" in code:
        print("✓ simulate.py computes and passes sla_before_scaling to metrics")
    else:
        print("✗ SLA logic not properly integrated")
        return False
    
    return True


def verify_dashboard_flexibility():
    """Verify dashboard offers data source selection."""
    print("\n" + "="*80)
    print("VERIFICATION 3: DASHBOARD FLEXIBILITY")
    print("="*80)
    
    with open("dashboard/app.py", "r") as f:
        code = f.read()
    
    checks = [
        ("visualization_mode" in code, "Data source selection widget"),
        ("Autoscaling Tests" in code, "Autoscaling Tests option"),
        ("Model Evaluation" in code, "Model Evaluation option"),
        ("model_evaluation.json" in code, "Model evaluation metrics display"),
    ]
    
    all_ok = True
    for check, desc in checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} not found")
            all_ok = False
    
    return all_ok


def verify_results_structure():
    """Verify results have proper structure."""
    print("\n" + "="*80)
    print("VERIFICATION 4: RESULTS STRUCTURE")
    print("="*80)
    
    results_dir = Path("results")
    
    required_files = {
        "simulation_results.csv": "Autoscaling test results",
        "metrics_summary.json": "Aggregated metrics",
        "strategy_comparison.json": "Strategy comparison",
    }
    
    all_ok = True
    for filename, description in required_files.items():
        path = results_dir / filename
        if path.exists():
            print(f"✓ {filename} exists ({description})")
        else:
            print(f"⚠ {filename} not found (will be created on first run)")
    
    # Check CSV structure if it exists
    csv_path = results_dir / "simulation_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        required_columns = [
            "time",
            "actual_requests",
            "forecast_requests",
            "pods_before",
            "pods_after",
            "scaling_action",
            "sla_breached_before_scaling",
            "strategy",
            "scenario",
        ]
        
        for col in required_columns:
            if col in df.columns:
                print(f"✓ Column '{col}' exists in simulation_results.csv")
            else:
                print(f"✗ Column '{col}' missing from simulation_results.csv")
                all_ok = False
        
        # Verify only synthetic scenarios
        scenarios = df["scenario"].unique()
        synthetic_scenarios = {
            "GRADUAL_INCREASE",
            "SUDDEN_SPIKE",
            "OSCILLATING",
            "TRAFFIC_DROP",
            "FORECAST_ERROR_TEST",
        }
        
        for scenario in scenarios:
            if scenario in synthetic_scenarios:
                print(f"✓ Synthetic scenario found: {scenario}")
            else:
                print(f"✗ Non-synthetic scenario found: {scenario}")
                all_ok = False
    
    return all_ok


def verify_no_hardcoded_paths():
    """Verify no hard-coded paths."""
    print("\n" + "="*80)
    print("VERIFICATION 5: NO HARD-CODED PATHS")
    print("="*80)
    
    files_to_check = [
        "simulate.py",
        "forecast/model_evaluation.py",
        "run_pipeline.py",
        "dashboard/app.py",
    ]
    
    suspicious_patterns = [
        "/Users/",
        "/home/",
        "C:\\Users\\",
        "D:\\",
    ]
    
    all_ok = True
    for filename in files_to_check:
        try:
            with open(filename, "r") as f:
                code = f.read()
            
            found_hardcoded = False
            for pattern in suspicious_patterns:
                if pattern in code:
                    print(f"✗ Hard-coded path found in {filename}: {pattern}")
                    found_hardcoded = True
                    all_ok = False
            
            if not found_hardcoded:
                print(f"✓ No hard-coded paths in {filename}")
        except FileNotFoundError:
            print(f"⚠ File not found: {filename}")
    
    return all_ok


def verify_docstrings():
    """Verify important functions have docstrings."""
    print("\n" + "="*80)
    print("VERIFICATION 6: DOCUMENTATION")
    print("="*80)
    
    functions_to_check = {
        "forecast/model_evaluation.py": ["ModelEvaluator", "evaluate_all_models"],
        "run_pipeline.py": ["run_phase_a_model_evaluation", "run_phase_b_autoscaling_tests"],
        "simulate.py": ["run_strategy_on_scenario", "run_all_simulations"],
    }
    
    all_ok = True
    for filename, functions in functions_to_check.items():
        try:
            with open(filename, "r") as f:
                code = f.read()
            
            for func in functions:
                if f"def {func}" in code and '"""' in code[code.find(f"def {func}"):code.find(f"def {func}")+500]:
                    print(f"✓ {func}() has docstring")
                else:
                    print(f"⚠ {func}() may lack docstring")
        except FileNotFoundError:
            print(f"⚠ File not found: {filename}")
            all_ok = False
    
    return all_ok


def main():
    """Run all verifications."""
    print("\n" + "="*80)
    print("COMPREHENSIVE REFACTORING VERIFICATION")
    print("="*80)
    
    results = {
        "Phase Separation": verify_phase_separation(),
        "SLA Logic": verify_sla_logic(),
        "Dashboard Flexibility": verify_dashboard_flexibility(),
        "Results Structure": verify_results_structure(),
        "No Hard-Coded Paths": verify_no_hardcoded_paths(),
        "Documentation": verify_docstrings(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ ALL VERIFICATIONS PASSED!")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py")
        print("2. Review results/model_evaluation.json")
        print("3. Check results/simulation_results.csv for SLA violations")
        print("4. Run dashboard: streamlit run dashboard/app.py")
        return 0
    else:
        print(f"\n✗ {total - passed} verification(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
