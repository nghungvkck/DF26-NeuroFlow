#!/usr/bin/env python3
"""
AUTOSCALING PIPELINE ORCHESTRATOR
==================================

Master script that runs both phases of the pipeline:

PHASE A — MODEL EVALUATION (Real Data)
  - Evaluate LSTM, XGBoost, Hybrid models on real historical data
  - Compute MAE, RMSE, MAPE for each model per timeframe
  - Determine best model per timeframe
  - Output: results/model_evaluation.json

PHASE B — AUTOSCALING SCENARIO TESTING (Synthetic Data)
  - Generate synthetic load scenarios
  - Test all autoscaling strategies on synthetic loads
  - Measure cost, SLA violations, stability
  - Output: results/simulation_results.csv

Usage:
    python run_pipeline.py [--phase-a-only | --phase-b-only]

Examples:
    python run_pipeline.py           # Run both phases
    python run_pipeline.py --phase-a-only   # Model evaluation only
    python run_pipeline.py --phase-b-only   # Autoscaling tests only
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Import both phases
from forecast.model_evaluation import ModelEvaluator
import simulate


def run_phase_a_model_evaluation(output_dir="results"):
    """
    PHASE A: Evaluate models on real data.
    
    - Load real data from processed_for_modeling_v2/
    - Evaluate LSTM, XGBoost, Hybrid models
    - Compute forecast accuracy metrics (MAE, RMSE, MAPE)
    - Identify best model per timeframe
    - Save results to model_evaluation.json
    """
    print("\n" + "=" * 80)
    print("PHASE A: MODEL EVALUATION ON REAL DATA")
    print("=" * 80)
    
    evaluator = ModelEvaluator(real_data_dir="processed_for_modeling_v2")
    results = evaluator.run_full_evaluation(output_dir=output_dir)
    
    if not results.get("evaluation_results"):
        print("⚠️  No models evaluated in Phase A")
        return {}
    
    return results


def run_phase_b_autoscaling_tests(output_dir="results"):
    """
    PHASE B: Test autoscaling on synthetic scenarios.
    
    - Generate synthetic load scenarios (5 types)
    - Test all strategies (REACTIVE, PREDICTIVE, CPU_BASED, HYBRID)
    - Measure metrics: cost, SLA violations, stability
    - Save results to simulation_results.csv and metrics_summary.json
    """
    print("\n" + "=" * 80)
    print("PHASE B: AUTOSCALING SCENARIO TESTING")
    print("=" * 80)
    
    # This will run simulate.py's main logic
    # Note: we need to temporarily redirect the output
    from autoscaling.scenarios import generate_all_scenarios
    
    scenarios = generate_all_scenarios(duration=200)
    print(f"✓ Generated {len(scenarios)} synthetic test scenarios")
    
    results = simulate.run_all_simulations(
        scenarios=scenarios,
        real_scenarios=None,
        run_synthetic=True,
        run_real=False,
    )
    
    simulate.print_summary(results)
    simulate.save_results(results, output_dir=output_dir)
    
    return results


def create_pipeline_summary(phase_a_results, phase_b_results, output_dir="results"):
    """
    Create a summary document comparing both phases.
    """
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "phase_a": {
            "name": "Model Evaluation (Real Data)",
            "description": "Forecast model performance on historical data",
            "output_file": "model_evaluation.json",
            "status": "completed" if phase_a_results else "skipped",
        },
        "phase_b": {
            "name": "Autoscaling Tests (Synthetic Data)",
            "description": "Autoscaling strategy performance on synthetic scenarios",
            "output_file": "simulation_results.csv",
            "status": "completed" if phase_b_results else "skipped",
        },
    }
    
    if phase_a_results and "best_models" in phase_a_results:
        summary["phase_a"]["best_models"] = phase_a_results["best_models"]
    
    if phase_b_results:
        # Count scenarios and strategies tested
        scenarios = set()
        strategies = set()
        for result in phase_b_results:
            scenarios.add(result.get("scenario"))
            strategies.add(result.get("strategy"))
        
        summary["phase_b"]["scenarios_tested"] = len(scenarios)
        summary["phase_b"]["strategies_tested"] = len(strategies)
        summary["phase_b"]["total_tests"] = len(phase_b_results)
    
    # Save summary
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Pipeline summary saved to {summary_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PHASE A: {summary['phase_a']['status'].upper()}")
    if "best_models" in summary["phase_a"]:
        print(f"  Best models per timeframe:")
        for timeframe, model in sorted(summary["phase_a"]["best_models"].items()):
            print(f"    {timeframe}: {model.upper()}")
    
    print(f"\nPHASE B: {summary['phase_b']['status'].upper()}")
    if "scenarios_tested" in summary["phase_b"]:
        print(f"  Scenarios tested: {summary['phase_b']['scenarios_tested']}")
        print(f"  Strategies tested: {summary['phase_b']['strategies_tested']}")
        print(f"  Total test runs: {summary['phase_b']['total_tests']}")
    
    print(f"{'='*80}\n")


def main():
    """Run the complete pipeline."""
    # Parse arguments
    run_phase_a = True
    run_phase_b = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--phase-a-only":
            run_phase_b = False
        elif sys.argv[1] == "--phase-b-only":
            run_phase_a = False
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python run_pipeline.py [--phase-a-only | --phase-b-only]")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("AUTOSCALING PIPELINE ORCHESTRATOR")
    print("=" * 80)
    print(f"{'='*80}\n")
    
    phase_a_results = {}
    phase_b_results = {}
    
    # Run Phase A
    if run_phase_a:
        try:
            phase_a_results = run_phase_a_model_evaluation()
        except Exception as e:
            print(f"\n❌ Phase A failed: {e}")
            if not run_phase_b:
                sys.exit(1)
    
    # Run Phase B
    if run_phase_b:
        try:
            phase_b_results = run_phase_b_autoscaling_tests()
        except Exception as e:
            print(f"\n❌ Phase B failed: {e}")
            if not run_phase_a:
                sys.exit(1)
    
    # Create summary
    create_pipeline_summary(phase_a_results, phase_b_results)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  Phase A: results/model_evaluation.json")
    print("  Phase B: results/simulation_results.csv")
    print("  Summary: results/pipeline_summary.json")
    print("\nNext steps:")
    print("  1. Review model_evaluation.json for best models")
    print("  2. Check simulation_results.csv for autoscaling test results")
    print("  3. Run dashboard: streamlit run dashboard/app.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
