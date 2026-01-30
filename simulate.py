"""
INTEGRATED AUTOSCALING SIMULATION
===================================
Complete pipeline demonstrating autoscaling optimization:

OBJECTIVE FUNCTION → SCALING POLICIES → TEST SCENARIOS → METRICS → OUTPUT

Runs multiple autoscaling strategies across:
1. SYNTHETIC scenarios (simulated load patterns)
2. REAL data (processed historical data from processed_for_modeling_v2)

Compares effectiveness and outputs comprehensive results.

Usage:
    python simulate.py [--synthetic-only | --real-only]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

from data.load_data import load_csv
from forecast.model_forecaster import ModelForecaster
from autoscaling.reactive import ReactiveAutoscaler
from autoscaling.predictive import PredictiveAutoscaler
from autoscaling.cpu_based import CPUBasedAutoscaler
from autoscaling.hybrid import HybridAutoscaler
from autoscaling.scenarios import generate_all_scenarios, Scenario
from autoscaling.objective import compute_total_objective
from cost.metrics import MetricsCollector, compare_strategies
from anomaly.anomaly_detection import zscore_detection


def run_strategy_on_scenario(
    strategy_name,
    autoscaler,
    forecaster,
    load_series,
    forecast_horizon=1,
    capacity_per_pod=500,
    step_minutes=5.0,
    data_source="synthetic",
    scenario_name="",
):
    """
    Simulate single autoscaling strategy on a load series.

    PIPELINE:
    1. Load traffic data (synthetic or real)
    2. Apply forecaster for each timestep
    3. Run autoscaler step-by-step
    4. Track decisions and compute metrics
    5. Return results

    Args:
        strategy_name: name of strategy (REACTIVE/PREDICTIVE/CPU_BASED/HYBRID)
        autoscaler: autoscaler instance
        forecaster: forecaster instance (ModelForecaster with hybrid/lstm/xgboost)
        load_series: pd.DataFrame with columns [timestamp, requests_count, ...]
        forecast_horizon: how many steps ahead to forecast
        capacity_per_pod: requests/s per pod
        step_minutes: time interval between samples
        data_source: "synthetic" or "real"
        scenario_name: name of scenario

    Returns:
        dict with results and metrics
    """
    # Initialize metrics collection
    metrics = MetricsCollector(capacity_per_pod, cost_per_pod_per_hour=0.05, step_minutes=step_minutes)

    # Initialize pod count
    current_pods = 5

    records = []
    actions = []

    # Simulation loop
    for t in range(len(load_series)):
        actual_requests = float(load_series.iloc[t]["requests_count"])

        # Anomaly detection
        window_start = max(0, t - 10)
        window_data = load_series.iloc[window_start : t + 1]["requests_count"].values
        z_anomaly = (
            zscore_detection(window_data, threshold=3.0)[-1] if len(window_data) > 0 else 0
        )

        # === FORECASTING ===
        # Use ML models from demo/models/ for forecast
        try:
            # Create a history dataframe for the forecaster
            history = load_series.iloc[: t + 1].copy()
            forecast_result = forecaster.predict(history, horizon=forecast_horizon)
            actual_forecast = forecast_result.yhat[0] if forecast_result.yhat else actual_requests
        except Exception as e:
            print(f"  [Warning: Forecast failed at t={t}: {e}]")
            actual_forecast = actual_requests

        # === SLA CHECK (BEFORE SCALING) ===
        # Check if current pod count cannot handle demand
        sla_breached_before_scaling = actual_requests > current_pods * capacity_per_pod

        # === SCALING DECISION (strategy-dependent) ===
        if strategy_name == "REACTIVE":
            new_pods, utilization, action = autoscaler.step(current_pods, actual_requests)
            reason = f"utilization={utilization:.2%}"

        elif strategy_name == "PREDICTIVE":
            new_pods, action = autoscaler.step(
                current_servers=current_pods,
                forecast_requests=actual_forecast,
                current_requests=actual_requests,
            )
            reason = "predictive decision"

        elif strategy_name == "CPU_BASED":
            new_pods, cpu_util, action = autoscaler.step(current_pods, actual_requests)
            reason = f"cpu={cpu_util:.2%}"

        elif strategy_name == "HYBRID":
            new_pods, action, reason = autoscaler.step(
                current_servers=current_pods,
                requests=actual_requests,
                forecast_requests=actual_forecast,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Record metrics (with SLA violation status BEFORE scaling)
        metrics.record(t, new_pods, actual_requests, action, sla_before_scaling=sla_breached_before_scaling)
        actions.append(action)

        timestamp = load_series.iloc[t]["timestamp"] if "timestamp" in load_series.columns else t
        records.append(
            {
                "time": t,
                "timestamp": timestamp,
                "actual_requests": actual_requests,
                "forecast_requests": actual_forecast,
                "forecast_error": actual_forecast - actual_requests,
                "pods_before": current_pods,
                "pods_after": new_pods,
                "scaling_action": action,
                "reason": reason,
                "z_anomaly": z_anomaly,
                "sla_breached_before_scaling": sla_breached_before_scaling,
            }
        )

        current_pods = new_pods

    # Compute objective function
    pods_history = [r["pods_after"] for r in records]
    requests_history = [r["actual_requests"] for r in records]

    objective = compute_total_objective(
        pod_history=pods_history,
        requests_history=requests_history,
        action_history=actions,
        capacity_per_pod=capacity_per_pod,
    )

    # Get aggregate metrics
    agg_metrics = metrics.compute_aggregate_metrics()

    return {
        "strategy": strategy_name,
        "scenario": scenario_name,
        "data_source": data_source,
        "records": records,
        "metrics": agg_metrics,
        "objective": objective,
    }


# REMOVED: Real data loading
# Real data is used ONLY for model evaluation in PHASE A (forecast/model_evaluation.py)
# Autoscaling scenario testing (PHASE B) uses ONLY synthetic generated scenarios
# See REFACTORING_PLAN.md for architecture details


def run_all_simulations(
    scenarios=None,
    real_scenarios=None,  # Deprecated - kept for backward compatibility only
    strategies=None,
    capacity_per_pod=500,
    run_synthetic=True,
    run_real=False,  # Always False in PHASE B - real data not used for autoscaling tests
):
    """
    Run autoscaling scenario tests on synthetic data.

    PHASE B of two-phase pipeline:
    - PHASE A: Model evaluation on real data (forecast/model_evaluation.py)
    - PHASE B: Autoscaling tests on synthetic scenarios (this function)

    Args:
        scenarios: list of Scenario objects (synthetic)
        real_scenarios: DEPRECATED - not used, kept for compatibility
        strategies: list of strategy names (default: all)
        capacity_per_pod: requests/s per pod
        run_synthetic: whether to run synthetic scenarios (should always be True)
        run_real: whether to run real data scenarios (should always be False)

    Returns:
        list of result dicts with autoscaling test results
    """
    if scenarios is None:
        scenarios = generate_all_scenarios(duration=200)

    if real_scenarios is None:
        real_scenarios = []

    if strategies is None:
        strategies = ["REACTIVE", "PREDICTIVE", "CPU_BASED", "HYBRID"]

    results = []

    # ============================================================
    # AUTOSCALING SCENARIO TESTING (SYNTHETIC DATA ONLY)
    # Note: Real data evaluation is in PHASE A (forecast/model_evaluation.py)
    # ============================================================
    if run_synthetic:
        print(f"\n{'='*80}")
        print("SYNTHETIC SCENARIOS")
        print(f"{'='*80}")

        for scenario in scenarios:
            print(f"\n{'='*70}")
            print(f"Scenario: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"{'='*70}")

            # Convert synthetic scenario to DataFrame format
            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range(start="2024-01-01", periods=len(scenario.load_series), freq="5min"),
                    "requests_count": scenario.load_series,
                }
            )

            for strategy in strategies:
                print(f"\n  Testing: {strategy}...", end=" ", flush=True)

                # Initialize strategy
                autoscaler = _create_autoscaler(strategy, capacity_per_pod)

                # Use hybrid model forecaster for consistency
                forecaster = ModelForecaster(
                    model_type="hybrid",
                    timeframe="5m",
                    forecast_horizon=1,
                )

                # Run simulation
                result = run_strategy_on_scenario(
                    strategy_name=strategy,
                    autoscaler=autoscaler,
                    forecaster=forecaster,
                    load_series=df,
                    forecast_horizon=1,
                    capacity_per_pod=capacity_per_pod,
                    data_source="synthetic",
                    scenario_name=scenario.name,
                )

                results.append(result)

                # Print summary
                metrics = result["metrics"]
                objective = result["objective"]
                print(f"✓ Done")
                print(
                    f"    Cost: ${metrics['total_cost']:.2f} | SLA: {metrics['sla_violation_rate']:.1%} | Events: {metrics['scaling_events']}"
                )
                print(f"    Objective: {objective['total']:.2f}")

    # ============================================================
    # REAL DATA SCENARIOS - REMOVED
    # Real data evaluation moved to PHASE A (forecast/model_evaluation.py)
    # ============================================================
    # This section previously tested autoscaling on real data, which mixed
    # two separate concerns. Now real data is used ONLY for model evaluation.


def _create_autoscaler(strategy: str, capacity_per_pod: int):
    """Helper function to create autoscaler instance."""
    if strategy == "REACTIVE":
        return ReactiveAutoscaler(
            capacity_per_server=capacity_per_pod,
            min_servers=2,
            max_servers=20,
            scale_out_th=0.7,
            scale_in_th=0.3,
            cooldown=5,
        )

    elif strategy == "PREDICTIVE":
        return PredictiveAutoscaler(
            capacity_per_server=capacity_per_pod,
            safety_margin=0.8,
            min_servers=2,
            max_servers=20,
            hysteresis=1,
            base_cooldown=5,
        )

    elif strategy == "CPU_BASED":
        return CPUBasedAutoscaler(
            capacity_per_server=capacity_per_pod,
            scale_out_cpu_th=0.75,
            scale_in_cpu_th=0.25,
            cooldown=5,
        )

    elif strategy == "HYBRID":
        return HybridAutoscaler(
            capacity_per_server=capacity_per_pod,
            min_servers=2,
            max_servers=20,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def save_results(results, output_dir="results"):
    """
    Save simulation results to CSV and JSON.

    Args:
        results: list of result dicts from run_all_simulations
        output_dir: directory to save files
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Flatten records for CSV
    all_records = []
    for result in results:
        for record in result["records"]:
            record_copy = record.copy()
            record_copy["strategy"] = result["strategy"]
            record_copy["scenario"] = result["scenario"]
            record_copy["data_source"] = result["data_source"]
            all_records.append(record_copy)

    df = pd.DataFrame(all_records)
    csv_path = Path(output_dir) / "simulation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved records to {csv_path}")

    # Summary metrics
    summary = {}
    for result in results:
        key = f"{result['strategy']}_{result['scenario']}_{result['data_source']}"
        summary[key] = {**result["metrics"], **result["objective"]}

    json_path = Path(output_dir) / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved metrics to {json_path}")

    # Strategy comparison (aggregated across scenarios by data source)
    comparison_by_source = {}
    for data_source in ["synthetic", "real"]:
        source_results = [r for r in results if r["data_source"] == data_source]
        if not source_results:
            continue

        comparison = {}
        for strategy in ["REACTIVE", "PREDICTIVE", "CPU_BASED", "HYBRID"]:
            strategy_results = [r for r in source_results if r["strategy"] == strategy]
            if strategy_results:
                avg_metrics = {}
                for key in strategy_results[0]["metrics"].keys():
                    try:
                        values = [r["metrics"][key] for r in strategy_results]
                        avg_metrics[key] = np.mean(
                            [v for v in values if isinstance(v, (int, float))]
                        )
                    except:
                        pass

                comparison[strategy] = avg_metrics

        comparison_by_source[data_source] = comparison

    comparison_path = Path(output_dir) / "strategy_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison_by_source, f, indent=2)
    print(f"✓ Saved comparison to {comparison_path}")


def print_summary(results):
    """Print human-readable summary of results by data source."""
    print(f"\n{'='*80}")
    print("SIMULATION SUMMARY")
    print(f"{'='*80}")

    # Group by data_source and scenario
    for data_source in ["synthetic", "real"]:
        source_results = [r for r in results if r["data_source"] == data_source]
        if not source_results:
            continue

        print(f"\n{data_source.upper()} DATA RESULTS")
        print("-" * 80)

        scenarios = {}
        for result in source_results:
            scenario_name = result["scenario"]
            if scenario_name not in scenarios:
                scenarios[scenario_name] = {}
            scenarios[scenario_name][result["strategy"]] = result["metrics"]

        for scenario_name, strats in scenarios.items():
            print(f"\n{scenario_name}:")
            print("-" * 80)
            print(
                f"{'Strategy':<15} {'Cost':<12} {'Avg Pods':<12} {'SLA Viol':<12} {'Events':<10} {'Oscillations':<12}"
            )
            print("-" * 80)

            for strategy_name, metrics in strats.items():
                print(
                    f"{strategy_name:<15} ${metrics['total_cost']:<11.2f} {metrics['average_pods']:<11.1f} "
                    f"{metrics['sla_violation_rate']:<11.1%} {metrics['scaling_events']:<9} {metrics['oscillation_count']:<11}"
                )


if __name__ == "__main__":
    print("=" * 80)
    print("AUTOSCALING SCENARIO TESTING (PHASE B)")
    print("=" * 80)
    print("\nInitializing...")

    # Generate all synthetic test scenarios
    scenarios = generate_all_scenarios(duration=200)
    print(f"✓ Generated {len(scenarios)} synthetic test scenarios")

    # Run all strategies on synthetic scenarios only
    # (Real data is evaluated separately in PHASE A: forecast/model_evaluation.py)
    results = run_all_simulations(
        scenarios=scenarios,
        real_scenarios=None,  # No real data in autoscaling tests
        run_synthetic=True,
        run_real=False,
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    print(f"\n{'='*80}")
    print("✓ Scenario testing complete! Check 'results/' directory for outputs.")
    print(f"\nNOTE: This is PHASE B (autoscaling scenario testing with synthetic data)")
    print(f"      For model evaluation on real data, run: python -m forecast.model_evaluation")
    print(f"{'='*80}")


