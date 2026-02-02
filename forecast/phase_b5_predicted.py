"""
PHASE B.5: AUTOSCALING ON PREDICTED DATA
=========================================

Test autoscaling strategies on pre-calculated predicted load values.

This phase uses LightGBM predictions from data/prediction/ to simulate:
1. What happens when forecasts are AVAILABLE before the data arrives
2. Compare performance: actual data vs predicted data
3. Understand impact of forecast quality on autoscaling

Useful for:
- Offline capacity planning with known forecasts
- Comparing ideal (perfect prediction) vs real (noisy prediction)
- Understanding forecast accuracy → autoscaling performance

Input:
    - data/prediction/lightgbm_1m_predictions.csv
    - data/prediction/lightgbm_5m_predictions.csv
    - data/prediction/lightgbm_15m_predictions.csv

Output:
    - results/phase_b5_predicted_results.csv
    - results/phase_b5_metrics_summary.json
    - results/phase_b5_vs_actual_comparison.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_csv
from forecast.model_forecaster import ModelForecaster
from autoscaling.reactive import ReactiveAutoscaler
from autoscaling.predictive import PredictiveAutoscaler
from autoscaling.cpu_based import CPUBasedAutoscaler
from autoscaling.hybrid import HybridAutoscaler
from autoscaling.objective import compute_total_objective
from cost.metrics import MetricsCollector, compare_strategies


def load_predicted_data(timeframe: str = "1m", data_dir: str = "data/prediction") -> pd.DataFrame:
    """
    Load predicted data from LightGBM predictions.
    
    Args:
        timeframe: "1m", "5m", or "15m"
        data_dir: directory containing prediction files
    
    Returns:
        DataFrame with columns: [actual, predicted, error, abs_error, error_pct]
    """
    file_path = Path(data_dir) / f"lightgbm_{timeframe}_predictions.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Add index as timestamp placeholder (for compatibility)
    df['timestamp'] = np.arange(len(df))
    df = df[['timestamp', 'actual', 'predicted', 'error', 'abs_error', 'error_pct']]
    
    return df


def run_strategy_on_predicted_load(
    strategy_name: str,
    autoscaler,
    predicted_data: pd.DataFrame,
    capacity_per_pod: float = 100.0,
    step_minutes: float = 5.0,
    timeframe: str = "1m",
    enable_advanced_metrics: bool = True,
) -> Dict:
    """
    Simulate autoscaling strategy on predicted load data.
    
    Uses the 'predicted' column as the load input (perfect forecast case),
    and 'actual' column for comparison.
    
    Args:
        strategy_name: "REACTIVE", "PREDICTIVE", "CPU_BASED", "HYBRID"
        autoscaler: autoscaler instance
        predicted_data: DataFrame with 'predicted' and 'actual' columns
        capacity_per_pod: capacity per pod
        step_minutes: time interval
        timeframe: for logging
        enable_advanced_metrics: enable K8s/AWS metrics
    
    Returns:
        Dict with results and metrics
    """
    metrics = MetricsCollector(
        capacity_per_pod,
        cost_per_pod_per_hour=0.05,
        step_minutes=step_minutes,
        enable_k8s_metrics=enable_advanced_metrics,
        enable_aws_metrics=enable_advanced_metrics,
        enable_borg_metrics=False
    )
    
    # Initialize pod count
    current_pods = 5
    records = []
    actions = []
    
    # Simulation loop using PREDICTED data as load
    for t in range(len(predicted_data)):
        predicted_load = float(predicted_data.iloc[t]["predicted"])
        actual_load = float(predicted_data.iloc[t]["actual"])
        
        # SLA check BEFORE scaling (using predicted data)
        sla_breached_before = predicted_load > current_pods * capacity_per_pod
        
        # === SCALING DECISION ===
        # For PREDICTED phase, we assume forecaster always returns the next predicted value
        # (as if we have perfect forecast)
        forecast_for_next = predicted_load
        
        if strategy_name == "REACTIVE":
            new_pods, utilization, action = autoscaler.step(current_pods, predicted_load)
            reason = f"utilization={utilization:.2%}"
        
        elif strategy_name == "PREDICTIVE":
            new_pods, action = autoscaler.step(
                current_servers=current_pods,
                forecast_requests=forecast_for_next,
                current_requests=predicted_load,
            )
            reason = "predictive on predicted data"
        
        elif strategy_name == "CPU_BASED":
            new_pods, cpu_util, action = autoscaler.step(current_pods, predicted_load)
            reason = f"cpu={cpu_util:.2%}"
        
        elif strategy_name == "HYBRID":
            new_pods, action, reason = autoscaler.step(
                current_servers=current_pods,
                requests=predicted_load,
                forecast_requests=forecast_for_next,
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Calculate CPU utilization
        cpu_utilization = predicted_load / (new_pods * capacity_per_pod) if new_pods > 0 else 0.0
        
        # Record metrics
        metrics.record(
            t, new_pods, predicted_load, action,
            sla_before_scaling=sla_breached_before,
            cpu_utilization=cpu_utilization
        )
        actions.append(action)
        
        records.append({
            'time': t,
            'timestamp': t,
            'predicted_load': predicted_load,
            'actual_load': actual_load,
            'forecast_error': predicted_data.iloc[t]['error'],
            'pods_before': current_pods,
            'pods_after': new_pods,
            'scaling_action': action,
            'reason': reason,
            'cpu_utilization': cpu_utilization,
            'sla_breached_before': sla_breached_before,
        })
        
        current_pods = new_pods
    
    # Compute objective function
    pods_history = [r['pods_after'] for r in records]
    requests_history = [r['predicted_load'] for r in records]
    
    objective = compute_total_objective(
        pod_history=pods_history,
        requests_history=requests_history,
        action_history=actions,
        capacity_per_pod=capacity_per_pod,
    )
    
    # Get aggregate metrics
    agg_metrics = metrics.compute_aggregate_metrics()
    
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'records': records,
        'metrics': agg_metrics,
        'objective': objective,
        'total_cost': objective['cost_component'],
        'sla_violations': objective['sla_violations'],
        'scaling_events': objective['scaling_events'],
    }


def run_phase_b5_predicted_autoscaling(
    output_dir: str = "results",
    timeframe: str = "1m",
) -> Dict:
    """
    PHASE B.5: Run autoscaling tests on predicted load data.
    
    Tests all 4 strategies on predicted load to understand:
    1. How well autoscaler works with pre-calculated forecasts
    2. Impact of forecast accuracy on autoscaling performance
    3. Offline capacity planning capability
    
    Args:
        output_dir: where to save results
        timeframe: "1m", "5m", or "15m"
    
    Returns:
        Dict with results
    """
    print("\n" + "=" * 80)
    print(f"PHASE B.5: AUTOSCALING ON PREDICTED DATA ({timeframe})")
    print("=" * 80)
    
    # Load predicted data
    try:
        predicted_data = load_predicted_data(timeframe)
        print(f"[OK] Loaded {len(predicted_data)} predictions for {timeframe}")
        print(f"  - Actual load range: {predicted_data['actual'].min():.0f} - {predicted_data['actual'].max():.0f}")
        print(f"  - Predicted load range: {predicted_data['predicted'].min():.0f} - {predicted_data['predicted'].max():.0f}")
        print(f"  - Mean prediction error: {predicted_data['error_pct'].mean():.2f}%")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return {}
    
    # Create autoscalers
    capacity_per_pod = 100
    autoscalers = {
        'REACTIVE': ReactiveAutoscaler(capacity_per_pod),
        'PREDICTIVE': PredictiveAutoscaler(capacity_per_pod),
        'CPU_BASED': CPUBasedAutoscaler(capacity_per_pod),
        'HYBRID': HybridAutoscaler(capacity_per_pod),
    }
    
    # Run all strategies
    print(f"\n[1/2] Running autoscaling strategies...")
    results = {}
    all_records = []
    metrics_summary = {}
    
    for strategy_name, autoscaler in autoscalers.items():
        print(f"  - {strategy_name}...", end=" ", flush=True)
        
        result = run_strategy_on_predicted_load(
            strategy_name=strategy_name,
            autoscaler=autoscaler,
            predicted_data=predicted_data,
            capacity_per_pod=capacity_per_pod,
            timeframe=timeframe,
        )
        
        results[strategy_name] = result
        metrics_summary[strategy_name] = {
            'total_cost': result['total_cost'],
            'avg_pods': result['metrics'].get('avg_pods', 0),
            'sla_violations': result['sla_violations'],
            'scaling_events': result['scaling_events'],
            'objective_value': result['objective']['total'],
        }
        
        # Add strategy and scenario to each record
        for record in result['records']:
            record['scenario'] = f"PREDICTED_{timeframe.upper()}"
            record['strategy'] = strategy_name
            all_records.append(record)
        
        print(f"Cost=${result['total_cost']:.2f}, Events={result['scaling_events']} [OK]")
    
    # Save results
    print(f"\n[2/2] Saving results...")
    
    # Save detailed records
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    records_df = pd.DataFrame(all_records)
    records_file = output_path / f"phase_b5_predicted_results_{timeframe}.csv"
    records_df.to_csv(records_file, index=False)
    print(f"  ✓ Saved {len(records_df)} records to {records_file.name}")
    
    # Save metrics summary
    metrics_file = output_path / f"phase_b5_metrics_summary_{timeframe}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  ✓ Saved metrics to {metrics_file.name}")
    
    # Save comparison analysis
    comparison = {
        'timeframe': timeframe,
        'data_source': 'predicted',
        'num_predictions': len(predicted_data),
        'prediction_quality': {
            'mean_error': float(predicted_data['error'].mean()),
            'mean_abs_error': float(predicted_data['abs_error'].mean()),
            'mean_error_pct': float(predicted_data['error_pct'].mean()),
            'std_error': float(predicted_data['error'].std()),
        },
        'strategy_performance': metrics_summary,
        'strategy_ranking': sorted(
            metrics_summary.items(),
            key=lambda x: x[1]['total_cost']
        ),
    }
    
    comparison_file = output_path / f"phase_b5_analysis_{timeframe}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"  ✓ Saved analysis to {comparison_file.name}")
    
    return {
        'phase': 'B.5',
        'timeframe': timeframe,
        'results': results,
        'metrics_summary': metrics_summary,
        'comparison': comparison,
    }


def run_all_timeframes(output_dir: str = "results") -> Dict:
    """
    Run PHASE B.5 for all available timeframes (1m, 5m, 15m).
    
    Args:
        output_dir: output directory
    
    Returns:
        Dict with all results
    """
    print("\n" + "=" * 80)
    print("PHASE B.5: AUTOSCALING ON PREDICTED DATA (ALL TIMEFRAMES)")
    print("=" * 80)
    
    all_results = {}
    timeframes = ['1m', '5m', '15m']
    
    for timeframe in timeframes:
        try:
            result = run_phase_b5_predicted_autoscaling(output_dir, timeframe)
            all_results[timeframe] = result
        except Exception as e:
            print(f"✗ Failed for {timeframe}: {e}")
            continue
    
    # Create summary across all timeframes
    if all_results:
        print(f"\n[Summary] Completed {len(all_results)} timeframes")
        
        # Aggregate metrics
        aggregate_metrics = {}
        for timeframe, result in all_results.items():
            for strategy, metrics in result['metrics_summary'].items():
                if strategy not in aggregate_metrics:
                    aggregate_metrics[strategy] = []
                aggregate_metrics[strategy].append(metrics)
        
        # Compute cross-timeframe average
        cross_timeframe_avg = {}
        for strategy, metric_list in aggregate_metrics.items():
            cross_timeframe_avg[strategy] = {
                'avg_cost': np.mean([m['total_cost'] for m in metric_list]),
                'avg_sla_violations': np.mean([m['sla_violations'] for m in metric_list]),
                'avg_scaling_events': np.mean([m['scaling_events'] for m in metric_list]),
            }
        
        # Save summary
        summary_file = Path(output_dir) / "phase_b5_cross_timeframe_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(cross_timeframe_avg, f, indent=2)
        
        print(f"✓ Saved cross-timeframe summary to phase_b5_cross_timeframe_summary.json")
        
        # Print ranking
        print("\n[Ranking] Best strategies across all timeframes:")
        ranking = sorted(
            cross_timeframe_avg.items(),
            key=lambda x: x[1]['avg_cost']
        )
        for i, (strategy, metrics) in enumerate(ranking, 1):
            print(f"  {i}. {strategy:12s} - Cost: ${metrics['avg_cost']:.2f}, "
                  f"SLA: {metrics['avg_sla_violations']:.0f}, "
                  f"Events: {metrics['avg_scaling_events']:.0f}")
    
    return all_results


if __name__ == "__main__":
    # Run for all timeframes
    results = run_all_timeframes()
    
    print("\n" + "=" * 80)
    print("PHASE B.5 COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - phase_b5_predicted_results_*.csv")
    print("  - phase_b5_metrics_summary_*.json")
    print("  - phase_b5_analysis_*.json")
    print("  - phase_b5_cross_timeframe_summary.json")
