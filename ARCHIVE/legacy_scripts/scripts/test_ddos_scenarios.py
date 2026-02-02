"""
TEST AUTOSCALING STRATEGIES WITH DDoS/SPIKE SCENARIOS
======================================================
Compare 4 strategies under various attack scenarios.

Tests:
1. REACTIVE - Threshold-based scaling
2. PREDICTIVE - Forecast-based scaling  
3. CPU_BASED - CPU utilization scaling
4. HYBRID (with anomaly detection) - Multi-layer with spike detection

Scenarios:
1. NORMAL - Baseline performance
2. SUDDEN_SPIKE - Instant 5x traffic (DDoS)
3. GRADUAL_SPIKE - Slow ramp (flash sale)
4. OSCILLATING_SPIKE - Multiple waves
5. SUSTAINED_DDOS - Long-duration attack
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoscaling.reactive import ReactiveAutoscaler
from autoscaling.predictive import PredictiveAutoscaler
from autoscaling.cpu_based import CPUBasedAutoscaler
from autoscaling.hybrid import HybridAutoscaler
from autoscaling.objective import compute_total_objective
from cost.metrics import MetricsCollector


def run_strategy_on_scenario(
    strategy_name: str,
    autoscaler,
    traffic_data: pd.DataFrame,
    capacity_per_pod: float = 100.0,
    enable_anomaly: bool = False
) -> dict:
    """
    Run single strategy on traffic scenario.
    
    Returns:
        dict with metrics and timeline
    """
    print(f"    Running {strategy_name}...", end=" ", flush=True)
    
    metrics = MetricsCollector(
        capacity_per_pod,
        cost_per_pod_per_hour=0.05,
        step_minutes=1.0,
        enable_k8s_metrics=True,
        enable_aws_metrics=True
    )
    
    current_pods = 5
    records = []
    actions = []
    
    # Simple forecast (moving average)
    forecast_window = 10
    
    for t in range(len(traffic_data)):
        actual_load = float(traffic_data.iloc[t]['requests_count'])
        
        # Compute forecast
        if t >= forecast_window:
            forecast = traffic_data.iloc[t-forecast_window:t]['requests_count'].mean()
        else:
            forecast = actual_load
        
        # SLA check before scaling
        sla_before = actual_load > current_pods * capacity_per_pod
        
        # Scaling decision
        if strategy_name == "REACTIVE":
            new_pods, utilization, action = autoscaler.step(current_pods, actual_load)
        
        elif strategy_name == "PREDICTIVE":
            new_pods, action = autoscaler.step(current_pods, forecast, actual_load)
        
        elif strategy_name == "CPU_BASED":
            new_pods, cpu_util, action = autoscaler.step(current_pods, actual_load)
        
        elif strategy_name == "HYBRID":
            new_pods, action, reason = autoscaler.step(current_pods, actual_load, forecast)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # CPU utilization
        cpu_util = actual_load / (new_pods * capacity_per_pod) if new_pods > 0 else 0
        
        # Record metrics
        metrics.record(
            t, new_pods, actual_load, action,
            sla_before_scaling=sla_before,
            cpu_utilization=cpu_util
        )
        
        actions.append(action)
        records.append({
            'time': t,
            'actual_load': actual_load,
            'forecast': forecast,
            'pods_before': current_pods,
            'pods_after': new_pods,
            'action': action,
            'cpu_utilization': cpu_util,
            'sla_breached': sla_before
        })
        
        current_pods = new_pods
    
    # Compute objective
    pods_history = [r['pods_after'] for r in records]
    requests_history = [r['actual_load'] for r in records]
    
    objective = compute_total_objective(
        pod_history=pods_history,
        requests_history=requests_history,
        action_history=actions,
        capacity_per_pod=capacity_per_pod
    )
    
    agg_metrics = metrics.compute_aggregate_metrics()
    
    # Calculate response time to spikes
    spike_response_time = calculate_spike_response(records)
    
    print(f"Cost=${objective['cost_component']:.2f}, SLA={objective['sla_violations']}, " +
          f"Spike Response={spike_response_time:.1f}min")
    
    return {
        'strategy': strategy_name,
        'records': records,
        'metrics': agg_metrics,
        'objective': objective,
        'spike_response_time': spike_response_time,
        'total_cost': objective['cost_component'],
        'sla_violations': objective['sla_violations'],
        'scaling_events': objective['scaling_events']
    }


def calculate_spike_response(records: list) -> float:
    """
    Calculate average response time to traffic spikes.
    
    Spike = traffic increases by >50% in one step
    Response time = steps until pods scale up
    """
    response_times = []
    
    for i in range(1, len(records)):
        prev_load = records[i-1]['actual_load']
        curr_load = records[i]['actual_load']
        
        if prev_load > 0:
            increase_rate = (curr_load - prev_load) / prev_load
            
            if increase_rate > 0.5:  # 50% spike
                # Find when pods scaled up
                for j in range(i, min(i + 30, len(records))):
                    if records[j]['action'] > 0:
                        response_times.append(j - i)
                        break
    
    return np.mean(response_times) if response_times else 0.0


def test_all_strategies_on_scenario(
    scenario_name: str,
    traffic_file: str,
    output_dir: str = "results/ddos_tests"
) -> dict:
    """
    Test all 4 strategies on one scenario.
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    # Load traffic data
    traffic_data = pd.read_csv(traffic_file)
    print(f"  Loaded {len(traffic_data)} samples")
    print(f"  Mean traffic: {traffic_data['requests_count'].mean():.1f}")
    print(f"  Max traffic: {traffic_data['requests_count'].max():.1f}")
    
    capacity_per_pod = 100
    
    # Initialize strategies
    strategies = {
        'REACTIVE': ReactiveAutoscaler(capacity_per_pod),
        'PREDICTIVE': PredictiveAutoscaler(capacity_per_pod),
        'CPU_BASED': CPUBasedAutoscaler(capacity_per_pod),
        'HYBRID': HybridAutoscaler(capacity_per_pod, enable_anomaly_detection=True)
    }
    
    # Run all strategies
    results = {}
    all_records = []
    
    for strategy_name, autoscaler in strategies.items():
        result = run_strategy_on_scenario(
            strategy_name, autoscaler, traffic_data, capacity_per_pod
        )
        results[strategy_name] = result
        
        # Add scenario info to records
        for record in result['records']:
            record['scenario'] = scenario_name
            record['strategy'] = strategy_name
            all_records.append(record)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed records
    records_df = pd.DataFrame(all_records)
    records_file = output_path / f"{scenario_name.lower()}_results.csv"
    records_df.to_csv(records_file, index=False)
    
    # Save metrics summary
    metrics_summary = {}
    for strategy_name, result in results.items():
        metrics_summary[strategy_name] = {
            'total_cost': result['total_cost'],
            'sla_violations': result['sla_violations'],
            'scaling_events': result['scaling_events'],
            'spike_response_time': result['spike_response_time'],
            'avg_pods': result['metrics'].get('avg_pods', 0)
        }
    
    metrics_file = output_path / f"{scenario_name.lower()}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n  Saved: {records_file.name}")
    
    return results


def generate_comparison_report(output_dir: str = "results/ddos_tests"):
    """
    Generate comprehensive comparison report across all scenarios.
    """
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    output_path = Path(output_dir)
    scenarios = ['NORMAL', 'SUDDEN_SPIKE', 'GRADUAL_SPIKE', 'OSCILLATING_SPIKE', 'SUSTAINED_DDOS']
    strategies = ['REACTIVE', 'PREDICTIVE', 'CPU_BASED', 'HYBRID']
    
    # Aggregate data
    comparison_data = {}
    
    for scenario in scenarios:
        metrics_file = output_path / f"{scenario.lower()}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                comparison_data[scenario] = json.load(f)
    
    # Create comparison tables
    print("\n" + "="*80)
    print("COST COMPARISON (Lower is better)")
    print("="*80)
    print(f"{'Strategy':<15} {'Normal':<12} {'Sudden':<12} {'Gradual':<12} {'Oscillating':<12} {'Sustained':<12}")
    print("-"*80)
    
    for strategy in strategies:
        row = [strategy]
        for scenario in scenarios:
            if scenario in comparison_data and strategy in comparison_data[scenario]:
                cost = comparison_data[scenario][strategy]['total_cost']
                row.append(f"${cost:.2f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}")
    
    print("\n" + "="*80)
    print("SLA VIOLATIONS (Lower is better)")
    print("="*80)
    print(f"{'Strategy':<15} {'Normal':<12} {'Sudden':<12} {'Gradual':<12} {'Oscillating':<12} {'Sustained':<12}")
    print("-"*80)
    
    for strategy in strategies:
        row = [strategy]
        for scenario in scenarios:
            if scenario in comparison_data and strategy in comparison_data[scenario]:
                sla = comparison_data[scenario][strategy]['sla_violations']
                row.append(f"{sla}")
            else:
                row.append("N/A")
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}")
    
    print("\n" + "="*80)
    print("SPIKE RESPONSE TIME (Lower is better, in minutes)")
    print("="*80)
    print(f"{'Strategy':<15} {'Sudden':<12} {'Gradual':<12} {'Oscillating':<12} {'Sustained':<12}")
    print("-"*80)
    
    for strategy in strategies:
        row = [strategy]
        for scenario in ['SUDDEN_SPIKE', 'GRADUAL_SPIKE', 'OSCILLATING_SPIKE', 'SUSTAINED_DDOS']:
            if scenario in comparison_data and strategy in comparison_data[scenario]:
                response = comparison_data[scenario][strategy]['spike_response_time']
                row.append(f"{response:.1f}min")
            else:
                row.append("N/A")
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    # Save comprehensive report
    report_file = output_path / "ddos_comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n  Saved comprehensive report: {report_file.name}")
    print("="*80)


def main():
    """Run all DDoS tests."""
    print("\n" + "="*80)
    print("DDoS/SPIKE AUTOSCALING TEST SUITE")
    print("="*80)
    
    data_dir = Path("data/synthetic_ddos")
    scenarios = [
        ('NORMAL', data_dir / 'normal_traffic.csv'),
        ('SUDDEN_SPIKE', data_dir / 'sudden_spike_traffic.csv'),
        ('GRADUAL_SPIKE', data_dir / 'gradual_spike_traffic.csv'),
        ('OSCILLATING_SPIKE', data_dir / 'oscillating_spike_traffic.csv'),
        ('SUSTAINED_DDOS', data_dir / 'sustained_ddos_traffic.csv')
    ]
    
    # Test each scenario
    for scenario_name, traffic_file in scenarios:
        if not traffic_file.exists():
            print(f"\n⚠ Skipping {scenario_name}: File not found")
            continue
        
        test_all_strategies_on_scenario(scenario_name, str(traffic_file))
    
    # Generate comparison report
    generate_comparison_report()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\nResults saved to: results/ddos_tests/")
    print("View dashboard to visualize results")


if __name__ == "__main__":
    main()
