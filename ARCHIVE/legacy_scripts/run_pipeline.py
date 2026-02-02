#!/usr/bin/env python3
"""
AUTOSCALING PIPELINE ORCHESTRATOR
==================================

Master script that runs all phases of the pipeline:

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

PHASE B.5 — AUTOSCALING ON PREDICTED DATA (NEW)
  - Test autoscaling on pre-calculated LightGBM predictions
  - Compare performance: actual vs predicted load
  - Understand impact of forecast quality on autoscaling
  - Output: results/phase_b5_*.csv, results/phase_b5_*.json

PHASE C — ANOMALY & COST ANALYSIS (Advanced Testing)
  - Test anomaly detection (DDoS, flash sales, failovers)
  - Evaluate cost models (K8s, AWS, GCP, spot instances)
  - Measure platform-specific metrics (HPA, Auto Scaling, Borg)
  - Output: results/anomaly_analysis.json, cost_breakdown.json

Usage:
    python run_pipeline.py [--phase-a-only | --phase-b-only | --phase-b5-only | --phase-c-only]

Examples:
    python run_pipeline.py                    # Run all phases
    python run_pipeline.py --phase-a-only    # Model evaluation only
    python run_pipeline.py --phase-b-only    # Autoscaling tests (synthetic) only
    python run_pipeline.py --phase-b5-only   # Autoscaling tests (predicted) only
    python run_pipeline.py --phase-c-only    # Anomaly & cost analysis only
    python run_pipeline.py --skip-phase-b5   # Run all except B.5
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Import both phases
from forecast.model_evaluation import ModelEvaluator
from forecast.phase_b5_predicted import run_all_timeframes as run_phase_b5_all_timeframes
import simulate

# Import HYBRID autoscaler (production-grade, selected based on analysis)
from autoscaling.hybrid_optimized import HybridAutoscalerOptimized

# Import new modules for Phase C
from anomaly.anomaly_detection import AnomalyDetector
from anomaly.simulate_anomaly import AnomalySimulator
from cost.cost_model import CloudCostModel, KubernetesCostModel, InstanceType, PriorityClass
from cost.metrics import MetricsCollector


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
    
    evaluator = ModelEvaluator(real_data_dir="data/real")
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


def run_phase_b5_autoscaling_on_predicted(output_dir="results"):
    """
    PHASE B.5: Test autoscaling on predicted load data.
    
    - Load LightGBM predictions from data/prediction/
    - Test all strategies on predicted load (perfect forecast case)
    - Measure cost, SLA, stability with high-quality forecasts
    - Compare against synthetic scenarios from Phase B
    - Output: results/phase_b5_*.csv, results/phase_b5_*.json
    """
    print("\n" + "=" * 80)
    print("PHASE B.5: AUTOSCALING ON PREDICTED DATA (LightGBM)")
    print("=" * 80)
    
    results = run_phase_b5_all_timeframes(output_dir)
    
    return results


def run_phase_c_anomaly_cost_analysis(output_dir="results"):
    """
    PHASE C: Advanced anomaly detection and cost analysis.
    
    - Test anomaly detection on various attack/failure scenarios
    - Compare cost models: Simple, AWS-style, Kubernetes, Borg
    - Evaluate platform-specific metrics (K8s HPA, AWS Auto Scaling)
    - Simulate real-world incidents (DDoS, flash sales, failovers)
    """
    print("\n" + "=" * 80)
    print("PHASE C: ANOMALY & COST ANALYSIS")
    print("=" * 80)
    
    from autoscaling.scenarios import generate_all_scenarios
    
    # Generate base traffic
    scenarios = generate_all_scenarios(duration=200)
    base_scenario = scenarios[0]  # Use GRADUAL_INCREASE as base
    base_traffic = base_scenario.load_series
    
    # ==================== ANOMALY DETECTION TESTS ====================
    print("\n[1/3] Testing Anomaly Detection Methods...")
    
    detector = AnomalyDetector(
        window_size=50,
        zscore_threshold=3.0,
        iqr_multiplier=1.5,
        rate_threshold=0.5
    )
    
    simulator = AnomalySimulator()
    
    anomaly_results = {}
    
    # Test different anomaly types
    anomaly_tests = [
        ('ddos', {'start': 50, 'duration': 20, 'intensity': 5.0}),
        ('flash_sale', {'start': 80, 'peak_time': 15, 'duration': 30, 'peak_intensity': 8.0}),
        ('service_failure', {'start': 120, 'duration': 15, 'drop_ratio': 0.8}),
        ('thundering_herd', {'start': 150, 'wave_count': 3, 'wave_spacing': 5, 'intensity': 3.0}),
        ('multi_region_failover', {'start': 30, 'ramp_duration': 20, 'intensity': 2.5}),
    ]
    
    for anomaly_type, params in anomaly_tests:
        # Inject anomaly
        method = getattr(simulator, f'inject_{anomaly_type}')
        modified_traffic, anomaly_mask = method(base_traffic.copy(), **params)
        
        # Detect using ensemble
        detected, methods = detector.detect_ensemble(modified_traffic, min_votes=2)
        
        # Compute detection metrics
        true_positives = np.sum((detected == 1) & (anomaly_mask == 1))
        false_positives = np.sum((detected == 1) & (anomaly_mask == 0))
        false_negatives = np.sum((detected == 0) & (anomaly_mask == 1))
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 0.001)
        
        anomaly_results[anomaly_type] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'detection_rate': float(true_positives / max(np.sum(anomaly_mask), 1))
        }
        
        print(f"  {anomaly_type:20s} - F1: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    # ==================== COST MODEL COMPARISON ====================
    print("\n[2/3] Comparing Cost Models...")
    
    # Simulate pod history (typical autoscaling pattern)
    pod_history = [5, 5, 6, 7, 8, 10, 12, 15, 18, 20, 20, 19, 18, 16, 14, 12, 10, 8, 7, 6, 5, 5] * 10
    
    cost_results = {}
    
    # 1. Simple linear cost (baseline)
    simple_cost = sum(pod_history) * 0.05 * (5.0 / 60.0)
    cost_results['simple_linear'] = {
        'total_cost': float(simple_cost),
        'cost_per_pod': 0.05,
        'model': 'baseline'
    }
    
    # 2. Cloud cost model (AWS/GCP/Azure style)
    cloud_model = CloudCostModel(
        on_demand_cost=0.05,
        reserved_cost=0.03,
        spot_cost=0.015,
        startup_cost=0.001,
        reserved_capacity=5
    )
    
    cloud_total, cloud_breakdown = cloud_model.compute_total_cost(
        pod_history, step_minutes=5.0, track_scaling_cost=True
    )
    
    cost_results['cloud_mixed'] = {
        'total_cost': float(cloud_total),
        'breakdown': {k: float(v) for k, v in cloud_breakdown.items()},
        'model': 'aws_gcp_azure_mixed_instances',
        'savings_vs_simple': float((simple_cost - cloud_total) / simple_cost * 100)
    }
    
    # 3. Kubernetes cost model
    k8s_model = KubernetesCostModel(
        node_cost_per_hour=0.10,
        pods_per_node=30,
        node_overhead_pods=3
    )
    
    k8s_result = k8s_model.compute_total_cost(pod_history, step_minutes=5.0)
    
    cost_results['kubernetes'] = {
        'total_cost': float(k8s_result['total_cost']),
        'avg_nodes': float(k8s_result['avg_nodes']),
        'packing_efficiency': float(k8s_result['packing_efficiency']),
        'wasted_capacity_pct': float(k8s_result['wasted_capacity']),
        'model': 'kubernetes_node_pools'
    }
    
    # 4. Google Borg priority-based
    borg_production = cloud_model.compute_borg_style_cost(
        pod_history, PriorityClass.PRODUCTION, step_minutes=5.0
    )
    borg_batch = cloud_model.compute_borg_style_cost(
        pod_history, PriorityClass.BATCH, step_minutes=5.0
    )
    
    cost_results['borg_production'] = {
        'total_cost': float(borg_production),
        'priority': 'production',
        'model': 'google_borg'
    }
    
    cost_results['borg_batch'] = {
        'total_cost': float(borg_batch),
        'priority': 'batch',
        'model': 'google_borg',
        'savings_vs_production': float((borg_production - borg_batch) / borg_production * 100)
    }
    
    print("  Cost Model Comparison:")
    for model_name, result in cost_results.items():
        cost = result['total_cost']
        print(f"    {model_name:20s} - ${cost:.4f}")
        if 'savings_vs_simple' in result:
            print(f"      → Savings: {result['savings_vs_simple']:.1f}% vs simple")
    
    # ==================== PLATFORM-SPECIFIC METRICS ====================
    print("\n[3/3] Testing Platform-Specific Metrics...")
    
    # Simulate metrics collection with K8s, AWS, Borg features
    platform_metrics = {}
    
    # Kubernetes HPA metrics
    k8s_collector = MetricsCollector(
        capacity_per_pod=500,
        cost_per_pod_per_hour=0.05,
        step_minutes=5.0,
        enable_k8s_metrics=True,
        enable_aws_metrics=False,
        enable_borg_metrics=False
    )
    
    for i in range(len(pod_history)):
        pods = pod_history[i]
        requests = pods * 500 * np.random.uniform(0.6, 0.9)  # 60-90% utilization
        cpu_util = requests / (pods * 500)
        
        k8s_collector.record(
            t=i, pods=pods, requests=requests, scaling_action=0,
            cpu_utilization=cpu_util
        )
    
    k8s_metrics = k8s_collector.compute_aggregate_metrics()
    platform_metrics['kubernetes_hpa'] = {
        'avg_cpu_utilization': k8s_metrics.get('k8s_avg_cpu_utilization', 0),
        'cpu_target_breaches': k8s_metrics.get('k8s_cpu_target_breaches', 0),
        'hpa_trigger_rate': k8s_metrics.get('k8s_hpa_trigger_rate', 0),
    }
    
    # AWS Auto Scaling metrics
    aws_collector = MetricsCollector(
        capacity_per_pod=500,
        cost_per_pod_per_hour=0.05,
        step_minutes=5.0,
        enable_k8s_metrics=False,
        enable_aws_metrics=True,
        enable_borg_metrics=False
    )
    
    for i in range(len(pod_history)):
        pods = pod_history[i]
        requests = pods * 500 * 0.75
        scaling_action = 1 if i % 20 == 0 else 0
        
        aws_collector.record(
            t=i, pods=pods, requests=requests, scaling_action=scaling_action
        )
    
    aws_metrics = aws_collector.compute_aggregate_metrics()
    platform_metrics['aws_auto_scaling'] = {
        'warm_up_time_ratio': aws_metrics.get('aws_warm_up_time_ratio', 0),
        'cooldown_effectiveness': aws_metrics.get('aws_cooldown_effectiveness', 0),
        'target_tracking_breaches': aws_metrics.get('aws_target_tracking_breaches', 0),
    }
    
    print("  Platform Metrics:")
    print(f"    K8s HPA avg CPU: {platform_metrics['kubernetes_hpa']['avg_cpu_utilization']:.1%}")
    print(f"    K8s HPA triggers: {platform_metrics['kubernetes_hpa']['hpa_trigger_rate']:.1%}")
    print(f"    AWS cooldown eff: {platform_metrics['aws_auto_scaling']['cooldown_effectiveness']:.1%}")
    
    # ==================== SAVE RESULTS ====================
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    phase_c_results = {
        'timestamp': datetime.now().isoformat(),
        'anomaly_detection': anomaly_results,
        'cost_models': cost_results,
        'platform_metrics': platform_metrics
    }
    
    # Save anomaly analysis
    anomaly_path = output_path / "anomaly_analysis.json"
    with open(anomaly_path, 'w') as f:
        json.dump({'anomaly_detection': anomaly_results}, f, indent=2)
    print(f"\n✓ Anomaly analysis saved to {anomaly_path}")
    
    # Save cost breakdown
    cost_path = output_path / "cost_breakdown.json"
    with open(cost_path, 'w') as f:
        json.dump({'cost_models': cost_results, 'platform_metrics': platform_metrics}, f, indent=2)
    print(f"✓ Cost breakdown saved to {cost_path}")
    
    return phase_c_results


def create_pipeline_summary(phase_a_results, phase_b_results, phase_c_results=None, output_dir="results"):
    """
    Create a summary document comparing all phases.
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
        "phase_c": {
            "name": "Anomaly & Cost Analysis",
            "description": "Advanced anomaly detection and cost optimization",
            "output_files": ["anomaly_analysis.json", "cost_breakdown.json"],
            "status": "completed" if phase_c_results else "skipped",
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
    
    print(f"\nPHASE C: {summary['phase_c']['status'].upper()}")
    if phase_c_results:
        anomaly_count = len(phase_c_results.get('anomaly_detection', {}))
        cost_models = len(phase_c_results.get('cost_models', {}))
        print(f"  Anomaly types tested: {anomaly_count}")
        print(f"  Cost models compared: {cost_models}")
        if 'anomaly_detection' in phase_c_results:
            avg_f1 = np.mean([r['f1_score'] for r in phase_c_results['anomaly_detection'].values()])
            print(f"  Avg anomaly detection F1: {avg_f1:.3f}")
    
    print(f"{'='*80}\n")


def main():
    """Run the complete pipeline."""
    # Parse arguments
    run_phase_a = True
    run_phase_b = True
    run_phase_b5 = True
    run_phase_c = True
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--phase-a-only":
            run_phase_b = run_phase_b5 = run_phase_c = False
        elif arg == "--phase-b-only":
            run_phase_a = run_phase_b5 = run_phase_c = False
        elif arg == "--phase-b5-only":
            run_phase_a = run_phase_b = run_phase_c = False
        elif arg == "--phase-c-only":
            run_phase_a = run_phase_b = run_phase_b5 = False
        elif arg == "--skip-phase-b5":
            run_phase_b5 = False
        elif arg in ["-h", "--help"]:
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python run_pipeline.py [--phase-a-only | --phase-b-only | --phase-b5-only | --phase-c-only | --skip-phase-b5]")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("AUTOSCALING PIPELINE ORCHESTRATOR")
    print("=" * 80)
    print(f"{'='*80}\n")
    
    phase_a_results = {}
    phase_b_results = {}
    phase_b5_results = {}
    phase_c_results = {}
    
    # Run Phase A
    if run_phase_a:
        try:
            phase_a_results = run_phase_a_model_evaluation()
        except Exception as e:
            print(f"\n❌ Phase A failed: {e}")
            if not (run_phase_b or run_phase_b5 or run_phase_c):
                sys.exit(1)
    
    # Run Phase B
    if run_phase_b:
        try:
            phase_b_results = run_phase_b_autoscaling_tests()
        except Exception as e:
            print(f"\n❌ Phase B failed: {e}")
            if not (run_phase_a or run_phase_b5 or run_phase_c):
                sys.exit(1)
    
    # Run Phase B.5 (NEW: Autoscaling on predicted data)
    if run_phase_b5:
        try:
            phase_b5_results = run_phase_b5_autoscaling_on_predicted()
        except Exception as e:
            print(f"\n❌ Phase B.5 failed: {e}")
            import traceback
            traceback.print_exc()
            if not (run_phase_a or run_phase_b or run_phase_c):
                sys.exit(1)
    
    # Run Phase C
    if run_phase_c:
        try:
            phase_c_results = run_phase_c_anomaly_cost_analysis()
        except Exception as e:
            print(f"\n❌ Phase C failed: {e}")
            import traceback
            traceback.print_exc()
            if not (run_phase_a or run_phase_b or run_phase_b5):
                sys.exit(1)
    
    # Create summary
    create_pipeline_summary(phase_a_results, phase_b_results, phase_c_results)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  Phase A: results/model_evaluation.json")
    print("  Phase B: results/simulation_results.csv, results/metrics_summary.json")
    if run_phase_b5:
        print("  Phase B.5: results/phase_b5_*.csv, results/phase_b5_*.json")
    print("  Phase C: results/anomaly_analysis.json, cost_breakdown.json")
    print("  Summary: results/pipeline_summary.json")
    print("\nNext steps:")
    print("  1. Review model_evaluation.json for best models")
    print("  2. Check simulation_results.csv for autoscaling test results (synthetic data)")
    if run_phase_b5:
        print("  3. Check phase_b5_*.json for autoscaling test results (predicted data)")
        print("  4. Compare Phase B vs B.5 for impact of forecast quality")
        print("  5. Analyze anomaly_analysis.json for detection performance")
        print("  6. Compare cost_breakdown.json for cost optimization opportunities")
        print("  7. Run dashboard: streamlit run dashboard/app.py")
    else:
        print("  3. Analyze anomaly_analysis.json for detection performance")
        print("  4. Compare cost_breakdown.json for cost optimization opportunities")
        print("  5. Run dashboard: streamlit run dashboard/app.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
