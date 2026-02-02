#!/usr/bin/env python3
"""
HYBRID AUTOSCALING PIPELINE
===========================

Production pipeline using HYBRID autoscaler with optimized cost model.

Based on comprehensive analysis (20 test scenarios):
- Selected Strategy: HYBRID (4-layer architecture)
- Cost Model: CloudCostModel with 2 reserved pods baseline
- Performance: 14 SLA violations (BEST), 4.7-5.5 min spike response

Configuration:
- Min Servers: 2 (reserved capacity)
- Max Servers: 20
- Cost: $0.05/pod/hour on-demand, $0.03/pod/hour reserved, $0.015/pod/hour spot
- Timeframe: 15-minute intervals (most realistic)

Usage:
    python run_hybrid_pipeline.py [--timeframe 15m]
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import HYBRID autoscaler (selected strategy)
from autoscaling.hybrid_optimized import HybridAutoscalerOptimized

# Import cost model
from cost.cost_model import CloudCostModel, InstanceType


class HybridPipeline:
    """
    Production pipeline with HYBRID autoscaler + optimized cost model.
    """
    
    def __init__(
        self,
        timeframe: str = "15m",
        capacity_per_server: int = 250,
        min_servers: int = 2,
        max_servers: int = 20,
    ):
        """
        Initialize pipeline.
        
        Args:
            timeframe: "1m", "5m", or "15m"
            capacity_per_server: Requests per pod per minute
            min_servers: Minimum pod count (reserved capacity)
            max_servers: Maximum pod count
        """
        self.timeframe = timeframe
        self.capacity_per_server = capacity_per_server
        self.min_servers = min_servers
        self.max_servers = max_servers
        
        # Initialize cost model (optimized for this workload)
        self.cost_model = CloudCostModel(
            on_demand_cost=0.05,       # $0.05/pod/hour
            reserved_cost=0.03,        # $0.03/pod/hour (40% savings)
            spot_cost=0.015,           # $0.015/pod/hour (70% savings)
            startup_cost=0.001,        # $0.001 cold start
            reserved_capacity=min_servers  # 2 pods always-on
        )
        
        # Initialize HYBRID autoscaler
        self.autoscaler = HybridAutoscalerOptimized(
            capacity_per_server=capacity_per_server,
            min_servers=min_servers,
            max_servers=max_servers
        )
    
    def load_data(self):
        """Load test data and predictions."""
        # Map timeframe to file
        timeframe_map = {
            "1m": "test_1m_autoscaling.csv",
            "5m": "test_5m_autoscaling.csv",
            "15m": "test_15m_autoscaling.csv"
        }
        
        # Load test data
        data_file = Path("data/real") / timeframe_map.get(self.timeframe, "test_15m_autoscaling.csv")
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        
        # Extract requests column
        if 'requests_count' in df.columns:
            requests = df['requests_count'].values
        elif 'requests' in df.columns:
            requests = df['requests'].values
        elif 'request_count' in df.columns:
            requests = df['request_count'].values
        else:
            raise ValueError("No requests column found in data")
        
        # Load predictions (pre-computed)
        predictions_file = Path(f"models/xgboost_{self.timeframe}_predictions.csv")
        predictions = None
        if predictions_file.exists():
            pred_df = pd.read_csv(predictions_file)
            if 'yhat' in pred_df.columns:
                predictions = pred_df['yhat'].values
                print(f"✅ Loaded {len(predictions)} predictions from {predictions_file}")
            elif 'predicted' in pred_df.columns:
                predictions = pred_df['predicted'].values
                print(f"✅ Loaded {len(predictions)} predictions from {predictions_file}")
        else:
            print(f"⚠️  No predictions file found at {predictions_file}")
        
        print(f"✅ Loaded {len(requests)} datapoints from {data_file}")
        return requests, predictions
    
    def run_simulation(self, requests: np.ndarray, predictions: np.ndarray = None):
        """
        Run HYBRID autoscaling simulation.
        
        Args:
            requests: Array of actual request counts
            predictions: Array of predicted request counts (pre-computed)
        
        Returns:
            results_df: DataFrame with simulation results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING HYBRID AUTOSCALING PIPELINE")
        print(f"{'='*80}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Strategy: HYBRID (4-layer: Anomaly → Emergency → Predictive → Reactive)")
        print(f"Cost Model: CloudCostModel (2 reserved + spot/on-demand)")
        print(f"Data points: {len(requests)}")
        print(f"Predictions: {'Yes (pre-computed)' if predictions is not None else 'No'}")
        print("="*80 + "\n")
        
        results = []
        current_servers = self.min_servers
        
        for i, request_count in enumerate(requests):
            # Use pre-computed forecast if available
            forecast_requests = None
            if predictions is not None and i < len(predictions):
                forecast_requests = predictions[i]
            
            # Run autoscaling decision
            new_servers, action, metrics = self.autoscaler.step(
                current_servers=current_servers,
                requests=request_count,
                forecast_requests=forecast_requests
            )
            
            # Calculate cost for this step
            timeframe_minutes = int(self.timeframe[:-1])  # Extract "15" from "15m"
            step_cost, cost_breakdown = self.cost_model.compute_step_cost(
                pod_count=new_servers,
                step_hours=timeframe_minutes / 60.0,
                strategy="spot_first"  # Use spot instances first (cost-effective)
            )
            
            # Record results
            results.append({
                'time': i,
                'requests': request_count,
                'forecast': forecast_requests if forecast_requests else 0,
                'pods_before': current_servers,
                'pods_after': new_servers,
                'action': action,
                'cpu': metrics['cpu'],
                'cost': step_cost,
                'cost_reserved': cost_breakdown.get('reserved', 0),
                'cost_spot': cost_breakdown.get('spot', 0),
                'cost_ondemand': cost_breakdown.get('on_demand', 0),
                'sla_violation': 1 if metrics['cpu'] > 0.95 else 0,
                'slo_violation': 1 if metrics['cpu'] > 0.85 else 0,
                'scaled': metrics.get('scaled', False),
                'scale_direction': metrics.get('scale_direction', 'NONE')
            })
            
            current_servers = new_servers
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(requests)} steps...")
        
        df = pd.DataFrame(results)
        return df
    
    def generate_report(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive report."""
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE - GENERATING REPORT")
        print("="*80 + "\n")
        
        # Calculate summary metrics
        summary = {
            'strategy': 'HYBRID',
            'timeframe': self.timeframe,
            'total_steps': len(results_df),
            'total_cost': float(results_df['cost'].sum()),
            'cost_breakdown': {
                'reserved': float(results_df['cost_reserved'].sum()),
                'spot': float(results_df['cost_spot'].sum()),
                'on_demand': float(results_df['cost_ondemand'].sum()),
            },
            'sla_violations': int(results_df['sla_violation'].sum()),
            'slo_violations': int(results_df['slo_violation'].sum()),
            'scaling_events': int(results_df['scaled'].sum()),
            'scale_up_events': int((results_df['scale_direction'] == 'UP').sum()),
            'scale_down_events': int((results_df['scale_direction'] == 'DOWN').sum()),
            'avg_pods': float(results_df['pods_after'].mean()),
            'min_pods': int(results_df['pods_after'].min()),
            'max_pods': int(results_df['pods_after'].max()),
            'avg_cpu': float(results_df['cpu'].mean()),
            'max_cpu': float(results_df['cpu'].max()),
            'forecast_mape': None,
        }
        
        # Calculate forecast MAPE if available
        if results_df['forecast'].sum() > 0:
            forecast_mask = results_df['forecast'] > 0
            if forecast_mask.sum() > 0:
                mape = np.mean(np.abs(
                    (results_df.loc[forecast_mask, 'requests'] - results_df.loc[forecast_mask, 'forecast']) / 
                    results_df.loc[forecast_mask, 'requests']
                )) * 100
                summary['forecast_mape'] = float(mape)
        
        # Print summary
        print("PERFORMANCE SUMMARY")
        print("-"*80)
        print(f"Total Cost:          ${summary['total_cost']:.2f}")
        print(f"  Reserved:          ${summary['cost_breakdown']['reserved']:.2f}")
        print(f"  Spot:              ${summary['cost_breakdown']['spot']:.2f}")
        print(f"  On-Demand:         ${summary['cost_breakdown']['on_demand']:.2f}")
        print(f"\nSLA Violations:      {summary['sla_violations']} (CPU > 95%)")
        print(f"SLO Violations:      {summary['slo_violations']} (CPU > 85%)")
        print(f"\nScaling Events:      {summary['scaling_events']}")
        print(f"  └─ Scale Up:       {summary['scale_up_events']}")
        print(f"  └─ Scale Down:     {summary['scale_down_events']}")
        print(f"\nPod Statistics:")
        print(f"  └─ Average:        {summary['avg_pods']:.1f}")
        print(f"  └─ Min:            {summary['min_pods']}")
        print(f"  └─ Max:            {summary['max_pods']}")
        print(f"\nCPU Statistics:")
        print(f"  └─ Average:        {summary['avg_cpu']:.1%}")
        print(f"  └─ Max:            {summary['max_cpu']:.1%}")
        
        if summary['forecast_mape']:
            print(f"\nForecast Quality:")
            print(f"  MAPE:              {summary['forecast_mape']:.2f}%")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / f"hybrid_results_{self.timeframe}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\nSaved results: {csv_file}")
        
        json_file = output_dir / f"hybrid_summary_{self.timeframe}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {json_file}")
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run HYBRID autoscaling pipeline")
    parser.add_argument(
        "--timeframe",
        choices=["1m", "5m", "15m"],
        default="15m",
        help="Timeframe for simulation (default: 15m, most realistic)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/hybrid_production",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Print header without Unicode characters (Windows console compatibility)
    print("\n" + "="*80)
    print("HYBRID AUTOSCALING PIPELINE - PRODUCTION DEPLOYMENT")
    print("="*80 + "\n")
    print("Strategy: HYBRID (4-layer decision hierarchy)")
    print("- Layer 0: Anomaly Detection (spike/DDoS protection)")
    print("- Layer 1: Emergency Detection (CPU > 95% protection)")
    print("- Layer 2: Predictive Scaling (forecast-based proactive)")
    print("- Layer 3: Reactive Scaling (request-based fallback)")
    print("\nCost Model: CloudCostModel (optimized)")
    print("- Reserved: 2 pods @ $0.03/pod/hour (baseline)")
    print("- Spot: 70% of burst @ $0.015/pod/hour (cost-effective)")
    print("- On-Demand: 30% of burst @ $0.05/pod/hour (reliability)")
    print("\nExpected Performance (based on Phase B.5 analysis):")
    print("- Cost: ~$57.79 per 15-day period")
    print("- SLA Violations: ~14 (BEST of all strategies)")
    print("- Spike Response: 4.7-5.5 minutes (FASTEST)")
    print("\n" + "="*80 + "\n")
    
    # Initialize pipeline
    pipeline = HybridPipeline(timeframe=args.timeframe)
    
    # Load data
    try:
        requests, predictions = pipeline.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Run simulation
    try:
        results_df = pipeline.run_simulation(requests, predictions)
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate report
    try:
        summary = pipeline.generate_report(results_df, Path(args.output_dir))
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80 + "\n")
    print(f"Results saved to: {args.output_dir}")
    print(f"View dashboard: streamlit run dashboard/app.py")
    print(f"\nRecommendation: Review {args.output_dir}/hybrid_summary_{args.timeframe}.json")
    print("="*80 + "\n")
if __name__ == "__main__":
    sys.exit(main())
