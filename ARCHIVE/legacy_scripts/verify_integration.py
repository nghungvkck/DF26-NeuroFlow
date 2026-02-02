#!/usr/bin/env python3
"""
Integration Verification Script
================================
Validates that all components of the model integration are working correctly.

Run this script to verify:
1. Models are in place and loadable
2. Real data is accessible
3. Synthetic scenarios work
4. All autoscaling strategies function
5. Forecasting works end-to-end

Usage:
    python verify_integration.py
"""

import sys
from pathlib import Path


def check_models():
    """Verify models are in place."""
    print("\n[CHECK 1] Models...")
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ✗ models/ directory not found")
        return False

    required_models = [
        "hybrid_model_package.pkl",
        "lstm_1m_best.keras",
        "lstm_5m_best.keras",
        "lstm_15m_best.keras",
        "xgboost_1m_model.json",
        "xgboost_5m_model.json",
        "xgboost_15m_model.json",
    ]

    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {model} not found")
            return False

    return True


def check_real_data():
    """Verify real data is in place."""
    print("\n[CHECK 2] Real Data...")
    data_dir = Path("data/real")
    if not data_dir.exists():
        print("  ✗ data/real/ directory not found")
        return False

    required_files = [
        "test_1m_autoscaling.csv",
        "test_5m_autoscaling.csv",
        "test_15m_autoscaling.csv",
        "train_1m_autoscaling.csv",
        "train_5m_autoscaling.csv",
        "train_15m_autoscaling.csv",
    ]

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            rows = sum(1 for _ in open(filepath)) - 1  # Subtract header
            print(f"  ✓ {filename} ({rows:,} rows)")
        else:
            print(f"  ✗ {filename} not found")
            return False

    return True


def check_imports():
    """Verify all imports work."""
    print("\n[CHECK 3] Python Imports...")
    
    try:
        from forecast.model_forecaster import ModelForecaster
        print("  ✓ ModelForecaster")
    except ImportError as e:
        print(f"  ✗ ModelForecaster: {e}")
        return False

    try:
        from forecast.model_base import BaseModel, ForecastResult
        print("  ✓ BaseModel, ForecastResult")
    except ImportError as e:
        print(f"  ✗ BaseModel: {e}")
        return False

    try:
        from autoscaling.scenarios import generate_all_scenarios
        print("  ✓ generate_all_scenarios")
    except ImportError as e:
        print(f"  ✗ generate_all_scenarios: {e}")
        return False

    try:
        from autoscaling.reactive import ReactiveAutoscaler
        from autoscaling.predictive import PredictiveAutoscaler
        from autoscaling.cpu_based import CPUBasedAutoscaler
        from autoscaling.hybrid import HybridAutoscaler
        print("  ✓ All autoscalers (REACTIVE, PREDICTIVE, CPU_BASED, HYBRID)")
    except ImportError as e:
        print(f"  ✗ Autoscalers: {e}")
        return False

    try:
        from cost.metrics import MetricsCollector
        print("  ✓ MetricsCollector")
    except ImportError as e:
        print(f"  ✗ MetricsCollector: {e}")
        return False

    return True


def check_forecasting():
    """Verify forecasting works."""
    print("\n[CHECK 4] Forecasting...")
    
    try:
        import pandas as pd
        from forecast.model_forecaster import ModelForecaster
        
        # Create sample data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "requests_count": [300 + i * 5 for i in range(100)],
        })
        
        # Test different model types
        for model_type in ["hybrid", "lstm", "xgboost"]:
            forecaster = ModelForecaster(model_type=model_type, timeframe="5m")
            result = forecaster.predict(df, horizon=1)
            
            if result.yhat and len(result.yhat) > 0:
                print(f"  ✓ {model_type}: {result.yhat[0]:.0f} req/s (status={result.metadata['status']})")
            else:
                print(f"  ✗ {model_type}: No predictions returned")
                return False
    except Exception as e:
        print(f"  ✗ Forecasting error: {e}")
        return False

    return True


def check_autoscaling():
    """Verify autoscaling works."""
    print("\n[CHECK 5] Autoscaling...")
    
    try:
        from autoscaling.reactive import ReactiveAutoscaler
        from autoscaling.predictive import PredictiveAutoscaler
        from autoscaling.cpu_based import CPUBasedAutoscaler
        from autoscaling.hybrid import HybridAutoscaler
        
        strategies = {
            "REACTIVE": ReactiveAutoscaler(capacity_per_server=500),
            "PREDICTIVE": PredictiveAutoscaler(capacity_per_server=500),
            "CPU_BASED": CPUBasedAutoscaler(capacity_per_server=500),
            "HYBRID": HybridAutoscaler(capacity_per_server=500),
        }
        
        for name, autoscaler in strategies.items():
            current_pods = 5
            actual_requests = 1500  # 3 pods would be needed
            
            if name == "REACTIVE":
                new_pods, util, action = autoscaler.step(current_pods, actual_requests)
            elif name == "PREDICTIVE":
                new_pods, action = autoscaler.step(current_pods, actual_requests, actual_requests)
            elif name == "CPU_BASED":
                new_pods, util, action = autoscaler.step(current_pods, actual_requests)
            elif name == "HYBRID":
                new_pods, action, reason = autoscaler.step(current_pods, actual_requests, actual_requests)
            
            print(f"  ✓ {name}: {current_pods} → {new_pods} pods")
    except Exception as e:
        print(f"  ✗ Autoscaling error: {e}")
        return False

    return True


def check_scenarios():
    """Verify scenario generation works."""
    print("\n[CHECK 6] Synthetic Scenarios...")
    
    try:
        from autoscaling.scenarios import generate_all_scenarios
        
        scenarios = generate_all_scenarios(duration=100)
        if len(scenarios) == 5:
            print(f"  ✓ Generated {len(scenarios)} scenarios:")
            for scenario in scenarios:
                print(f"    - {scenario.name}: {len(scenario.load_series)} samples")
        else:
            print(f"  ✗ Expected 5 scenarios, got {len(scenarios)}")
            return False
    except Exception as e:
        print(f"  ✗ Scenario generation error: {e}")
        return False

    return True


def main():
    """Run all checks."""
    print("="*80)
    print("INTEGRATION VERIFICATION")
    print("="*80)
    
    checks = [
        ("Models", check_models),
        ("Real Data", check_real_data),
        ("Imports", check_imports),
        ("Forecasting", check_forecasting),
        ("Autoscaling", check_autoscaling),
        ("Scenarios", check_scenarios),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
    
    all_passed = all(results.values())
    
    print("="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nIntegration is complete and working correctly!")
        print("\nNext steps:")
        print("  1. Run: python simulate.py")
        print("  2. View: streamlit run dashboard/app.py")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
