# Model Integration & Real Data Evaluation

## Overview

The autoscaling pipeline has been updated to:

1. **Replace ARIMA forecaster** with production ML models from `demo/models/`
2. **Support real data evaluation** alongside synthetic scenarios
3. **Unify results format** with a `data_source` field for easy comparison

## Changes Made

### PART 1: Model Replacement

#### Before

- Used ARIMA forecaster (`ARIMAForecaster` from `forecast/arima_forecaster.py`)
- Limited to simple statistical time series forecasting
- No support for hybrid/ensemble approaches

#### After

- Uses ML models from `demo/models/`:
  - **Hybrid Model** (LSTM + residual forecasting)
  - **XGBoost** (1m, 5m, 15m timeframes)
  - **LSTM** (1m, 5m, 15m timeframes)
- New abstraction layer: `BaseModel` interface
- New implementation: `ModelForecaster` wrapper
- Utility functions: `forecast_utils.py` (model loading, forecasting)

#### New Files

```
forecast/
├── model_base.py          # BaseModel abstract class + ForecastResult dataclass
├── model_forecaster.py    # ModelForecaster concrete implementation
├── forecast_utils.py      # Model loading, inference, path resolution
├── base_forecast.py       # [DEPRECATED] Old ARIMA base
└── arima_forecaster.py    # [DEPRECATED] Old ARIMA implementation
```

#### Model Directory

```
models/
├── hybrid_model_package.pkl      # Hybrid model with scaler
├── lstm_1m_best.keras
├── lstm_5m_best.keras
├── lstm_15m_best.keras
├── xgboost_1m_model.json
├── xgboost_5m_model.json
├── xgboost_15m_model.json
├── xgboost_*_metadata.json       # Feature lists
├── xgboost_*_scaler.pkl          # Feature scalers
└── xgboost_*_predictions.csv     # Pre-computed predictions
```

### PART 2: Real Data Support

#### Before

- Only synthetic scenarios (GRADUAL_INCREASE, SUDDEN_SPIKE, OSCILLATING, TRAFFIC_DROP, FORECAST_ERROR_TEST)
- 200 timesteps per scenario
- Deterministic behavior

#### After

- **Synthetic scenarios** (5 types, 200 steps each) ✓ [PRESERVED]
- **Real data** from processed historical datasets ✓ [NEW]
  - `test_1m_autoscaling.csv`
  - `test_5m_autoscaling.csv`
  - `test_15m_autoscaling.csv`
  - `train_1m_autoscaling.csv`
  - `train_5m_autoscaling.csv`
  - `train_15m_autoscaling.csv`

#### Real Data Directory

```
data/
├── synthetic/              # [PLACEHOLDER, currently in-memory]
└── real/
    ├── test_1m_autoscaling.csv
    ├── test_5m_autoscaling.csv
    ├── test_15m_autoscaling.csv
    ├── train_1m_autoscaling.csv
    ├── train_5m_autoscaling.csv
    └── train_15m_autoscaling.csv
```

### PART 3: Pipeline Changes

#### Updated simulate.py

**New function: `load_real_data_scenarios()`**

```python
def load_real_data_scenarios() -> list[tuple[str, str, pd.DataFrame]]:
    """Load real processed data from data/real/ directory."""
    # Returns: [(scenario_name, data_source_label, dataframe), ...]
```

**Updated function: `run_strategy_on_scenario()`**

```python
def run_strategy_on_scenario(
    strategy_name,
    autoscaler,
    forecaster,
    load_series,           # NEW: pd.DataFrame instead of Scenario object
    forecast_horizon=1,
    capacity_per_pod=500,
    step_minutes=5.0,
    data_source="synthetic",  # NEW: track data source
    scenario_name="",
):
    """Simulate single autoscaling strategy on a load series."""
    # Now accepts both synthetic and real data in DataFrame format
    # Uses ModelForecaster instead of ARIMAForecaster
    # Tracks data_source in results
```

**Updated function: `run_all_simulations()`**

```python
def run_all_simulations(
    scenarios=None,
    real_scenarios=None,    # NEW: real data scenarios
    strategies=None,
    capacity_per_pod=500,
    run_synthetic=True,     # NEW: toggle synthetic runs
    run_real=True,          # NEW: toggle real data runs
):
    """Run all strategies across synthetic AND real scenarios."""
```

**Updated function: `save_results()`**

```
Changes:
- Results CSV now includes data_source column
- Metrics summary JSON keys now include data_source
- Strategy comparison separated by data_source (synthetic vs real)
```

**Updated function: `print_summary()`**

```
Changes:
- Now groups results by data_source
- Separate summary sections for synthetic and real data
```

**Main execution**

```python
if __name__ == "__main__":
    # Parse command line arguments
    # --synthetic-only  : Run only synthetic scenarios
    # --real-only       : Run only real data scenarios
    # (default: run both)

    results = run_all_simulations(
        run_synthetic=run_synthetic,
        run_real=run_real,
    )
```

### PART 4: Results Format

#### CSV Output: `results/simulation_results.csv`

**Old columns:**

- time, actual_requests, forecast_requests, forecast_error
- pods, scaling_action, reason, z_anomaly, sla_breached
- strategy, scenario

**New columns (added):**

- `data_source` : "synthetic" | "real"
- `timestamp` : Full timestamp (for real data alignment)

#### JSON Output: `results/metrics_summary.json`

**Old structure:**

```json
{
  "REACTIVE_GRADUAL_INCREASE": { metrics },
  "REACTIVE_SUDDEN_SPIKE": { metrics },
  ...
}
```

**New structure:**

```json
{
  "REACTIVE_GRADUAL_INCREASE_synthetic": { metrics },
  "REACTIVE_SUDDEN_SPIKE_synthetic": { metrics },
  "REACTIVE_test_5m_autoscaling_real": { metrics },
  "REACTIVE_test_1m_autoscaling_real": { metrics },
  ...
}
```

#### Strategy Comparison: `results/strategy_comparison.json`

**Old structure:**

```json
{
  "REACTIVE": { avg_metrics },
  "PREDICTIVE": { avg_metrics },
  ...
}
```

**New structure:**

```json
{
  "synthetic": {
    "REACTIVE": { avg_metrics_synthetic },
    "PREDICTIVE": { avg_metrics_synthetic },
    ...
  },
  "real": {
    "REACTIVE": { avg_metrics_real },
    "PREDICTIVE": { avg_metrics_real },
    ...
  }
}
```

## Usage

### Run Complete Pipeline (Synthetic + Real)

```bash
python simulate.py
```

### Run Synthetic Only

```bash
python simulate.py --synthetic-only
```

### Run Real Data Only

```bash
python simulate.py --real-only
```

### View Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard now includes a data source selector to compare synthetic vs real data results.

## Key Architecture Decisions

### 1. DataFrame-Based Load Series

- **Why**: Real data is already in DataFrame format; standardizing on this reduces conversion overhead
- **Impact**: `run_strategy_on_scenario()` now accepts `load_series: pd.DataFrame` instead of `scenario: Scenario`
- **Backward Compatibility**: Synthetic scenarios are converted to DataFrame format automatically

### 2. ModelForecaster Abstraction

- **Why**: Allows plugging in different forecasting models without changing autoscaling logic
- **Design**: `BaseModel` abstract class with `predict(history_df, horizon) -> ForecastResult`
- **Flexibility**: Can easily add Prophet, Prophet+LSTM, or other models by implementing BaseModel

### 3. Unified Results Format

- **Why**: Enables cross-dataset comparison (synthetic vs real) in dashboards and analysis
- **Method**: Added `data_source` field to all result records
- **Benefit**: Same dashboard can filter/compare across data sources

### 4. Lazy Model Loading

- **Why**: Models are large (~800MB for LSTM); don't load unnecessary ones
- **Implementation**: `forecast_utils.py` loads models on-demand by timeframe
- **Fallback**: Uses heuristic forecasting if model fails to load

## Migration Guide (For User Code)

### Old Code (ARIMA-based)

```python
from forecast.arima_forecaster import ARIMAForecaster

forecaster = ARIMAForecaster(order=(2, 1, 2))
forecaster.fit(train_data)
predictions = forecaster.predict(horizon=5)
```

### New Code (ML models)

```python
from forecast.model_forecaster import ModelForecaster

forecaster = ModelForecaster(
    model_type="hybrid",  # or "lstm", "xgboost"
    timeframe="5m",       # or "1m", "15m"
    forecast_horizon=5,
)

# History must be a DataFrame with requests_count column
result = forecaster.predict(history_df, horizon=5)
predictions = result.yhat  # list of floats
```

## Backward Compatibility

- Old `forecast/arima_forecaster.py` and `forecast/base_forecast.py` still present (but unused)
- Autoscaling policies unchanged (REACTIVE, PREDICTIVE, CPU_BASED, HYBRID)
- Metrics collection unchanged
- Dashboard mostly compatible (added data source selector)

## Testing

### Unit Tests

```bash
# Test ModelForecaster
python -c "
from forecast.model_forecaster import ModelForecaster
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=50, freq='5min'),
    'requests_count': [300 + i*5 for i in range(50)]
})

forecaster = ModelForecaster(model_type='hybrid', timeframe='5m')
result = forecaster.predict(df, horizon=1)
print(f'✓ Prediction: {result.yhat[0]:.0f} req/s')
"
```

### Integration Tests

```bash
# Run small simulation
python simulate.py --synthetic-only
# Check results/simulation_results.csv for data_source column
```

## Known Limitations

1. **Real data forecast accuracy depends on model training**: Models were trained on different data; performance on test sets may vary
2. **XGBoost predictions use CSV fallback**: For efficiency, uses pre-computed predictions from CSV; may not match test set exactly
3. **Hybrid model requires scaler**: If hybrid_model_package.pkl corrupts, falls back to heuristic
4. **Small duration scenarios may fail**: Scenario generator has edge cases with very small duration (<50 steps)

## Future Enhancements

1. **Online learning**: Retrain models as new data arrives
2. **Multi-step ahead forecasts**: Currently predicts 1 step; add horizon > 1 support
3. **Forecast confidence intervals**: Track prediction uncertainty
4. **Cost optimization**: Jointly optimize cost and SLA instead of using objective function sum
5. **AutoML**: Auto-select best model per timeframe/scenario
