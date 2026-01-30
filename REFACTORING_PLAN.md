# Refactoring Plan: Fix Data Responsibilities & SLA Logic

## Issues to Fix

### 1. **Real Data Misuse**

- ❌ Real data is being used for autoscaling scenario testing
- ❌ Real data scenario names (test_1m, train_1m) are treated like test scenarios
- ✅ Real data should ONLY be used for model training/validation metrics

### 2. **SLA Violation Logic Bug**

- ❌ SLA is computed AFTER scaling decisions (line: `pods = new_pods`)
- ❌ SLA should be computed BEFORE scaling decisions take effect
- ❌ Results show SLA violations are always 0
- ✅ Must check if `actual_requests > CURRENT_pods * capacity` BEFORE scaling

### 3. **Two Conflicting Responsibilities**

- ❌ `simulate.py` mixes:
  - Model evaluation on real data
  - Autoscaling scenario testing on synthetic data
- ✅ Must split into two separate pipelines:
  - **PHASE A**: Model evaluation (real data only)
  - **PHASE B**: Scenario testing (synthetic data only)

### 4. **UI Inflexibility**

- ❌ Dashboard assumes all scenarios are testable with filtering
- ❌ No clear separation of data sources in visualization
- ✅ Must add proper controls for data source selection

## Solution Architecture

### PHASE A: Model Evaluation (Real Data Only)

```
Real Data (processed_for_modeling_v2/)
    ↓
Load real data
    ↓
For each timeframe (1m, 5m, 15m):
  - Extract requests_count
  - Run forecaster on each window
  - Compute MAE, RMSE, MAPE
  - Save metrics per model type
    ↓
Output: model_evaluation_results.json
  - Per-model performance on real data
  - Used for model selection
```

### PHASE B: Autoscaling Testing (Synthetic Scenarios Only)

```
Synthetic Scenarios (generated)
    ↓
For each scenario (GRADUAL_INCREASE, SPIKE, etc.):
  - Generate synthetic load
  - For each strategy (REACTIVE, PREDICTIVE, CPU_BASED, HYBRID):
    - Get forecast using trained model
    - CHECK SLA BEFORE scaling
    - Apply scaling decision
    - Record: pods, SLA breach, cost, actions
      ↓
Output: simulation_results.csv
  - SLA violations NON-ZERO for spike scenarios
  - Cost/stability metrics across strategies
```

## File Changes Required

### 1. New File: `forecast/model_evaluation.py`

- Load real data from `processed_for_modeling_v2/`
- Run forecasting validation on each file
- Compute MAE, RMSE, MAPE per timeframe
- Output results to JSON

### 2. Refactor: `simulate.py`

- **Remove** all real data scenario processing
- Keep ONLY synthetic scenario testing
- Fix SLA logic to check BEFORE scaling
- Update command-line args: `--scenarios-only` (not `--synthetic-only`)

### 3. Refactor: `cost/metrics.py`

- Clarify SLA computation happens at DECISION POINT
- Add pre-scaling SLA check

### 4. Refactor: `dashboard/app.py`

- Add data source selector:
  - "Autoscaling Tests" (synthetic scenarios only)
  - "Model Evaluation" (real data forecast metrics only)
- Update tabs based on data source

### 5. New File: `run_pipeline.py`

- Master orchestrator
- Execute both PHASE A (model eval) and PHASE B (scenario tests)
- Separate result files:
  - `results/model_evaluation.json`
  - `results/simulation_results.csv`

## Key Implementation Details

### SLA Violation Fix

**BEFORE (Wrong):**

```python
# Line runs AFTER scaling
new_pods, ...  = autoscaler.step(current_pods, ...)
sla_breached = actual_requests > new_pods * capacity_per_pod  # ❌ After scaling!
current_pods = new_pods
```

**AFTER (Correct):**

```python
# Check SLA BEFORE scaling
sla_breached_before = actual_requests > current_pods * capacity_per_pod  # ✓ Before scaling!

# Then apply scaling
new_pods, ... = autoscaler.step(current_pods, ...)

# Record both
records.append({
    "sla_breached_before_scaling": sla_breached_before,
    "pods_before": current_pods,
    "pods_after": new_pods,
    ...
})
current_pods = new_pods
```

### Data Separation

**Real Data (PHASE A - Model Evaluation):**

```python
def evaluate_models_on_real_data():
    real_data_dir = Path("processed_for_modeling_v2")

    for csv_file in real_data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)

        for model_type in ["hybrid", "lstm", "xgboost"]:
            metrics = evaluate_forecast_accuracy(
                model_type=model_type,
                data=df,
                horizon=1
            )
            # Save metrics
```

**Synthetic Data (PHASE B - Scenario Testing):**

```python
def test_autoscaling_scenarios():
    scenarios = generate_all_scenarios()  # ONLY synthetic!

    for scenario in scenarios:
        synthetic_load = scenario.load_series  # Generated, NOT real

        for strategy in ["REACTIVE", ...]:
            test_strategy_on_scenario(...)
```

## Verification Checklist

Before considering this complete:

- [ ] Real data files are NOT in simulation_results.csv
- [ ] Simulation results ONLY contain GRADUAL_INCREASE, SUDDEN_SPIKE, etc.
- [ ] SLA violations are NON-ZERO for SUDDEN_SPIKE scenario
- [ ] SLA violations are LOW for GRADUAL_INCREASE
- [ ] Dashboard shows "Autoscaling Tests" with synthetic data only
- [ ] Dashboard can show "Model Evaluation" metrics separately
- [ ] No hard-coded file paths
- [ ] No TODOs in code
- [ ] Code runs without errors

## Timeline

1. Create model_evaluation.py
2. Refactor simulate.py (remove real data, fix SLA)
3. Update dashboard.py (data source selection)
4. Create run_pipeline.py (master orchestrator)
5. Test and verify all checkpoints
