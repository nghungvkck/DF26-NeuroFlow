# DataFlow 2026: Autoscaling Optimization Pipeline

## Tổng quan

Hệ thống mô phỏng autoscaling hoàn chỉnh, dễ demo, gồm 3 pha:

1. **Phase A**: Đánh giá model dự báo trên dữ liệu thực
2. **Phase B**: Test autoscaling trên kịch bản synthetic
3. **Phase C**: Anomaly detection + cost analysis nâng cao

Mục tiêu: tối ưu **chi phí**, **SLA**, và **độ ổn định** khi tải biến động.

---

## Yêu cầu

- Python 3.9+ (khuyến nghị 3.11)
- Các thư viện: pandas, numpy, scikit-learn, statsmodels, plotly, streamlit, tensorflow (nếu dùng LSTM)

Nếu thiếu, cài nhanh:

```bash
pip install -r requirements.txt
```

---

## Demo nhanh (khuyến nghị)

### Bước 1: Chạy toàn bộ pipeline

```bash
python run_pipeline.py
```

Tạo ra các file kết quả trong thư mục results/.

### Bước 2: Mở dashboard

```bash
streamlit run dashboard/app.py
```

Mở trình duyệt: http://localhost:8501

---

## Demo chi tiết (theo pha)

### ✅ Phase A: Model Evaluation (dữ liệu thực)

Nguồn dữ liệu thực nằm ở:

```
data/real/
```

Chạy riêng Phase A:

```bash
python run_pipeline.py --phase-a-only
```

Kết quả:

- results/model_evaluation.json

### ✅ Phase B: Autoscaling Tests (synthetic)

Chạy riêng Phase B:

```bash
python run_pipeline.py --phase-b-only
```

Kết quả:

- results/simulation_results.csv
- results/metrics_summary.json

**Lưu ý:** capacity_per_pod mặc định = 100 để SLA violations thực tế.

### ✅ Phase C: Anomaly & Cost Analysis

Chạy riêng Phase C:

```bash
python run_pipeline.py --phase-c-only
```

Kết quả:

- results/anomaly_analysis.json
- results/cost_breakdown.json

---

## Dashboard: 3 chế độ hiển thị

### 1) Autoscaling Tests
- Load vs Forecast
- Pod timeline
- Cost analysis
- SLA violations
- Anomaly detection
- Advanced metrics (K8s HPA, AWS Auto Scaling)

### 2) Model Evaluation
- Best model theo timeframe (1m/5m/15m)
- MAE / RMSE / MAPE

### 3) Anomaly & Cost Analysis
- F1/Precision/Recall cho anomaly detection
- So sánh cost models (cloud, k8s, borg)
- Platform metrics (HPA, Auto Scaling)

---

## Cấu trúc thư mục chính

```
.
├── autoscaling/        # Policies + scenarios
├── anomaly/            # Anomaly detection + simulation
├── cost/               # Cost models + metrics
├── forecast/           # Forecasting models
├── data/real/          # Dữ liệu thực để đánh giá model
├── dashboard/          # Streamlit UI
├── simulate.py         # Simulation logic
├── run_pipeline.py     # Orchestrator
└── results/            # Output files
```

---

## Outputs quan trọng

- results/model_evaluation.json
- results/simulation_results.csv
- results/metrics_summary.json
- results/anomaly_analysis.json
- results/cost_breakdown.json
- results/pipeline_summary.json

---

## Lỗi thường gặp

### 1) Phase A báo "No real data files"
→ Kiểm tra thư mục [data/real](data/real) có dữ liệu hay chưa.

### 2) SLA violations luôn 0
→ Đảm bảo `capacity_per_pod = 100` (đã fix trong [simulate.py](simulate.py)).

### 3) Dashboard báo thiếu results
→ Chạy `python run_pipeline.py` trước.

---

## Mẹo demo ổn định

- Chạy Phase B trước để có dữ liệu hiển thị nhanh.
- Dùng dashboard tab **Anomaly Detection** để demo robustness.
- Dùng tab **Cost Models** để nói về tối ưu chi phí cloud.

---

## TL;DR (demo 30s)

```bash
python run_pipeline.py
streamlit run dashboard/app.py
```

---

Nếu cần mở rộng: xem các file trong folders autoscaling/, anomaly/, cost/, forecast/.
│   ├── hysteresis.py        # Anti-flapping mechanisms
│   └── scenarios.py         # Load scenario generators
├── cost/
│   ├── cost_model.py        # Cost computation
│   └── metrics.py           # Metrics collection & aggregation
├── forecast/
│   ├── base_forecast.py     # Forecaster interface
│   └── arima_forecaster.py  # ARIMA implementation
├── data/
│   ├── load_data.py         # Data loading utilities
│   └── *.csv                # Sample training data
├── anomaly/
│   ├── anomaly_detection.py # Anomaly detection (Z-score)
│   └── simulate_anomaly.py  # Anomaly injection
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── simulate.py              # Main simulation runner
├── README.md                # This file
└── results/                 # Output directory
    ├── simulation_results.csv    # Detailed results
    ├── metrics_summary.json      # Aggregated metrics
    └── strategy_comparison.json  # Cross-strategy comparison
```

---

## Results Interpretation

### Example Output (Gradual Increase Scenario):

```
GRADUAL_INCREASE:
Strategy    Cost      Avg Pods   SLA Viol   Events   Oscillations
REACTIVE    $1.74     2.1        0.0%       19       0
PREDICTIVE  $1.67     2.0        0.0%       1        0      ← BEST
CPU_BASED   $13.90    16.7       0.0%       32       0
HYBRID      $7.99     9.6        0.0%       34       0
```

**Key Insights:**

- **PREDICTIVE** excels: Fewer scaling events, lower cost
- **CPU_BASED** over-provisions: 16.7 pods vs 2.1 (unnecessary cost)
- **HYBRID** balances: More events but moderate cost
- **REACTIVE** stable: Responsive but 30+ scale decisions

---

## How to Use Each Component

### 1. Running Simulations

```python
from autoscaling.scenarios import generate_all_scenarios
from autoscaling.hybrid import HybridAutoscaler
from simulate import run_strategy_on_scenario

scenarios = generate_all_scenarios(duration=300)
autoscaler = HybridAutoscaler(capacity_per_server=500)

result = run_strategy_on_scenario(
    strategy_name="HYBRID",
    autoscaler=autoscaler,
    forecaster=forecaster,
    scenario=scenarios[0],
    capacity_per_pod=500
)

metrics = result['metrics']
objective = result['objective']
```

### 2. Adding Custom Scenarios

```python
from autoscaling.scenarios import Scenario
import numpy as np

# Create custom load pattern
load = np.linspace(100, 1000, 500) + np.random.normal(0, 50, 500)

scenario = Scenario(
    name="MY_SCENARIO",
    description="Custom load pattern",
    load_series=load,
    forecast_errors=np.random.normal(0.1, 0.1, 500)
)
```

### 3. Implementing Custom Policy

```python
class MyAutoscaler:
    def step(self, current_servers, requests, forecast_requests=None):
        # Your logic here
        decision = ...
        return new_servers, action, reason
```

---

## Performance Summary

### What Works Best:

✅ **Predictive** - Best cost & stability (1-3 events vs 30+)  
✅ **Hybrid** - Most robust to forecast errors  
✅ **Reactive** - Simple, reliable baseline  
❌ **CPU-Based** - Over-provisions by 5-8x

### Across Scenarios:

| Scenario         | Winner     | Key Insight                           |
| ---------------- | ---------- | ------------------------------------- |
| Gradual Increase | PREDICTIVE | Forecast accuracy high, fewer events  |
| Sudden Spike     | CPU_BASED  | Immediate detection via CPU threshold |
| Oscillating      | PREDICTIVE | Captures pattern, minimal flapping    |
| Traffic Drop     | REACTIVE   | Quick scale-down, good efficiency     |
| Forecast Error   | HYBRID     | Handles unreliability gracefully      |

---

## Dashboard Features

Access `http://localhost:8501` (after `streamlit run dashboard/app.py`):

1. **Load & Forecast Tab**
   - Actual vs predicted traffic
   - Forecast accuracy metrics
   - Anomaly detection

2. **Pod Timeline Tab**
   - Scaling decisions over time
   - Strategy comparison
   - Event counts

3. **Cost Analysis Tab**
   - Cumulative cost curves
   - Cost breakdown by strategy
   - Resource efficiency

4. **SLA Violations Tab**
   - Service breach timeline
   - Violation statistics
   - Impact quantification

5. **Metrics Comparison Tab**
   - Comprehensive metrics table
   - Multi-dimensional radar chart
   - Normalized performance scores

---

## Extension Points

### 1. New Autoscaling Strategy

Add to `autoscaling/new_policy.py`:

```python
class MyPolicy:
    def step(self, current_servers, requests, forecast_requests=None):
        # Implement decision logic
        return new_servers, action, reason
```

Then add to `simulate.py` `run_strategy_on_scenario()` function.

### 2. New Load Scenario

Add to `autoscaling/scenarios.py` ScenarioGenerator:

```python
@staticmethod
def my_scenario(...):
    load = ...  # Your pattern
    return Scenario(name="MY_SCENARIO", ..., load_series=load)
```

### 3. Custom Metrics

Extend `cost/metrics.py` MetricsCollector with new metrics.

### 4. Different Forecaster

Implement `forecast/base_forecast.py` interface with your model.

---

## Configuration

### Cost Parameters (simulate.py)

```python
cost_per_pod_per_hour = 0.05  # AWS EC2 pricing
capacity_per_pod = 500         # Requests/min per pod
step_minutes = 5.0             # Simulation timestep
```

### Autoscaler Parameters

Each strategy has configurable thresholds. See individual files:

- REACTIVE: `scale_out_th=0.7, scale_in_th=0.3`
- PREDICTIVE: `safety_margin=0.8, hysteresis=1`
- CPU_BASED: `cpu_critical_th=0.95`
- HYBRID: Multi-layer thresholds + forecast reliability

---

## Output Files

After `python simulate.py`:

**simulation_results.csv**

- One row per timestep per strategy per scenario
- Columns: time, actual_requests, forecast, pods, scaling_action, reason, sla_breached
- Use for detailed analysis and debugging

**metrics_summary.json**

- Aggregated metrics for all combinations
- Includes: cost, SLA, events, utilization statistics

**strategy_comparison.json**

- Average metrics across scenarios
- Shows which strategy wins for each metric

---

## Dependencies

```
pandas, numpy, scikit-learn, statsmodels, plotly, streamlit
```

Install: `pip install -r requirements.txt` (if available) or individually.

---

## FAQs

**Q: Why does PREDICTIVE have so few scaling events?**
A: It uses forecast to scale proactively to exact required capacity, avoiding reactive overshooting.

**Q: Why is CPU_BASED so expensive?**
A: CPU threshold (75%) requires more headroom than actual request capacity (70%), leading to over-provisioning.

**Q: Can I use this in production?**
A: The architecture is production-ready. Integrate with your Kubernetes/cloud platform's metrics and scaling API.

**Q: How accurate must the forecaster be?**
A: HYBRID adapts to forecast reliability. Even 10-20% errors are handled gracefully.

**Q: How do I handle multiple metrics (cost vs SLA)?**
A: Adjust weights in `objective.py` `compute_total_objective()`:

```python
weights = {'cost': 2.0, 'sla': 1.0, 'stability': 0.5}
```

---

## References

- ARIMA Forecasting: statsmodels documentation
- Autoscaling: Kubernetes VPA/HPA papers
- Multi-objective: NSGA-II optimization framework
- Metrics: NIST/ACME autoscaling benchmarks

---

**Last Updated:** January 30, 2026  
**Status:** Production-ready with comprehensive pipeline implementation
