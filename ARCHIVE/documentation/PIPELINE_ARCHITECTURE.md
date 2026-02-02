# Pipeline Analysis & Architecture

## 📊 Tổng Quan Pipeline

Hệ thống được thiết kế với **3 pha** xử lý:

```
┌─────────────────────────────────────────────────────────────┐
│          AUTOSCALING OPTIMIZATION PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE A: MODEL EVALUATION (Dữ liệu thực)                 │
│  ────────────────────────────────────────                 │
│  Input: data/real/*.csv (lịch sử traffic)                │
│  Processing:                                              │
│    ├─ Load real data                                      │
│    ├─ Evaluate LSTM, XGBoost, Hybrid models              │
│    └─ Compute MAE, RMSE, MAPE metrics                    │
│  Output: results/model_evaluation.json                    │
│                                                             │
│  ▼                                                           │
│                                                             │
│  PHASE B: AUTOSCALING SCENARIO TESTING (Dữ liệu synthetic) │
│  ─────────────────────────────────────────────────────────│
│  Input: Generated scenarios (5 types)                      │
│  Processing:                                              │
│    ├─ Generate 5 kịch bản tải (gradual, spike, etc)     │
│    ├─ Test 4 strategies (reactive, predictive, etc)      │
│    ├─ For each combo: Run simulate.py                    │
│    │  ├─ Dự báo tải (forecaster)                        │
│    │  ├─ Quyết định scaling (autoscaler)                │
│    │  ├─ Tính metrics (cost, SLA, stability)            │
│    │  └─ Phát hiện anomaly                               │
│    └─ Aggregate kết quả                                  │
│  Output: results/simulation_results.csv                   │
│          results/metrics_summary.json                     │
│                                                             │
│  ▼                                                           │
│                                                             │
│  PHASE C: ADVANCED ANALYSIS (Anomaly & Cost)              │
│  ─────────────────────────────────────────────           │
│  Input: Simulation results                                │
│  Processing:                                              │
│    ├─ Test anomaly detection (DDoS, flash sales)         │
│    ├─ Evaluate cost models (K8s, AWS, GCP)              │
│    └─ Measure platform metrics                           │
│  Output: results/anomaly_analysis.json                    │
│          results/cost_breakdown.json                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Chi Tiết Quy Trình Trong Mỗi Pha

### PHASE A: Model Evaluation (forecast/model_evaluation.py)

**Mục đích:** Đánh giá độ chính xác của các ML models trên dữ liệu thực.

```python
# Quy trình:
1. Load dữ liệu thực từ data/real/*.csv
   └─ train_1m.csv, train_5m.csv, train_15m.csv (dữ liệu huấn luyện)
   └─ test_1m.csv, test_5m.csv, test_15m.csv (dữ liệu kiểm tra)

2. Cho mỗi timeframe (1m, 5m, 15m):
   ├─ Load model đã được lưu từ models/
   │  └─ LSTM: lstm_1m.keras
   │  └─ XGBoost: xgboost_1m_model.json
   │  └─ Hybrid: (LSTM + residual learning)
   │
   ├─ Dự báo trên test set
   │  └─ Tính MAE, RMSE, MAPE
   │
   └─ Xác định mô hình tốt nhất

3. Output: model_evaluation.json
   {
     "1m": {"best_model": "LSTM", "mae": 5.2, "rmse": 7.1},
     "5m": {"best_model": "XGBoost", "mae": 12.3, "rmse": 15.6},
     ...
   }
```

---

### PHASE B: Autoscaling Testing (simulate.py)

**Mục đích:** Test các autoscaling strategies trên kịch bản tế.

#### B1: Tạo Scenarios (autoscaling/scenarios.py)

```python
# 5 kịch bản test:
1. GRADUAL_INCREASE
   └─ Load tăng dần: từ 50 → 300 requests/s
   └─ Kiểm tra policy tăng pod từ từ

2. SUDDEN_SPIKE
   └─ Load tăng đột ngột: 100 → 400 requests/s
   └─ Kiểm tra khả năng phản ứng nhanh

3. OSCILLATING
   └─ Load dao động: 100 ↔ 300 ↔ 100 requests/s
   └─ Kiểm tra tránh "flapping" (scaling liên tục)

4. TRAFFIC_DROP
   └─ Load giảm đột ngột: 300 → 50 requests/s
   └─ Kiểm tra scale-down hiệu quả

5. FORECAST_ERROR_TEST
   └─ Load không theo pattern dự báo
   └─ Kiểm tra khả năng chịu lỗi forecast
```

#### B2: Simulate Core Loop (simulate.py - run_strategy_on_scenario)

```python
for t in range(len(load_series)):  # Cho mỗi timestep
    
    # 1. LẤY DỮ LIỆU THỰC
    actual_requests = load_series[t]
    
    # 2. PHÁT HIỆN ANOMALY
    is_anomaly = anomaly_detector.detect(actual_requests)
    
    # 3. DỰ BÁO (Forecasting)
    forecast = forecaster.predict(history, horizon=1)
    
    # 4. QUYẾT ĐỊNH SCALING (Autoscaler)
    new_pods, action, reason = autoscaler.step(
        current_pods=current_pods,
        requests=actual_requests,
        forecast=forecast
    )
    
    # 5. TÍNH METRICS
    metrics.record(
        t=t,
        pods=new_pods,
        requests=actual_requests,
        scaling_action=action,
        sla_before_scaling=sla_violated_before
    )
    
    # 6. CẬP NHẬT TRẠNG THÁI
    current_pods = new_pods
    records.append({
        'timestamp': t,
        'pods': new_pods,
        'requests': actual_requests,
        'forecast': forecast,
        'action': action,
        ...
    })

# 7. TỔNG HỢP KẾT QUẢ
return {
    'strategy': strategy_name,
    'scenario': scenario_name,
    'records': records,
    'metrics': metrics.aggregate(),
    'total_cost': sum(pods_history),
    'sla_violations': violations_count,
    'scaling_events': actions_count
}
```

#### B3: 4 Autoscaling Strategies

```python
STRATEGY 1: REACTIVE (autoscaling/reactive.py)
└─ Phản ứng với tải hiện tại
└─ if requests > threshold: scale_up()
└─ Đơn giản, độ trễ cao

STRATEGY 2: PREDICTIVE (autoscaling/predictive.py)
└─ Dự báo tải trong tương lai
└─ if forecast_requests > threshold: scale_up()
└─ Proactive, cần dự báo chính xác

STRATEGY 3: CPU_BASED (autoscaling/cpu_based.py)
└─ Dựa trên CPU utilization
└─ if cpu_utilization > 70%: scale_up()
└─ Truyền thống, có thể over-provision

STRATEGY 4: HYBRID (autoscaling/hybrid.py)
└─ Kết hợp nhiều yếu tố: requests, forecast, CPU, history
└─ Thích ứng, có anti-flapping mechanism
└─ Tối ưu nhất cho production
```

#### B4: Tính Metrics (cost/metrics.py)

```python
Cho mỗi strategy × scenario:

1. COST METRICS
   ├─ total_cost = sum(pod_count) × cost_per_hour × time_interval
   └─ avg_pods = mean(pod_history)

2. SLA METRICS
   ├─ sla_violations = count(requests > capacity)
   ├─ sla_violation_rate = violations / total_steps
   └─ time_to_handle = thời gian để xử lý SLA

3. STABILITY METRICS
   ├─ scaling_events = count(pods changed)
   ├─ oscillation_count = count(scale up then down)
   └─ pod_change_rate = |pods_t - pods_t-1|

4. K8S HPA METRICS
   ├─ cpu_utilization = requests / capacity
   ├─ target_tracking_events = count(target breached)
   └─ warm_up_time = thời gian instance warming

5. AWS AUTO SCALING METRICS
   ├─ cooldown_effectiveness = pods released / pods_max
   └─ warm_up_overhead = extra capacity during warm-up
```

---

### PHASE C: Advanced Analysis (anomaly + cost)

```python
1. ANOMALY DETECTION TESTING
   ├─ Inject anomalies: DDoS, flash sales, failovers
   ├─ Test detection: Z-score, IQR, rate-change
   └─ Measure detection accuracy

2. COST MODEL TESTING
   ├─ Kubernetes cost model
   ├─ AWS EC2 cost model
   ├─ Google Cloud cost model
   └─ Spot instances (dynamic pricing)

3. OUTPUT
   └─ results/anomaly_analysis.json
   └─ results/cost_breakdown.json
```

---

## ⚙️ Objective Function (Multi-Objective Optimization)

```python
# autoscaling/objective.py

MINIMIZE: w_cost × Cost + w_sla × SLA + w_stability × Stability

Chi tiết từng component:

1. COST COMPONENT
   Cost = Σ(pods_t × cost_per_hour × step_hours)
   └─ Mục tiêu: Sử dụng ít pods nhất

2. SLA COMPONENT
   SLA_t = 1 nếu requests_t > pods_t × capacity
   SLA_cost = Σ(SLA_t) × penalty
   └─ Mục tiêu: Tránh SLA violations

3. STABILITY COMPONENT
   Scaling_events = Σ(|action_t|)
   Stability_cost = Σ(scaling_events) × penalty
   └─ Mục tiêu: Tránh flapping (scaling liên tục)

4. AGGREGATION (mặc định weights = {cost: 1, sla: 1, stability: 1})
   Total = 1.0 × Cost + 1.0 × SLA + 1.0 × Stability
```

---

## 🚨 CÁC VẤNĐỀ ĐƯỢC PHÁT HIỆN & TÌNH TRẠNG

### Vấn Đề 1: SLA Violation Logic (Đã fix ✅)
**Problem:** SLA được tính SAU khi scaling, nên luôn = 0
**Nguyên nhân:** Tính toán SLA sau khi pods đã tăng
**Fix:** Thêm metric `sla_before_scaling` để track SLA TRƯỚC decision
**Tình trạng:** ✅ Đã fix trong MetricsCollector

### Vấn Đề 2: Real Data vs Synthetic Data Mixing (Đã fix ✅)
**Problem:** Autoscaling tests chạy trên dữ liệu thực, gây nhầm lẫn
**Nguyên nhân:** Không tách rõ PHASE A và PHASE B
**Fix:** Tạo model_evaluation.py (PHASE A) riêng, simulate.py chỉ chạy synthetic
**Tình trạng:** ✅ Đã fix trong run_pipeline.py

### Vấn Đề 3: Forecaster Integration (Đã fix ✅)
**Problem:** Sử dụng ARIMA không tối ưu
**Nguyên nhân:** ARIMA chậy và không chính xác cho short-term forecast
**Fix:** Thay bằng ML models (LSTM, XGBoost, Hybrid)
**Tình trạng:** ✅ Đã fix trong model_forecaster.py

### Vấn Đề 4: Capacity Per Pod (Cần review ⚠️)
**Current value:** capacity_per_pod = 100 requests/s
**Problem:** Giá trị này có hợp lý không? Test data chạy ~300 requests max
**Impact:** Nếu quá thấp → quá nhiều SLA, quá cao → không thấy SLA
**Recommendation:** Xem dữ liệu thực để calibrate

---

## 📈 Data Flow (Chi Tiết)

```
DATA SOURCES
├── data/real/
│   ├── train_1m.csv ──┐
│   ├── train_5m.csv  ─┼─→ PHASE A: ModelEvaluator
│   ├── train_15m.csv ─┤   │
│   ├── test_1m.csv   ─┤   └─→ model_evaluation.json
│   ├── test_5m.csv   ─┤
│   └── test_15m.csv ──┘

└── autoscaling/scenarios.py
    └─→ generate_all_scenarios()
        └─→ PHASE B: simulate.py
            ├─ run_strategy_on_scenario()
            ├─ run_strategy_on_scenario()
            └─ run_strategy_on_scenario() × (4 strategies × 5 scenarios)
            
FORECASTING
├── models/lstm_*.keras
├── models/xgboost_*.json
└── forecast/model_forecaster.py
    └─→ Predict load for next step

AUTOSCALING DECISIONS
├── autoscaling/reactive.py      → Decision making
├── autoscaling/predictive.py    │
├── autoscaling/cpu_based.py     │
└── autoscaling/hybrid.py        ↓

METRICS COLLECTION
└── cost/metrics.py
    ├── Cost: Σ(pods × cost_per_hour)
    ├── SLA: violations count
    ├── Stability: scaling events
    └── Platform metrics (K8s, AWS)
    
OUTPUT
├── results/simulation_results.csv      (Detailed records)
├── results/metrics_summary.json        (Aggregated metrics)
├── results/strategy_comparison.json    (Winner analysis)
├── results/model_evaluation.json       (PHASE A)
└── results/anomaly_analysis.json       (PHASE C)
```

---

## 🔧 Cách Chạy Pipeline

```bash
# RUN ALL PHASES
python run_pipeline.py

# RUN SPECIFIC PHASE
python run_pipeline.py --phase-a-only    # Model evaluation only
python run_pipeline.py --phase-b-only    # Autoscaling tests only
python run_pipeline.py --phase-c-only    # Anomaly & cost analysis

# RUN SIMULATE DIRECTLY (PHASE B chỉ)
python simulate.py

# VIEW DASHBOARD
streamlit run dashboard/app.py
```

---

## 📊 Output Files

```
results/
├── model_evaluation.json
│   └─ Accuracy metrics (MAE, RMSE, MAPE) per model per timeframe
│
├── simulation_results.csv
│   └─ Row = 1 timestep × 1 scenario × 1 strategy
│   └─ Columns: scenario, strategy, t, pods, requests, forecast, cost, sla, action
│
├── metrics_summary.json
│   └─ Aggregated metrics per strategy:
│   │  {
│   │    "REACTIVE": {
│   │      "total_cost": 1.74,
│   │      "avg_pods": 2.1,
│   │      "sla_violations": 0,
│   │      "scaling_events": 19
│   │    }
│   │  }
│
├── strategy_comparison.json
│   └─ Win count & ranking per strategy
│
├── anomaly_analysis.json
│   └─ Anomaly detection results (PHASE C)
│
└── cost_breakdown.json
    └─ Cost by platform (K8s, AWS, GCP)
```

---

## ✅ Architecture Assessment

### Điểm Tốt ✅

1. **Clear separation**: PHASE A, B, C tách rõ
2. **Reproducible**: Synthetic scenarios, fixed random seed
3. **Comprehensive metrics**: Cost, SLA, stability, platform-specific
4. **Multi-strategy comparison**: Fair comparison giữa policies
5. **Real data integration**: Phase A đánh giá trên dữ liệu thực
6. **Production-ready**: Anti-flapping, multi-objective, hybrid policy

### Điểm Cần Cải Thiện ⚠️

1. **Capacity Per Pod**: Cần calibrate dựa trên real data
2. **Cost Parameters**: Needs tuning based on actual cloud costs
3. **Weights in Objective**: Currently equal, có thể cần custom weights
4. **Forecaster Error Handling**: Fallback to heuristic nếu model fail
5. **Anomaly Threshold**: Z-score = 3.0 có quá cao?

---

**Kết Luận:** Pipeline rất tốt, logic hợp lý. Chỉ cần tuning parameters.
