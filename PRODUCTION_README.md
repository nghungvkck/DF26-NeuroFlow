# PRODUCTION GUIDE - Getting Started

## 🚀 Quick Start

### 1. Run HYBRID Autoscaling Pipeline

```bash
# 15-minute intervals (recommended - most realistic)
python run_hybrid_pipeline.py --timeframe 15m

# Or other timeframes
python run_hybrid_pipeline.py --timeframe 5m
python run_hybrid_pipeline.py --timeframe 1m
```

**Duration**: ~30 seconds for 908 timesteps (9.5 days of data)

**Output**:
```
Results saved to: results/hybrid_production/
├── hybrid_results_15m.csv      # Detailed metrics per timestep
└── hybrid_summary_15m.json     # Summary: cost, SLA, scaling events
```

### 2. View Dashboard

```bash
streamlit run dashboard/app.py
```

Opens interactive visualization at `http://localhost:8501`

### 3. Check Results

```bash
# View cost summary
cat results/hybrid_production/hybrid_summary_15m.json

# View detailed results
head -20 results/hybrid_production/hybrid_results_15m.csv
```

---

## 📊 Expected Results

### Metrics Summary

```json
{
  "strategy": "HYBRID",
  "timeframe": "15m",
  "total_cost": 13.62,
  "cost_breakdown": {
    "reserved": 13.62,
    "spot": 0.0,
    "on_demand": 0.0
  },
  "sla_violations": 0,
  "slo_violations": 0,
  "scaling_events": 0,
  "avg_pods": 2.0,
  "max_pods": 2,
  "max_cpu": 0.164
}
```

### Interpretation

- **Total Cost: $13.62** → Low cost (baseline only, no spikes)
- **Reserved: $13.62** → All cost from 2 pods always-on
- **SLA Violations: 0** → Perfect reliability (CPU never > 95%)
- **Avg Pods: 2.0** → Stayed at minimum (low traffic)
- **Expected with spikes**: $57.79 (Phase B.5 analysis)

---

## 🏗️ Architecture Overview

### HYBRID Autoscaler (4-Layer Decision)

```
Request Count → Layer 0 (Anomaly?) → Layer 1 (Emergency?) 
                                      ↓
                                    Layer 2 (Predictive?) 
                                      ↓
                                    Layer 3 (Reactive)
                                      ↓
                                 Pod Count Decision
```

### Cost Model (3-Tier Pricing)

```
0-2 pods    → Reserved only ($0.03/pod/hour)
3+ pods     → Reserved + Spot (70%) + On-demand (30%)
             Spot: $0.015/hour | On-demand: $0.05/hour
```

---

## 🔧 Configuration

### Adjust Autoscaler Parameters

Edit `run_hybrid_pipeline.py`:

```python
pipeline = HybridPipeline(
    timeframe="15m",              # 1m, 5m, or 15m
    capacity_per_server=5000,     # Requests per pod per minute
    min_servers=2,                # Minimum pods (reserved)
    max_servers=20                # Maximum pods (cost ceiling)
)
```

### Adjust Cost Model

Edit `run_hybrid_pipeline.py`:

```python
self.cost_model = CloudCostModel(
    on_demand_cost=0.05,          # $/pod/hour
    reserved_cost=0.03,           # 40% cheaper
    spot_cost=0.015,              # 70% cheaper
    startup_cost=0.001,           # Cold start
    reserved_capacity=2           # Baseline pods
)
```

### Adjust HYBRID Layers

Edit `autoscaling/hybrid_optimized.py`:

```python
# Layer 0: Anomaly detection sensitivity
self.anomaly_threshold = 2  # 2/4 voting (lower = more sensitive)

# Layer 1: Emergency threshold
self.emergency_threshold = 0.95  # CPU > 95% triggers scale-up

# Layer 2: Predictive safety margin
self.forecast_margin = 1.8  # 80% buffer (higher = conservative)

# Layer 3: Reactive thresholds
self.scale_out_threshold = 0.70   # CPU > 70% scale up
self.scale_in_threshold = 0.30    # CPU < 30% scale down
```

---

## 📁 Workspace Structure

```
dataFlow-2026/                  (Production folder)
├── run_hybrid_pipeline.py       ⭐ Main entry point
├── requirements.txt             (Python dependencies)
├── PRODUCTION_README.md         (This file)
├── HYBRID_DEPLOYMENT.md         (Deployment details)
├── COST_MODEL_SELECTION.md      (Cost analysis)
├── QUICK_REFERENCE.md           (Tips & tricks)
│
├── autoscaling/
│   ├── hybrid_optimized.py      ⭐ HYBRID strategy (500+ lines)
│   ├── cpu_based.py             (Alternative strategy)
│   ├── reactive.py              (Alternative strategy)
│   ├── predictive.py            (Alternative strategy)
│   └── ...
│
├── cost/
│   ├── cost_model.py            ⭐ CloudCostModel (production)
│   └── metrics.py               (Cost tracking)
│
├── forecast/
│   ├── model_forecaster.py      (Forecast predictions)
│   ├── model_base.py            (Base classes)
│   └── ...
│
├── data/
│   ├── real/
│   │   ├── test_1m_autoscaling.csv
│   │   ├── test_5m_autoscaling.csv
│   │   └── test_15m_autoscaling.csv  ⭐ Main test data
│   └── ...
│
├── models/
│   ├── xgboost_15m_model.json
│   ├── xgboost_15m_predictions.csv  ⭐ Pre-computed forecast
│   ├── lstm_15m_best.keras
│   └── ...
│
├── dashboard/
│   └── app.py                   (Streamlit visualization)
│
├── evaluation/
│   └── cost_report_generator.py (Cost reporting)
│
├── anomaly/
│   ├── anomaly_detection.py     (Spike/DDoS detection)
│   └── ...
│
├── results/
│   └── hybrid_production/
│       ├── hybrid_results_15m.csv
│       └── hybrid_summary_15m.json
│
└── ARCHIVE/                     (Legacy files)
    ├── documentation/           (Old .md files)
    └── legacy_scripts/          (Old scripts)
```

---

## ✅ Workflow

### Phase 1: Run Pipeline

```bash
python run_hybrid_pipeline.py --timeframe 15m
```

**What happens**:
1. Loads test data (908 timesteps, 9.5 days)
2. Loads pre-computed predictions
3. Runs HYBRID autoscaler for each timestep
4. Calculates cost using CloudCostModel
5. Saves results to `results/hybrid_production/`

**Time**: ~30 seconds

### Phase 2: Analyze Results

```bash
cat results/hybrid_production/hybrid_summary_15m.json
```

**Key metrics**:
- Total cost
- SLA/SLO violations
- Pod scaling events
- CPU statistics

### Phase 3: Visualize (Optional)

```bash
streamlit run dashboard/app.py
```

**Interactive dashboard**:
- Cost trends
- Pod scaling patterns
- CPU utilization
- Violation timeline

---

## 🎯 Key Files Explained

### run_hybrid_pipeline.py
**Purpose**: Production entry point  
**What it does**:
- Initializes HybridAutoscalerOptimized
- Initializes CloudCostModel
- Loads test data and predictions
- Runs simulation for each timestep
- Generates cost breakdown and report

**Key config**:
```python
HybridPipeline(
    timeframe="15m",           # Test timeframe
    min_servers=2,             # Matches cost model reserved_capacity
    max_servers=20             # Cost ceiling
)
```

### autoscaling/hybrid_optimized.py
**Purpose**: HYBRID autoscaler implementation  
**What it does**:
- Layer 0: Anomaly detection (4-method ensemble)
- Layer 1: Emergency detection (CPU > 95%)
- Layer 2: Predictive scaling (forecast-based)
- Layer 3: Reactive scaling (request-based)

**Example usage**:
```python
autoscaler = HybridAutoscalerOptimized(
    capacity_per_server=5000,
    min_servers=2,
    max_servers=20
)

new_servers, action, metrics = autoscaler.step(
    current_servers=2,
    requests=1200,
    forecast_requests=1350
)
```

### cost/cost_model.py
**Purpose**: Multi-tier cloud cost model  
**What it does**:
- Tracks reserved baseline cost
- Allocates burst as 70% spot + 30% on-demand
- Calculates cold start penalties
- Provides cost breakdown

**Example usage**:
```python
cost_model = CloudCostModel(
    reserved_capacity=2,
    on_demand_cost=0.05,
    reserved_cost=0.03,
    spot_cost=0.015
)

cost, breakdown = cost_model.compute_step_cost(
    pod_count=5,
    step_hours=0.25  # 15 minutes
)
```

---

## 🔍 Troubleshooting

### Problem: Cost seems too low

**Reason**: Test data has no spikes (steady 1100-1200 requests)  
**Solution**: Check Phase B.5 analysis for DDoS test results ($57.79)

### Problem: No scaling events

**Reason**: Load is below max capacity for all 908 timesteps  
**Expected**: Avg pods = 2 (minimum), max pods = 2 (no bursts)

### Problem: Dashboard not loading

**Solution**:
```bash
# Install Streamlit first
pip install streamlit

# Then run
streamlit run dashboard/app.py
```

### Problem: Forecast predictions missing

**Reason**: Pre-computed predictions file may not exist  
**Solution**: Check `models/xgboost_15m_predictions.csv` exists  
**Workaround**: Pipeline runs fine without forecasts (uses default)

---

## 📈 Performance Benchmarks

### HYBRID Strategy Performance (Phase B.5)

| Metric | Value | Status |
|--------|-------|--------|
| Total Cost (15-day) | $57.79 | ✅ BEST |
| SLA Violations | 14 | ✅ BEST |
| Spike Response Time | 4.7-5.5 min | ✅ FASTEST |
| Avg Pods | 3.2 | ✅ Optimal |
| Max Pods | 12 | ✅ Cost-effective |

### Compared to Alternatives

| Strategy | Cost | SLA Violations | Response Time |
|----------|------|----------------|----------------|
| HYBRID | $57.79 | 14 | 4.7-5.5 min ✅ |
| REACTIVE | $59.47 | 41 | 7-12 min |
| PREDICTIVE | $65.83 | 27 | 5-8 min |
| CPU_BASED | $171.26 | 19 | 6-9 min |

---

## 🚀 Next Steps

1. **Validate Results**: Review `hybrid_summary_15m.json` metrics
2. **Test Different Timeframes**: Try `--timeframe 5m` or `--timeframe 1m`
3. **Adjust Parameters**: Tweak thresholds in `hybrid_optimized.py`
4. **Monitor Dashboard**: Visualize trends with Streamlit
5. **Compare Costs**: Use COST_MODEL_SELECTION.md for reference

---

## 📚 Documentation

- **HYBRID_DEPLOYMENT.md** - Detailed deployment & configuration
- **COST_MODEL_SELECTION.md** - Cost analysis & model comparison
- **QUICK_REFERENCE.md** - Common tasks & tips
- **README.md** - Project overview

---

## ⚠️ Known Limitations

1. **Test data is synthetic**: No real DDoS spikes, steady traffic ~1100-1200 requests
2. **Predictions pre-computed**: Real-time forecast disabled (uses existing predictions)
3. **Single timeframe test**: 15-minute intervals, 9.5-day duration
4. **Fixed pod capacity**: 5000 requests/pod/minute (not adjusted per test)

---

## 💡 Tips

- Use `--timeframe 15m` for production (most realistic)
- Cost breakdown shows allocation: reserved + spot + on-demand
- Monitor `sla_violations` for SLA compliance
- Check `scaling_events` to understand autoscaler behavior
- Use dashboard for visual analysis of trends

---

## 🎓 Learning Resources

- **autoscaling/hybrid_optimized.py** - Study 4-layer decision hierarchy
- **cost/cost_model.py** - Understand multi-tier pricing
- **COST_MODEL_SELECTION.md** - Learn why CloudCostModel chosen
- **results/hybrid_production/** - Analyze actual results

---

**Ready for production!** 🎉

```bash
python run_hybrid_pipeline.py --timeframe 15m
```
