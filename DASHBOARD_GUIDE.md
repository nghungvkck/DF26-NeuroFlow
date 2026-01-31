# 📊 Dashboard User Guide

## Overview

Interactive Streamlit dashboard với **3 visualization modes** để phân tích kết quả autoscaling pipeline.

---

## 🚀 Quick Start

### 1. Chạy Pipeline (tạo dữ liệu)
```bash
python run_pipeline.py
```

### 2. Khởi động Dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Truy cập
Mở browser tại: **http://localhost:8501**

---

## 🎯 Visualization Modes

### **Mode 1: Autoscaling Tests** (Phase B)

Test các chiến lược autoscaling trên synthetic scenarios.

#### **7 Tabs:**

1. **📊 Load & Forecast**
   - Traffic pattern vs predicted load
   - Forecast accuracy visualization
   - Error distribution

2. **📈 Pod Timeline**
   - Pod scaling decisions over time
   - Strategy comparison
   - Scaling events count

3. **💰 Cost Analysis**
   - Cumulative cost by strategy
   - Cost efficiency comparison
   - Resource utilization

4. **🚨 SLA Violations**
   - Service breach timeline
   - Violation statistics per strategy
   - Impact quantification

5. **📋 Metrics Comparison**
   - Comprehensive metrics table
   - Total cost, avg pods, violations
   - Scaling events, utilization stats

6. **🔴 Anomaly Detection** (NEW!)
   - Anomaly timeline with markers
   - Detection statistics by strategy
   - Anomaly types distribution
   - Scaling response to anomalies

7. **🎯 Advanced Metrics** (NEW!)
   - Kubernetes HPA metrics (CPU utilization, target breaches)
   - AWS Auto Scaling metrics (warm-up, cooldown effectiveness)
   - Cost model comparison (simple, cloud, K8s, Borg)

#### **Filters:**
- **Scenario**: Select load pattern (Gradual Increase, Sudden Spike, etc.)
- **Strategies**: Choose strategies to compare (multi-select)

---

### **Mode 2: Model Evaluation** (Phase A)

Đánh giá forecast models trên real historical data.

#### **Content:**
- **Best Models**: Top performing model per timeframe (1m, 5m, 15m)
- **Detailed Metrics**: MAE, RMSE, MAPE by model
- **Comparison**: LSTM vs XGBoost vs Hybrid

---

### **Mode 3: Anomaly & Cost Analysis** (Phase C - NEW!)

Advanced analysis về anomaly detection và cost optimization.

#### **3 Tabs:**

1. **🔴 Anomaly Detection**
   - Performance metrics by anomaly type:
     - DDoS Attack
     - Flash Sale
     - Service Failure
     - Thundering Herd
     - Multi-region Failover
   - F1 Score, Precision, Recall comparison
   - Detection rate visualization
   - Key insights và recommendations

2. **💰 Cost Models**
   - Cost comparison across 5 models:
     - Simple Linear (baseline)
     - Cloud Mixed (AWS/GCP/Azure style)
     - Kubernetes (node packing)
     - Borg Production
     - Borg Batch
   - Cost breakdown by component (reserved, on-demand, spot, startup)
   - Savings visualization
   - Kubernetes packing efficiency metrics

3. **📊 Platform Metrics**
   - **Kubernetes HPA**: CPU utilization, target breaches, trigger rate
   - **AWS Auto Scaling**: Warm-up ratio, cooldown effectiveness
   - Platform best practices và insights

---

## 📂 Required Files

Dashboard đọc dữ liệu từ `results/` directory:

### **Phase A (Model Evaluation):**
```
results/model_evaluation.json
```

### **Phase B (Autoscaling Tests):**
```
results/simulation_results.csv
results/metrics_summary.json
```

### **Phase C (Anomaly & Cost Analysis):**
```
results/anomaly_analysis.json
results/cost_breakdown.json
```

### **Summary:**
```
results/pipeline_summary.json
```

---

## 🎨 Dashboard Features

### **Interactive Charts:**
- ✅ Hover tooltips for detailed info
- ✅ Zoom and pan capabilities
- ✅ Legend toggle (click to hide/show)
- ✅ Export to PNG

### **Data Tables:**
- ✅ Sortable columns
- ✅ Full-width responsive design
- ✅ Formatted numbers (currency, percentages)

### **Filters:**
- ✅ Scenario selector
- ✅ Multi-strategy comparison
- ✅ Real-time updates

---

## 🔧 Troubleshooting

### **"⚠️ No results found"**
```bash
# Run pipeline first
python run_pipeline.py
```

### **"⚠️ Phase C results not found"**
```bash
# Run Phase C analysis
python run_pipeline.py --phase-c-only
```

### **"Advanced metrics not available"**
Chạy simulation với `enable_advanced_metrics=True` (default trong pipeline mới)

### **Dashboard không load:**
```bash
# Check terminal for errors
# Reinstall streamlit if needed:
pip install --upgrade streamlit plotly pandas
```

---

## 💡 Tips & Best Practices

### **Performance:**
- Load dữ liệu từ 1 scenario trước khi compare tất cả
- Use filter để giảm data points hiển thị
- Close unused tabs trong browser

### **Analysis:**
- **Compare strategies** trên same scenario để fair comparison
- **Check Anomaly Detection** tab để hiểu scaling behavior trong extreme events
- **Review Cost Models** để tìm cost optimization opportunities

### **Interpretation:**
- **F1 > 0.8**: Excellent anomaly detection
- **Packing Efficiency > 80%**: Good Kubernetes node utilization
- **Cooldown Effectiveness > 70%**: AWS Auto Scaling working well
- **Savings > 30%**: Significant cost optimization potential

---

## 📊 Example Analysis Workflow

1. **Start with Autoscaling Tests**
   - Select "GRADUAL_INCREASE" scenario
   - Compare all 4 strategies
   - Check Metrics Comparison tab

2. **Deep Dive into Anomaly Detection**
   - Switch to Tab 6 - Anomaly Detection
   - Identify anomaly patterns
   - Check scaling response rates

3. **Analyze Cost Optimization**
   - Switch to Mode 3 - Anomaly & Cost Analysis
   - Review Cost Models tab
   - Compare savings across models

4. **Platform-Specific Tuning**
   - Check Advanced Metrics (Tab 7)
   - Review K8s HPA metrics
   - Optimize AWS cooldown settings

---

## 🎯 Key Metrics to Watch

### **Autoscaling Performance:**
- Total Cost: Lower is better
- SLA Violations: Should be 0 or minimal
- Scaling Events: Fewer = more stable
- Mean Utilization: 70-85% is optimal

### **Anomaly Detection:**
- F1 Score: > 0.7 is good, > 0.8 is excellent
- Precision: Minimize false positives
- Recall: Catch all real anomalies

### **Cost Optimization:**
- Savings vs Simple: Target > 20%
- Packing Efficiency: Target > 75%
- Wasted Capacity: Target < 25%

---

## 🚀 Next Steps

1. **Experiment with different scenarios** trong Autoscaling Tests
2. **Compare cost models** trong Anomaly & Cost Analysis
3. **Tune strategies** based on metrics
4. **Export charts** for presentations
5. **Share insights** với team

---

**Dashboard Version:** 3.0 (với Anomaly Detection & Cost Analysis)
**Last Updated:** January 31, 2026
