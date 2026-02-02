# Autoscaling Demo - Quick Start Guide

Complete demo of HYBRID autoscaler with Streamlit dashboard and FastAPI recommendation engine.

## Overview

**Two components working together:**

1. **FastAPI Server** (Port 8000)
   - REST API endpoint: `/recommend-scaling`
   - Analyzes scaling decisions with detailed reasoning
   - Explains why scaling up/down/no-change

2. **Streamlit Dashboard** (Port 8501)
   - Interactive visualization of simulated load & autoscaling
   - 4 charts: Load timeline, Pod scaling, Thresholds, Cost vs SLA
   - Call FastAPI endpoint to get recommendations
   - Displays detailed decision layers and reasoning

## Setup

### Requirements

```bash
pip install fastapi uvicorn pydantic streamlit plotly pandas numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Running the Demo

### Step 1: Start the FastAPI Server

Open a terminal and run:

```bash
python api_server.py
```

Expected output:
```
╔═══════════════════════════════════════════════════════════╗
║  AUTOSCALING RECOMMENDATION API                           ║
║  Listening on http://localhost:8000                       ║
╚═══════════════════════════════════════════════════════════╝

Endpoints:
- GET  http://localhost:8000/              (Documentation)
- GET  http://localhost:8000/health        (Health check)
- POST http://localhost:8000/recommend-scaling (Main endpoint)
```

### Step 2: Start the Streamlit Dashboard

Open a **second** terminal and run:

```bash
streamlit run dashboard_demo.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 3: Use the Dashboard

1. Open http://localhost:8501 in your browser
2. You'll see:
   - **Load Timeline**: How many requests come in, with forecast vs actual
   - **Pod Scaling**: Number of pods over time (blue line) with scaling events (triangles)
   - **Thresholds**: CPU utilization vs SLA/SLO/Target thresholds
   - **Cost vs SLA**: Cost vs compliance percentage

3. Scroll down to **"Scaling Recommendation (FastAPI)"** section
4. Use the slider to pick any timestep
5. Click **"Get Recommendation"** button
6. See:
   - Current pods, Recommended pods, Action (SCALE UP/DOWN/NO CHANGE)
   - **Decision Layers**: 
     - Layer 0: Anomaly Detection (spike detection)
     - Layer 1: Emergency (SLA breach detection)
     - Layer 2: Predictive (forecast-based)
     - Layer 3: Reactive (load-based)
   - **Explanation**: Human-readable reasoning
   - **Cost Impact**: How cost changes with recommendation

## Example Scenarios

### Scenario 1: Load Spike

1. Move slider to timestep 100-110 (first spike: 1500 requests)
2. Click "Get Recommendation"

**Expected output:**
- Anomaly detected at spike
- Predictive layer recommends scaling UP
- Cost increases, but SLA compliance improves

### Scenario 2: Stable Load

1. Move slider to timestep 50 (baseline load)
2. Click "Get Recommendation"

**Expected output:**
- No anomaly
- System stable
- Recommendation: NO CHANGE
- Cost unchanged

### Scenario 3: High Load + Forecast Spike

1. Move slider to timestep 200-220 (second spike: 2000 requests)
2. Click "Get Recommendation"

**Expected output:**
- Large spike with forecast predicting it
- Predictive layer: "Forecast shows 2000 reqs → need 5 pods"
- Scale UP recommendation
- Cost increases

## What's Happening

### Data Flow

```
Dashboard
    ↓
Select timestep (load, forecast, current pods)
    ↓
User clicks "Get Recommendation"
    ↓
POST /recommend-scaling to FastAPI
    {
        "current_pods": 3,
        "requests": 15000,
        "forecast": 18000,
        "capacity_per_pod": 5000
    }
    ↓
FastAPI Analyzer (HYBRID 4-layer)
    Layer 0: Anomaly Detection
    Layer 1: Emergency (SLA breach)
    Layer 2: Predictive (forecast-based)
    Layer 3: Reactive (utilization-based)
    ↓
Return Recommendation
    {
        "recommended_pods": 4,
        "action": "scale-up",
        "reasons": [...],
        "explanation": "...",
        "estimated_cost_impact": {...}
    }
    ↓
Dashboard displays with formatting
    - Decision layers in expanders
    - Clear explanation
    - Cost impact metrics
```

### HYBRID Autoscaler Logic

The API uses a **4-layer decision hierarchy** (like a firewall):

**Layer 0: Anomaly Detection**
- Detects unusual spikes (>50% deviation from forecast)
- If detected: Alert to prepare for scale

**Layer 1: Emergency (SLA Breach)**
- Monitors current utilization
- If utilization > 95%: **CRITICAL - Scale UP immediately**
- Highest priority, highest confidence

**Layer 2: Predictive (Forecast-based)**
- Uses ML forecast to predict future load
- Proactively scales before spike arrives
- Better user experience, lower latency

**Layer 3: Reactive (Load-based)**
- Traditional scaling on current load
- Scale UP if utilization > 80%
- Scale DOWN if utilization < 30%
- Fallback when predictive not triggered

### Cost Model

**3-tier pricing** (CloudCostModel):

- **Reserved**: First 2 pods @ $0.03/pod/hour (40% discount)
- **Spot**: 70% of burst pods @ $0.015/pod/hour (70% discount)
- **On-Demand**: 30% of burst pods @ $0.05/pod/hour (full price fallback)

Example: 4 pods for 1 hour
```
Reserved: 2 pods × $0.03 = $0.06
Spot:     2 pods × 0.7 × $0.015 = $0.021
On-Demand: 2 pods × 0.3 × $0.05 = $0.03
Total: $0.111/hour
```

## Simulated Data

The dashboard generates realistic simulation data:

```python
def generate_demo_data(num_steps=288):
    # 24 hours × 15-min intervals = 288 timesteps
    base_load = 1200 requests
    hourly_pattern = Daily sine curve (peak at 12 PM)
    noise = Random fluctuation (±5%)
    spike_1 = 1500 requests (t=100-110, ~25 hours into day)
    spike_2 = 2000 requests (t=200-220, ~50 hours into day)
    
    forecast = ML prediction (slightly offset from actual)
    pods = HYBRID scaling decisions
    cost = CloudCostModel calculation
```

## API Endpoint Reference

### POST /recommend-scaling

**Request:**
```json
{
    "current_pods": 3,
    "requests": 15000,
    "forecast": 18000,
    "capacity_per_pod": 5000
}
```

**Response:**
```json
{
    "current_pods": 3,
    "recommended_pods": 4,
    "action": "scale-up",
    "reasons": [
        {
            "factor": "Predictive Scaling",
            "current_value": "Forecast: 18000 requests",
            "threshold": "Current capacity: 15000",
            "decision": "Proactive: Scale UP 1 pod (prepare for spike)"
        }
    ],
    "confidence": 0.8,
    "explanation": "RECOMMENDATION: SCALE UP from 3 to 4 pods\n\nREASON: Proactive: Scale UP 1 pod (prepare for spike)\n\n...",
    "estimated_cost_impact": {
        "current_hourly_cost": 0.135,
        "new_hourly_cost": 0.156,
        "cost_difference": 0.021,
        "cost_change_percent": 15.6
    }
}
```

### GET /health

Simple health check.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

## Testing with curl

If you want to test the API directly without the dashboard:

```bash
# Scale-up scenario
curl -X POST http://localhost:8000/recommend-scaling \
  -H "Content-Type: application/json" \
  -d '{
    "current_pods": 3,
    "requests": 15000,
    "forecast": 18000,
    "capacity_per_pod": 5000
  }'

# Scale-down scenario
curl -X POST http://localhost:8000/recommend-scaling \
  -H "Content-Type: application/json" \
  -d '{
    "current_pods": 8,
    "requests": 3000,
    "forecast": 3500,
    "capacity_per_pod": 5000
  }'

# Health check
curl http://localhost:8000/health
```

## Troubleshooting

### "Cannot connect to API server"

Make sure FastAPI server is running:
```bash
python api_server.py
```

### "Port 8000 already in use"

Change port in api_server.py:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use 8001 instead
```

### "ModuleNotFoundError: No module named 'fastapi'"

Install dependencies:
```bash
pip install fastapi uvicorn
```

### Dashboard not loading

Make sure Streamlit is installed and run from correct directory:
```bash
cd /path/to/dataFlow-2026
pip install streamlit
streamlit run dashboard_demo.py
```

## Code Files

- **api_server.py** (400+ lines)
  - FastAPI app with `/recommend-scaling` endpoint
  - HybridAutoscalerAnalyzer class (4-layer logic)
  - Detailed reasoning for each decision

- **dashboard_demo.py** (500+ lines)
  - Streamlit dashboard
  - Simulated data generation
  - 4 interactive Plotly charts
  - API integration

## Next Steps

### Extend the demo:

1. **Add real data**: Replace simulated data with actual metrics
2. **Store history**: Save recommendations to database
3. **Add more strategies**: Compare HYBRID vs REACTIVE vs PREDICTIVE
4. **Webhook integration**: Send recommendations to Kubernetes API
5. **Alert system**: Notify on SLA violations

### Production deployment:

1. Use Docker containers for API and dashboard
2. Add authentication (API key or JWT)
3. Add metrics logging (Prometheus)
4. Add monitoring (Grafana)
5. Scale horizontally (multiple API instances behind load balancer)

## Reference

- [HYBRID Strategy Details](HYBRID_DEPLOYMENT.md)
- [Cost Model Analysis](COST_MODEL_SELECTION.md)
- [Production README](PRODUCTION_README.md)

---

**Created**: 2024
**Version**: 1.0
**Demo Status**: ✅ Ready to use
