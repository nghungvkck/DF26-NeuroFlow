# DataFlow Autoscaling Demo

Interactive dashboard for traffic forecasting and autoscaling decisions using machine learning models.

## Features

- **Traffic Overview**: Visualize request patterns, burst detection, and events
- **Forecast Models**: LSTM (Hybrid), XGBoost, and LightGBM predictions
- **Autoscaling Decisions**: Real-time scaling recommendations based on predictions
- **Cost Analysis**: Compare reactive vs predictive scaling costs
- **REST API**: HTTP endpoints for programmatic access to predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

### REST API Server

```bash
python api.py
```

Access API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
Demo/
├── app/
│   ├── dashboard.py           # Main Streamlit app
│   ├── forecast_tab_simple.py # Simple forecast view
│   ├── forecast_tab_plotly.py # Advanced Plotly visualization
│   └── api_demo_tab.py        # API demo interface
├── utils/
│   ├── forecast.py      # Model forecasting logic
│   ├── scaling.py       # Autoscaling decision logic
│   └── load_data.py     # Data loading utilities
├── data/                # CSV data files
├── models/              # Pre-trained models
├── api.py               # REST API server
└── requirements.txt     # Dependencies
```

## Models

- **LSTM (Hybrid)**: Prophet baseline + LSTM residuals
- **XGBoost**: Rolling feature-based forecast
- **LightGBM**: Gradient boosting model

## Data Format

CSV files require:

- `timestamp`: ISO 8601 datetime
- `requests_count`: Number of requests (numeric)
- Optional: `is_burst`, `is_event`, seasonal features

## Key Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **Cost Savings**: Predictive vs reactive scaling comparison

---

## REST API Documentation

### Quick Start

```bash
# Start API server
python api.py

# Test health check
curl http://localhost:8000/health
```

### API Endpoints

**Health Check**

```bash
GET /              # API information
GET /health        # Health status
```

**Forecasting**

```bash
POST /forecast/predict     # Forward forecasting
POST /forecast/backtest    # Historical backtesting
POST /forecast/predict/csv # Predict from CSV upload
```

**Metrics & Models**

```bash
GET /metrics/{model_type}/{timeframe}  # Get pre-computed metrics
GET /models                            # List available models
```

### Example: Forward Prediction

```bash
curl -X POST "http://localhost:8000/forecast/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"ds": "1995-08-01 00:00:00", "y": 245}],
    "horizon": 12,
    "model_type": "xgboost",
    "timeframe": "5m"
  }'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/forecast/predict",
    json={
        "data": [{"ds": "1995-08-01 00:00:00", "y": 245}],
        "horizon": 12,
        "model_type": "xgboost",
        "timeframe": "5m"
    }
)

result = response.json()
if result['success']:
    print(result['predictions'])
```

### Valid Parameters

**model_type**: `hybrid`, `xgboost`, `lightgbm`  
**timeframe**: `1m`, `5m`, `15m`  
**horizon**: 1-100 (prediction steps)  
**step**: 1-50 (backtest step size)

### Production Deployment

```bash
# Using Gunicorn
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t forecast-api .
docker run -p 8000:8000 forecast-api
```

### Error Handling

HTTP Status Codes:

- 200: Success
- 400: Invalid parameters
- 404: Model/metrics not found
- 500: Server error

### Troubleshooting

**Model not found**: Ensure model files exist in `models/` directory  
**Import errors**: Run `pip install -r requirements.txt`  
**Port in use**: Change port with `uvicorn api:app --port 8001`
