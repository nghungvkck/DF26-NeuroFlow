# Autoscaling Forecasting & Optimization

Time-series forecasting + hybrid autoscaling optimization with Streamlit demo and FastAPI.

## Project Structure

```
├── data/                       # Test datasets (1m, 5m, 15m)
│   ├── test_1m_autoscaling.csv
│   ├── test_5m_autoscaling.csv
│   └── test_15m_autoscaling.csv
├── raw_data/                   # Raw Apache HTTP logs
│   ├── train.txt              # Training logs (~2.9M requests)
│   └── test.txt               # Test logs
├── forecasting/                # Forecasting module
│   ├── train/                  # Training scripts for 3 models
│   │   ├── train_lightgbm.py
│   │   ├── train_xgboost.py
│   │   ├── train_hybrid.py
│   │   └── common.py           # Shared utilities
│   ├── inference/              # Inference module for predictions
│   │   └── predictor.py        # ModelPredictor for loading & predicting
│   ├── preprocess/             # Data preprocessing pipeline
│   │   ├── pipeline.py         # Main orchestrator
│   │   ├── data_loader.py
│   │   ├── parser.py
│   │   ├── normalizer.py
│   │   ├── aggregator.py
│   │   ├── feature_engineering.py
│   │   └── missing_handler.py
│   ├── evaluate/               # Model evaluation
│   │   └── evaluate.py         # MetricEvaluator
│   ├── models/                 # Model class definitions
│   │   ├── base_model.py
│   │   ├── hybrid_model.py
│   │   ├── lstm_model.py
│   │   ├── prophet_model.py
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   └── model_factory.py
│   ├── artifacts/              # Models, metrics, predictions
│   │   ├── models/            # Trained models (.txt, .json, .pkl, .h5)
│   │   ├── metrics/           # Evaluation metrics (JSON, CSV)
│   │   └── predictions/       # Predictions CSV files
│   ├── utils/                  # Utility modules
│   ├── main.py
│   └── artifacts.py            # ArtifactManager for output organization
├── optimization/               # Hybrid autoscaling logic
│   ├── hybrid_autoscaler.py
│   ├── anomaly_detection.py
│   ├── cost_model.py
│   └── metrics.py
├── demo/                       # Streamlit dashboard + FastAPI
│   ├── app/
│   │   ├── dashboard.py        # Tabs: Overview, Forecast, Optimization, API Demo
│   │   ├── forecast_tab_simple.py
│   │   └── api_demo_tab.py
│   ├── utils/
│   │   ├── forecast.py
│   │   ├── metrics_forecast.py
│   │   └── load_data.py
│   ├── api.py                  # FastAPI (forecast + recommend-scaling)
│   ├── requirements.txt
│   └── README.md
├── notebook/                   # Jupyter notebooks
│   ├── eda_analysis.ipynb
│   └── pre_process.ipynb
├── configs/                    # Configuration files
│   └── train_config.yaml
├── pyproject.toml              # Project dependencies
├── main.py                     # Full pipeline entry point
├── infer.py                    # Standalone inference script
└── README.md
```

## Key Features

- **Forecasting**: XGBoost, LightGBM, Hybrid (Prophet + LSTM)
- **3 Timeframes**: 1m / 5m / 15m
- **Optimization**: 4-layer hybrid autoscaler (Anomaly → Emergency → Predictive → Reactive)
- **Cost Model**: Reserved + Spot + On-demand
- **Dashboard**: Streamlit tabs (Overview, Forecast, Optimization, API Demo)
- **API**: FastAPI endpoints for forecast + scaling recommendations

## Pipeline Overview

**Complete Data Processing Flow:**

1. **Data Loading** → Read raw Apache logs from `raw_data/train.txt` and `raw_data/test.txt`
2. **Parsing** → Extract: host, timestamp, method, URL, status, bytes from Apache Common Log Format
3. **Normalization** → Convert to UTC datetime, sort by timestamp
4. **Aggregation** → Combine logs into time windows (1m, 5m, 15m) with metrics
5. **Reindexing** → Fill missing timestamps, interpolate event periods (Aug 1-3 1995)
6. **Feature Engineering** → Add 13 features:
   - Temporal: hour_of_day, day_of_week, hour_sin, hour_cos
   - Event: is_event, is_burst, burst_ratio
   - Lag: lag_requests_5m, lag_requests_15m, lag_requests_6h, lag_requests_1d
   - Rolling: rolling_mean_1h, rolling_max_1h
7. **Train/Test Split** → Before Aug 23 (train) vs Aug 23+ (test)
8. **Model Training** → 3 models × 3 timeframes (9 total):
   - XGBoost: `xgboost_1m.json`, `xgboost_5m.json`, `xgboost_15m.json`
   - LightGBM: `lightgbm_1m.txt`, `lightgbm_5m.txt`, `lightgbm_15m.txt`
   - Hybrid: `lstm_1m.keras`, `lstm_5m.keras`, `lstm_15m.keras`
9. **Evaluation** → Calculate metrics (MAE, RMSE, MAPE, SMAPE, R2)
10. **Output** → Save to `forecasting/artifacts/`:
    - Models: `artifacts/models/`
    - Metrics: `artifacts/metrics/`
    - Predictions: `artifacts/predictions/`

## Installation

```bash
pip install -e .
```

Installs the `forecasting` package and all dependencies from `pyproject.toml`.

## Quick Start

### 1. Train Models (All Timeframes)

```bash
python main.py
```

or individual models:

```bash
python forecasting/train/train_lightgbm.py
python forecasting/train/train_xgboost.py
python forecasting/train/train_hybrid.py
```

Models saved to `forecasting/artifacts/models/`.

### 2. Launch Dashboard

```bash
cd demo
streamlit run app/dashboard.py
```

- Load test data from `../data/`
- View predictions from `forecasting/artifacts/predictions`
- Run autoscaling simulation with forecast input

### 3. Start API Server

```bash
cd demo
python api.py
```

Access at: http://localhost:8000/docs

Key endpoints:

- `POST /forecast/metrics`
- `POST /recommend-scaling`

## Predictions Artifacts

`forecasting/artifacts/predictions/` contains:

- `lightgbm_{1m,5m,15m}_predictions.csv`
- `xgboost_{1m,5m,15m}_predictions.csv`
- `hybrid_{1m,5m,15m}_predictions.csv`

## Configuration

Edit `configs/train_config.yaml` to customize:

- Raw data paths (`raw_data_train`, `raw_data_test`)
- Time windows (`windows`: [1m, 5m, 15m])
- Feature engineering parameters (lag, rolling, burst thresholds)
- Model hyperparameters (n_estimators, learning_rate, etc.)

## Output Structure

After training, `forecasting/artifacts/` contains:

```

artifacts/
├── models/
│ ├── lightgbm_1m.txt
│ ├── xgboost_1m.json
│ ├── prophet_1m.pkl
│ ├── lstm_1m.keras
│ ├── lstm_scaler_1m.pkl
│ ... (similar for 5m, 15m windows)
├── metrics/
│ ├── lightgbm_evaluation.csv
│ ├── xgboost_evaluation.csv
│ └── hybrid_evaluation.csv
└── predictions/
   ├── lightgbm_1m_predictions.csv
   ├── xgboost_1m_predictions.csv
   └── hybrid_1m_predictions.csv

```

## Dependencies

- **Data & ML**: pandas, numpy, scikit-learn, lightgbm, xgboost, prophet, tensorflow/keras
- **Time Series**: statsmodels
- **Config**: pyyaml
- **Demo**: streamlit, plotly, altair
- **API**: fastapi, uvicorn, pydantic

See `pyproject.toml` for full dependency list.

```

```
