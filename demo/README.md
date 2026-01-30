# DataFlow Autoscaling Demo

Interactive dashboard for traffic forecasting and autoscaling decisions using machine learning models.

## Features

- **Traffic Overview**: Visualize request patterns, burst detection, and events
- **Forecast Models**: LSTM (Hybrid), XGBoost, and LightGBM predictions
- **Autoscaling Decisions**: Real-time scaling recommendations based on predictions
- **Cost Analysis**: Compare reactive vs predictive scaling costs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app/dashboard.py
```

## Project Structure

```
Demo/
├── app/
│   ├── dashboard.py           # Main Streamlit app
│   ├── forecast_tab_simple.py # Simple forecast view
│   └── forecast_tab_plotly.py # Advanced Plotly visualization
├── utils/
│   ├── forecast.py      # Model forecasting logic
│   ├── scaling.py       # Autoscaling decision logic
│   └── load_data.py     # Data loading utilities
├── data/                # CSV data files
├── models/              # Pre-trained models
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
