"""
Metrics-based forecasting module.

Handles:
- Converting system metrics (aggregated monitoring data) to ML-ready features
- Feature engineering internally (lag, rolling stats, time-based features)
- LightGBM model forecasting at system metric level

This module abstracts ML complexity - users provide timestamps and request counts,
and the module handles all feature engineering internally.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd


def _build_features_for_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw metrics data.
    
    Input: DataFrame with 'timestamp' and 'requests' columns
    Output: DataFrame with ML features needed for LightGBM model
    
    LightGBM model expects these 12 features:
    - hour_of_day, day_of_week, hour_sin, hour_cos (time-based)
    - is_burst, burst_ratio (burst detection)
    - lag_requests_5m, lag_requests_15m, lag_requests_6h, lag_requests_1d (lag)
    - rolling_max_1h, rolling_mean_1h (rolling statistics)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Rename 'requests' to 'requests_count' for consistency with model training
    if 'requests' in df.columns:
        df.rename(columns={'requests': 'requests_count'}, inplace=True)
    
    # Time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # Lag features (with NaN for initial rows)
    # 5m lag: 1 period
    df['lag_requests_5m'] = df['requests_count'].shift(1)
    
    # 15m lag: 3 periods
    df['lag_requests_15m'] = df['requests_count'].shift(3)
    
    # 6h lag: 72 periods (5-minute intervals in 6 hours)
    df['lag_requests_6h'] = df['requests_count'].shift(72)
    
    # 1 day lag: 288 periods (5-minute intervals in 24 hours)
    df['lag_requests_1d'] = df['requests_count'].shift(288)
    
    # Rolling features
    # rolling_mean_1h: 12 periods (5-minute intervals in 1 hour)
    df['rolling_mean_1h'] = df['requests_count'].rolling(window=12, min_periods=1).mean()
    
    # rolling_max_1h: 12 periods
    df['rolling_max_1h'] = df['requests_count'].rolling(window=12, min_periods=1).max()
    
    # Burst detection (simple heuristic)
    # is_burst: 1 if requests > rolling_mean_1h + 2*std
    df['rolling_std_1h_temp'] = df['requests_count'].rolling(window=12, min_periods=1).std()
    df['is_burst'] = (
        (df['requests_count'] > df['rolling_mean_1h'] + 2 * df['rolling_std_1h_temp']).astype(int)
    )
    
    # burst_ratio: how much above rolling mean?
    df['burst_ratio'] = (
        (df['requests_count'] - df['rolling_mean_1h']) / (df['rolling_mean_1h'] + 1)
    ).clip(lower=-1, upper=10)
    
    # Fill NaN values
    df = df.ffill().bfill().fillna(0)
    
    # Clean up temporary columns
    if 'rolling_std_1h_temp' in df.columns:
        df = df.drop(columns=['rolling_std_1h_temp'])
    
    return df


def forecast_metrics_lightgbm(
    history: list[dict],
    horizon_steps: int,
    model_dir: Optional[str] = None,
) -> tuple[bool, str, Optional[list[dict]]]:
    """
    Forecast next `horizon_steps` periods using LightGBM model.
    
    Args:
        history: List of dicts with 'timestamp' and 'requests' keys
        horizon_steps: Number of future periods to forecast (1-100)
        model_dir: Directory containing trained LightGBM model
    
    Returns:
        (success: bool, message: str, forecast: list[dict] or None)
        
    Forecast format:
        [
            {"step": 1, "predicted_requests": 450},
            {"step": 2, "predicted_requests": 460},
            {"step": 3, "predicted_requests": 470}
        ]
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return False, "LightGBM not installed", None
    
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    try:
        # Validate input
        if not history or len(history) < 6:
            return False, "History must contain at least 6 data points", None
        
        if horizon_steps < 1 or horizon_steps > 100:
            return False, "horizon_steps must be between 1 and 100", None
        
        # Convert input to DataFrame and engineer features
        df_history = pd.DataFrame(history)
        df_features = _build_features_for_metrics(df_history)
        
        # Load LightGBM model
        model_path = os.path.join(model_dir, "lightgbm_5m_model.txt")
        if not os.path.exists(model_path):
            # Try alternative paths
            model_path = os.path.join(model_dir, "lightgbm_5m_model.pkl")
        
        if not os.path.exists(model_path):
            return False, f"LightGBM model not found at {model_path}", None
        
        try:
            model = lgb.Booster(model_file=model_path)
        except:
            # Try pickle format
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                return False, f"Failed to load model: {str(e)}", None
        
        # Feature columns that model expects (in specific order)
        feature_cols = [
            'hour_of_day', 'day_of_week', 'hour_sin', 'hour_cos',
            'is_burst', 'burst_ratio',
            'lag_requests_5m', 'lag_requests_15m',
            'rolling_max_1h', 'rolling_mean_1h',
            'lag_requests_6h', 'lag_requests_1d'
        ]
        
        # Make sure all feature columns exist
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        
        # Rolling forecast (predict step-by-step, update features)
        predictions = []
        df_forecast = df_features.copy()
        last_timestamp = pd.to_datetime(df_forecast['timestamp'].iloc[-1])
        
        for step in range(horizon_steps):
            # Get last row's features
            last_row = df_forecast.iloc[-1]
            
            # Extract feature vector
            X = np.array([[last_row[col] for col in feature_cols]])
            
            # Predict
            pred = float(model.predict(X)[0])
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Update features for next iteration
            next_timestamp = last_timestamp + pd.Timedelta(minutes=5)
            last_timestamp = next_timestamp
            
            new_row = last_row.copy()
            new_row['timestamp'] = next_timestamp
            new_row['requests_count'] = pred
            
            # Update time-based features
            new_row['hour_of_day'] = next_timestamp.hour
            new_row['day_of_week'] = next_timestamp.dayofweek
            new_row['hour_sin'] = np.sin(2 * np.pi * next_timestamp.hour / 24)
            new_row['hour_cos'] = np.cos(2 * np.pi * next_timestamp.hour / 24)
            
            # Update lags
            new_row['lag_requests_5m'] = df_forecast.iloc[-1]['requests_count']  # Previous value
            new_row['lag_requests_15m'] = df_forecast.iloc[-3]['requests_count'] if len(df_forecast) >= 3 else pred
            new_row['lag_requests_6h'] = df_forecast.iloc[-72]['requests_count'] if len(df_forecast) >= 72 else pred
            new_row['lag_requests_1d'] = df_forecast.iloc[-288]['requests_count'] if len(df_forecast) >= 288 else pred
            
            # Update rolling features
            recent_1h = df_forecast['requests_count'].tail(12).tolist() + [pred]
            new_row['rolling_mean_1h'] = np.mean(recent_1h[-12:])
            new_row['rolling_max_1h'] = np.max(recent_1h[-12:])
            
            # Update burst features
            rolling_std = np.std(recent_1h[-12:])
            new_row['is_burst'] = int(pred > new_row['rolling_mean_1h'] + 2 * rolling_std)
            new_row['burst_ratio'] = ((pred - new_row['rolling_mean_1h']) / (new_row['rolling_mean_1h'] + 1)).clip(-1, 10)
            
            # Append to forecast dataframe
            df_forecast = pd.concat(
                [df_forecast, pd.DataFrame([new_row])],
                ignore_index=True
            )
        
        # Format response
        forecast_response = [
            {"step": i + 1, "predicted_requests": round(pred, 1)}
            for i, pred in enumerate(predictions)
        ]
        
        return True, "Forecast generated successfully", forecast_response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Forecasting error: {str(e)}", None
