from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = str(BASE_DIR / "models")
HYBRID_LSTM_DIR = DEFAULT_MODEL_DIR
HYBRID_PACKAGE_PATH = os.path.join(DEFAULT_MODEL_DIR, "hybrid_model_package.pkl")
XGBOOST_MODEL_DIR = DEFAULT_MODEL_DIR
XGBOOST_METADATA_DIR = DEFAULT_MODEL_DIR
XGBOOST_PREDICTIONS_DIR = DEFAULT_MODEL_DIR


def _get_timestamp_series(df: pd.DataFrame) -> pd.Series | None:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"])
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index)
    return None


def _infer_step(df: pd.DataFrame) -> pd.Timedelta:
    ts = _get_timestamp_series(df)
    if ts is None or len(ts) < 2:
        return pd.Timedelta(minutes=5)
    inferred = pd.infer_freq(ts)
    if inferred is None:
        deltas = ts.diff().dropna()
        if deltas.empty:
            return pd.Timedelta(minutes=5)
        return deltas.median()
    return pd.tseries.frequencies.to_offset(inferred).delta


def _build_future_timestamps(last_ts: pd.Timestamp, step: pd.Timedelta, horizon: int) -> pd.Series:
    return pd.date_range(start=last_ts + step, periods=horizon, freq=step)


def discover_models(model_dir: str | None = None) -> dict[str, list[str]]:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    models = {"lstm": [], "xgboost": []}

    if not os.path.isdir(model_dir):
        return models

    for file in os.listdir(model_dir):
        if file.startswith("lstm_") and file.endswith(".keras"):
            models["lstm"].append(file)
        elif file.startswith("xgboost_") and (file.endswith(".pkl") or file.endswith(".json")):
            models["xgboost"].append(file)

    return models


def extract_timeframe(filename: str) -> str:
    if "_1m_" in filename:
        return "1m"
    if "_5m_" in filename:
        return "5m"
    if "_15m_" in filename:
        return "15m"
    return "5m"


def load_lstm_model(model_path: str) -> Any:
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")
        return None


def load_xgboost_model(model_path: str) -> Any:
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")
        return None


def _resolve_xgboost_model_path(timeframe: str, model_dir: str | None = None) -> str | None:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    filename = f"xgboost_{timeframe}_model.json"
    candidates = [
        os.path.join(model_dir, filename),
        os.path.join(XGBOOST_MODEL_DIR, filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def _load_xgboost_feature_list(timeframe: str, model_dir: str | None = None) -> list[str] | None:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    filename = f"xgboost_{timeframe}_metadata.json"
    candidates = [
        os.path.join(model_dir, filename),
        os.path.join(XGBOOST_METADATA_DIR, filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                import json
                with open(path, "r") as f:
                    meta_obj = json.load(f)
                features = meta_obj.get("features")
                if isinstance(features, list) and features:
                    return features
            except Exception as e:
                print(f"Failed to load XGBoost metadata: {e}")
                return None

    return None


def _load_hybrid_scaler_and_window(model_dir: str | None = None) -> tuple[Any | None, int | None]:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    package_path = os.path.join(model_dir, "hybrid_model_package.pkl")
    if not os.path.exists(package_path):
        return None, None
    try:
        with open(package_path, "rb") as f:
            package = pickle.load(f)
        window_size = int(package.get("lstm_config", {}).get("window_size", 24))
        return package, window_size
    except Exception as e:
        print(f"Failed to load hybrid package for scaler: {e}")
        return None, None


def _resolve_hybrid_model_path(timeframe: str, model_dir: str | None = None) -> str | None:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    filename = f"lstm_{timeframe}_best.keras"
    candidates = [
        os.path.join(model_dir, filename),
        os.path.join(HYBRID_LSTM_DIR, filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def _predict_lstm(model: Any, df: pd.DataFrame, horizon: int) -> list[float] | None:
    try:
        lookback = min(12, len(df))
        last_sequence = df["requests_count"].iloc[-lookback:].values.astype(np.float32)
        X = last_sequence.reshape(1, lookback, 1)

        predictions = []
        for _ in range(horizon):
            pred = model.predict(X, verbose=0)
            pred_value = float(pred[0, 0])
            predictions.append(max(0.0, pred_value))
            X = np.append(X[:, 1:, :], [[[pred_value]]], axis=1)

        return predictions
    except Exception as e:
        print(f"LSTM prediction failed: {e}")
        return None


def _predict_xgboost_rolling(
    model: Any,
    df: pd.DataFrame,
    horizon: int,
    scaler: Any | None = None,
    feature_list: list[str] | None = None,
) -> list[float] | None:
    try:
        import xgboost as xgb

        feature_cols = feature_list or [
            "is_burst",
            "burst_ratio",
            "is_event",
            "hour_of_day",
            "day_of_week",
            "hour_sin",
            "hour_cos",
            "lag_requests_1h",
            "lag_requests_1d",
            "rolling_mean_1h",
            "rolling_std_1h",
            "rolling_mean_5m",
            "rolling_std_5m",
        ]

        if not feature_cols:
            return None

        predictions = []
        df_extended = df.copy()

        for _ in range(horizon):
            last_row = []
            for col in feature_cols:
                if col in df_extended.columns:
                    value = df_extended.iloc[-1][col]
                    if pd.isna(value):
                        value = 0.0
                    last_row.append(float(value))
                else:
                    last_row.append(0.0)

            last_row_array = np.array(last_row).reshape(1, -1)

            if scaler is not None and hasattr(scaler, "transform"):
                last_row_array = scaler.transform(last_row_array)

            dmatrix = xgb.DMatrix(last_row_array, feature_names=feature_cols)

            pred = model.predict(dmatrix)[0]
            pred_value = max(0.0, float(pred))
            predictions.append(pred_value)

            new_row = df_extended.iloc[-1].copy()
            new_row["requests_count"] = pred_value

            if "rolling_mean_5m" in df_extended.columns:
                recent_5m = df_extended["requests_count"].tail(5).tolist() + [pred_value]
                new_row["rolling_mean_5m"] = np.mean(recent_5m[-5:])
                new_row["rolling_std_5m"] = np.std(recent_5m[-5:])

            if "rolling_mean_1h" in df_extended.columns:
                recent_1h = df_extended["requests_count"].tail(12).tolist() + [pred_value]
                new_row["rolling_mean_1h"] = np.mean(recent_1h[-12:])
                new_row["rolling_std_1h"] = np.std(recent_1h[-12:])

            df_extended = pd.concat([df_extended, pd.DataFrame([new_row])], ignore_index=True)

        return predictions
    except Exception as e:
        print(f"XGBoost rolling prediction failed: {e}")
        return None


def _predict_xgboost_from_csv(
    df: pd.DataFrame,
    horizon: int,
    timeframe: str,
    model_dir: str | None = None,
) -> list[float] | None:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    candidates = [
        os.path.join(model_dir, f"xgboost_{timeframe}_predictions.csv"),
        os.path.join(XGBOOST_PREDICTIONS_DIR, f"xgboost_{timeframe}_predictions.csv"),
        os.path.join(model_dir, "xgboost_test_predictions.csv"),
        os.path.join(XGBOOST_PREDICTIONS_DIR, "xgboost_test_predictions.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                pred_df = pd.read_csv(path)
                if "predicted" not in pred_df.columns:
                    continue

                if "timestamp" in pred_df.columns and "timestamp" in df.columns:
                    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                    df_ts = pd.to_datetime(df["timestamp"]).iloc[-horizon:]
                    aligned = pred_df[pred_df["timestamp"].isin(df_ts)]
                    if not aligned.empty:
                        preds = aligned.sort_values("timestamp")["predicted"].astype(float).values
                        return [max(0.0, float(v)) for v in preds[-horizon:]]

                preds = pred_df["predicted"].astype(float).values
                if len(preds) == 0:
                    continue

                if horizon <= len(preds):
                    return [max(0.0, float(v)) for v in preds[-horizon:]]

                last_val = float(preds[-1])
                padded = list(preds) + [last_val] * (horizon - len(preds))
                return [max(0.0, float(v)) for v in padded[-horizon:]]
            except Exception as e:
                print(f"Failed to load XGBoost predictions CSV: {e}")
                return None

    return None


def _predict_hybrid(
    model: Any,
    df: pd.DataFrame,
    horizon: int,
    scaler: Any | None,
    window_size: int | None,
) -> list[float] | None:
    try:
        if scaler is None:
            return _predict_lstm(model, df, horizon)

        lookback = min(window_size or 24, len(df))
        last_values = df["requests_count"].iloc[-lookback:].values.astype(np.float32)
        scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()
        X = scaled.reshape(1, lookback, 1)

        predictions = []
        for _ in range(horizon):
            pred_scaled = model.predict(X, verbose=0)
            pred_value_scaled = float(pred_scaled[0, 0])
            predictions.append(pred_value_scaled)
            X = np.append(X[:, 1:, :], [[[pred_value_scaled]]], axis=1)

        predictions = np.array(predictions).reshape(-1, 1)
        yhat_residual = scaler.inverse_transform(predictions).flatten()
        return [float(v) for v in yhat_residual]
    except Exception as e:
        print(f"Hybrid prediction failed: {e}")
        return None


def _heuristic_forecast(df: pd.DataFrame, horizon: int) -> list[float]:
    baseline = float(df["rolling_mean_1h"].iloc[-1]) if "rolling_mean_1h" in df.columns else float(df["requests_count"].iloc[-1])

    recent = df["requests_count"].tail(24).values
    trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

    np.random.seed(42)
    yhat = []
    current = baseline
    for _ in range(horizon):
        noise = np.random.normal(0, baseline * 0.02)
        current = current + trend + noise
        current = max(1.0, current)
        yhat.append(float(current))

    return yhat


def forecast_next(
    df: pd.DataFrame,
    forecast_horizon: int,
    model_type: str = "lstm",
    timeframe: str = "5m",
    model_dir: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be a positive integer")

    ts = _get_timestamp_series(df)
    if ts is None or ts.empty:
        raise ValueError("Input data must contain a timestamp column or DateTimeIndex")

    last_ts = pd.to_datetime(ts.iloc[-1])
    step = _infer_step(df)
    future_ts = _build_future_timestamps(last_ts, step, forecast_horizon)

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    yhat = None
    status = "heuristic"

    try:
        if model_type == "hybrid":
            model_path = _resolve_hybrid_model_path(timeframe, model_dir)
            package, window_size = _load_hybrid_scaler_and_window(model_dir)
            scaler = None
            if package is not None:
                scaler = package.get("lstm_models", {}).get(timeframe, {}).get("scaler")
            if model_path:
                model = load_lstm_model(model_path)
                if model is not None:
                    residuals = _predict_hybrid(model, df, forecast_horizon, scaler, window_size)
                    if residuals is not None:
                        baseline = _heuristic_forecast(df, forecast_horizon)
                        yhat = [max(0.0, float(b) + float(r)) for b, r in zip(baseline, residuals)]
                        status = f"hybrid_{timeframe}"
        elif model_type == "lstm":
            model_filename = f"{model_type}_{timeframe}_best.keras"
            model_path = os.path.join(model_dir, model_filename)
            if os.path.exists(model_path):
                model = load_lstm_model(model_path)
                if model is not None:
                    yhat = _predict_lstm(model, df, forecast_horizon)
                    if yhat is not None:
                        status = f"lstm_{timeframe}"
        elif model_type == "xgboost":
            yhat = _predict_xgboost_from_csv(df, forecast_horizon, timeframe, model_dir)
            if yhat is not None:
                status = f"xgboost_csv_{timeframe}"
            else:
                model_path = _resolve_xgboost_model_path(timeframe, model_dir)
                scaler_path = os.path.join(model_dir, f"xgboost_{timeframe}_scaler.pkl")
                scaler = None
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                    except Exception as e:
                        print(f"Failed to load XGBoost scaler: {e}")
                if model_path:
                    model = load_xgboost_model(model_path)
                    if model is not None:
                        feature_list = _load_xgboost_feature_list(timeframe, model_dir)
                        yhat = _predict_xgboost_rolling(model, df, forecast_horizon, scaler, feature_list)
                        if yhat is not None:
                            status = f"xgboost_{timeframe}"
    except Exception as e:
        print(f"Model loading error: {e}")

    if yhat is None:
        yhat = _heuristic_forecast(df, forecast_horizon)
        status = "heuristic"

    return pd.DataFrame({"timestamp": future_ts, "yhat": yhat}), status
