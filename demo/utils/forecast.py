from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


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


def _find_model_file(model_dir: str) -> str | None:
    if not os.path.isdir(model_dir):
        return None
    for ext in (".keras", ".pkl"):
        candidates = sorted([f for f in os.listdir(model_dir) if f.endswith(ext)])
        if candidates:
            return os.path.join(model_dir, candidates[0])
    return None


def _load_model(path: str) -> Any:
    if path.endswith(".keras"):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TensorFlow is required to load .keras models") from exc
        return load_model(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_with_model(model: Any, last_value: float, horizon: int) -> list[float] | None:
    try:
        X = pd.DataFrame({"requests_count": [last_value]})
        y_pred = model.predict(X)
    except Exception:
        try:
            import numpy as np

            X = np.array([[[last_value]]])
            y_pred = model.predict(X)
        except Exception:
            return None

    if isinstance(y_pred, pd.DataFrame):
        y_values = y_pred.values.ravel().tolist()
    elif isinstance(y_pred, pd.Series):
        y_values = y_pred.values.tolist()
    else:
        try:
            y_values = list(y_pred.ravel())
        except Exception:
            y_values = list(y_pred)

    if not y_values:
        return None

    if len(y_values) < horizon:
        y_values += [y_values[-1]] * (horizon - len(y_values))
    return [float(v) for v in y_values[:horizon]]


def forecast_next(
    df: pd.DataFrame,
    forecast_horizon: int,
    model_dir: str | None = None,
) -> pd.DataFrame:
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be a positive integer")

    ts = _get_timestamp_series(df)
    if ts is None or ts.empty:
        raise ValueError("Input data must contain a timestamp column or DateTimeIndex")

    last_ts = pd.to_datetime(ts.iloc[-1])
    step = _infer_step(df)
    future_ts = _build_future_timestamps(last_ts, step, forecast_horizon)

    last_value = float(df["requests_count"].iloc[-1])
    baseline = float(df["rolling_mean_1h"].iloc[-1]) if "rolling_mean_1h" in df.columns else last_value

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    yhat: list[float] | None = None
    model_path = _find_model_file(model_dir)
    if model_path is not None:
        try:
            model = _load_model(model_path)
            yhat = _predict_with_model(model, last_value, forecast_horizon)
        except Exception:
            yhat = None

    if yhat is None:
        yhat = [baseline] * forecast_horizon

    return pd.DataFrame({"timestamp": future_ts, "yhat": yhat})