from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from forecasting.train.common import (
    iter_windows,
    load_config,
    prepare_window_data,
)
from forecasting.evaluate.evaluate import MetricEvaluator
from forecasting.artifacts import ArtifactManager


def _period_from_window(window: str) -> int:
    if window == "1m":
        return 24 * 60
    if window == "5m":
        return 24 * 12
    if window == "15m":
        return 24 * 4
    return 24


def _prepare_prophet_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]):
    y_train_full = train_df["requests_count"].values
    y_test = test_df["requests_count"].values

    split_idx = int(len(y_train_full) * 0.8)
    y_train = y_train_full[:split_idx]
    y_valid = y_train_full[split_idx:]

    df_train = pd.DataFrame({"ds": train_df["timestamp"][:split_idx], "y": y_train})
    df_valid = pd.DataFrame({"ds": train_df["timestamp"][split_idx:], "y": y_valid})
    df_test = pd.DataFrame({"ds": test_df["timestamp"], "y": y_test})

    for f in features:
        df_train[f] = train_df[f][:split_idx].fillna(0)
        df_valid[f] = train_df[f][split_idx:].fillna(0)
        df_test[f] = test_df[f].fillna(0)

    return df_train, df_valid, df_test, y_train, y_valid, y_test


def _build_prophet(features: list[str]) -> Prophet:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.3,
    )
    for f in features:
        model.add_regressor(f)
    return model


def _build_lstm(window_size: int, lstm_units: int = 32) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, activation="relu")),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, activation="relu")),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(lstm_units, activation="relu")),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _make_sequences(arr: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(arr) - window_size):
        X.append(arr[i:i + window_size])
        y.append(arr[i + window_size])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y


def run(config_path: str = "configs/train_config.yaml"):
    cfg = load_config(config_path)
    metrics = cfg.get("evaluation", {}).get("metrics", ["mae", "rmse", "mape", "smape"])
    evaluator = MetricEvaluator(metrics=metrics)

    artifacts = ArtifactManager(Path(__file__).parent.parent / "artifacts")

    all_metrics = []
    for window, train_path, test_path, window_rule in iter_windows(cfg):
        train_df, test_df, features = prepare_window_data(
            cfg, train_path, test_path, window_rule
        )

        df_train, df_valid, df_test, y_train, y_valid, y_test = _prepare_prophet_frames(
            train_df, test_df, features
        )

        prophet = _build_prophet(features)
        prophet.fit(df_train)

        pred_train = prophet.predict(df_train)["yhat"].values
        pred_valid = prophet.predict(df_valid)["yhat"].values
        pred_test = prophet.predict(df_test)["yhat"].values

        period = _period_from_window(window)
        decomp_train = sm.tsa.seasonal_decompose(
            pd.Series(y_train - pred_train, index=df_train["ds"]),
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )
        decomp_valid = sm.tsa.seasonal_decompose(
            pd.Series(y_valid - pred_valid, index=df_valid["ds"]),
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )
        decomp_test = sm.tsa.seasonal_decompose(
            pd.Series(y_test - pred_test, index=df_test["ds"]),
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )

        train_target = (decomp_train.trend + decomp_train.resid).bfill().ffill().values
        valid_target = (decomp_valid.trend + decomp_valid.resid).bfill().ffill().values
        test_target = (decomp_test.trend + decomp_test.resid).bfill().ffill().values

        scaler = MinMaxScaler()
        scaler.fit(train_target.reshape(-1, 1))
        train_scaled = scaler.transform(train_target.reshape(-1, 1)).flatten()
        valid_scaled = scaler.transform(valid_target.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_target.reshape(-1, 1)).flatten()

        window_size = 24
        X_train, y_train_seq = _make_sequences(train_scaled, window_size)
        X_valid, y_valid_seq = _make_sequences(valid_scaled, window_size)

        lstm = _build_lstm(window_size=window_size, lstm_units=32)
        lstm.fit(
            X_train,
            y_train_seq,
            validation_data=(X_valid, y_valid_seq),
            epochs=100,
            batch_size=32,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0,
        )

        train_pred_scaled = lstm.predict(X_train, verbose=0).flatten()
        valid_pred_scaled = lstm.predict(X_valid, verbose=0).flatten()
        train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        valid_pred = scaler.inverse_transform(valid_pred_scaled.reshape(-1, 1)).flatten()

        current_batch = train_scaled[-window_size:].reshape(1, window_size, 1)
        test_pred_scaled = []
        for _ in range(len(test_scaled)):
            pred = lstm.predict(current_batch, verbose=0)[0, 0]
            test_pred_scaled.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)
        test_pred_scaled = np.array(test_pred_scaled)
        test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

        train_resid = train_pred + decomp_train.seasonal.values[window_size:]
        valid_resid = valid_pred + decomp_valid.seasonal.values[window_size:]
        test_resid = test_pred + decomp_test.seasonal.values

        hybrid_train = pred_train[window_size:] + train_resid
        hybrid_valid = pred_valid[window_size:] + valid_resid
        hybrid_test = pred_test + test_resid

        metrics_result = evaluator.evaluate(y_test, hybrid_test)
        artifacts.save_metrics(metrics_result, "hybrid", window)
        
        metrics_frame = evaluator.to_frame(metrics_result, "hybrid", window)
        all_metrics.append(metrics_frame)

        pred_df = pd.DataFrame({
            "timestamp": df_test["ds"].values,
            "y_true": y_test,
            "y_pred": hybrid_test,
        })
        artifacts.save_predictions(pred_df, "hybrid", window)

        prophet_path = artifacts.get_model_path("prophet", window, ext=".pkl")
        lstm_path = artifacts.get_model_path("lstm", window, ext=".keras")
        scaler_path = artifacts.get_model_path("lstm_scaler", window, ext=".pkl")
        
        with open(prophet_path, "wb") as f:
            pickle.dump(prophet, f)
        lstm.save(str(lstm_path))
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        artifacts.save_evaluation_summary(metrics_df, "hybrid")


if __name__ == "__main__":
    run()
