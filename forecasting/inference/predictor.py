from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from forecasting.preprocess.pipeline import DataPipeline


class ModelPredictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._models = {}
        self._scaler_cache = {}

    def load_model(self, model_name: str, window: str) -> Any:
        cache_key = f"{model_name}_{window}"
        if cache_key in self._models:
            return self._models[cache_key]

        if model_name == "lightgbm":
            path = self.model_dir / f"lightgbm_{window}.txt"
            model = lgb.Booster(model_file=str(path))
        elif model_name == "xgboost":
            path = self.model_dir / f"xgboost_{window}.json"
            model = xgb.Booster(model_file=str(path))
        elif model_name == "hybrid":
            prophet_path = self.model_dir / f"prophet_{window}.pkl"
            lstm_path = self.model_dir / f"lstm_{window}.keras"
            with open(prophet_path, "rb") as f:
                prophet_model = pickle.load(f)
            try:
                from tensorflow.keras.models import load_model as keras_load_model
            except ImportError:
                from keras.models import load_model as keras_load_model
            lstm_model = keras_load_model(str(lstm_path))
            model = {"prophet": prophet_model, "lstm": lstm_model}
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self._models[cache_key] = model
        return model

    def predict(
        self,
        model_name: str,
        window: str,
        data: pd.DataFrame,
        features: list[str],
    ) -> np.ndarray:
        model = self.load_model(model_name, window)

        if model_name in ["lightgbm", "xgboost"]:
            preds = model.predict(data[features].values)
        elif model_name == "hybrid":
            prophet_preds = self._predict_hybrid_prophet(model["prophet"], data)
            lstm_preds = self._predict_hybrid_lstm(model["lstm"], data, features)
            preds = (prophet_preds + lstm_preds) / 2

        return preds

    def _predict_hybrid_prophet(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        future = data[["timestamp"]].rename(columns={"timestamp": "ds"})
        forecast = model.predict(future)
        return forecast["yhat"].values

    def _predict_hybrid_lstm(
        self,
        model: Any,
        data: pd.DataFrame,
        features: list[str],
    ) -> np.ndarray:
        X = data[features].values
        return model.predict(X, verbose=0)

    def predict_batch(
        self,
        model_name: str,
        window: str,
        data: pd.DataFrame,
        time_column: str = "timestamp",
        target_column: str = "requests_count",
    ) -> pd.DataFrame:
        pipeline = DataPipeline(
            window_rule=self._window_rule_from_name(window),
            time_column=time_column,
            target_column=target_column,
        )
        processed_data, features = pipeline.run(data)

        preds = self.predict(model_name, window, processed_data, features)

        result = processed_data[[time_column, target_column]].copy()
        result[f"{model_name}_pred"] = preds
        return result

    @staticmethod
    def _window_rule_from_name(window: str) -> str | None:
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
        }
        return mapping.get(window)
