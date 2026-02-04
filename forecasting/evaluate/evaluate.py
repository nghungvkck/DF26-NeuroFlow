from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


@dataclass
class MetricEvaluator:
    """Step 8: Evaluate predictions and save results."""

    metrics: Iterable[str]

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        results: dict[str, float] = {}
        for m in self.metrics:
            if m.lower() == "mae":
                results["mae"] = mae(y_true, y_pred)
            elif m.lower() == "rmse":
                results["rmse"] = rmse(y_true, y_pred)
            elif m.lower() == "mape":
                results["mape"] = mape(y_true, y_pred)
            elif m.lower() == "smape":
                results["smape"] = smape(y_true, y_pred)
            elif m.lower() == "r2":
                results["r2"] = r2(y_true, y_pred)
        return results

    def to_frame(self, results: dict[str, float], model_name: str, window_name: str) -> pd.DataFrame:
        data = {"model": model_name, "window": window_name}
        data.update(results)
        return pd.DataFrame([data])
