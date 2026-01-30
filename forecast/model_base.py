from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ForecastResult:
    yhat: List[float]
    timestamps: List[pd.Timestamp]
    metadata: Dict[str, Any]


class BaseModel(ABC):
    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon

    @abstractmethod
    def predict(self, history_df: pd.DataFrame, horizon: int | None = None) -> ForecastResult:
        raise NotImplementedError
