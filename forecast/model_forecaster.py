from __future__ import annotations

from typing import Optional

import pandas as pd

from .forecast_utils import forecast_next
from .model_base import BaseModel, ForecastResult


class ModelForecaster(BaseModel):
    def __init__(
        self,
        model_type: str = "hybrid",
        timeframe: str = "5m",
        model_dir: Optional[str] = None,
        forecast_horizon: int = 1,
    ) -> None:
        super().__init__(forecast_horizon=forecast_horizon)
        self.model_type = model_type
        self.timeframe = timeframe
        self.model_dir = model_dir

    def predict(self, history_df: pd.DataFrame, horizon: int | None = None) -> ForecastResult:
        effective_horizon = horizon if horizon is not None else self.forecast_horizon
        forecast_df, status = forecast_next(
            history_df,
            effective_horizon,
            model_type=self.model_type,
            timeframe=self.timeframe,
            model_dir=self.model_dir,
        )

        return ForecastResult(
            yhat=forecast_df["yhat"].tolist(),
            timestamps=[pd.to_datetime(ts) for ts in forecast_df["timestamp"].tolist()],
            metadata={
                "status": status,
                "model_type": self.model_type,
                "timeframe": self.timeframe,
            },
        )
