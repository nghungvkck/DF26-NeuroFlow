from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import pandas as pd
from prophet import Prophet


@dataclass
class ProphetModel:
    """Prophet baseline model."""

    params: dict
    model: Prophet = field(init=False)

    def __post_init__(self):
        self.model = Prophet(**self.params)

    def fit(self, df: pd.DataFrame, time_column: str, target_column: str):
        train_df = df[[time_column, target_column]].rename(columns={
            time_column: "ds",
            target_column: "y",
        })
        train_df["ds"] = pd.to_datetime(train_df["ds"]).dt.tz_localize(None)
        self.model.fit(train_df)

    def predict(self, df: pd.DataFrame, time_column: str) -> pd.Series:
        future = df[[time_column]].rename(columns={time_column: "ds"})
        future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)
        forecast = self.model.predict(future)
        return forecast["yhat"]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f)
