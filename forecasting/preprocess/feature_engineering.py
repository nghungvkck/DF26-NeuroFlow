from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer:
	"""Step 6: Feature engineering aligned with pre_process notebook."""

	time_column: str = "timestamp"
	target_column: str = "requests_count"

	def _infer_step_minutes(self, df: pd.DataFrame) -> int:
		diffs = df[self.time_column].diff().dropna()
		if diffs.empty:
			return 5
		return int(diffs.mode().iloc[0].total_seconds() / 60)

	def run(
		self,
		df: pd.DataFrame,
		train_df: pd.DataFrame | None = None,
		window_name: str | None = None,
	) -> pd.DataFrame:
		if self.time_column not in df.columns:
			return df
		df = df.copy()
		df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True, errors="coerce")
		df = df.dropna(subset=[self.time_column])

		base = df[self.target_column]
		df["log_requests"] = np.log1p(base)

		df["hour_of_day"] = df[self.time_column].dt.hour
		df["day_of_week"] = df[self.time_column].dt.dayofweek
		df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
		df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

		df["is_event"] = 0
		event_start = pd.Timestamp("1995-08-01 00:00:00", tz="UTC")
		event_end = pd.Timestamp("1995-08-03 23:59:59", tz="UTC")
		mask = (df[self.time_column] >= event_start) & (df[self.time_column] <= event_end)
		df.loc[mask, "is_event"] = 1

		base_train = base if train_df is None else train_df[self.target_column]
		q1_train = base_train.quantile(0.25)
		q3_train = base_train.quantile(0.75)
		iqr_train = q3_train - q1_train
		burst_threshold = q3_train + 1.5 * iqr_train

		window_1h = self._rolling_1h_window(window_name)
		df["rolling_mean_1h"] = base.rolling(window=window_1h, min_periods=1).mean().shift(1)
		df["is_burst"] = (base.shift(1) > burst_threshold).astype(int)
		df["burst_ratio"] = base.shift(1) / (df["rolling_mean_1h"] + 1e-6)

		step_minutes = self._infer_step_minutes(df)
		step_minutes = max(step_minutes, 1)

		lag_5m = max(int(5 / step_minutes), 1)
		lag_15m = max(int(15 / step_minutes), 1)
		lag_6h = max(int(360 / step_minutes), 1)
		lag_1d = max(int(1440 / step_minutes), 1)

		if "lag_requests_5m" not in df.columns:
			df["lag_requests_5m"] = base.shift(lag_5m)
		if "lag_requests_15m" not in df.columns:
			df["lag_requests_15m"] = base.shift(lag_15m)
		if "lag_requests_6h" not in df.columns:
			df["lag_requests_6h"] = base.shift(lag_6h)
		if "lag_requests_1d" not in df.columns:
			df["lag_requests_1d"] = base.shift(lag_1d)

		rolling_base = base.shift(1)
		df["rolling_max_1h"] = rolling_base.rolling(window_1h, min_periods=1).max()

		return df

	def _rolling_1h_window(self, window_name: str | None) -> int:
		if window_name == "1m":
			return 60
		if window_name == "15m":
			return 4
		return 12

	def feature_columns(self, df: pd.DataFrame) -> list[str]:
		base_cols = [
			"is_burst",
			"burst_ratio",
			"is_event",
			"hour_of_day",
			"day_of_week",
			"hour_sin",
			"hour_cos",
			"lag_requests_5m",
			"lag_requests_15m",
			"rolling_max_1h",
			"rolling_mean_1h",
			"lag_requests_6h",
			"lag_requests_1d",
		]
		return [c for c in base_cols if c in df.columns]

