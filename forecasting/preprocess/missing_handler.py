from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MissingValueHandler:
	"""Step 5: Handle missing values and time gaps."""

	time_column: str = "timestamp"

	def run(
		self,
		df: pd.DataFrame,
		required_cols: list[str] | None = None,
		window_name: str | None = None,
		interpolate_storm: bool = True,
		drop_na: bool = True,
	) -> pd.DataFrame:
		df = df.copy()
		df = self.reindex_and_fill(df, window_name)
		if interpolate_storm:
			df = self._interpolate_storm_period(df)

		rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
		if rolling_cols:
			df[rolling_cols] = df[rolling_cols].ffill()
		if drop_na:
			if required_cols:
				df = df.dropna(subset=required_cols)
			else:
				df = df.dropna()
		return df

	def reindex_and_fill(self, df: pd.DataFrame, window_name: str | None) -> pd.DataFrame:
		if self.time_column not in df.columns:
			return df
		df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True, errors="coerce")
		df = df.dropna(subset=[self.time_column])
		df = df.sort_values(self.time_column).reset_index(drop=True)

		freq_map = {"1m": "1T", "5m": "5T", "15m": "15T"}
		freq = freq_map.get(window_name or "15m", "15T")

		full_range = pd.date_range(
			start=df[self.time_column].min(),
			end=df[self.time_column].max(),
			freq=freq,
		)
		df_reindexed = df.set_index(self.time_column).reindex(full_range)
		df_reindexed.index.name = self.time_column
		df_reindexed = df_reindexed.reset_index()

		df_reindexed["requests_count"] = df_reindexed["requests_count"].fillna(0)
		if "total_bytes" in df_reindexed.columns:
			df_reindexed["total_bytes"] = df_reindexed["total_bytes"].fillna(0)
		if "error_rate" in df_reindexed.columns:
			df_reindexed["error_rate"] = df_reindexed["error_rate"].fillna(0)
		if "window_size" in df.columns and "window_size" not in df_reindexed.columns:
			df_reindexed["window_size"] = window_name

		return df_reindexed

	def _interpolate_storm_period(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.time_column not in df.columns:
			return df
		df = df.copy()
		df[self.time_column] = pd.to_datetime(df[self.time_column]).dt.tz_localize(None)
		df = df.sort_values(self.time_column).reset_index(drop=True)

		storm_start = pd.Timestamp("1995-08-01 14:52:01")
		storm_end = pd.Timestamp("1995-08-03 04:36:13")
		ts = df[self.time_column]
		storm_mask = (ts >= storm_start) & (ts <= storm_end)

		df.loc[storm_mask & (df["requests_count"] == 0), "requests_count"] = pd.NA

		df_indexed = df.set_index(self.time_column)
		df_indexed["requests_count"] = df_indexed["requests_count"].interpolate(
			method="time", limit_direction="both"
		)
		df_interpolated = df_indexed.reset_index()
		df_interpolated["requests_count"] = (
			df_interpolated["requests_count"].round().astype(int).clip(lower=0)
		)
		return df_interpolated

