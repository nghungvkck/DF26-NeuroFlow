from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeWindowAggregator:
	window_rule: str
	time_column: str = "timestamp"
	window_name: str | None = None

	def run(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.time_column not in df.columns:
			return df

		df = df.copy()
		df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True, errors="coerce")
		df = df.dropna(subset=[self.time_column])
		df = df.set_index(self.time_column)

		agg_dict = {"host": "count"}
		if "bytes" in df.columns:
			agg_dict["bytes"] = "sum"
		if "status_code" in df.columns:
			df["error_flag"] = (df["status_code"] >= 500).astype(int)
			agg_dict["error_flag"] = "mean"

		out = df.resample(self.window_rule).agg(agg_dict)
		out = out.rename(columns={
			"host": "requests_count",
			"bytes": "total_bytes",
			"error_flag": "error_rate",
		})
		out["error_rate"] = out.get("error_rate", 0).round(4)
		out = out.reset_index()
		if self.window_name:
			out["window_size"] = self.window_name
		return out

