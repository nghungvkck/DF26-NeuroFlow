from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataNormalizer:
	time_column: str = "timestamp"
	timezone: str = "UTC"

	def run(self, df: pd.DataFrame) -> pd.DataFrame:
		if "timestamp_str" in df.columns:
			df[self.time_column] = pd.to_datetime(
				df["timestamp_str"],
				format="%d/%b/%Y:%H:%M:%S %z",
				utc=True,
				errors="coerce"
			)
			df = df.drop(columns=["timestamp_str"], errors="ignore")
		elif self.time_column in df.columns:
			df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True, errors="coerce")
		
		df = df.dropna(subset=[self.time_column])
		df = df.sort_values(self.time_column).reset_index(drop=True)
		return df

