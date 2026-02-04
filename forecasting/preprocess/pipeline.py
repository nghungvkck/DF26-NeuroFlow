from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecasting.preprocess.aggregator import TimeWindowAggregator
from forecasting.preprocess.data_loader import DataLoader
from forecasting.preprocess.feature_engineering import FeatureEngineer
from forecasting.preprocess.missing_handler import MissingValueHandler
from forecasting.preprocess.normalizer import DataNormalizer
from forecasting.preprocess.parser import ApacheLogParser


@dataclass
class DataPipeline:
	"""Unified pipeline used by all models."""

	window_rule: str | None = None
	time_column: str = "timestamp"
	target_column: str = "requests_count"

	def run_raw(self, paths: list[str]) -> pd.DataFrame:
		loader = DataLoader(time_column=self.time_column, target_column=self.target_column)
		parser = ApacheLogParser()
		normalizer = DataNormalizer(time_column=self.time_column)

		frames = []
		for path in paths:
			df = loader.read(path)
			df = parser.run(df)
			df = normalizer.run(df)
			frames.append(df)

		if not frames:
			return pd.DataFrame()
		out = pd.concat(frames, ignore_index=True)
		out = out.sort_values(self.time_column).reset_index(drop=True)
		return out

	def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.window_rule:
			aggregator = TimeWindowAggregator(
				window_rule=self.window_rule,
				time_column=self.time_column,
				window_name=self._window_name_from_rule(),
			)
			return aggregator.run(df)
		return df

	def run(self, path: str) -> tuple[pd.DataFrame, list[str]]:
		feature_engineer = FeatureEngineer(time_column=self.time_column, target_column=self.target_column)
		missing_handler = MissingValueHandler(time_column=self.time_column)

		df = self.run_raw([path])
		df = self.aggregate(df)
		df = missing_handler.reindex_and_fill(df, window_name=self._window_name_from_rule())
		df = feature_engineer.run(df, window_name=self._window_name_from_rule())
		features = feature_engineer.feature_columns(df)
		return df, features

	def _window_name_from_rule(self) -> str | None:
		mapping = {
			"1min": "1m",
			"5min": "5m",
			"15min": "15m",
		}
		return mapping.get(self.window_rule or "", None)

