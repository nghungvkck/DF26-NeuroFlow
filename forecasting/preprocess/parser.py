from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class ApacheLogParser:
	input_column: str = "raw_log"

	LOG_PATTERN = re.compile(
		r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d{3}) (\S+)'
	)

	def run(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.input_column not in df.columns:
			if "value" in df.columns:
				self.input_column = "value"
			else:
				return df

		parsed = df[self.input_column].apply(self._parse_line)
		parsed_df = pd.DataFrame(parsed.tolist(), columns=[
			"host",
			"timestamp_str",
			"method",
			"url",
			"protocol",
			"status_code",
			"bytes",
		])
		
		valid_mask = parsed_df["timestamp_str"].notna()
		parsed_df = parsed_df[valid_mask].copy()
		
		parsed_df["status_code"] = pd.to_numeric(parsed_df["status_code"], errors="coerce")
		parsed_df["bytes"] = pd.to_numeric(parsed_df["bytes"], errors="coerce").fillna(0)
		return parsed_df

	def _parse_line(self, line: str | None):
		if not isinstance(line, str):
			return [None] * 7
		match = self.LOG_PATTERN.match(line)
		if not match:
			return [None] * 7
		return list(match.groups())

