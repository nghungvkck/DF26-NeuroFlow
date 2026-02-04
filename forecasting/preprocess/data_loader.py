from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataLoader:
	time_column: str = "timestamp"
	target_column: str = "requests_count"

	def read(self, path: str | Path) -> pd.DataFrame:
		file_path = Path(path)
		if not file_path.is_absolute():
			candidate = ROOT / file_path
			if candidate.exists():
				file_path = candidate
		if not file_path.exists():
			raise FileNotFoundError(f"File not found: {file_path}")

		if file_path.suffix.lower() == ".txt":
			with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
				lines = [line.strip() for line in f if line.strip()]
			return pd.DataFrame({"raw_log": lines})
		elif file_path.suffix.lower() == ".csv":
			return pd.read_csv(file_path)
		elif file_path.suffix.lower() == ".parquet":
			return pd.read_parquet(file_path)
		else:
			raise ValueError(f"Unsupported file type: {file_path.suffix}")

