from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.preprocess.pipeline import DataPipeline
from forecasting.preprocess.missing_handler import MissingValueHandler
from forecasting.preprocess.feature_engineering import FeatureEngineer




WINDOW_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1H",
}


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_windows(cfg: dict) -> Iterable[tuple[str, str, str, str]]:
    train_raw = cfg["data"]["paths"]["train_raw"]
    test_raw = cfg["data"]["paths"]["test_raw"]

    for window_name, window_rule in WINDOW_RULES.items():
        if window_name != "1h":
            yield window_name, train_raw, test_raw, window_rule


def prepare_window_data(cfg: dict, train_path: str, test_path: str, window_rule: str | None):
    pipeline = DataPipeline(
        window_rule=window_rule,
        time_column=cfg["data"]["time_column"],
        target_column=cfg["data"]["target"],
    )

    raw_df = pipeline.run_raw([train_path, test_path])
    agg_df = pipeline.aggregate(raw_df)

    time_col = cfg["data"]["time_column"]
    agg_df[time_col] = pd.to_datetime(agg_df[time_col], utc=True, errors="coerce")
    agg_df = agg_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    train_end = pd.Timestamp("1995-08-22 23:59:59", tz="UTC")
    test_start = pd.Timestamp("1995-08-23 00:00:00", tz="UTC")

    train_df = agg_df[agg_df[time_col] < train_end].copy()
    test_df = agg_df[agg_df[time_col] >= test_start].copy()

    window_name = {"1min": "1m", "5min": "5m", "15min": "15m"}.get(window_rule or "", None)
    missing_handler = MissingValueHandler(time_column=time_col)
    train_df = missing_handler.reindex_and_fill(train_df, window_name=window_name)
    test_df = missing_handler.reindex_and_fill(test_df, window_name=window_name)

    feature_engineer = FeatureEngineer(time_column=time_col, target_column=cfg["data"]["target"])
    train_df = feature_engineer.run(train_df, train_df=train_df, window_name=window_name)
    test_df = feature_engineer.run(test_df, train_df=train_df, window_name=window_name)
    features = feature_engineer.feature_columns(train_df)

    return train_df, test_df, features


def load_preprocessed_data(cfg: dict, train_path: str, test_path: str):
    train_df = pd.read_csv(train_path, parse_dates=[cfg["data"]["time_column"]])
    test_df = pd.read_csv(test_path, parse_dates=[cfg["data"]["time_column"]])
    train_df = train_df.sort_values(cfg["data"]["time_column"]).reset_index(drop=True)
    test_df = test_df.sort_values(cfg["data"]["time_column"]).reset_index(drop=True)
    return train_df, test_df


def split_xy(df: pd.DataFrame, features: list[str], target: str):
    X = df[features]
    y = df[target]
    return X, y


def time_split(df: pd.DataFrame, ratio: float):
    split_idx = int(len(df) * (1 - ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
