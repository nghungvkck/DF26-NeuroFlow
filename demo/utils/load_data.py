from __future__ import annotations

import os

import pandas as pd


def load_traffic_data(file_name: str, data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Available: {sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])}"
        )
    df = pd.read_csv(file_path)

    if "timestamp" not in df.columns and "ds" in df.columns:
        df = df.rename(columns={"ds": "timestamp"})
    if "requests_count" not in df.columns and "y_true" in df.columns:
        df = df.rename(columns={"y_true": "requests_count"})
    if "requests_count" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": "requests_count"})
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df