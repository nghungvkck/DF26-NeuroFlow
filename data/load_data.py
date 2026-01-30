import pandas as pd

def load_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # đảm bảo numeric
    numeric_cols = df.columns.drop("timestamp")
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df
