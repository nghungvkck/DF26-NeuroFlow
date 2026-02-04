from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.train.common import (
    iter_windows,
    load_config,
    prepare_window_data,
)
from forecasting.evaluate.evaluate import MetricEvaluator
from forecasting.artifacts import ArtifactManager
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def run(config_path: str = "configs/train_config.yaml"):
    cfg = load_config(config_path)
    metrics = cfg.get("evaluation", {}).get("metrics", ["mae", "rmse", "mape", "smape"])
    evaluator = MetricEvaluator(metrics=metrics)

    artifacts = ArtifactManager(Path(__file__).parent.parent / "artifacts")

    all_metrics = []
    for window, train_path, test_path, window_rule in iter_windows(cfg):
        train_df, test_df, features = prepare_window_data(
            cfg, train_path, test_path, window_rule
        )
        target = cfg["data"]["target"]

        X_train_full = train_df[features].fillna(0).values
        y_train_full = train_df[target].values
        X_test = test_df[features].fillna(0).values
        y_test = test_df[target].values

        split_idx = int(len(X_train_full) * 0.8)
        X_train = X_train_full[:split_idx]
        y_train = y_train_full[:split_idx]
        X_valid = X_train_full[split_idx:]
        y_valid = y_train_full[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_valid_scaled, y_valid)],
            verbose=False,
        )
        preds = model.predict(X_test_scaled)

        model_path = artifacts.get_model_path("xgboost", window, ext=".json")
        model.save_model(str(model_path))

        metrics_result = evaluator.evaluate(y_test, preds)
        artifacts.save_metrics(metrics_result, "xgboost", window)
        
        metrics_frame = evaluator.to_frame(metrics_result, "xgboost", window)
        all_metrics.append(metrics_frame)

        pred_df = pd.DataFrame({
            "timestamp": test_df[cfg["data"]["time_column"]].values,
            "y_true": y_test,
            "y_pred": preds,
        })
        artifacts.save_predictions(pred_df, "xgboost", window)

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        artifacts.save_evaluation_summary(metrics_df, "xgboost")


if __name__ == "__main__":
    run()
