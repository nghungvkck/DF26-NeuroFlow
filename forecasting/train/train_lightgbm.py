from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.train.common import (
    ensure_dir,
    iter_windows,
    load_config,
    prepare_window_data,
)
from forecasting.evaluate.evaluate import MetricEvaluator
from forecasting.artifacts import ArtifactManager
import lightgbm as lgb


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

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 63,
            "min_data_in_leaf": 62,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": 42,
            "n_estimators": 1000,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_test)
        model_path = artifacts.get_model_path("lightgbm", window, ext=".txt")
        model.save_model(str(model_path))

        metrics_result = evaluator.evaluate(y_test.to_numpy(), preds)
        artifacts.save_metrics(metrics_result, "lightgbm", window)
        
        metrics_frame = evaluator.to_frame(metrics_result, "lightgbm", window)
        all_metrics.append(metrics_frame)

        pred_df = pd.DataFrame({
            "timestamp": test_df[cfg["data"]["time_column"]].values,
            "y_true": y_test.values,
            "y_pred": preds,
        })
        artifacts.save_predictions(pred_df, "lightgbm", window)

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        artifacts.save_evaluation_summary(metrics_df, "lightgbm")


if __name__ == "__main__":
    run()
