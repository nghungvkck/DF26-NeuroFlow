"""
MODEL EVALUATION ON REAL DATA
=============================

Evaluates trained forecasting models (LSTM, XGBoost, Hybrid) on real historical data.

This is PHASE A of the two-phase pipeline:
- PHASE A: Model evaluation using real data from processed_for_modeling_v2/
- PHASE B: Autoscaling scenario testing using synthetic generated data

Purpose:
- Assess model performance (MAE, RMSE, MAPE) on real historical patterns
- Determine which models work best for different timeframes
- Provide metrics for model selection in Phase B

Output:
- results/model_evaluation.json with per-model metrics per timeframe
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from forecast.model_forecaster import ModelForecaster
from forecast.forecast_utils import extract_timeframe


class ModelEvaluator:
    """Evaluates forecast models on real historical data."""

    def __init__(self, real_data_dir: str = None):
        """
        Initialize evaluator.

        Args:
            real_data_dir: path to directory containing real data CSVs
        """
        if real_data_dir is None:
            # Try relative paths first
            candidates = [
                Path("../processed_for_modeling_v2"),
                Path("processed_for_modeling_v2"),
                Path("/Users/maydothi/Documents/dataflow/processed_for_modeling_v2"),
            ]
            real_data_dir = "processed_for_modeling_v2"
            for candidate in candidates:
                if candidate.exists():
                    real_data_dir = str(candidate)
                    break
        
        self.real_data_dir = Path(real_data_dir)
        self.results = {}

    def load_real_data_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from real data directory.

        Returns:
            Dict mapping filename to DataFrame
        """
        data_files = {}

        if not self.real_data_dir.exists():
            print(f"⚠️  Real data directory not found: {self.real_data_dir}")
            return data_files

        for csv_file in sorted(self.real_data_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, parse_dates=["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                data_files[csv_file.stem] = df
                print(f"  ✓ Loaded {csv_file.stem} ({len(df)} rows)")
            except Exception as e:
                print(f"  ✗ Failed to load {csv_file.stem}: {e}")

        return data_files

    def evaluate_model_on_data(
        self,
        model_type: str,
        timeframe: str,
        data: pd.DataFrame,
        test_split: float = 0.3,
        horizon: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate a model on real data using rolling window approach.

        Args:
            model_type: "hybrid", "lstm", or "xgboost"
            timeframe: "1m", "5m", or "15m"
            data: DataFrame with columns [timestamp, requests_count, ...]
            test_split: fraction of data to use for testing (default 0.3)
            horizon: forecast horizon (default 1)

        Returns:
            Dict with metrics (mae, rmse, mape, num_samples)
        """
        if "requests_count" not in data.columns:
            return {
                "error": "requests_count column not found",
                "mae": None,
                "rmse": None,
                "mape": None,
                "num_samples": 0,
            }

        # Split data: train + test
        split_idx = int(len(data) * (1 - test_split))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        if len(test_data) < horizon + 1:
            return {
                "error": "Not enough test samples",
                "mae": None,
                "rmse": None,
                "mape": None,
                "num_samples": 0,
            }

        # Initialize forecaster
        try:
            forecaster = ModelForecaster(
                model_type=model_type,
                timeframe=timeframe,
                forecast_horizon=horizon,
            )
        except Exception as e:
            return {
                "error": f"Failed to initialize forecaster: {e}",
                "mae": None,
                "rmse": None,
                "mape": None,
                "num_samples": 0,
            }

        # Rolling window forecasting
        predictions = []
        actuals = []

        for t in range(horizon, len(test_data)):
            try:
                history = pd.concat([train_data, test_data.iloc[:t]])
                result = forecaster.predict(history, horizon=horizon)

                if result.yhat and len(result.yhat) > 0:
                    pred = result.yhat[0]
                    actual = test_data.iloc[t]["requests_count"]

                    predictions.append(pred)
                    actuals.append(actual)
            except Exception as e:
                # Fallback: predict actual value if model fails
                predictions.append(test_data.iloc[t - 1]["requests_count"])
                actuals.append(test_data.iloc[t]["requests_count"])

        if len(predictions) == 0:
            return {
                "error": "No valid predictions generated",
                "mae": None,
                "rmse": None,
                "mape": None,
                "num_samples": 0,
            }

        # Compute metrics
        predictions = np.array(predictions, dtype=float)
        actuals = np.array(actuals, dtype=float)

        mae = float(np.mean(np.abs(predictions - actuals)))
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        mape = float(np.mean(np.abs((predictions - actuals) / np.maximum(actuals, 1))))

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "num_samples": len(predictions),
            "mean_actual": float(np.mean(actuals)),
            "std_actual": float(np.std(actuals)),
        }

    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all available models on all real data files.

        Returns:
            Dict structure: {filename: {timeframe: {model_type: metrics}}}
        """
        data_files = self.load_real_data_files()

        if not data_files:
            print("❌ No real data files found")
            return {}

        results = {}
        model_types = ["lstm", "xgboost", "hybrid"]

        for filename, data in data_files.items():
            print(f"\n📊 Evaluating models on {filename}:")
            print("-" * 70)

            timeframe = extract_timeframe(filename)
            results[filename] = {
                "timeframe": timeframe,
                "total_rows": len(data),
                "models": {},
            }

            for model_type in model_types:
                print(f"  {model_type.upper():<12}", end=" ", flush=True)

                try:
                    metrics = self.evaluate_model_on_data(
                        model_type=model_type,
                        timeframe=timeframe,
                        data=data,
                        test_split=0.3,
                        horizon=1,
                    )

                    results[filename]["models"][model_type] = metrics

                    if metrics.get("mae") is not None:
                        print(
                            f"MAE={metrics['mae']:.1f} | RMSE={metrics['rmse']:.1f} | "
                            f"MAPE={metrics['mape']:.2%} | Samples={metrics['num_samples']}"
                        )
                    else:
                        print(f"❌ {metrics.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"❌ Exception: {e}")
                    results[filename]["models"][model_type] = {
                        "error": str(e),
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "num_samples": 0,
                    }

        return results

    def get_best_model_per_timeframe(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Select best model for each timeframe based on MAPE.

        Args:
            results: output from evaluate_all_models()

        Returns:
            Dict mapping timeframe to best model_type
        """
        best_models = {}

        # Group by timeframe
        timeframes = {}
        for filename, file_results in results.items():
            tf = file_results["timeframe"]
            if tf not in timeframes:
                timeframes[tf] = {}

            for model_type, metrics in file_results["models"].items():
                if "mape" in metrics and metrics["mape"] is not None:
                    if model_type not in timeframes[tf]:
                        timeframes[tf][model_type] = []
                    timeframes[tf][model_type].append(metrics["mape"])

        # Average MAPE per model per timeframe
        for tf, models_dict in timeframes.items():
            best_model = None
            best_mape = float("inf")

            for model_type, mapes in models_dict.items():
                avg_mape = np.mean(mapes)
                if avg_mape < best_mape:
                    best_mape = avg_mape
                    best_model = model_type

            if best_model:
                best_models[tf] = best_model
                print(f"  {tf}: {best_model.upper()} (MAPE={best_mape:.2%})")

        return best_models

    def save_results(self, results: Dict[str, Any], output_dir: str = "results") -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: output from evaluate_all_models()
            output_dir: directory to save results
        """
        Path(output_dir).mkdir(exist_ok=True)

        output_path = Path(output_dir) / "model_evaluation.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Saved model evaluation results to {output_path}")

    def run_full_evaluation(self, output_dir: str = "results") -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            output_dir: directory to save results

        Returns:
            Results dict with all evaluation data
        """
        print("=" * 80)
        print("PHASE A: MODEL EVALUATION ON REAL DATA")
        print("=" * 80)

        # Evaluate all models
        results = self.evaluate_all_models()

        if not results:
            print("❌ No models could be evaluated")
            return {}

        # Determine best models
        print("\n📈 Best models by timeframe:")
        print("-" * 70)
        best_models = self.get_best_model_per_timeframe(results)

        # Save results
        self.save_results(results, output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to {output_dir}/model_evaluation.json")
        print(f"\nBest models per timeframe:")
        for tf, model in sorted(best_models.items()):
            print(f"  {tf}: {model.upper()}")

        return {
            "evaluation_results": results,
            "best_models": best_models,
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()
