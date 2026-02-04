from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
import pandas as pd


class ArtifactManager:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.metrics_dir = self.base_dir / "metrics"
        self.predictions_dir = self.base_dir / "predictions"
        self._ensure_dirs()

    def _ensure_dirs(self):
        for dir_path in [self.models_dir, self.metrics_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        metrics: dict[str, float],
        model_name: str,
        window: str,
    ) -> Path:
        path = self.metrics_dir / f"{model_name}_{window}_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        return path

    def save_predictions(
        self,
        df: pd.DataFrame,
        model_name: str,
        window: str,
    ) -> Path:
        path = self.predictions_dir / f"{model_name}_{window}_predictions.csv"
        df.to_csv(path, index=False)
        return path

    def save_evaluation_summary(
        self,
        df: pd.DataFrame,
        model_name: str,
    ) -> Path:
        path = self.metrics_dir / f"{model_name}_evaluation.csv"
        df.to_csv(path, index=False)
        return path

    def save_artifact_metadata(
        self,
        metadata: dict[str, Any],
    ) -> Path:
        path = self.base_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        return path

    def get_model_path(self, model_name: str, window: str, ext: str = ".txt") -> Path:
        return self.models_dir / f"{model_name}_{window}{ext}"

    def list_artifacts(self) -> dict[str, list[Path]]:
        return {
            "models": list(self.models_dir.glob("*")),
            "metrics": list(self.metrics_dir.glob("*")),
            "predictions": list(self.predictions_dir.glob("*")),
        }
