from pathlib import Path
import pandas as pd

from forecasting.utils.config import PipelineConfig
from forecasting.models.model_factory import ModelFactory


class TrainingPipeline:
    """
    Pipeline điều phối toàn bộ quá trình training forecasting model.
    """

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            Đường dẫn tới file config.yaml
        """
        self.root_dir = Path(__file__).resolve().parents[1]
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.root_dir / cfg_path
        self.cfg = PipelineConfig(str(cfg_path))

    def load_data(self, time_window: str):
        """
        Tải dữ liệu theo time window (1m, 5m, 15m).
        """
        train_path = self.cfg.data.paths[f"train_{time_window}"]
        test_path = self.cfg.data.paths[f"test_{time_window}"]

        train = pd.read_csv(self.root_dir / train_path)
        test = pd.read_csv(self.root_dir / test_path)

        return train, test

    def train(self, model_name: str, time_window: str):
        """
        Huấn luyện model.

        Parameters
        ----------
        model_name : str
            Tên model (lightgbm, xgboost, ...)
        time_window : str
            Time window (1m, 5m, 15m)

        Returns
        -------
        model
            Trained model instance
        """
        train, test = self.load_data(time_window)

        model_params = getattr(self.cfg.model, model_name, {})
        model = ModelFactory.create(model_name, params=model_params)

        print(f"Training {model_name} model on {time_window} data...")

        # TODO: Extract features, train model

        return model

    def run(self) -> dict:
        """Orchestrate training (placeholder)."""
        return {}


if __name__ == "__main__":
    pipeline = TrainingPipeline(config_path="configs/train_config.yaml")
    metrics = pipeline.run()
    print(metrics)