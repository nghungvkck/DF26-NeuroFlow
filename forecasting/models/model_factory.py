from typing import Dict, Type
from forecasting.models.base_model import BaseTimeSeriesModel


class ModelFactory:
    """
    Factory tạo model dựa trên registry.
    """

    _REGISTRY: Dict[str, Type[BaseTimeSeriesModel]] = {}

    @classmethod
    def register(cls, model_cls: Type[BaseTimeSeriesModel]):
        """
        Đăng ký model class vào registry
        """
        name = getattr(model_cls, "name", None)

        if name in cls._REGISTRY:
            raise KeyError(f"Model '{name}' already registered")

        cls._REGISTRY[name] = model_cls

    @classmethod
    def create(cls, config: dict) -> BaseTimeSeriesModel:
        """
        Parameters
        ----------
        config : dict
            {
              "name": "xgboost",
              "params": {...}
            }
        """
        name = config["name"]
        params = config.get("params", {})

        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available models: {list(cls._REGISTRY.keys())}"
            )

        return cls._REGISTRY[name](params)

    @classmethod
    def available_models(cls):
        return list(cls._REGISTRY.keys())


if __name__ == "__main__":

    model_cfg = {
        "name": "hybrid_prophet_lstm",
        "params": {
            "window_size": 60,
            "lstm_units": 32,
            "period": 1440
        }
    }

    model = ModelFactory.create(model_cfg)

    # model.fit(X_train, y_train)
