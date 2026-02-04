import yaml
from types import SimpleNamespace
from pathlib import Path


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


class PipelineConfig:
    """
    Load và quản lý config cho training pipeline
    """

    def __init__(self, path: str):
        raw = self._load(path)
        ns = _to_ns(raw)

        self.data = ns.data
        self.feature = ns.feature
        self.model = ns.model
        self.evaluation = ns.evaluation

    @staticmethod
    def _load(path: str) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
