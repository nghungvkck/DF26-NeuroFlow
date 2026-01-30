from abc import ABC, abstractmethod
import pandas as pd

class BaseForecaster(ABC):
    """
    Interface chung cho mọi mô hình dự báo
    """

    @abstractmethod
    def fit(self, ts: pd.Series):
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """
        Return forecast for next `horizon` steps
        """
        pass
  
