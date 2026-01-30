import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .base_forecast import BaseForecaster

class ARIMAForecaster(BaseForecaster):
    def __init__(self, order=(2,1,2)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, ts: pd.Series):
        self.model = ARIMA(ts, order=self.order)
        self.fitted = self.model.fit()

    def predict(self, horizon: int) -> pd.Series:
        return self.fitted.forecast(steps=horizon)
