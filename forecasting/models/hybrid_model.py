from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from forecasting.models.lstm_model import ResidualLSTM
from forecasting.models.prophet_model import ProphetModel


@dataclass
class HybridProphetLSTM:
    """Hybrid model: Prophet + LSTM residuals."""

    prophet_params: dict
    lstm_hidden: int
    lstm_layers: int
    lstm_steps: int
    epochs: int
    batch_size: int

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, time_column: str, target_column: str):
        self.prophet = ProphetModel(self.prophet_params)
        self.prophet.fit(train_df, time_column, target_column)

        train_pred = self.prophet.predict(train_df, time_column).to_numpy()
        valid_pred = self.prophet.predict(valid_df, time_column).to_numpy()

        residuals_train = train_df[target_column].to_numpy() - train_pred
        residuals_valid = valid_df[target_column].to_numpy() - valid_pred

        self.scaler = MinMaxScaler()
        self.scaler.fit(residuals_train.reshape(-1, 1))
        scaled_train = self.scaler.transform(residuals_train.reshape(-1, 1)).flatten()
        scaled_valid = self.scaler.transform(residuals_valid.reshape(-1, 1)).flatten()

        lstm_helper = ResidualLSTM(self.lstm_steps, self.lstm_hidden, self.lstm_layers)
        self.lstm_model = lstm_helper.build()

        X_train, y_train = lstm_helper.create_sequences(scaled_train, self.lstm_steps)
        X_valid, y_valid = lstm_helper.create_sequences(scaled_valid, self.lstm_steps)

        self.lstm_model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, df: pd.DataFrame, time_column: str, target_column: str) -> np.ndarray:
        prophet_pred = self.prophet.predict(df, time_column).to_numpy()
        residuals = df[target_column].to_numpy() - prophet_pred
        scaled_residuals = self.scaler.transform(residuals.reshape(-1, 1)).flatten()

        lstm_helper = ResidualLSTM(self.lstm_steps, self.lstm_hidden, self.lstm_layers)
        X_seq, _ = lstm_helper.create_sequences(scaled_residuals, self.lstm_steps)
        if len(X_seq) == 0:
            return prophet_pred

        pred_residuals = self.lstm_model.predict(X_seq, verbose=0).flatten()
        pred_residuals = self.scaler.inverse_transform(pred_residuals.reshape(-1, 1)).flatten()

        aligned_prophet = prophet_pred[self.lstm_steps:]
        return aligned_prophet + pred_residuals

    def save(self, prophet_path: str | Path, lstm_path: str | Path, scaler_path: str | Path):
        self.prophet.save(prophet_path)
        ResidualLSTM.save_model(self.lstm_model, lstm_path)
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as f:
            pickle.dump(self.scaler, f)
