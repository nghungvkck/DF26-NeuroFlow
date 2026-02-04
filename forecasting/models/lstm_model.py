from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras


@dataclass
class ResidualLSTM:
    """LSTM model for residual forecasting."""

    input_steps: int
    hidden_units: int
    layers: int

    def build(self) -> keras.Model:
        inputs = keras.Input(shape=(self.input_steps, 1))
        x = inputs
        for i in range(self.layers - 1):
            x = keras.layers.LSTM(self.hidden_units, return_sequences=True)(x)
        x = keras.layers.LSTM(self.hidden_units)(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    @staticmethod
    def create_sequences(series: np.ndarray, steps: int):
        X, y = [], []
        for i in range(len(series) - steps):
            X.append(series[i:i + steps])
            y.append(series[i + steps])
        X = np.array(X).reshape(-1, steps, 1)
        y = np.array(y)
        return X, y

    @staticmethod
    def save_model(model: keras.Model, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
