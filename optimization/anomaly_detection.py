from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class AnomalyDetector:
    def __init__(
        self,
        window_size: int = 50,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        rate_threshold: float = 0.8,
        min_votes: int = 2,
    ) -> None:
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.rate_threshold = rate_threshold
        self.min_votes = min_votes
        self.history = deque(maxlen=window_size)

    def update_online(self, value: float) -> Tuple[bool, str, float]:
        self.history.append(value)
        if len(self.history) < 10:
            return False, "insufficient_data", 0.0

        ts = np.array(self.history, dtype=float)
        votes = 0
        methods_triggered = []

        if len(ts) >= 2:
            prev = ts[-2]
            if prev > 0:
                rate = abs((value - prev) / prev)
                if rate > self.rate_threshold:
                    votes += 1
                    methods_triggered.append(f"rate({rate:.2f})")

        mean = np.mean(ts[:-1])
        std = np.std(ts[:-1])
        if std > 0:
            z = abs((value - mean) / std)
            if z > self.zscore_threshold:
                votes += 1
                methods_triggered.append(f"zscore({z:.2f})")

        q1 = np.percentile(ts[:-1], 25)
        q3 = np.percentile(ts[:-1], 75)
        iqr = q3 - q1
        upper = q3 + self.iqr_multiplier * iqr
        lower = q1 - self.iqr_multiplier * iqr
        if value > upper or value < lower:
            votes += 1
            gap = max(abs(value - upper), abs(lower - value))
            methods_triggered.append(f"iqr({gap:.0f})")

        ma_window = min(10, len(ts) - 1)
        if len(ts) > ma_window:
            ma = np.mean(ts[-ma_window-1:-1])
            ma_std = np.std(ts[-ma_window-1:-1])
            if ma_std > 0:
                ma_dev = abs(value - ma) / ma_std
                if ma_dev > self.zscore_threshold:
                    votes += 1
                    methods_triggered.append(f"ma_dev({ma_dev:.2f})")

        if votes >= self.min_votes:
            method = f"ensemble_{votes}/4:" + "+".join(methods_triggered)
            confidence = float(votes) / 4.0
            return True, method, confidence

        return False, "normal", 0.0
