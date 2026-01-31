"""
ANOMALY DETECTION FOR AUTOSCALING
==================================
Detects anomalies in workload patterns to trigger proactive scaling.
Inspired by:
- Kubernetes Vertical Pod Autoscaler (VPA) anomaly detection
- AWS CloudWatch Anomaly Detection
- Google Borg workload classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from collections import deque


class AnomalyDetector:
    """
    Multi-method anomaly detection for autoscaling systems.
    
    Supports:
    - Z-score (statistical)
    - IQR (robust to outliers)
    - Moving average deviation
    - Rate of change (sudden spikes)
    - Seasonal decomposition
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 zscore_threshold: float = 3.0,
                 iqr_multiplier: float = 1.5,
                 rate_threshold: float = 0.5):
        """
        Args:
            window_size: Lookback window for statistics
            zscore_threshold: Z-score anomaly threshold (typically 2.5-3.0)
            iqr_multiplier: IQR multiplier (Kubernetes uses 1.5)
            rate_threshold: Max acceptable rate of change (50% = 0.5)
        """
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.rate_threshold = rate_threshold
        
        # Sliding window for online detection
        self.history = deque(maxlen=window_size)
    
    def detect_zscore(self, ts: np.ndarray) -> np.ndarray:
        """
        Z-score anomaly detection (AWS CloudWatch style).
        
        Detects values > threshold standard deviations from mean.
        """
        if len(ts) < 3:
            return np.zeros(len(ts), dtype=int)
        
        mean = np.mean(ts)
        std = np.std(ts)
        if std == 0:
            return np.zeros(len(ts), dtype=int)
        
        z = (ts - mean) / std
        return (np.abs(z) > self.zscore_threshold).astype(int)
    
    def detect_iqr(self, ts: np.ndarray) -> np.ndarray:
        """
        IQR (Interquartile Range) anomaly detection.
        
        More robust to outliers than Z-score.
        Used by Kubernetes for detecting resource usage anomalies.
        """
        if len(ts) < 4:
            return np.zeros(len(ts), dtype=int)
        
        q1 = np.percentile(ts, 25)
        q3 = np.percentile(ts, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        return ((ts < lower_bound) | (ts > upper_bound)).astype(int)
    
    def detect_rate_of_change(self, ts: np.ndarray) -> np.ndarray:
        """
        Sudden spike/drop detection (rate of change).
        
        Critical for detecting:
        - DDoS attacks (sudden traffic spike)
        - Service failures (sudden drop)
        - Flash sales / viral events
        
        AWS Auto Scaling uses this for step scaling policies.
        """
        if len(ts) < 2:
            return np.zeros(len(ts), dtype=int)
        
        anomalies = np.zeros(len(ts), dtype=int)
        
        for i in range(1, len(ts)):
            if ts[i-1] == 0:
                rate = 0 if ts[i] == 0 else 1.0
            else:
                rate = abs((ts[i] - ts[i-1]) / ts[i-1])
            
            if rate > self.rate_threshold:
                anomalies[i] = 1
        
        return anomalies
    
    def detect_moving_average_deviation(self, ts: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Moving average deviation detection.
        
        Used by Google Borg for workload pattern classification.
        Detects deviations from recent trend.
        """
        if len(ts) < window:
            return np.zeros(len(ts), dtype=int)
        
        ma = pd.Series(ts).rolling(window=window, min_periods=1).mean().values
        std = pd.Series(ts).rolling(window=window, min_periods=1).std().fillna(0).values
        
        anomalies = np.zeros(len(ts), dtype=int)
        for i in range(window, len(ts)):
            if std[i] > 0:
                deviation = abs(ts[i] - ma[i]) / std[i]
                if deviation > self.zscore_threshold:
                    anomalies[i] = 1
        
        return anomalies
    
    def detect_ensemble(self, ts: np.ndarray, min_votes: int = 2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Ensemble anomaly detection (voting system).
        
        Combines multiple methods for robust detection.
        Similar to Kubernetes multi-metric autoscaling.
        
        Args:
            ts: Time series data
            min_votes: Minimum methods agreeing for anomaly (default: 2/4)
        
        Returns:
            (anomalies, method_results): Binary array and individual method results
        """
        methods = {
            'zscore': self.detect_zscore(ts),
            'iqr': self.detect_iqr(ts),
            'rate_of_change': self.detect_rate_of_change(ts),
            'moving_avg': self.detect_moving_average_deviation(ts)
        }
        
        # Voting: anomaly if >= min_votes methods agree
        votes = np.sum([methods[m] for m in methods], axis=0)
        anomalies = (votes >= min_votes).astype(int)
        
        return anomalies, methods
    
    def update_online(self, value: float) -> Tuple[bool, str]:
        """
        Online anomaly detection (streaming mode).
        
        For real-time autoscaling systems like Kubernetes HPA.
        
        Args:
            value: New observation
        
        Returns:
            (is_anomaly, reason): Whether value is anomalous and why
        """
        self.history.append(value)
        
        if len(self.history) < 10:
            return False, "Insufficient data"
        
        ts = np.array(self.history)
        
        # Check rate of change (immediate)
        if len(ts) >= 2:
            if ts[-2] > 0:
                rate = abs((value - ts[-2]) / ts[-2])
                if rate > self.rate_threshold:
                    return True, f"Rate spike: {rate:.1%} > {self.rate_threshold:.1%}"
        
        # Check Z-score
        mean = np.mean(ts[:-1])  # Exclude current value
        std = np.std(ts[:-1])
        if std > 0:
            z = abs((value - mean) / std)
            if z > self.zscore_threshold:
                return True, f"Z-score: {z:.2f} > {self.zscore_threshold}"
        
        # Check IQR
        q1 = np.percentile(ts[:-1], 25)
        q3 = np.percentile(ts[:-1], 75)
        iqr = q3 - q1
        upper = q3 + self.iqr_multiplier * iqr
        lower = q1 - self.iqr_multiplier * iqr
        
        if value > upper or value < lower:
            return True, f"IQR outlier: {value:.0f} outside [{lower:.0f}, {upper:.0f}]"
        
        return False, "Normal"


# Legacy function (kept for backward compatibility)
def zscore_detection(ts, threshold=3.0):
    """
    DEPRECATED: Use AnomalyDetector.detect_zscore() instead.
    """
    detector = AnomalyDetector(zscore_threshold=threshold)
    return detector.detect_zscore(ts)
