"""
HYBRID AUTOSCALING POLICY
===========================
Multi-level decision hierarchy for robustness.

Priority order:
1. ANOMALY: Detect DDoS/spikes and scale aggressively
2. EMERGENCY: If CPU > critical threshold → scale out immediately
3. PREDICTIVE: Use forecast-based decision if available
4. REACTIVE: Fallback to real-time request threshold
5. HOLD: No scaling decision needed

This ensures:
- Proactive spike/DDoS detection and mitigation
- Emergency protection against sudden spikes
- Proactive scaling when forecast is reliable
- Graceful degradation to reactive when forecast fails
"""

import numpy as np
import sys
import os

# Add parent directory to path for anomaly detector import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anomaly.anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False


class HybridAutoscaler:
    """
    Multi-layer autoscaling combining CPU emergency detection,
    predictive forecasting, and reactive fallback.
    """
    
    # Standard safety margin: 20% headroom for all strategies
    FORECAST_SAFETY_MARGIN = 0.80
    
    def __init__(
        self,
        capacity_per_server,
        min_servers=2,
        max_servers=20,
        # Anomaly detection layer
        enable_anomaly_detection=True,
        anomaly_rate_threshold=0.5,     # 50% spike = anomaly
        anomaly_scale_multiplier=1.5,   # Scale by 1.5x on anomaly
        # Emergency layer (CPU-based)
        cpu_critical_th=0.95,           # Immediate scale-out trigger
        cpu_per_request=0.05,
        # Predictive layer
        predictive_weight=0.9,          # Weight between predictive and reactive
        forecast_safety_margin=None,    # Use FORECAST_SAFETY_MARGIN if None
        # Reactive layer
        reactive_scale_out_th=0.7,
        reactive_scale_in_th=0.3,
        # Stability
        hysteresis=1,
        base_cooldown=5,
        # Forecast confidence
        forecast_error_window=20        # Recent steps to assess forecast reliability
    ):
        self.C = capacity_per_server
        self.min = min_servers
        self.max = max_servers
        
        # Anomaly detection parameters
        self.enable_anomaly_detection = enable_anomaly_detection and ANOMALY_DETECTION_AVAILABLE
        self.anomaly_scale_multiplier = anomaly_scale_multiplier
        if self.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector(
                window_size=50,
                zscore_threshold=2.5,
                rate_threshold=anomaly_rate_threshold
            )
        else:
            self.anomaly_detector = None
        
        # Emergency parameters
        self.cpu_critical_th = cpu_critical_th
        self.cpu_per_request = cpu_per_request
        
        # Predictive parameters
        self.predictive_weight = predictive_weight
        # Use standardized safety margin (20% headroom)
        self.forecast_safety_margin = forecast_safety_margin or self.FORECAST_SAFETY_MARGIN
        
        # Reactive parameters
        self.reactive_scale_out_th = reactive_scale_out_th
        self.reactive_scale_in_th = reactive_scale_in_th
        
        # Stability parameters
        self.hysteresis = hysteresis
        self.base_cooldown = base_cooldown
        
        # Forecast reliability tracking
        self.forecast_error_window = forecast_error_window
        self.recent_errors = []  # Forecast errors
        self.recent_traffic = []  # Actual traffic
        
        # State
        self.cooldown_timer = 0
        self.last_decision_reason = ""
    
    def _estimate_cpu(self, requests, servers):
        """Estimate CPU utilization."""
        if servers == 0:
            return 0
        return (requests * self.cpu_per_request) / servers
    
    def _forecast_reliability(self):
        """
        Assess forecast reliability based on recent MAPE (Mean Absolute Percentage Error).
        
        FIXED: Calculate actual percentage errors, not just absolute errors.
        MAPE = mean(|error_i / actual_i|) for actual_i > 0
        
        Returns:
            reliability_score: 0-1 (1 = perfect, 0 = unreliable)
        """
        if len(self.recent_errors) < 5:
            return 0.5  # Default confidence if insufficient history
        
        errors = np.array(self.recent_errors[-self.forecast_error_window:])
        traffic = np.array(self.recent_traffic[-self.forecast_error_window:])
        
        # Ensure arrays are same length
        min_len = min(len(errors), len(traffic))
        errors = errors[-min_len:]
        traffic = traffic[-min_len:]
        
        # Calculate percentage errors only where traffic > 0
        valid_idx = traffic > 0
        if not np.any(valid_idx):
            return 0.5
        
        # MAPE: Mean Absolute Percentage Error
        pct_errors = np.abs(errors[valid_idx] / traffic[valid_idx])
        mape = np.mean(pct_errors)
        
        # Convert MAPE to reliability (e.g., 20% error → 80% reliable)
        # Bounds: [0, 1]
        reliability = max(0, min(1, 1 - mape))
        return reliability
    
    def _anomaly_decision(self, requests, current_servers):
        """
        LAYER 0: ANOMALY DETECTION
        Detect DDoS attacks, flash sales, sudden spikes.
        
        Returns:
            (servers, reason) or (None, None) if no anomaly
        """
        if not self.anomaly_detector:
            return None, None
        
        # Check for anomaly
        is_anomaly, anomaly_reason = self.anomaly_detector.update_online(requests)
        
        if is_anomaly:
            # Aggressive scaling on spike/DDoS
            # Scale up by multiplier (default 1.5x)
            target_servers = int(current_servers * self.anomaly_scale_multiplier)
            target_servers = max(self.min, min(target_servers, self.max))
            
            if target_servers > current_servers:
                reason = f"ANOMALY_SCALE_OUT: {anomaly_reason}"
                return target_servers, reason
        
        return None, None
    
    def _emergency_decision(self, requests, current_servers):
        """
        LAYER 1: EMERGENCY
        Override all other decisions if CPU is critically high.
        
        Returns:
            (new_servers, reason) or (None, None) if not triggered
        """
        cpu = self._estimate_cpu(requests, current_servers)
        
        if cpu > self.cpu_critical_th:
            new_servers = min(current_servers + 2, self.max)  # Aggressive scale-out
            return new_servers, f"EMERGENCY: CPU={cpu:.2%} > {self.cpu_critical_th:.2%}"
        
        return None, None
    
    def _predictive_decision(self, forecast_requests, current_servers):
        """
        LAYER 2: PREDICTIVE
        Use forecast to proactively scale based on predicted load.
        
        Returns:
            (new_servers, reason) or (None, None) if not reliable
        """
        reliability = self._forecast_reliability()
        
        # Only use predictive if forecast is reasonably reliable
        if reliability < 0.3:
            return None, None
        
        # Calculate required servers for forecasted load
        required_servers = int(np.ceil(
            forecast_requests / (self.C * self.forecast_safety_margin)
        ))
        required_servers = np.clip(required_servers, self.min, self.max)
        
        # Only act if significantly different from current
        if abs(required_servers - current_servers) > self.hysteresis:
            return required_servers, f"PREDICTIVE: forecast={forecast_requests:.0f} req/s, reliability={reliability:.2%}"
        
        return None, None
    
    def _reactive_decision(self, requests, current_servers):
        """
        LAYER 3: REACTIVE
        Fallback to real-time utilization if predictive not triggered.
        
        Returns:
            (new_servers, reason) or (None, None) if no action needed
        """
        utilization = requests / (current_servers * self.C)
        
        if utilization > self.reactive_scale_out_th:
            new_servers = min(current_servers + 1, self.max)
            return new_servers, f"REACTIVE_SCALE_OUT: utilization={utilization:.2%}"
        
        elif utilization < self.reactive_scale_in_th:
            new_servers = max(current_servers - 1, self.min)
            return new_servers, f"REACTIVE_SCALE_IN: utilization={utilization:.2%}"
        
        return None, None
    
    def step(self, current_servers, requests, forecast_requests):
        """
        Multi-layer decision process.
        
        Args:
            current_servers: current pod count
            requests: actual request rate
            forecast_requests: predicted request rate from forecaster
        
        Returns:
            (new_servers, action, reason)
                new_servers: updated pod count
                action: +1 (scale out), -1 (scale in), 0 (hold)
                reason: explanation of decision
        """
        action = 0
        new_servers = current_servers
        reason = "HOLD"
        
        # Cooldown enforcement
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            self.last_decision_reason = f"COOLDOWN: {self.cooldown_timer} steps remaining"
            return current_servers, 0, self.last_decision_reason
        
        # Track forecast error
        if requests > 0:
            forecast_error = abs(forecast_requests - requests) / requests
            self.recent_errors.append(forecast_error)
            if len(self.recent_errors) > self.forecast_error_window:
                self.recent_errors.pop(0)
        
        # Track traffic
        self.recent_traffic.append(requests)
        if len(self.recent_traffic) > self.forecast_error_window:
            self.recent_traffic.pop(0)
        
        # --- Layer 0: Anomaly Detection (DDoS/Spike) ---
        if self.enable_anomaly_detection:
            anomaly_servers, anomaly_reason = self._anomaly_decision(requests, current_servers)
            if anomaly_servers is not None:
                new_servers = anomaly_servers
                action = +1  # Always scale out on anomaly
                reason = anomaly_reason
                self.cooldown_timer = self.base_cooldown // 2  # Shorter cooldown for anomalies
                self.last_decision_reason = reason
                return new_servers, action, reason
        
        # --- Layer 1: Emergency ---
        emergency_servers, emergency_reason = self._emergency_decision(requests, current_servers)
        if emergency_servers is not None:
            new_servers = emergency_servers
            action = +1
            reason = emergency_reason
            self.cooldown_timer = self.base_cooldown
            self.last_decision_reason = reason
            return new_servers, action, reason
        
        # --- Layer 2: Predictive ---
        predictive_servers, predictive_reason = self._predictive_decision(forecast_requests, current_servers)
        if predictive_servers is not None:
            new_servers = predictive_servers
            action = 1 if predictive_servers > current_servers else -1
            reason = predictive_reason
            self.cooldown_timer = self.base_cooldown
            self.last_decision_reason = reason
            return new_servers, action, reason
        
        # --- Layer 3: Reactive ---
        reactive_servers, reactive_reason = self._reactive_decision(requests, current_servers)
        if reactive_servers is not None:
            new_servers = reactive_servers
            action = 1 if reactive_servers > current_servers else -1
            reason = reactive_reason
            self.cooldown_timer = self.base_cooldown
            self.last_decision_reason = reason
            return new_servers, action, reason
        
        # --- Layer 4: Hold (no action) ---
        self.last_decision_reason = "HOLD: No scaling trigger"
        return current_servers, 0, self.last_decision_reason
