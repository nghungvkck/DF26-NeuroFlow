"""
HYBRID AUTOSCALING POLICY
===========================
Multi-level decision hierarchy for robustness.

Priority order:
1. EMERGENCY: If CPU > critical threshold → scale out immediately
2. PREDICTIVE: Use forecast-based decision if available
3. REACTIVE: Fallback to real-time request threshold
4. HOLD: No scaling decision needed

This ensures:
- Emergency protection against sudden spikes
- Proactive scaling when forecast is reliable
- Graceful degradation to reactive when forecast fails
"""

import numpy as np


class HybridAutoscaler:
    """
    Multi-layer autoscaling combining CPU emergency detection,
    predictive forecasting, and reactive fallback.
    """
    
    def __init__(
        self,
        capacity_per_server,
        min_servers=2,
        max_servers=20,
        # Emergency layer (CPU-based)
        cpu_critical_th=0.95,           # Immediate scale-out trigger
        cpu_per_request=0.05,
        # Predictive layer
        predictive_weight=0.9,          # Weight between predictive and reactive
        forecast_safety_margin=0.85,
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
        
        # Emergency parameters
        self.cpu_critical_th = cpu_critical_th
        self.cpu_per_request = cpu_per_request
        
        # Predictive parameters
        self.predictive_weight = predictive_weight
        self.forecast_safety_margin = forecast_safety_margin
        
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
        
        Returns:
            reliability_score: 0-1 (1 = perfect, 0 = unreliable)
        """
        if len(self.recent_errors) < 5:
            return 0.5  # Default confidence if insufficient history
        
        errors = np.array(self.recent_errors[-self.forecast_error_window:])
        mape = np.mean(np.abs(errors))
        # Convert MAPE to reliability score: 0.2 mape → 0.9 reliability
        reliability = max(0, min(1, 1 - mape * 2))
        return reliability
    
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
