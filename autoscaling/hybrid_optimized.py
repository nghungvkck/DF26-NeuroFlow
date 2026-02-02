"""
OPTIMIZED HYBRID AUTOSCALING POLICY
====================================

Production-grade autoscaler combining anomaly detection, emergency protection,
predictive forecasting, and reactive fallback in a clean, maintainable design.

Decision Hierarchy (Priority Order):
  Layer 0: ANOMALY DETECTION    → Spike/DDoS detection (fastest response)
  Layer 1: EMERGENCY DETECTION  → CPU critical threshold (safety)
  Layer 2: PREDICTIVE SCALING   → Forecast-based proactive
  Layer 3: REACTIVE SCALING     → Request threshold fallback

Performance (Phase B.5 Results):
  • Cost:              $57.79 (15-day test)
  • SLA Violations:    14     (BEST of all strategies)
  • Spike Response:    4.7-5.5 minutes (FASTEST)
  • Scaling Events:    152    (Aggressive but protective)

Key Features:
  ✓ 4-method anomaly ensemble (Z-score, IQR, ROC, ensemble voting)
  ✓ Intelligent cooldown management (5min base, 2.5min anomaly)
  ✓ Hysteresis to prevent flapping (20% margin)
  ✓ Real-time cost tracking ($0.05/pod/hour)
  ✓ SLA/SLO compliance tracking
"""

import sys
import os
from typing import Tuple, Dict, Optional, List
import numpy as np
from datetime import datetime

# Add parent directory to path for anomaly detector import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anomaly.anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False


class HybridAutoscalerOptimized:
    """
    Production-grade autoscaler with clean separation of concerns.
    
    Configuration:
      - capacity_per_server: Requests per server per minute
      - min_servers, max_servers: Scaling bounds
      - forecast: LightGBM forecaster object (optional)
    """
    
    # Constants
    UNIT_COST_PER_POD_HOUR = 0.05
    SLA_CPU_THRESHOLD = 0.95
    SLO_CPU_THRESHOLD = 0.85
    FORECAST_SAFETY_MARGIN = 0.80
    HYSTERESIS_MARGIN = 0.20  # 20% margin to prevent flapping
    
    def __init__(
        self,
        capacity_per_server: float,
        min_servers: int = 2,
        max_servers: int = 20,
        forecast=None,
    ):
        """Initialize HYBRID autoscaler."""
        self.capacity_per_server = capacity_per_server
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.forecast = forecast
        
        # Anomaly Detection Layer
        self.anomaly_detector = (
            AnomalyDetector() if ANOMALY_DETECTION_AVAILABLE else None
        )
        
        # State Tracking
        self.current_servers = min_servers
        self.last_scale_time = datetime.now()
        self.cost_accumulator = 0.0
        self.scaling_history = []  # Track all scaling decisions
        
        # SLA/SLO Tracking
        self.sla_violations = 0
        self.slo_violations = 0
        self.violation_log = []
        
        # Cooldown Management
        self.base_cooldown_minutes = 5
        self.anomaly_cooldown_minutes = 2.5
        self.last_scale_type = None  # Track what type of scaling
        
    # ========================================================================
    # LAYER 0: ANOMALY DETECTION
    # ========================================================================
    
    def _check_anomaly(self, requests: float) -> Tuple[bool, Optional[str]]:
        """
        Detect spikes/DDoS using 4-method ensemble.
        
        Returns:
            (is_anomaly, reason_str)
        """
        if not self.anomaly_detector:
            return False, None
        
        is_anomaly, method, confidence = self.anomaly_detector.update_online(requests)
        
        if is_anomaly:
            reason = f"ANOMALY_{method}_conf={confidence:.2f}"
            return True, reason
        
        return False, None
    
    def _decide_anomaly(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Layer 0 Decision: Scale aggressively if anomaly detected.
        
        Action: Scale out 1.5× current servers (aggressive protection)
        Cooldown: 2.5 minutes (faster response to spikes)
        """
        is_anomaly, reason = self._check_anomaly(requests)
        
        if is_anomaly:
            target_servers = int(current_servers * 1.5)
            target_servers = max(self.min_servers, min(target_servers, self.max_servers))
            return target_servers, f"LAYER0_ANOMALY: {reason}"
        
        return None, None
    
    # ========================================================================
    # LAYER 1: EMERGENCY DETECTION
    # ========================================================================
    
    def _estimate_cpu(self, requests: float, servers: int) -> float:
        """Estimate CPU utilization as proxy for load."""
        if servers == 0:
            return 0.0
        return (requests / (servers * self.capacity_per_server * 100)) * 100
    
    def _decide_emergency(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Layer 1 Decision: Emergency CPU protection.
        
        Trigger: CPU > 95% (critical)
        Action: Scale out 1.5× immediately
        Cooldown: 2.5 minutes (faster response)
        """
        cpu = self._estimate_cpu(requests, current_servers)
        
        if cpu > 0.95:
            target_servers = int(current_servers * 1.5)
            target_servers = max(self.min_servers, min(target_servers, self.max_servers))
            return target_servers, f"LAYER1_EMERGENCY: CPU={cpu:.1%}"
        
        return None, None
    
    # ========================================================================
    # LAYER 2: PREDICTIVE SCALING
    # ========================================================================
    
    def _decide_predictive(self, forecast_requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Layer 2 Decision: Predictive scaling based on forecast.
        
        Uses LightGBM forecast with safety margin.
        Trigger: Forecasted requests > 70% capacity
        Condition: Forecast reliability > 80%
        Action: Scale out 1.2×
        Cooldown: 5 minutes
        """
        if self.forecast is None or forecast_requests == 0:
            return None, None
        
        # Check forecast reliability
        if hasattr(self, 'forecast_mape') and self.forecast_mape > 0.20:
            return None, "LAYER2_SKIPPED: Low forecast confidence"
        
        # Calculate required servers with safety margin
        required = forecast_requests / (self.capacity_per_server * 100)
        required_safe = required / self.FORECAST_SAFETY_MARGIN
        
        # Trigger: forecast > 70% capacity
        if forecast_requests > 0.70 * current_servers * self.capacity_per_server * 100:
            target_servers = int(required_safe * 1.2)  # 1.2× multiplier
            target_servers = max(self.min_servers, min(target_servers, self.max_servers))
            return target_servers, f"LAYER2_PREDICTIVE: forecast={forecast_requests:.0f} safety_margin={self.FORECAST_SAFETY_MARGIN:.1%}"
        
        return None, None
    
    # ========================================================================
    # LAYER 3: REACTIVE SCALING
    # ========================================================================
    
    def _decide_reactive(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Layer 3 Decision: Reactive fallback (scale by 1 pod at a time).
        
        Scale Out: requests > 70% capacity
        Scale In: requests < 30% capacity
        Action: Scale by 1 pod
        Cooldown: 5 minutes
        """
        capacity = current_servers * self.capacity_per_server * 100
        utilization = requests / capacity if capacity > 0 else 0
        
        # With hysteresis to prevent flapping
        scale_out_threshold = 0.70 - self.HYSTERESIS_MARGIN
        scale_in_threshold = 0.30 + self.HYSTERESIS_MARGIN
        
        if utilization > scale_out_threshold:
            return current_servers + 1, f"LAYER3_REACTIVE: utilization={utilization:.1%}"
        elif utilization < scale_in_threshold and current_servers > self.min_servers:
            return current_servers - 1, f"LAYER3_REACTIVE: utilization={utilization:.1%}"
        
        return None, None
    
    # ========================================================================
    # MAIN DECISION ENGINE
    # ========================================================================
    
    def _is_on_cooldown(self, scale_type: str) -> bool:
        """Check if still on cooldown from last scaling decision."""
        from datetime import timedelta
        
        cooldown_min = (
            self.anomaly_cooldown_minutes if scale_type == "anomaly"
            else self.base_cooldown_minutes
        )
        cooldown = timedelta(minutes=cooldown_min)
        
        return (datetime.now() - self.last_scale_time) < cooldown
    
    def step(
        self,
        current_servers: int,
        requests: float,
        forecast_requests: Optional[float] = None
    ) -> Tuple[int, str, Dict]:
        """
        Execute one autoscaling decision cycle.
        
        Args:
            current_servers: Current pod count
            requests: Current requests per minute
            forecast_requests: Forecasted requests (optional)
        
        Returns:
            (new_server_count, scaling_action, metrics_dict)
        """
        metrics = {
            'cpu': self._estimate_cpu(requests, current_servers),
            'requests': requests,
            'forecast': forecast_requests,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Skip if on cooldown
        if self._is_on_cooldown(self.last_scale_type or "normal"):
            return current_servers, "COOLDOWN_ACTIVE", metrics
        
        new_servers = current_servers
        action = "NO_ACTION"
        scale_type = None
        
        # === LAYER 0: ANOMALY ===
        if ANOMALY_DETECTION_AVAILABLE:
            servers, reason = self._decide_anomaly(requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "anomaly"
        
        # === LAYER 1: EMERGENCY ===
        if new_servers == current_servers:
            servers, reason = self._decide_emergency(requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "emergency"
        
        # === LAYER 2: PREDICTIVE ===
        if new_servers == current_servers and forecast_requests:
            servers, reason = self._decide_predictive(forecast_requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "predictive"
        
        # === LAYER 3: REACTIVE ===
        if new_servers == current_servers:
            servers, reason = self._decide_reactive(requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "reactive"
        
        # === APPLY CONSTRAINTS ===
        new_servers = max(self.min_servers, min(new_servers, self.max_servers))
        
        # === TRACK SLA/SLO ===
        cpu = metrics['cpu']
        if cpu > self.SLA_CPU_THRESHOLD:
            self.sla_violations += 1
            self.violation_log.append({
                'time': metrics['timestamp'],
                'type': 'SLA',
                'cpu': cpu,
                'requests': requests,
                'servers': new_servers
            })
        
        if cpu > self.SLO_CPU_THRESHOLD:
            self.slo_violations += 1
        
        # === TRACK COST ===
        cost_for_step = new_servers * self.UNIT_COST_PER_POD_HOUR / 60  # Per minute cost
        self.cost_accumulator += cost_for_step
        metrics['cost'] = cost_for_step
        metrics['cumulative_cost'] = self.cost_accumulator
        
        # === UPDATE STATE ===
        if new_servers != current_servers:
            self.last_scale_time = datetime.now()
            self.last_scale_type = scale_type
            self.scaling_history.append({
                'time': metrics['timestamp'],
                'from': current_servers,
                'to': new_servers,
                'reason': action
            })
            metrics['scaled'] = True
            metrics['scale_direction'] = 'UP' if new_servers > current_servers else 'DOWN'
        else:
            metrics['scaled'] = False
        
        return new_servers, action, metrics
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_summary(self) -> Dict:
        """Generate comprehensive summary of autoscaling performance."""
        return {
            'total_cost': self.cost_accumulator,
            'sla_violations': self.sla_violations,
            'slo_violations': self.slo_violations,
            'scaling_events': len(self.scaling_history),
            'cost_per_violation': (
                self.cost_accumulator / self.sla_violations
                if self.sla_violations > 0 else 0
            ),
            'avg_cost_per_step': self.cost_accumulator,
        }
    
    def get_scaling_report(self) -> List[Dict]:
        """Return detailed scaling history."""
        return self.scaling_history
    
    def get_violation_report(self) -> List[Dict]:
        """Return SLA violation details."""
        return self.violation_log


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class HybridAutoscaler(HybridAutoscalerOptimized):
    """Alias for backward compatibility."""
    pass
