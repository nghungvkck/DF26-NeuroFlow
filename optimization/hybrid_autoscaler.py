from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .anomaly_detection import AnomalyDetector


class HybridAutoscaler:
    UNIT_COST_PER_SERVER_HOUR = 0.05
    SLA_CPU_THRESHOLD = 0.95
    SLO_CPU_THRESHOLD = 0.85
    FORECAST_SAFETY_MARGIN = 0.80
    HYSTERESIS_MARGIN = 0.20
    MAX_SCALE_INCREMENT = 5

    def __init__(
        self,
        capacity_per_server: float,
        min_servers: int = 2,
        max_servers: int = 20,
        enable_anomaly: bool = True,
        cooldown_steps: int = 5,
    ) -> None:
        self.capacity_per_server = capacity_per_server
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.anomaly_detector = AnomalyDetector() if enable_anomaly else None
        self.cooldown_steps = cooldown_steps

        self.current_servers = min_servers
        self.last_scale_step = -999
        self.current_step = 0
        self.last_scale_type = "normal"

        self.cost_accumulator = 0.0
        self.scaling_history: List[Dict] = []
        self.sla_violations = 0
        self.slo_violations = 0
        self.violation_log: List[Dict] = []

    def _estimate_cpu(self, requests: float, servers: int) -> float:
        if servers == 0:
            return 0.0
        return (requests / (servers * self.capacity_per_server)) * 100

    def _is_on_cooldown(self) -> bool:
        steps_since_last_scale = self.current_step - self.last_scale_step
        return steps_since_last_scale < self.cooldown_steps

    def _decide_anomaly(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        if not self.anomaly_detector:
            return None, None
        is_anomaly, method, confidence = self.anomaly_detector.update_online(requests)
        if is_anomaly:
            increment = min(int(current_servers * 0.3), self.MAX_SCALE_INCREMENT)
            target = current_servers + increment
            target = max(self.min_servers, min(target, self.max_servers))
            reason = f"LAYER0_ANOMALY:{method}:{confidence:.2f}"
            return target, reason
        return None, None

    def _decide_emergency(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        cpu = self._estimate_cpu(requests, current_servers)
        if cpu >= 95.0:
            increment = min(int(current_servers * 0.3), self.MAX_SCALE_INCREMENT)
            target = current_servers + increment
            target = max(self.min_servers, min(target, self.max_servers))
            return target, f"LAYER1_EMERGENCY:CPU={cpu:.1f}%"
        return None, None

    def _decide_predictive(self, forecast_requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        if forecast_requests <= 0:
            return None, None
        required = forecast_requests / self.capacity_per_server
        required_safe = required / self.FORECAST_SAFETY_MARGIN
        if forecast_requests > 0.70 * current_servers * self.capacity_per_server:
            target = int(required_safe * 1.2)
            target = max(self.min_servers, min(target, self.max_servers))
            return target, f"LAYER2_PREDICTIVE:forecast={forecast_requests:.0f}"
        return None, None

    def _decide_reactive(self, requests: float, current_servers: int) -> Tuple[Optional[int], Optional[str]]:
        capacity = current_servers * self.capacity_per_server
        utilization = requests / capacity if capacity > 0 else 0
        scale_out_threshold = 0.70 - self.HYSTERESIS_MARGIN
        scale_in_threshold = 0.30 + self.HYSTERESIS_MARGIN
        if utilization > scale_out_threshold:
            return current_servers + 1, f"LAYER3_REACTIVE:util={utilization:.1%}"
        if utilization < scale_in_threshold and current_servers > self.min_servers:
            return current_servers - 1, f"LAYER3_REACTIVE:util={utilization:.1%}"
        return None, None

    def step(
        self,
        current_servers: int,
        requests: float,
        forecast_requests: Optional[float] = None,
    ) -> Tuple[int, str, Dict]:
        self.current_step += 1
        
        metrics = {
            "cpu": self._estimate_cpu(requests, current_servers),
            "requests": requests,
            "forecast": forecast_requests or 0,
            "timestamp": self.current_step,
        }

        if self._is_on_cooldown():
            return current_servers, "COOLDOWN", metrics

        new_servers = current_servers
        action = "NO_ACTION"
        scale_type = "normal"

        servers, reason = self._decide_anomaly(requests, current_servers)
        if servers is not None:
            new_servers = servers
            action = reason
            scale_type = "anomaly"

        if new_servers == current_servers:
            servers, reason = self._decide_emergency(requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "emergency"

        if new_servers == current_servers and forecast_requests is not None:
            servers, reason = self._decide_predictive(forecast_requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "predictive"

        if new_servers == current_servers:
            servers, reason = self._decide_reactive(requests, current_servers)
            if servers is not None:
                new_servers = servers
                action = reason
                scale_type = "reactive"

        new_servers = max(self.min_servers, min(new_servers, self.max_servers))
        cpu = metrics["cpu"]

        if cpu >= 95.0:
            self.sla_violations += 1
            self.violation_log.append(
                {
                    "step": self.current_step,
                    "cpu": cpu,
                    "requests": requests,
                    "servers": new_servers,
                }
            )

        if cpu >= 85.0:
            self.slo_violations += 1

        if new_servers != current_servers:
            self.last_scale_step = self.current_step
            self.last_scale_type = scale_type
            self.scaling_history.append(
                {
                    "step": self.current_step,
                    "from": current_servers,
                    "to": new_servers,
                    "reason": action,
                }
            )
            metrics["scaled"] = True
            metrics["scale_direction"] = "UP" if new_servers > current_servers else "DOWN"
        else:
            metrics["scaled"] = False
            metrics["scale_direction"] = "NONE"

        return new_servers, action, metrics

    def get_summary(self) -> Dict:
        return {
            "total_cost": self.cost_accumulator,
            "sla_violations": self.sla_violations,
            "slo_violations": self.slo_violations,
            "scaling_events": len(self.scaling_history),
        }

    def get_scaling_report(self) -> List[Dict]:
        return self.scaling_history

    def get_violation_report(self) -> List[Dict]:
        return self.violation_log
