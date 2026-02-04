from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MetricsSnapshot:
    timestamp: int
    servers: int
    requests: float
    cost: float
    cost_breakdown: Dict[str, float]
    cpu_utilization: float
    sla_violated: bool
    slo_violated: bool
    scaling_action: int


class MetricsCollector:
    def __init__(self, capacity_per_server: int, step_minutes: float = 15.0) -> None:
        self.capacity = capacity_per_server
        self.step_minutes = step_minutes
        self.snapshots: List[MetricsSnapshot] = []

    def record(
        self,
        t: int,
        servers: int,
        requests: float,
        cost: float,
        cost_breakdown: Dict[str, float],
        scaling_action: int = 0,
    ) -> None:
        cpu = (requests / (servers * self.capacity)) * 100 if servers > 0 else 0.0
        sla_violated = cpu >= 95.0
        slo_violated = cpu >= 85.0

        snapshot = MetricsSnapshot(
            timestamp=t,
            servers=servers,
            requests=requests,
            cost=cost,
            cost_breakdown=cost_breakdown,
            cpu_utilization=cpu,
            sla_violated=sla_violated,
            slo_violated=slo_violated,
            scaling_action=scaling_action,
        )
        self.snapshots.append(snapshot)

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        if not self.snapshots:
            return {}

        snapshots = self.snapshots
        total_cost = sum(s.cost for s in snapshots)
        cost_reserved = sum(s.cost_breakdown.get("reserved", 0) for s in snapshots)
        cost_spot = sum(s.cost_breakdown.get("spot", 0) for s in snapshots)
        cost_ondemand = sum(s.cost_breakdown.get("on_demand", 0) for s in snapshots)

        avg_servers = np.mean([s.servers for s in snapshots])
        min_servers = int(np.min([s.servers for s in snapshots]))
        max_servers = int(np.max([s.servers for s in snapshots]))

        cost_per_server = total_cost / sum(s.servers for s in snapshots) if sum(s.servers for s in snapshots) > 0 else 0

        sla_violations = sum(1 for s in snapshots if s.sla_violated)
        slo_violations = sum(1 for s in snapshots if s.slo_violated)
        sla_violation_rate = sla_violations / len(snapshots)
        slo_violation_rate = slo_violations / len(snapshots)

        actions = np.array([s.scaling_action for s in snapshots])
        scaling_events = int(np.sum(actions != 0))
        scale_ups = int(np.sum(actions > 0))
        scale_downs = int(np.sum(actions < 0))
        oscillations = sum(1 for i in range(1, len(actions)) if actions[i] * actions[i - 1] < 0)

        utils = np.array([s.cpu_utilization for s in snapshots])
        avg_cpu = float(np.mean(utils))
        max_cpu = float(np.max(utils))
        min_cpu = float(np.min(utils))
        efficiency = float(1.0 - np.mean(np.maximum(0, 0.95 - utils)))

        return {
            "total_cost": float(total_cost),
            "cost_reserved": float(cost_reserved),
            "cost_spot": float(cost_spot),
            "cost_ondemand": float(cost_ondemand),
            "cost_per_server": float(cost_per_server),
            "avg_servers": float(avg_servers),
            "min_servers": min_servers,
            "max_servers": max_servers,
            "sla_violations": sla_violations,
            "sla_violation_rate": float(sla_violation_rate),
            "slo_violations": slo_violations,
            "slo_violation_rate": float(slo_violation_rate),
            "scaling_events": scaling_events,
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "oscillations": oscillations,
            "avg_cpu": avg_cpu,
            "max_cpu": max_cpu,
            "min_cpu": min_cpu,
            "efficiency": efficiency,
            "total_timesteps": len(snapshots),
        }
