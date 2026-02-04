from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class CostBreakdown:
    reserved: float = 0.0
    on_demand: float = 0.0
    spot: float = 0.0
    startup: float = 0.0

    def total(self) -> float:
        return self.reserved + self.on_demand + self.spot + self.startup


class CloudCostModel:
    def __init__(
        self,
        on_demand_cost: float = 0.05,
        reserved_cost: float = 0.03,
        spot_cost: float = 0.015,
        startup_cost: float = 0.001,
        reserved_capacity: int = 2,
        spot_ratio: float = 0.7,
    ) -> None:
        self.on_demand_cost = on_demand_cost
        self.reserved_cost = reserved_cost
        self.spot_cost = spot_cost
        self.startup_cost = startup_cost
        self.reserved_capacity = reserved_capacity
        self.spot_ratio = spot_ratio

    def compute_step_cost(self, server_count: int, step_hours: float) -> Tuple[float, Dict[str, float]]:
        breakdown = CostBreakdown()
        reserved_servers = min(server_count, self.reserved_capacity)
        breakdown.reserved = reserved_servers * self.reserved_cost * step_hours

        remaining = max(0, server_count - reserved_servers)
        if remaining > 0:
            spot_servers = int(remaining * self.spot_ratio)
            on_demand_servers = remaining - spot_servers
            breakdown.spot = spot_servers * self.spot_cost * step_hours
            breakdown.on_demand = on_demand_servers * self.on_demand_cost * step_hours

        total = breakdown.total()
        return total, {
            "reserved": breakdown.reserved,
            "spot": breakdown.spot,
            "on_demand": breakdown.on_demand,
            "startup": breakdown.startup,
        }
