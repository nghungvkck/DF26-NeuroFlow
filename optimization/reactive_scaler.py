from __future__ import annotations


class ReactiveOnlyScaler:
    def __init__(
        self,
        capacity_per_server: float,
        min_servers: int,
        max_servers: int,
        upper_threshold: float,
        lower_threshold: float,
        consecutive_steps: int,
        cooldown_steps: int,
        initial_servers: int,
    ) -> None:
        self.capacity_per_server = capacity_per_server
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.consecutive_steps = max(1, consecutive_steps)
        self.cooldown_steps = max(0, cooldown_steps)
        self.servers = max(self.min_servers, min(self.max_servers, initial_servers))
        self._above_count = 0
        self._cooldown = 0

    def step(self, requests: float) -> int:
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.servers

        capacity = self.servers * self.capacity_per_server
        utilization = requests / capacity if capacity > 0 else 0.0

        if utilization > self.upper_threshold:
            self._above_count += 1
        else:
            self._above_count = 0

        if utilization > self.upper_threshold and self._above_count >= self.consecutive_steps:
            self.servers = min(self.max_servers, self.servers + 1)
            self._cooldown = self.cooldown_steps
            self._above_count = 0
            return self.servers

        if utilization < self.lower_threshold and self.servers > self.min_servers:
            self.servers = max(self.min_servers, self.servers - 1)
            self._cooldown = self.cooldown_steps

        return self.servers
