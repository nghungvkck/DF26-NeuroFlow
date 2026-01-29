from .hysteresis import adaptive_cooldown
import numpy as np

class PredictiveAutoscaler:
    def __init__(
        self,
        capacity_per_server,
        safety_margin=0.8,
        min_servers=2,
        max_servers=20,
        hysteresis=1,
        base_cooldown=5,
        window_size=10
    ):
        self.C = capacity_per_server
        self.alpha = safety_margin
        self.min = min_servers
        self.max = max_servers
        self.hysteresis = hysteresis
        self.base_cooldown = base_cooldown
        self.window_size = window_size

        self.cooldown_timer = 0
        self.recent_traffic = []

    def required_servers(self, forecast_requests):
        return int(np.ceil(forecast_requests / (self.C * self.alpha)))

    def step(self, current_servers, forecast_requests, current_requests):
        """
        current_requests: dùng để tính độ biến động
        """
        action = 0

        # --- cập nhật traffic window ---
        self.recent_traffic.append(current_requests)
        if len(self.recent_traffic) > self.window_size:
            self.recent_traffic.pop(0)

        # --- cooldown ---
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return current_servers, action

        # --- forecast -> required servers ---
        N_req = self.required_servers(forecast_requests)

        # --- hysteresis (chống dao động) ---
        if abs(N_req - current_servers) <= self.hysteresis:
            return current_servers, action

        # --- scale decision ---
        if N_req > current_servers:
            current_servers = min(N_req, self.max)
            action = +1
        else:
            current_servers = max(N_req, self.min)
            action = -1

        # --- adaptive cooldown ---
        self.cooldown_timer = adaptive_cooldown(
            self.base_cooldown,
            self.recent_traffic
        )

        return current_servers, action
