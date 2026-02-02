from .hysteresis import adaptive_cooldown
import numpy as np

class PredictiveAutoscaler:
    """
    Predictive autoscaling based on load forecasting.
    
    Uses forecast to proactively scale before traffic spike arrives.
    """
    
    # Standard safety margin: 20% headroom for all strategies
    FORECAST_SAFETY_MARGIN = 0.80
    
    def __init__(
        self,
        capacity_per_server,
        safety_margin=None,  # Use FORECAST_SAFETY_MARGIN if None
        min_servers=2,
        max_servers=20,
        hysteresis=1,
        base_cooldown=5,
        window_size=10
    ):
        self.C = capacity_per_server
        # Use standardized safety margin (20% headroom)
        self.alpha = safety_margin if safety_margin is not None else self.FORECAST_SAFETY_MARGIN
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
        Predictive autoscaling step.
        
        Args:
            current_servers: current pod count
            forecast_requests: predicted load for next step
            current_requests: actual current load (for variance tracking)
        
        Returns:
            (new_servers, action)
        """
        action = 0

        # --- Update traffic history for variance tracking ---
        self.recent_traffic.append(current_requests)
        if len(self.recent_traffic) > self.window_size:
            self.recent_traffic.pop(0)

        # --- Cooldown enforcement ---
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return current_servers, action

        # --- Calculate required servers for forecast ---
        N_req = self.required_servers(forecast_requests)

        # --- Hysteresis: avoid flapping with small changes ---
        if abs(N_req - current_servers) <= self.hysteresis:
            return current_servers, action

        # --- Make scaling decision ---
        if N_req > current_servers:
            current_servers = min(N_req, self.max)
            action = +1
        else:
            current_servers = max(N_req, self.min)
            action = -1

        # --- Set adaptive cooldown AFTER decision ---
        self.cooldown_timer = adaptive_cooldown(
            self.base_cooldown,
            self.recent_traffic
        )

        return current_servers, action
