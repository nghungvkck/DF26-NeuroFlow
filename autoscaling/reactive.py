class ReactiveAutoscaler:
    def __init__(
        self,
        capacity_per_server,
        min_servers=2,
        max_servers=20,
        scale_out_th=0.7,
        scale_in_th=0.3,
        cooldown=5
    ):
        self.C = capacity_per_server
        self.min = min_servers
        self.max = max_servers
        self.scale_out_th = scale_out_th
        self.scale_in_th = scale_in_th
        self.cooldown = cooldown
        self.cooldown_timer = 0

    def step(self, current_servers, requests):
        """
        One timestep decision
        """
        utilization = requests / (current_servers * self.C)
        action = 0

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return current_servers, utilization, action

        if utilization > self.scale_out_th:
            current_servers = min(current_servers + 1, self.max)
            self.cooldown_timer = self.cooldown
            action = +1

        elif utilization < self.scale_in_th:
            current_servers = max(current_servers - 1, self.min)
            self.cooldown_timer = self.cooldown
            action = -1

        return current_servers, utilization, action
