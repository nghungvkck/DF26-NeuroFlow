"""
CPU-BASED AUTOSCALING POLICY
=============================
Threshold-based scaling using CPU utilization as baseline policy.

Used as baseline for comparison with reactive (request-based) and
predictive approaches. Simulates traditional infrastructure monitoring.
"""

import numpy as np


class CPUBasedAutoscaler:
    """
    Baseline autoscaler using CPU utilization thresholds.
    
    Typical usage in traditional cloud platforms:
    - Monitor average CPU across pods
    - Scale out if CPU > scale_out_threshold
    - Scale in if CPU < scale_in_threshold
    """
    
    def __init__(
        self,
        capacity_per_server,
        cpu_per_request=0.05,  # CPU utilization per request
        min_servers=2,
        max_servers=20,
        scale_out_cpu_th=0.75,     # Scale out at 75% CPU
        scale_in_cpu_th=0.25,      # Scale in at 25% CPU
        cooldown=5
    ):
        """
        Args:
            capacity_per_server: requests/min per server
            cpu_per_request: CPU units consumed per request (0-1)
            min_servers: minimum pod count
            max_servers: maximum pod count
            scale_out_cpu_th: CPU threshold to trigger scale-out
            scale_in_cpu_th: CPU threshold to trigger scale-in
            cooldown: cooldown time between scaling decisions
        """
        self.C = capacity_per_server
        self.cpu_per_request = cpu_per_request
        self.min = min_servers
        self.max = max_servers
        self.scale_out_cpu_th = scale_out_cpu_th
        self.scale_in_cpu_th = scale_in_cpu_th
        self.cooldown = cooldown
        self.cooldown_timer = 0
    
    def _estimate_cpu(self, requests, current_servers):
        """
        Estimate average CPU utilization across servers.
        
        Args:
            requests: current request rate
            current_servers: number of running pods
        
        Returns:
            cpu_utilization: CPU % (0-1)
        """
        if current_servers == 0:
            return 0
        total_cpu = requests * self.cpu_per_request
        cpu_per_server = total_cpu / current_servers
        return cpu_per_server
    
    def step(self, current_servers, requests):
        """
        One timestep decision based on CPU thresholds.
        
        Args:
            current_servers: current pod count
            requests: current request rate
        
        Returns:
            new_servers: updated pod count
            cpu_utilization: computed CPU utilization
            action: scaling action (+1, 0, -1)
        """
        action = 0
        cpu = self._estimate_cpu(requests, current_servers)
        
        # Cooldown enforcement
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return current_servers, cpu, action
        
        # Scale-out decision: CPU above threshold
        if cpu > self.scale_out_cpu_th:
            current_servers = min(current_servers + 1, self.max)
            self.cooldown_timer = self.cooldown
            action = +1
        
        # Scale-in decision: CPU below threshold
        elif cpu < self.scale_in_cpu_th:
            current_servers = max(current_servers - 1, self.min)
            self.cooldown_timer = self.cooldown
            action = -1
        
        return current_servers, cpu, action


def estimate_requests_from_cpu(cpu_utilization, servers, cpu_per_request):
    """
    Reverse: estimate request load from CPU metric.
    
    Useful for monitoring dashboards that display implied load.
    """
    return cpu_utilization * servers / cpu_per_request
