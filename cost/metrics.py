import numpy as np

def sla_violation_rate(requests, servers, capacity):
    violations = requests > (servers * capacity)
    return np.mean(violations)

def overprovision_ratio(requests, servers, capacity):
    over = (servers * capacity - requests) / (servers * capacity)
    return np.mean(over)

def speed_of_scale(overload_times, recovery_times):
    return np.mean(recovery_times - overload_times)
