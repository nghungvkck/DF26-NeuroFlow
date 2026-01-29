import numpy as np

def adaptive_cooldown(base, traffic_window):
    """
    traffic_window: recent requests
    """
    sigma = np.std(traffic_window)
    return max(1, int(base / (1 + sigma)))
