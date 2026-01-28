import numpy as np

def inject_ddos(
    ts,
    start,
    duration,
    intensity=5.0
):
    """
    ts: np.array requests
    intensity: multiplier
    """
    ts = ts.copy()
    anomaly = np.zeros(len(ts))

    end = min(start + duration, len(ts))
    ts[start:end] *= intensity
    anomaly[start:end] = 1

    return ts, anomaly
