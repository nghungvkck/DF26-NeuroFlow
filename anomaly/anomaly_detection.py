import numpy as np

def zscore_detection(ts, threshold=3.0):
    mean = np.mean(ts)
    std = np.std(ts)
    z = (ts - mean) / std
    return (np.abs(z) > threshold).astype(int)
