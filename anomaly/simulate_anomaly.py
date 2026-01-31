"""
ANOMALY SIMULATION FOR AUTOSCALING TESTING
==========================================
Simulates real-world anomalies to test autoscaler robustness.

Anomaly types:
- DDoS attacks (sudden traffic spike)
- Flash sales / viral events (gradual ramp)
- Service failures (sudden drop)
- Thundering herd (cascading spike)
- Diurnal pattern disruption
- Multi-region failover
"""

import numpy as np
from typing import Tuple, Dict, Optional


class AnomalySimulator:
    """
    Simulates various production anomalies for autoscaler testing.
    
    Based on real-world scenarios from:
    - Kubernetes production incidents
    - AWS Auto Scaling case studies
    - Google SRE workload books
    """
    
    @staticmethod
    def inject_ddos(ts: np.ndarray, 
                    start: int, 
                    duration: int, 
                    intensity: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        DDoS attack: sudden sustained traffic spike.
        
        Args:
            ts: Original time series
            start: Attack start timestep
            duration: Attack duration (timesteps)
            intensity: Traffic multiplier (5x = 500% increase)
        
        Returns:
            (modified_ts, anomaly_mask)
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        end = min(start + duration, len(ts))
        ts[start:end] *= intensity
        anomaly[start:end] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_flash_sale(ts: np.ndarray, 
                         start: int, 
                         peak_time: int,
                         duration: int,
                         peak_intensity: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flash sale / viral event: gradual ramp-up to peak, then decay.
        
        Realistic pattern:
        - Slow initial ramp (curiosity)
        - Sharp peak (FOMO)
        - Exponential decay (sold out)
        
        Common in e-commerce, video streaming, game launches.
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        end = min(start + duration, len(ts))
        
        for i in range(start, end):
            t = i - start
            
            # Ramp-up phase (sigmoid)
            if t < peak_time:
                progress = t / peak_time
                multiplier = 1 + (peak_intensity - 1) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
            # Decay phase (exponential)
            else:
                decay_rate = 0.05
                multiplier = 1 + (peak_intensity - 1) * np.exp(-decay_rate * (t - peak_time))
            
            ts[i] *= multiplier
            anomaly[i] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_service_failure(ts: np.ndarray, 
                              start: int, 
                              duration: int,
                              drop_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Service failure: sudden traffic drop (database crash, network partition).
        
        Autoscaler should:
        1. Quickly scale down (avoid wasted cost)
        2. Be ready for recovery spike
        3. Not oscillate during recovery
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        end = min(start + duration, len(ts))
        ts[start:end] *= (1 - drop_ratio)  # Drop to 10% of normal
        anomaly[start:end] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_thundering_herd(ts: np.ndarray, 
                              start: int,
                              wave_count: int = 3,
                              wave_spacing: int = 5,
                              intensity: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thundering herd: cascading retry waves after service recovery.
        
        Pattern:
        - Initial spike (backlog)
        - Secondary wave (retry backoff)
        - Tertiary wave (exponential backoff)
        
        Common after:
        - Cache invalidation (cache stampede)
        - Database failover
        - API rate limit lift
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        for wave in range(wave_count):
            wave_start = start + wave * wave_spacing
            wave_intensity = intensity * (0.7 ** wave)  # Decreasing waves
            
            if wave_start < len(ts):
                ts[wave_start] *= wave_intensity
                anomaly[wave_start] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_diurnal_disruption(ts: np.ndarray, 
                                 start: int,
                                 duration: int,
                                 phase_shift: float = np.pi) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diurnal pattern disruption: timezone shift, DST change, special event.
        
        Challenges forecasting models that rely on historical patterns.
        Tests autoscaler's ability to adapt to pattern shifts.
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        end = min(start + duration, len(ts))
        
        for i in range(start, end):
            # Add phase-shifted sinusoidal component
            t = i - start
            period = 24  # 24 timesteps = 1 day (if 1h timesteps)
            shift = 1 + 0.3 * np.sin(2 * np.pi * t / period + phase_shift)
            
            ts[i] *= shift
            anomaly[i] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_multi_region_failover(ts: np.ndarray, 
                                    start: int,
                                    ramp_duration: int = 10,
                                    intensity: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-region failover: gradual traffic shift from failed region.
        
        Realistic cloud scenario:
        - Region A fails
        - Traffic gradually routes to Region B (DNS TTL, connection draining)
        - Region B sees 2-3x normal load
        
        Tests:
        - Gradual scaling (not too aggressive)
        - Sustained load handling
        - No oscillation during ramp
        """
        ts = ts.copy()
        anomaly = np.zeros(len(ts))
        
        end = min(start + ramp_duration, len(ts))
        
        for i in range(start, end):
            # Linear ramp-up
            progress = (i - start) / ramp_duration
            multiplier = 1 + (intensity - 1) * progress
            
            ts[i] *= multiplier
            anomaly[i] = 1
        
        # Sustained high load after ramp
        if end < len(ts):
            sustained_end = min(end + ramp_duration * 2, len(ts))
            ts[end:sustained_end] *= intensity
            anomaly[end:sustained_end] = 1
        
        return ts, anomaly
    
    @staticmethod
    def inject_composite_anomaly(ts: np.ndarray, 
                                anomaly_configs: list) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Inject multiple overlapping anomalies (realistic scenario).
        
        Args:
            ts: Original time series
            anomaly_configs: List of (type, params) tuples
                Example: [('ddos', {'start': 10, 'duration': 5, 'intensity': 3}),
                         ('flash_sale', {'start': 30, 'peak_time': 10, 'duration': 20})]
        
        Returns:
            (modified_ts, anomaly_masks_dict)
        """
        simulator = AnomalySimulator()
        modified_ts = ts.copy()
        masks = {}
        
        for anomaly_type, params in anomaly_configs:
            method = getattr(simulator, f'inject_{anomaly_type}', None)
            if method:
                modified_ts, mask = method(modified_ts, **params)
                masks[anomaly_type] = mask
        
        return modified_ts, masks


# Legacy function (kept for backward compatibility)
def inject_ddos(ts, start, duration, intensity=5.0):
    """
    DEPRECATED: Use AnomalySimulator.inject_ddos() instead.
    """
    return AnomalySimulator.inject_ddos(ts, start, duration, intensity)
