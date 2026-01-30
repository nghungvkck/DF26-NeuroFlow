"""
SCENARIO SIMULATOR
==================
Synthetic load scenarios for testing autoscaling strategies.

Scenarios:
1. GRADUAL_INCREASE: Linear traffic ramp-up over time
2. SUDDEN_SPIKE: Sharp traffic jump (e.g., viral event)
3. OSCILLATING: Regular sinusoidal load (e.g., diurnal pattern with noise)
4. DROP: Sudden traffic drop and recovery
5. FORECAST_ERROR: Base load + forecast errors to test robustness

Each scenario returns a time series that feeds into autoscaling logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Scenario:
    """Scenario definition with load pattern."""
    name: str
    description: str
    load_series: np.ndarray
    forecast_errors: np.ndarray = None
    
    def __post_init__(self):
        if self.forecast_errors is None:
            self.forecast_errors = np.zeros_like(self.load_series)


class ScenarioGenerator:
    """Generate synthetic load scenarios for autoscaling testing."""
    
    @staticmethod
    def gradual_increase(
        base_load=100,
        peak_load=500,
        duration=100,
        noise_level=0.05
    ) -> Scenario:
        """
        SCENARIO 1: Gradual Load Increase
        
        Simulates: Business hours ramp-up, marketing campaign effect
        Pattern: Linear increase from base to peak
        
        Args:
            base_load: initial request rate
            peak_load: final request rate
            duration: number of timesteps
            noise_level: random noise std as % of load
        
        Returns:
            Scenario with load time series
        """
        load = np.linspace(base_load, peak_load, duration)
        noise = np.random.normal(0, noise_level * load, duration)
        load = np.maximum(load + noise, base_load * 0.5)
        
        return Scenario(
            name="GRADUAL_INCREASE",
            description=f"Linear ramp from {base_load} to {peak_load} req/s over {duration} steps",
            load_series=load,
            forecast_errors=np.random.normal(0, 0.1, duration)
        )
    
    @staticmethod
    def sudden_spike(
        base_load=100,
        spike_load=800,
        spike_start=50,
        spike_duration=20,
        duration=150,
        noise_level=0.05
    ) -> Scenario:
        """
        SCENARIO 2: Sudden Traffic Spike
        
        Simulates: DDoS, viral event, flash sale
        Pattern: Base → sudden jump → gradual return
        
        Args:
            base_load: normal request rate
            spike_load: peak during spike
            spike_start: when spike begins
            spike_duration: how long spike lasts
            duration: total timesteps
            noise_level: random noise
        
        Returns:
            Scenario with spike pattern
        """
        load = np.full(duration, base_load, dtype=float)
        
        # Spike phase
        spike_end = min(spike_start + spike_duration, duration)
        spike_phase = np.linspace(spike_load, base_load, spike_end - spike_start)
        load[spike_start:spike_end] = spike_phase
        
        # Add noise
        noise = np.random.normal(0, noise_level * load, duration)
        load = np.maximum(load + noise, base_load * 0.3)
        
        # Forecast error: forecaster doesn't see the spike coming
        forecast_errors = np.zeros(duration)
        forecast_errors[spike_start:spike_end] = 0.5  # 50% underprediction
        forecast_errors += np.random.normal(0, 0.08, duration)
        
        return Scenario(
            name="SUDDEN_SPIKE",
            description=f"Spike from {base_load} to {spike_load} req/s at t={spike_start}",
            load_series=load,
            forecast_errors=forecast_errors
        )
    
    @staticmethod
    def oscillating(
        base_load=200,
        amplitude=150,
        period=20,
        duration=200,
        noise_level=0.05
    ) -> Scenario:
        """
        SCENARIO 3: Oscillating Load
        
        Simulates: Diurnal pattern, batch job cycles, periodic user behavior
        Pattern: Sinusoidal load with noise
        
        Args:
            base_load: average request rate
            amplitude: oscillation amplitude
            period: cycle length in timesteps
            duration: total timesteps
            noise_level: noise level
        
        Returns:
            Scenario with oscillating pattern
        """
        t = np.arange(duration)
        load = base_load + amplitude * np.sin(2 * np.pi * t / period)
        
        # Add realistic noise
        noise = np.random.normal(0, noise_level * np.abs(load), duration)
        load = np.maximum(load + noise, base_load * 0.5)
        
        # Forecast is slightly delayed
        forecast_errors = np.random.normal(0.05, 0.12, duration)  # 5% bias + noise
        
        return Scenario(
            name="OSCILLATING",
            description=f"Sinusoidal load (base={base_load}, amplitude={amplitude}, period={period})",
            load_series=load,
            forecast_errors=forecast_errors
        )
    
    @staticmethod
    def traffic_drop(
        base_load=300,
        drop_load=50,
        drop_start=80,
        drop_duration=30,
        duration=200,
        recovery_slope=0.3,
        noise_level=0.05
    ) -> Scenario:
        """
        SCENARIO 4: Traffic Drop & Recovery
        
        Simulates: Service outage, user migration, maintenance window recovery
        Pattern: Base → sudden drop → gradual recovery
        
        Args:
            base_load: normal request rate
            drop_load: minimum during drop
            drop_start: when drop begins
            drop_duration: drop duration
            duration: total timesteps
            recovery_slope: how fast traffic recovers
            noise_level: noise level
        
        Returns:
            Scenario with drop and recovery
        """
        load = np.full(duration, base_load, dtype=float)
        
        drop_end = min(drop_start + drop_duration, duration)
        recovery_end = min(drop_end + int(20 / recovery_slope), duration)
        
        # Drop phase
        load[drop_start:drop_end] = drop_load
        
        # Recovery phase
        recovery_length = recovery_end - drop_end
        if recovery_length > 0:
            recovery = np.linspace(drop_load, base_load, recovery_length)
            load[drop_end:recovery_end] = recovery
        
        # Add noise
        noise = np.random.normal(0, noise_level * (load + 10), duration)
        load = np.maximum(load + noise, 10)
        
        # Forecast error: recovers slower than actual load
        forecast_errors = np.zeros(duration)
        forecast_errors[drop_start:drop_end] = -0.3  # Overpredicts drop
        forecast_errors[drop_end:recovery_end] = -0.2  # Underpredicts recovery
        forecast_errors += np.random.normal(0, 0.08, duration)
        
        return Scenario(
            name="TRAFFIC_DROP",
            description=f"Drop to {drop_load} req/s at t={drop_start}, recovery over {recovery_end - drop_end} steps",
            load_series=load,
            forecast_errors=forecast_errors
        )
    
    @staticmethod
    def forecast_error_test(
        base_load=200,
        amplitude=100,
        period=15,
        duration=200,
        underprediction_rate=0.15,
        overprediction_rate=0.10,
        anomaly_frequency=0.05
    ) -> Scenario:
        """
        SCENARIO 5: Forecast Error Resilience Test
        
        Simulates: Real-world forecast errors
        Pattern: Base oscillating load + systematic bias + random anomalies
        
        Args:
            base_load: average load
            amplitude: oscillation amplitude
            period: oscillation period
            duration: total timesteps
            underprediction_rate: % underprediction bias
            overprediction_rate: % overprediction bias
            anomaly_frequency: probability of anomaly per step
        
        Returns:
            Scenario with realistic forecast errors
        """
        t = np.arange(duration)
        load = base_load + amplitude * np.sin(2 * np.pi * t / period)
        load = np.maximum(load + np.random.normal(0, 0.08 * load, duration), base_load * 0.3)
        
        # Create realistic forecast errors
        forecast_errors = np.zeros(duration)
        
        # Systematic bias (alternating under/over)
        for i in range(duration):
            if i % 40 < 20:
                forecast_errors[i] = underprediction_rate
            else:
                forecast_errors[i] = -overprediction_rate
        
        # Random noise
        forecast_errors += np.random.normal(0, 0.08, duration)
        
        # Occasional anomalies (spikes forecaster misses)
        anomaly_indices = np.where(np.random.rand(duration) < anomaly_frequency)[0]
        forecast_errors[anomaly_indices] = np.random.choice([-0.5, 0.5], len(anomaly_indices))
        
        return Scenario(
            name="FORECAST_ERROR_TEST",
            description=f"Oscillating load with {underprediction_rate:.1%} underprediction bias and anomalies",
            load_series=load,
            forecast_errors=forecast_errors
        )


# Convenience generator
def generate_all_scenarios(duration=200) -> List[Scenario]:
    """
    Generate all test scenarios with standard parameters.
    
    Returns:
        List of Scenario objects
    """
    return [
        ScenarioGenerator.gradual_increase(duration=duration),
        ScenarioGenerator.sudden_spike(duration=duration),
        ScenarioGenerator.oscillating(duration=duration),
        ScenarioGenerator.traffic_drop(duration=duration),
        ScenarioGenerator.forecast_error_test(duration=duration),
    ]
