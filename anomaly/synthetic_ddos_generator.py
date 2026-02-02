"""
SYNTHETIC DDoS/SPIKE DATA GENERATOR
====================================
Generate realistic traffic patterns with various attack scenarios for testing autoscaling.

Scenarios:
1. NORMAL: Regular daily pattern with slight variations
2. GRADUAL_SPIKE: Slow traffic increase (flash sale)
3. SUDDEN_SPIKE: Instant 5x traffic jump (DDoS attack)
4. OSCILLATING_SPIKE: Multiple waves of attacks
5. SUSTAINED_DDOS: Long-duration high traffic
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def generate_normal_traffic(
    duration_minutes: int = 1440,  # 24 hours
    base_rate: float = 100.0,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate normal daily traffic pattern.
    
    Pattern:
    - Low traffic at night (3am-7am): 50% of base
    - Peak traffic during business hours (9am-5pm): 150% of base
    - Evening traffic (6pm-11pm): 100% of base
    """
    timestamps = np.arange(duration_minutes)
    hours = (timestamps / 60) % 24
    
    # Sinusoidal daily pattern
    daily_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Add business hours peak
    business_hours = (hours >= 9) & (hours <= 17)
    daily_pattern[business_hours] *= 1.5
    
    traffic = base_rate * daily_pattern
    
    # Add noise
    noise = np.random.normal(0, noise_level * base_rate, duration_minutes)
    traffic += noise
    
    return np.maximum(traffic, 0)


def inject_sudden_spike(
    traffic: np.ndarray,
    spike_start: int,
    spike_duration: int = 30,
    spike_multiplier: float = 5.0
) -> Tuple[np.ndarray, Dict]:
    """
    Inject sudden traffic spike (DDoS attack simulation).
    
    Args:
        traffic: Base traffic pattern
        spike_start: Start index for spike
        spike_duration: Duration in minutes
        spike_multiplier: Traffic increase multiplier
    
    Returns:
        (modified_traffic, metadata)
    """
    traffic = traffic.copy()
    spike_end = min(spike_start + spike_duration, len(traffic))
    
    # Instant spike
    traffic[spike_start:spike_end] *= spike_multiplier
    
    metadata = {
        'type': 'SUDDEN_SPIKE',
        'start': spike_start,
        'end': spike_end,
        'duration': spike_end - spike_start,
        'multiplier': spike_multiplier,
        'peak_traffic': float(np.max(traffic[spike_start:spike_end]))
    }
    
    return traffic, metadata


def inject_gradual_spike(
    traffic: np.ndarray,
    spike_start: int,
    ramp_duration: int = 60,
    peak_duration: int = 30,
    spike_multiplier: float = 3.0
) -> Tuple[np.ndarray, Dict]:
    """
    Inject gradual traffic increase (flash sale simulation).
    
    Traffic gradually increases over ramp_duration, stays at peak, then gradually decreases.
    """
    traffic = traffic.copy()
    
    # Ramp up
    ramp_up_end = min(spike_start + ramp_duration, len(traffic))
    ramp_up = np.linspace(1, spike_multiplier, ramp_up_end - spike_start)
    traffic[spike_start:ramp_up_end] *= ramp_up
    
    # Peak
    peak_end = min(ramp_up_end + peak_duration, len(traffic))
    traffic[ramp_up_end:peak_end] *= spike_multiplier
    
    # Ramp down
    ramp_down_end = min(peak_end + ramp_duration, len(traffic))
    ramp_down = np.linspace(spike_multiplier, 1, ramp_down_end - peak_end)
    traffic[peak_end:ramp_down_end] *= ramp_down
    
    metadata = {
        'type': 'GRADUAL_SPIKE',
        'start': spike_start,
        'end': ramp_down_end,
        'ramp_duration': ramp_duration,
        'peak_duration': peak_duration,
        'multiplier': spike_multiplier
    }
    
    return traffic, metadata


def inject_oscillating_spikes(
    traffic: np.ndarray,
    num_spikes: int = 5,
    spike_duration: int = 15,
    spike_interval: int = 120,
    spike_multiplier: float = 4.0,
    start_offset: int = 100
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Inject multiple waves of spikes (repeated DDoS attacks).
    """
    traffic = traffic.copy()
    metadata_list = []
    
    for i in range(num_spikes):
        spike_start = start_offset + i * spike_interval
        if spike_start >= len(traffic):
            break
        
        spike_end = min(spike_start + spike_duration, len(traffic))
        traffic[spike_start:spike_end] *= spike_multiplier
        
        metadata_list.append({
            'type': 'OSCILLATING_SPIKE',
            'spike_number': i + 1,
            'start': spike_start,
            'end': spike_end,
            'multiplier': spike_multiplier
        })
    
    return traffic, metadata_list


def inject_sustained_ddos(
    traffic: np.ndarray,
    attack_start: int,
    attack_duration: int = 180,
    attack_multiplier: float = 3.5
) -> Tuple[np.ndarray, Dict]:
    """
    Inject sustained high traffic (long-duration DDoS).
    """
    traffic = traffic.copy()
    attack_end = min(attack_start + attack_duration, len(traffic))
    
    # Sustained high traffic with some variation
    sustained_traffic = traffic[attack_start:attack_end] * attack_multiplier
    noise = np.random.normal(0, 0.05 * np.mean(sustained_traffic), attack_end - attack_start)
    traffic[attack_start:attack_end] = sustained_traffic + noise
    
    metadata = {
        'type': 'SUSTAINED_DDOS',
        'start': attack_start,
        'end': attack_end,
        'duration': attack_end - attack_start,
        'multiplier': attack_multiplier
    }
    
    return traffic, metadata


def generate_scenario(
    scenario_type: str,
    duration_minutes: int = 1440,
    base_rate: float = 100.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate complete traffic scenario with metadata.
    
    Args:
        scenario_type: One of ['NORMAL', 'SUDDEN_SPIKE', 'GRADUAL_SPIKE', 
                                'OSCILLATING_SPIKE', 'SUSTAINED_DDOS']
        duration_minutes: Total simulation duration
        base_rate: Base traffic rate (requests/minute)
    
    Returns:
        (dataframe, metadata)
    """
    # Generate base traffic
    traffic = generate_normal_traffic(duration_minutes, base_rate)
    
    metadata = {
        'scenario_type': scenario_type,
        'duration_minutes': duration_minutes,
        'base_rate': base_rate
    }
    
    # Apply scenario-specific modifications
    if scenario_type == 'SUDDEN_SPIKE':
        spike_start = duration_minutes // 3  # 1/3 through
        traffic, spike_meta = inject_sudden_spike(
            traffic, spike_start, spike_duration=30, spike_multiplier=5.0
        )
        metadata['spikes'] = [spike_meta]
    
    elif scenario_type == 'GRADUAL_SPIKE':
        spike_start = duration_minutes // 3
        traffic, spike_meta = inject_gradual_spike(
            traffic, spike_start, ramp_duration=60, peak_duration=30, spike_multiplier=3.0
        )
        metadata['spikes'] = [spike_meta]
    
    elif scenario_type == 'OSCILLATING_SPIKE':
        traffic, spike_metas = inject_oscillating_spikes(
            traffic, num_spikes=5, spike_duration=15, spike_interval=120, spike_multiplier=4.0
        )
        metadata['spikes'] = spike_metas
    
    elif scenario_type == 'SUSTAINED_DDOS':
        attack_start = duration_minutes // 4
        traffic, attack_meta = inject_sustained_ddos(
            traffic, attack_start, attack_duration=180, attack_multiplier=3.5
        )
        metadata['spikes'] = [attack_meta]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': np.arange(duration_minutes),
        'requests_count': traffic,
        'scenario': scenario_type
    })
    
    # Add statistics
    metadata['statistics'] = {
        'mean_traffic': float(np.mean(traffic)),
        'max_traffic': float(np.max(traffic)),
        'min_traffic': float(np.min(traffic)),
        'std_traffic': float(np.std(traffic))
    }
    
    return df, metadata


def generate_all_scenarios(
    output_dir: str = "data/synthetic_ddos",
    duration_minutes: int = 720,  # 12 hours
    base_rate: float = 100.0
) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Generate all DDoS/spike scenarios and save to files.
    
    Returns:
        Dictionary of {scenario_name: (dataframe, metadata)}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenarios = ['NORMAL', 'SUDDEN_SPIKE', 'GRADUAL_SPIKE', 'OSCILLATING_SPIKE', 'SUSTAINED_DDOS']
    results = {}
    
    print("=" * 80)
    print("GENERATING SYNTHETIC DDoS/SPIKE SCENARIOS")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n[{scenario}] Generating...", end=" ")
        
        df, metadata = generate_scenario(scenario, duration_minutes, base_rate)
        results[scenario] = (df, metadata)
        
        # Save CSV
        csv_file = output_path / f"{scenario.lower()}_traffic.csv"
        df.to_csv(csv_file, index=False)
        
        # Save metadata
        import json
        meta_file = output_path / f"{scenario.lower()}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        stats = metadata['statistics']
        print(f"✓ Mean: {stats['mean_traffic']:.1f}, Max: {stats['max_traffic']:.1f}")
        print(f"    Saved: {csv_file.name}")
    
    print("\n" + "=" * 80)
    print(f"Generated {len(scenarios)} scenarios in {output_dir}/")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Generate all scenarios
    results = generate_all_scenarios(
        output_dir="data/synthetic_ddos",
        duration_minutes=720,  # 12 hours
        base_rate=100.0
    )
    
    # Display summary
    print("\nSCENARIO SUMMARY:")
    for scenario_name, (df, metadata) in results.items():
        print(f"\n{scenario_name}:")
        print(f"  Duration: {metadata['duration_minutes']} minutes")
        print(f"  Mean traffic: {metadata['statistics']['mean_traffic']:.1f}")
        print(f"  Max traffic: {metadata['statistics']['max_traffic']:.1f}")
        if 'spikes' in metadata:
            print(f"  Spike events: {len(metadata['spikes'])}")
