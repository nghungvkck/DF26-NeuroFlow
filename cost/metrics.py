"""
COMPREHENSIVE METRICS MODULE
=============================
Tracks and aggregates cost, performance, and stability metrics.

Metrics computed:
1. COST METRICS: Total cost, average pods, overprovision ratio
2. PERFORMANCE METRICS: SLA violation rate, mean response delay
3. STABILITY METRICS: Number of scaling events, oscillation count
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class MetricsSnapshot:
    """Single timestep metrics snapshot."""
    timestamp: int
    pods: int
    requests: float
    cost_accumulated: float
    sla_violated: bool
    sla_violated_before_scaling: bool  # NEW: SLA violation BEFORE scaling decision
    scaling_action: int  # -1, 0, +1


class MetricsCollector:
    """Collects and aggregates metrics throughout simulation."""
    
    def __init__(self, capacity_per_pod, cost_per_pod_per_hour, step_minutes=5.0):
        self.capacity = capacity_per_pod
        self.cost_per_hour = cost_per_pod_per_hour
        self.step_minutes = step_minutes
        self.step_hours = step_minutes / 60.0
        
        self.snapshots: List[MetricsSnapshot] = []
    
    def record(self, t, pods, requests, scaling_action, sla_before_scaling=False):
        """
        Record timestep metrics.
        
        Args:
            t: timestep
            pods: pod count AFTER scaling decision
            requests: actual request count
            scaling_action: scaling decision (-1, 0, +1)
            sla_before_scaling: whether SLA was violated BEFORE scaling (important!)
        """
        # Compute cost for this timestep (based on post-scaling pod count)
        step_cost = pods * self.cost_per_hour * self.step_hours
        
        # Accumulated cost
        accumulated = (self.snapshots[-1].cost_accumulated if self.snapshots else 0) + step_cost
        
        # SLA check: requests > capacity? (after scaling)
        sla_violated_after = requests > (pods * self.capacity)
        
        snapshot = MetricsSnapshot(
            timestamp=t,
            pods=pods,
            requests=requests,
            cost_accumulated=accumulated,
            sla_violated=sla_violated_after,
            sla_violated_before_scaling=sla_before_scaling,
            scaling_action=scaling_action
        )
        self.snapshots.append(snapshot)
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute all aggregate metrics."""
        if len(self.snapshots) == 0:
            return {}
        
        snapshots = self.snapshots
        
        # ==================== COST METRICS ====================
        total_cost = snapshots[-1].cost_accumulated
        avg_pods = np.mean([s.pods for s in snapshots])
        
        # Overprovision ratio: how much extra capacity was idle
        capacities = np.array([s.pods * self.capacity for s in snapshots])
        requests = np.array([s.requests for s in snapshots])
        overprovision_ratio = np.mean((capacities - requests) / np.maximum(capacities, 1))
        
        # ==================== PERFORMANCE METRICS ====================
        # Count SLA violations BEFORE scaling (when demand exceeded current capacity)
        sla_violations = np.sum([int(s.sla_violated_before_scaling) for s in snapshots])
        sla_violation_rate = sla_violations / len(snapshots) if len(snapshots) > 0 else 0
        
        # Reaction time: delay between SLA breach and scale response
        reaction_delays = []
        for i in range(1, len(snapshots)):
            if snapshots[i-1].sla_violated_before_scaling and snapshots[i].scaling_action > 0:
                reaction_delays.append(1)  # Immediate response
            elif not snapshots[i-1].sla_violated_before_scaling and snapshots[i].sla_violated_before_scaling:
                reaction_delays.append(0)  # Just breached
        
        mean_reaction_delay = np.mean(reaction_delays) if reaction_delays else 0
        
        # ==================== STABILITY METRICS ====================
        actions = np.array([s.scaling_action for s in snapshots])
        
        # Total scaling events (non-zero actions)
        scaling_events = np.sum(actions != 0)
        
        # Oscillation count: alternating scale-up/down (flapping)
        oscillations = 0
        for i in range(1, len(actions)):
            if actions[i] * actions[i-1] < 0:  # Sign change = flapping
                oscillations += 1
        
        # Scale-out ratio: how often we scale out vs scale in
        scale_outs = np.sum(actions > 0)
        scale_ins = np.sum(actions < 0)
        scale_out_ratio = scale_outs / max(scale_outs + scale_ins, 1)
        
        # ==================== UTILIZATION METRICS ====================
        utilizations = requests / np.maximum(capacities, 1)
        mean_utilization = np.mean(utilizations)
        max_utilization = np.max(utilizations)
        min_utilization = np.min(utilizations)
        
        return {
            # Cost
            'total_cost': float(total_cost),
            'average_pods': float(avg_pods),
            'overprovision_ratio': float(overprovision_ratio),
            'cost_per_timestep': float(total_cost / len(snapshots)) if len(snapshots) > 0 else 0,
            
            # Performance
            'sla_violations': int(sla_violations),
            'sla_violation_rate': float(sla_violation_rate),
            'mean_reaction_delay': float(mean_reaction_delay),
            
            # Stability
            'scaling_events': int(scaling_events),
            'oscillation_count': int(oscillations),
            'scale_out_ratio': float(scale_out_ratio),
            'scale_ups': int(scale_outs),
            'scale_downs': int(scale_ins),
            
            # Utilization
            'mean_utilization': float(mean_utilization),
            'max_utilization': float(max_utilization),
            'min_utilization': float(min_utilization),
            
            # Summary
            'total_timesteps': len(snapshots),
        }


# Legacy functions (kept for backward compatibility)

def sla_violation_rate(requests, servers, capacity):
    """
    Legacy: Compute SLA violation rate.
    
    DEPRECATED: Use MetricsCollector instead
    """
    violations = requests > (servers * capacity)
    return np.mean(violations)


def overprovision_ratio(requests, servers, capacity):
    """
    Legacy: Compute overprovision ratio.
    
    DEPRECATED: Use MetricsCollector instead
    """
    over = (servers * capacity - requests) / np.maximum(servers * capacity, 1)
    return np.mean(over)


def speed_of_scale(overload_times, recovery_times):
    """
    Legacy: Compute mean recovery time.
    
    DEPRECATED: Use MetricsCollector instead
    """
    return np.mean(recovery_times - overload_times)


# Utility functions

def compare_strategies(results_dict: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compare multiple strategy results side-by-side.
    
    Args:
        results_dict: {strategy_name: metrics_dict, ...}
    
    Returns:
        Comparison dataframe-like structure
    """
    comparison = {}
    
    for strategy_name, metrics in results_dict.items():
        comparison[strategy_name] = {
            'Cost ($)': f"{metrics['total_cost']:.2f}",
            'Avg Pods': f"{metrics['average_pods']:.1f}",
            'SLA Violations': f"{metrics['sla_violations']} ({metrics['sla_violation_rate']:.1%})",
            'Scaling Events': f"{metrics['scaling_events']}",
            'Oscillations': f"{metrics['oscillation_count']}",
            'Mean Util.': f"{metrics['mean_utilization']:.1%}",
        }
    
    return comparison

