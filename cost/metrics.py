"""
COST & PERFORMANCE METRICS
===========================
Simplified metrics module focused on CloudCostModel integration.

Metrics tracked:
1. COST METRICS: Total cost (reserved/spot/on-demand), average pods, efficiency
2. SLA/SLO METRICS: Violation count and rates
3. SCALING METRICS: Events, direction, oscillations
4. UTILIZATION METRICS: CPU, pod efficiency

Integrated with CloudCostModel (3-tier: reserved + spot/on-demand)
Validates cost calculation against autoscaler decisions
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class MetricsSnapshot:
    """Single timestep metrics - aligned with CloudCostModel."""
    timestamp: int
    pods: int
    requests: float
    cost: float  # Step cost (reserved + spot/on-demand)
    cost_breakdown: Dict[str, float]  # {reserved, spot, on_demand}
    cpu_utilization: float  # Computed: requests / (pods * capacity)
    sla_violated: bool  # CPU > 0.95 (SLA threshold)
    slo_violated: bool  # CPU > 0.85 (SLO threshold)
    scaling_action: int  # -1 (scale-in), 0 (no-op), +1 (scale-out)


class MetricsCollector:
    """Collects metrics aligned with CloudCostModel."""
    
    def __init__(self, capacity_per_pod: int, step_minutes: float = 15.0):
        """
        Initialize metrics collector.
        
        Args:
            capacity_per_pod: Requests per pod per minute
            step_minutes: Timestep duration (typically 15 for production)
        """
        self.capacity = capacity_per_pod
        self.step_minutes = step_minutes
        self.snapshots: List[MetricsSnapshot] = []
    
    def record(self, t: int, pods: int, requests: float, cost: float, 
               cost_breakdown: Dict[str, float], scaling_action: int = 0):
        """
        Record single timestep metrics.
        
        Args:
            t: Timestep number
            pods: Pod count after scaling
            requests: Request count
            cost: Step cost (from CloudCostModel.compute_step_cost)
            cost_breakdown: Cost breakdown {reserved, spot, on_demand}
            scaling_action: -1 (scale-in), 0 (no-op), +1 (scale-out)
        """
        # Compute utilization
        cpu = requests / (pods * self.capacity) if pods > 0 else 0.0
        
        # Check SLA/SLO (standard cloud metrics)
        sla_violated = cpu > 0.95  # SLA: CPU must be < 95%
        slo_violated = cpu > 0.85  # SLO: Target < 85%
        
        snapshot = MetricsSnapshot(
            timestamp=t,
            pods=pods,
            requests=requests,
            cost=cost,
            cost_breakdown=cost_breakdown,
            cpu_utilization=cpu,
            sla_violated=sla_violated,
            slo_violated=slo_violated,
            scaling_action=scaling_action
        )
        self.snapshots.append(snapshot)
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics from all snapshots."""
        if not self.snapshots:
            return {}
        
        snapshots = self.snapshots
        
        # ========== COST METRICS ==========
        total_cost = sum(s.cost for s in snapshots)
        
        # Cost breakdown by instance type
        cost_reserved = sum(s.cost_breakdown.get('reserved', 0) for s in snapshots)
        cost_spot = sum(s.cost_breakdown.get('spot', 0) for s in snapshots)
        cost_ondemand = sum(s.cost_breakdown.get('on_demand', 0) for s in snapshots)
        
        avg_pods = np.mean([s.pods for s in snapshots])
        min_pods = np.min([s.pods for s in snapshots])
        max_pods = np.max([s.pods for s in snapshots])
        
        # Cost efficiency: cost per pod per timestep
        cost_per_pod = total_cost / sum(s.pods for s in snapshots) if sum(s.pods for s in snapshots) > 0 else 0
        
        # ========== SLA/SLO METRICS ==========
        sla_violations = sum(1 for s in snapshots if s.sla_violated)
        slo_violations = sum(1 for s in snapshots if s.slo_violated)
        
        sla_violation_rate = sla_violations / len(snapshots)
        slo_violation_rate = slo_violations / len(snapshots)
        
        # ========== SCALING METRICS ==========
        actions = np.array([s.scaling_action for s in snapshots])
        
        scaling_events = np.sum(actions != 0)
        scale_ups = np.sum(actions > 0)
        scale_downs = np.sum(actions < 0)
        
        # Oscillation: rapid scale-up/down cycles (flapping)
        oscillations = sum(1 for i in range(1, len(actions)) 
                          if actions[i] * actions[i-1] < 0)
        
        # ========== UTILIZATION METRICS ==========
        utils = np.array([s.cpu_utilization for s in snapshots])
        
        avg_cpu = np.mean(utils)
        max_cpu = np.max(utils)
        min_cpu = np.min(utils)
        
        # Efficiency: how well we utilized capacity (avoid over-provisioning)
        efficiency = 1.0 - np.mean(np.maximum(0, 0.95 - utils))  # 0.95 is ideal
        
        # Return consolidated metrics
        return {
            # Cost metrics (primary)
            'total_cost': float(total_cost),
            'cost_reserved': float(cost_reserved),
            'cost_spot': float(cost_spot),
            'cost_ondemand': float(cost_ondemand),
            'cost_per_pod': float(cost_per_pod),
            
            # Pod metrics
            'avg_pods': float(avg_pods),
            'min_pods': int(min_pods),
            'max_pods': int(max_pods),
            
            # SLA/SLO metrics (performance)
            'sla_violations': int(sla_violations),
            'sla_violation_rate': float(sla_violation_rate),
            'slo_violations': int(slo_violations),
            'slo_violation_rate': float(slo_violation_rate),
            
            # Scaling metrics (stability)
            'scaling_events': int(scaling_events),
            'scale_ups': int(scale_ups),
            'scale_downs': int(scale_downs),
            'oscillations': int(oscillations),
            
            # Utilization metrics
            'avg_cpu': float(avg_cpu),
            'max_cpu': float(max_cpu),
            'min_cpu': float(min_cpu),
            'efficiency': float(efficiency),
            
            # Meta
            'total_timesteps': len(snapshots),
        }


# ============================================================================
# LEGACY FUNCTIONS (Backward Compatibility)
# ============================================================================

def sla_violation_rate(requests, servers, capacity):
    """
    DEPRECATED: Use MetricsCollector.compute_aggregate_metrics() instead.
    
    Compute SLA violation rate.
    """
    violations = requests > (servers * capacity)
    return np.mean(violations)


def compare_strategies(results_dict: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compare multiple strategy results side-by-side.
    
    Args:
        results_dict: {strategy_name: metrics_dict, ...}
    
    Returns:
        Comparison structure with key metrics
    
    Example:
        >>> results = {
        ...     'HYBRID': {'total_cost': 57.79, 'sla_violations': 14, ...},
        ...     'REACTIVE': {'total_cost': 59.47, 'sla_violations': 41, ...}
        ... }
        >>> compare_strategies(results)
    """
    comparison = {}
    
    for strategy_name, metrics in results_dict.items():
        comparison[strategy_name] = {
            'Cost': f"${metrics['total_cost']:.2f}",
            'Avg Pods': f"{metrics['avg_pods']:.1f}",
            'SLA Violations': f"{metrics['sla_violations']}",
            'Scaling Events': f"{metrics['scaling_events']}",
            'Oscillations': f"{metrics['oscillations']}",
            'Avg CPU': f"{metrics['avg_cpu']:.1%}",
        }
    
    return comparison


