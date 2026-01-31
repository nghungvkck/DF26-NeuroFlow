"""
COMPREHENSIVE METRICS MODULE
=============================
Tracks and aggregates cost, performance, and stability metrics.

Metrics computed:
1. COST METRICS: Total cost, average pods, overprovision ratio
2. PERFORMANCE METRICS: SLA violation rate, mean response delay
3. STABILITY METRICS: Number of scaling events, oscillation count
4. KUBERNETES HPA METRICS: Resource utilization, target tracking
5. AWS AUTO SCALING METRICS: Warm-up time, cooldown effectiveness
6. GOOGLE BORG METRICS: Priority enforcement, resource efficiency
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


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
    
    # Kubernetes HPA metrics
    cpu_utilization: Optional[float] = None  # CPU utilization percentage (0-1)
    memory_utilization: Optional[float] = None  # Memory utilization (0-1)
    custom_metric_value: Optional[float] = None  # Custom metric (e.g., queue depth)
    
    # AWS Auto Scaling metrics
    warm_up_active: bool = False  # Instance warming up (grace period)
    cooldown_active: bool = False  # Cooldown period active
    target_tracking_breach: bool = False  # Target metric breached
    
    # Google Borg metrics
    priority_preemptions: int = 0  # Low priority pods preempted
    resource_quota_breach: bool = False  # Quota exceeded


class MetricsCollector:
    """Collects and aggregates metrics throughout simulation."""
    
    def __init__(self, capacity_per_pod, cost_per_pod_per_hour, step_minutes=5.0,
                 enable_k8s_metrics=True, enable_aws_metrics=True, enable_borg_metrics=False):
        self.capacity = capacity_per_pod
        self.cost_per_hour = cost_per_pod_per_hour
        self.step_minutes = step_minutes
        self.step_hours = step_minutes / 60.0
        
        # Feature flags for different metric systems
        self.enable_k8s_metrics = enable_k8s_metrics
        self.enable_aws_metrics = enable_aws_metrics
        self.enable_borg_metrics = enable_borg_metrics
        
        # Kubernetes HPA state
        self.target_cpu_utilization = 0.7  # Default HPA target: 70%
        self.target_memory_utilization = 0.8
        
        # AWS Auto Scaling state
        self.warm_up_time = 300 / step_minutes  # 5 min warm-up (in timesteps)
        self.cooldown_time = 300 / step_minutes  # 5 min cooldown
        self.last_scale_out_time = -999
        self.last_scale_in_time = -999
        
        # Google Borg state
        self.resource_quota = 100  # Max pods allowed
        self.preemption_count = 0
        
        self.snapshots: List[MetricsSnapshot] = []
    
    def record(self, t, pods, requests, scaling_action, sla_before_scaling=False,
               cpu_utilization=None, memory_utilization=None, custom_metric=None):
        """
        Record timestep metrics.
        
        Args:
            t: timestep
            pods: pod count AFTER scaling decision
            requests: actual request count
            scaling_action: scaling decision (-1, 0, +1)
            sla_before_scaling: whether SLA was violated BEFORE scaling (important!)
            cpu_utilization: Optional CPU utilization (Kubernetes HPA)
            memory_utilization: Optional memory utilization (Kubernetes HPA)
            custom_metric: Optional custom metric value (Kubernetes HPA)
        """
        # Compute cost for this timestep (based on post-scaling pod count)
        step_cost = pods * self.cost_per_hour * self.step_hours
        
        # Accumulated cost
        accumulated = (self.snapshots[-1].cost_accumulated if self.snapshots else 0) + step_cost
        
        # SLA check: requests > capacity? (after scaling)
        sla_violated_after = requests > (pods * self.capacity)
        
        # Kubernetes HPA metrics
        k8s_cpu = cpu_utilization if cpu_utilization is not None else (requests / (pods * self.capacity) if pods > 0 else 0)
        k8s_memory = memory_utilization
        k8s_custom = custom_metric
        
        # AWS Auto Scaling metrics
        aws_warm_up = False
        aws_cooldown = False
        if self.enable_aws_metrics:
            if scaling_action > 0:
                self.last_scale_out_time = t
                aws_warm_up = True
            elif scaling_action < 0:
                self.last_scale_in_time = t
            
            # Check if still in warm-up from previous scale-out
            if t - self.last_scale_out_time <= self.warm_up_time:
                aws_warm_up = True
            
            # Check if in cooldown period
            if t - self.last_scale_out_time <= self.cooldown_time or t - self.last_scale_in_time <= self.cooldown_time:
                aws_cooldown = True
        
        aws_target_breach = k8s_cpu > self.target_cpu_utilization if k8s_cpu is not None else False
        
        # Google Borg metrics
        borg_preemptions = 0
        borg_quota_breach = False
        if self.enable_borg_metrics:
            if scaling_action < 0 and pods < self.resource_quota * 0.5:
                # Simulate preemption of low-priority pods
                borg_preemptions = abs(scaling_action)
                self.preemption_count += borg_preemptions
            
            if pods > self.resource_quota:
                borg_quota_breach = True
        
        snapshot = MetricsSnapshot(
            timestamp=t,
            pods=pods,
            requests=requests,
            cost_accumulated=accumulated,
            sla_violated=sla_violated_after,
            sla_violated_before_scaling=sla_before_scaling,
            scaling_action=scaling_action,
            cpu_utilization=k8s_cpu,
            memory_utilization=k8s_memory,
            custom_metric_value=k8s_custom,
            warm_up_active=aws_warm_up,
            cooldown_active=aws_cooldown,
            target_tracking_breach=aws_target_breach,
            priority_preemptions=borg_preemptions,
            resource_quota_breach=borg_quota_breach
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
        
        # ==================== KUBERNETES HPA METRICS ====================
        k8s_metrics = {}
        if self.enable_k8s_metrics:
            cpu_utils = [s.cpu_utilization for s in snapshots if s.cpu_utilization is not None]
            if cpu_utils:
                k8s_metrics['k8s_avg_cpu_utilization'] = float(np.mean(cpu_utils))
                k8s_metrics['k8s_max_cpu_utilization'] = float(np.max(cpu_utils))
                k8s_metrics['k8s_cpu_target_breaches'] = int(np.sum(np.array(cpu_utils) > self.target_cpu_utilization))
                k8s_metrics['k8s_cpu_target_breach_rate'] = float(k8s_metrics['k8s_cpu_target_breaches'] / len(cpu_utils))
            
            # HPA effectiveness: How often HPA would trigger scaling
            hpa_would_scale = 0
            for s in snapshots:
                if s.cpu_utilization is not None:
                    if s.cpu_utilization > self.target_cpu_utilization * 1.1:  # 10% threshold
                        hpa_would_scale += 1
            k8s_metrics['k8s_hpa_trigger_rate'] = float(hpa_would_scale / len(snapshots))
        
        # ==================== AWS AUTO SCALING METRICS ====================
        aws_metrics = {}
        if self.enable_aws_metrics:
            warm_up_time = np.sum([s.warm_up_active for s in snapshots])
            cooldown_time = np.sum([s.cooldown_active for s in snapshots])
            
            aws_metrics['aws_warm_up_time_ratio'] = float(warm_up_time / len(snapshots))
            aws_metrics['aws_cooldown_time_ratio'] = float(cooldown_time / len(snapshots))
            aws_metrics['aws_target_tracking_breaches'] = int(np.sum([s.target_tracking_breach for s in snapshots]))
            
            # Cooldown effectiveness: Did cooldown prevent unnecessary scaling?
            cooldown_effectiveness = 0
            for i in range(1, len(snapshots)):
                if snapshots[i].cooldown_active and snapshots[i].scaling_action == 0:
                    cooldown_effectiveness += 1
            aws_metrics['aws_cooldown_effectiveness'] = float(cooldown_effectiveness / max(cooldown_time, 1))
        
        # ==================== GOOGLE BORG METRICS ====================
        borg_metrics = {}
        if self.enable_borg_metrics:
            total_preemptions = self.preemption_count
            quota_breaches = np.sum([s.resource_quota_breach for s in snapshots])
            
            borg_metrics['borg_total_preemptions'] = int(total_preemptions)
            borg_metrics['borg_quota_breaches'] = int(quota_breaches)
            borg_metrics['borg_quota_breach_rate'] = float(quota_breaches / len(snapshots))
            
            # Resource efficiency (Borg's primary metric)
            borg_metrics['borg_resource_efficiency'] = float(mean_utilization)
        
        # Combine all metrics
        result = {
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
        
        # Add platform-specific metrics
        result.update(k8s_metrics)
        result.update(aws_metrics)
        result.update(borg_metrics)
        
        return result


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

