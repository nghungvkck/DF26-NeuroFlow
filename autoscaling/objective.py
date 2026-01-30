"""
OBJECTIVE FUNCTION MODULE
==========================
Explicit multi-objective cost function for autoscaling optimization.

Minimizes:
1. Infrastructure cost (pods × cost_per_hour)
2. SLA violations (requests exceed capacity)
3. Scaling instability (flapping/oscillation)

This is the quantitative target that guides all scaling decisions.
"""

import numpy as np


def compute_cost_objective(
    pod_history,
    cost_per_pod_per_hour,
    step_minutes=5.0
):
    """
    COST COMPONENT: Total infrastructure cost
    
    Args:
        pod_history: list/array of pod counts over time
        cost_per_pod_per_hour: float, cost per pod per hour
        step_minutes: time interval between steps (default 5 min)
    
    Returns:
        float: Total cost in currency units
    """
    step_hours = step_minutes / 60.0
    total_cost = sum(pod_history) * cost_per_pod_per_hour * step_hours
    return total_cost


def compute_sla_violation_cost(
    requests_history,
    pods_history,
    capacity_per_pod,
    sla_penalty_per_violation=100.0
):
    """
    SLA VIOLATION COMPONENT: Penalty for each timestep where SLA is breached
    
    SLA_t = 1 if requests_t > pods_t × capacity
    Cost_SLA = sum(SLA_t) × penalty_per_violation
    
    Args:
        requests_history: list/array of request counts over time
        pods_history: list/array of pod counts over time
        capacity_per_pod: float, requests/min per pod
        sla_penalty_per_violation: penalty cost per SLA breach
    
    Returns:
        float: Total SLA violation cost
    """
    violations = 0
    for t in range(len(requests_history)):
        capacity = pods_history[t] * capacity_per_pod
        if requests_history[t] > capacity:
            violations += 1
    
    sla_cost = violations * sla_penalty_per_violation
    return sla_cost, violations


def compute_stability_cost(
    action_history,
    flapping_penalty_per_event=50.0
):
    """
    STABILITY COMPONENT: Penalty for scaling events (flapping)
    
    Minimizes the number of scale-up/down decisions to prevent:
    - Rapid pod allocation/deallocation
    - SLA violations during transitions
    - Wasted resource churn
    
    Args:
        action_history: list/array of scaling actions (+1, 0, -1)
        flapping_penalty_per_event: cost per scale decision
    
    Returns:
        float: Total stability cost
        int: Number of scaling events
    """
    # Count non-zero actions (scaling events)
    scaling_events = np.sum(np.array(action_history) != 0)
    stability_cost = scaling_events * flapping_penalty_per_event
    
    return stability_cost, scaling_events


def compute_total_objective(
    pod_history,
    requests_history,
    action_history,
    capacity_per_pod,
    cost_per_pod_per_hour=0.05,
    sla_penalty_per_violation=100.0,
    flapping_penalty_per_event=50.0,
    step_minutes=5.0,
    weights=None
):
    """
    TOTAL OBJECTIVE FUNCTION (multi-objective aggregation)
    
    Minimize: w_cost × Cost + w_sla × SLA + w_stability × Stability
    
    Args:
        pod_history: list of pod counts
        requests_history: list of request counts
        action_history: list of scaling actions
        capacity_per_pod: capacity per pod
        cost_per_pod_per_hour: cost parameter
        sla_penalty_per_violation: SLA cost parameter
        flapping_penalty_per_event: stability cost parameter
        step_minutes: time interval
        weights: dict with keys 'cost', 'sla', 'stability' (default equal weights)
    
    Returns:
        dict with components: {
            'total': float,
            'cost_component': float,
            'sla_component': float,
            'stability_component': float,
            'sla_violations': int,
            'scaling_events': int
        }
    """
    # Default equal weighting
    if weights is None:
        weights = {'cost': 1.0, 'sla': 1.0, 'stability': 1.0}
    
    # Compute individual components
    cost = compute_cost_objective(pod_history, cost_per_pod_per_hour, step_minutes)
    sla_cost, violations = compute_sla_violation_cost(
        requests_history, pod_history, capacity_per_pod, sla_penalty_per_violation
    )
    stability_cost, events = compute_stability_cost(
        action_history, flapping_penalty_per_event
    )
    
    # Weighted aggregation
    total_objective = (
        weights['cost'] * cost +
        weights['sla'] * sla_cost +
        weights['stability'] * stability_cost
    )
    
    return {
        'total': total_objective,
        'cost_component': cost,
        'sla_component': sla_cost,
        'stability_component': stability_cost,
        'sla_violations': int(violations),
        'scaling_events': int(events),
    }
