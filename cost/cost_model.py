"""
COST MODEL FOR CLOUD AUTOSCALING
=================================
Production-grade cost modeling for autoscaling systems.

Supports:
- On-demand instances (AWS EC2, GCP Compute, Azure VMs)
- Reserved capacity (AWS Reserved Instances, GCP Committed Use)
- Spot/Preemptible instances (AWS Spot, GCP Preemptible, Azure Low Priority)
- Kubernetes node pools with mixed instance types
- Multi-tier pricing (Google Borg priority classes)
- Startup/shutdown costs (cold start penalty)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class InstanceType(Enum):
    """Instance pricing tiers."""
    ON_DEMAND = "on_demand"          # Pay-as-you-go (most expensive)
    RESERVED = "reserved"            # 1-3 year commitment (40-60% cheaper)
    SPOT = "spot"                    # Interruptible (70-90% cheaper)
    PREEMPTIBLE = "preemptible"      # GCP term for spot
    SAVINGS_PLAN = "savings_plan"    # AWS flexible commitment


class PriorityClass(Enum):
    """Google Borg style priority classes."""
    PRODUCTION = "production"        # Critical workload (highest cost)
    BATCH = "batch"                  # Deferrable work (can use spot)
    BEST_EFFORT = "best_effort"      # Lowest priority (preemptible only)


@dataclass
class PricingTier:
    """Pricing configuration for instance type."""
    instance_type: InstanceType
    cost_per_hour: float
    startup_cost: float = 0.0        # Cold start penalty (EBS, network setup)
    shutdown_cost: float = 0.0       # Graceful termination cost
    interruption_rate: float = 0.0   # Spot interruption probability (0-1)
    min_commit_hours: int = 0        # Reserved capacity commitment


class CloudCostModel:
    """
    Multi-tier cloud cost model.
    
    Supports:
    - Mixed instance types (on-demand + spot)
    - Reserved capacity baseline
    - Kubernetes-style node pool management
    - Cold start penalties
    - Spot interruption handling
    """
    
    def __init__(self, 
                 on_demand_cost: float = 0.05,
                 reserved_cost: float = 0.03,
                 spot_cost: float = 0.015,
                 startup_cost: float = 0.001,
                 reserved_capacity: int = 0):
        """
        Args:
            on_demand_cost: $/hour for on-demand instance
            reserved_cost: $/hour for reserved instance
            spot_cost: $/hour for spot instance
            startup_cost: One-time cost for launching instance
            reserved_capacity: Baseline reserved capacity (always running)
        """
        self.pricing = {
            InstanceType.ON_DEMAND: PricingTier(InstanceType.ON_DEMAND, on_demand_cost, startup_cost),
            InstanceType.RESERVED: PricingTier(InstanceType.RESERVED, reserved_cost, 0.0),
            InstanceType.SPOT: PricingTier(InstanceType.SPOT, spot_cost, startup_cost * 0.5, interruption_rate=0.05),
        }
        
        self.reserved_capacity = reserved_capacity
        self.total_startup_cost = 0.0
        self.total_interruption_cost = 0.0
    
    def compute_step_cost(self, 
                         pod_count: int, 
                         step_hours: float,
                         strategy: str = "auto") -> Tuple[float, Dict[str, float]]:
        """
        Compute cost for single timestep with intelligent instance selection.
        
        Args:
            pod_count: Total pods needed
            step_hours: Timestep duration (minutes / 60)
            strategy: "auto", "on_demand", "spot_first", "reserved_first"
        
        Returns:
            (total_cost, breakdown): Cost and breakdown by instance type
        """
        breakdown = {
            'reserved': 0.0,
            'on_demand': 0.0,
            'spot': 0.0,
            'startup': 0.0,
            'interruption': 0.0
        }
        
        # Reserved capacity (always running)
        reserved_pods = min(pod_count, self.reserved_capacity)
        breakdown['reserved'] = reserved_pods * self.pricing[InstanceType.RESERVED].cost_per_hour * step_hours
        
        # Remaining demand
        remaining_pods = max(0, pod_count - reserved_pods)
        
        if remaining_pods > 0:
            if strategy == "auto" or strategy == "spot_first":
                # Kubernetes style: Fill spot capacity first (cheaper)
                spot_ratio = 0.7  # 70% spot, 30% on-demand for reliability
                spot_pods = int(remaining_pods * spot_ratio)
                on_demand_pods = remaining_pods - spot_pods
                
                breakdown['spot'] = spot_pods * self.pricing[InstanceType.SPOT].cost_per_hour * step_hours
                breakdown['on_demand'] = on_demand_pods * self.pricing[InstanceType.ON_DEMAND].cost_per_hour * step_hours
                
                # Spot interruption risk (average cost)
                breakdown['interruption'] = spot_pods * self.pricing[InstanceType.SPOT].interruption_rate * 0.01
                
            else:  # on_demand only
                breakdown['on_demand'] = remaining_pods * self.pricing[InstanceType.ON_DEMAND].cost_per_hour * step_hours
        
        total = sum(breakdown.values())
        return total, breakdown
    
    def compute_total_cost(self, 
                          pod_history: List[int], 
                          step_minutes: float = 5.0,
                          track_scaling_cost: bool = True) -> Tuple[float, Dict]:
        """
        Compute total cost over time with scaling penalties.
        
        Args:
            pod_history: List of pod counts over time
            step_minutes: Duration of each timestep
            track_scaling_cost: Include startup/shutdown costs
        
        Returns:
            (total_cost, detailed_breakdown)
        """
        step_hours = step_minutes / 60.0
        total = 0.0
        breakdown = {'reserved': 0.0, 'on_demand': 0.0, 'spot': 0.0, 'startup': 0.0}
        
        prev_pods = 0
        
        for pods in pod_history:
            # Runtime cost
            step_cost, step_breakdown = self.compute_step_cost(pods, step_hours)
            total += step_cost
            
            for key in step_breakdown:
                breakdown[key] = breakdown.get(key, 0.0) + step_breakdown[key]
            
            # Scaling cost (cold start)
            if track_scaling_cost and pods > prev_pods:
                scale_out_count = pods - prev_pods
                startup_cost = scale_out_count * self.pricing[InstanceType.ON_DEMAND].startup_cost
                total += startup_cost
                breakdown['startup'] += startup_cost
            
            prev_pods = pods
        
        breakdown['total'] = total
        return total, breakdown
    
    def compute_borg_style_cost(self, 
                               pod_history: List[int],
                               priority_class: PriorityClass,
                               step_minutes: float = 5.0) -> float:
        """
        Google Borg style priority-based costing.
        
        Priority classes:
        - PRODUCTION: Always on-demand (reliability)
        - BATCH: Prefer spot (cost optimization)
        - BEST_EFFORT: Spot only (no SLA)
        """
        step_hours = step_minutes / 60.0
        total = 0.0
        
        for pods in pod_history:
            if priority_class == PriorityClass.PRODUCTION:
                # Production: Reserved + on-demand
                reserved_pods = min(pods, self.reserved_capacity)
                on_demand_pods = max(0, pods - reserved_pods)
                
                total += reserved_pods * self.pricing[InstanceType.RESERVED].cost_per_hour * step_hours
                total += on_demand_pods * self.pricing[InstanceType.ON_DEMAND].cost_per_hour * step_hours
                
            elif priority_class == PriorityClass.BATCH:
                # Batch: Spot with on-demand fallback
                spot_pods = int(pods * 0.8)
                on_demand_pods = pods - spot_pods
                
                total += spot_pods * self.pricing[InstanceType.SPOT].cost_per_hour * step_hours
                total += on_demand_pods * self.pricing[InstanceType.ON_DEMAND].cost_per_hour * step_hours
                
            else:  # BEST_EFFORT
                # Best effort: Spot only
                total += pods * self.pricing[InstanceType.SPOT].cost_per_hour * step_hours
        
        return total


class KubernetesCostModel:
    """
    Kubernetes-specific cost model.
    
    Accounts for:
    - Node overhead (kubelet, system pods)
    - Pod packing efficiency
    - Node pool fragmentation
    - Cluster autoscaler delays
    """
    
    def __init__(self,
                 node_cost_per_hour: float = 0.10,
                 pods_per_node: int = 30,
                 node_overhead_pods: int = 3,
                 scale_up_delay_minutes: int = 2):
        """
        Args:
            node_cost_per_hour: Cost per node (e.g., AWS m5.large)
            pods_per_node: Max pods per node (K8s limit: 110)
            node_overhead_pods: System pods (kube-proxy, CNI, etc.)
            scale_up_delay_minutes: Cluster autoscaler reaction time
        """
        self.node_cost = node_cost_per_hour
        self.pods_per_node = pods_per_node
        self.overhead = node_overhead_pods
        self.scale_delay = scale_up_delay_minutes
        
        self.usable_pods_per_node = pods_per_node - node_overhead_pods
    
    def compute_node_cost(self, pod_count: int, step_hours: float) -> Tuple[float, int]:
        """
        Compute cost accounting for node packing.
        
        Returns:
            (cost, node_count): Total cost and number of nodes needed
        """
        # Ceiling division: need full node even if partially used
        nodes_needed = int(np.ceil(pod_count / self.usable_pods_per_node))
        cost = nodes_needed * self.node_cost * step_hours
        
        return cost, nodes_needed
    
    def compute_total_cost(self, pod_history: List[int], step_minutes: float = 5.0) -> Dict:
        """
        Compute K8s cluster cost with efficiency metrics.
        """
        step_hours = step_minutes / 60.0
        total = 0.0
        total_nodes = 0
        total_pods = 0
        
        for pods in pod_history:
            cost, nodes = self.compute_node_cost(pods, step_hours)
            total += cost
            total_nodes += nodes
            total_pods += pods
        
        avg_nodes = total_nodes / len(pod_history) if pod_history else 0
        avg_pods = total_pods / len(pod_history) if pod_history else 0
        
        # Packing efficiency: how well we utilize nodes
        packing_efficiency = avg_pods / (avg_nodes * self.usable_pods_per_node) if avg_nodes > 0 else 0
        
        return {
            'total_cost': total,
            'avg_nodes': avg_nodes,
            'avg_pods': avg_pods,
            'packing_efficiency': packing_efficiency,
            'wasted_capacity': (1 - packing_efficiency) * 100  # percentage
        }


# Legacy function (kept for backward compatibility)
def compute_cost(server_history, cost_per_server_per_hour, step_minutes):
    """
    DEPRECATED: Use CloudCostModel.compute_total_cost() instead.
    """
    total = 0.0
    for n in server_history:
        total += n * cost_per_server_per_hour * (step_minutes / 60)
    return total
