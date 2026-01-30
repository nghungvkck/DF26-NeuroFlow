"""
HYSTERESIS & COOLDOWN MODULE
=============================
Anti-flapping mechanisms for stable autoscaling.

Includes:
1. Adaptive cooldown based on traffic volatility
2. Majority-based hysteresis to prevent rapid oscillation
3. Decision smoothing to stabilize scaling
"""

import numpy as np
from collections import deque


def adaptive_cooldown(base, traffic_window):
    """
    Adaptive cooldown based on traffic volatility.
    
    High traffic variance → longer cooldown to avoid flapping
    Low traffic variance → shorter cooldown for faster response
    
    Args:
        base: base cooldown in timesteps
        traffic_window: recent request history
    
    Returns:
        int: adjusted cooldown in timesteps
    """
    if len(traffic_window) == 0:
        return base
    
    sigma = np.std(traffic_window)
    
    # Higher volatility = longer cooldown
    # Cooldown formula: base / (1 + sigma) ∈ [base/10, base]
    adjusted = max(1, int(base / (1 + sigma / np.mean(traffic_window) if np.mean(traffic_window) > 0 else 1)))
    
    return adjusted


class MajorityHysteresis:
    """
    Majority-voting hysteresis to prevent flapping.
    
    Requires N out of M recent decisions to agree before
    accepting a scaling decision. This prevents single
    anomalies from causing unnecessary scaling.
    """
    
    def __init__(self, window_size=3, required_majority=2):
        """
        Args:
            window_size: how many recent decisions to track (default 3)
            required_majority: minimum decisions agreeing (default 2)
        """
        self.window_size = window_size
        self.required_majority = required_majority
        self.decision_history = deque(maxlen=window_size)
    
    def should_scale(self, current_decision):
        """
        Apply majority voting to scale decisions.
        
        Args:
            current_decision: current scaling decision (+1, 0, -1)
        
        Returns:
            (should_scale: bool, reason: str)
        """
        self.decision_history.append(current_decision)
        
        # Not enough history yet
        if len(self.decision_history) < self.required_majority:
            return False, f"Insufficient history: {len(self.decision_history)}/{self.required_majority}"
        
        # Count votes
        decisions = list(self.decision_history)
        decision_counts = {
            1: decisions.count(1),      # Scale-out votes
            0: decisions.count(0),      # Hold votes
            -1: decisions.count(-1),    # Scale-in votes
        }
        
        max_count = max(decision_counts.values())
        
        # Majority threshold reached?
        if max_count >= self.required_majority:
            # Find which decision won
            winning_decision = [d for d, c in decision_counts.items() if c == max_count][0]
            
            if winning_decision != 0:
                action_str = "SCALE_OUT" if winning_decision > 0 else "SCALE_IN"
                return True, f"Majority ({max_count}/{self.window_size}): {action_str}"
            else:
                return False, f"Majority ({max_count}/{self.window_size}): HOLD"
        
        return False, f"No majority: {decision_counts}"
    
    def reset(self):
        """Clear decision history."""
        self.decision_history.clear()


def smooth_scaling_decisions(action_sequence, smoothing_window=3):
    """
    Post-process scaling decisions to reduce noise.
    
    Converts: [+1, +1, 0, +1] → [+1, +1, +1, +1]
    Smooths isolated decisions that contradict local trend.
    
    Args:
        action_sequence: list of scaling actions
        smoothing_window: window size for trend detection
    
    Returns:
        list: smoothed action sequence
    """
    if len(action_sequence) < smoothing_window:
        return action_sequence
    
    smoothed = action_sequence.copy()
    
    for i in range(smoothing_window // 2, len(action_sequence) - smoothing_window // 2):
        window = action_sequence[i - smoothing_window // 2 : i + smoothing_window // 2 + 1]
        # Trend is the sign of median or mode
        trend = np.sign(np.median(window))
        
        # If isolated against trend, smooth
        if trend != 0 and action_sequence[i] == 0:
            smoothed[i] = int(trend)
    
    return smoothed
