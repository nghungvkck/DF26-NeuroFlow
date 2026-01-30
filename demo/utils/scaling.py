from __future__ import annotations


_cooldown_remaining = 0


def decide_scaling(
    current_requests: float,
    predicted_requests: float,
    upper_threshold: float,
    lower_threshold: float,
    cooldown_steps: int = 3,
) -> tuple[str, str]:
    # Decide scaling action

    if cooldown_steps < 0:
        raise ValueError("cooldown_steps must be >= 0")

    if _cooldown_remaining > 0:
        _cooldown_remaining -= 1
        return "hold", f"Cooldown active ({_cooldown_remaining} steps remaining)"

    if predicted_requests > upper_threshold:
        _cooldown_remaining = cooldown_steps
        return "scale_up", "Predicted load above upper threshold"
    if predicted_requests < lower_threshold:
        _cooldown_remaining = cooldown_steps
        return "scale_down", "Predicted load below lower threshold"

    return "hold", "Predicted load within thresholds"