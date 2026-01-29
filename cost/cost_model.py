def compute_cost(server_history, cost_per_server_per_hour, step_minutes):
    total = 0.0
    for n in server_history:
        total += n * cost_per_server_per_hour * (step_minutes / 60)
    return total
