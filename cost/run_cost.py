import json

def compute_cost(history, step_seconds,
                 cpu_price_per_hour=0.05,
                 ram_price_per_gb_hour=0.01):
    total = 0.0
    step_hours = step_seconds / 3600

    for step in history:
        for s in step:
            cpu_cost = s["cpu"] * cpu_price_per_hour
            ram_cost = (s["ram"] / 1024) * ram_price_per_gb_hour
            total += (cpu_cost + ram_cost) * step_hours

    return total


if __name__ == "__main__":
    with open("/home/nghung/Learn/DeepLearning/server-cluster/metrics/server_history.json") as f:
        data = json.load(f)

    total = compute_cost(data["history"], data["step_seconds"])
    print(f"💰 Total cost: ${total:.6f}")
