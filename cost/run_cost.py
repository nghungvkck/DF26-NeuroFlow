from metrics.server_tracker import collect_server_history
from cost.cost import compute_cost
from cost.cost_config import COST_PER_SERVER_PER_HOUR

DURATION = 60        # 1 phút test
STEP_SECONDS = 5     # lấy mẫu mỗi 5 giây

server_history = collect_server_history(DURATION, STEP_SECONDS)

total_cost = compute_cost(
    server_history,
    COST_PER_SERVER_PER_HOUR,
    step_minutes=STEP_SECONDS / 60
)

print("Server history:", server_history)
print("Total cost: $", round(total_cost, 5))
