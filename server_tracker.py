import subprocess
import time

def get_running_servers(service_name="app"):
    result = subprocess.check_output(
        ["docker", "ps", "--filter", f"name={service_name}", "--format", "{{.ID}}"]
    )
    lines = result.decode().strip().splitlines()
    return len(lines)

def collect_server_history(duration_seconds, step_seconds):
    history = []
    steps = int(duration_seconds / step_seconds)

    for _ in range(steps):
        n = get_running_servers("app")
        history.append(n)
        time.sleep(step_seconds)

    return history
