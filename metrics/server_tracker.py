import subprocess
import time
import json

def get_server_stats(service_keyword="server-cluster-app"):
    cmd = [
        "docker", "stats",
        "--no-stream",
        "--format",
        "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"
    ]

    output = subprocess.check_output(cmd).decode().strip().splitlines()
    servers = []

    for line in output:
        name, cpu, mem = line.split(maxsplit=2)

        # ✅ chỉ lấy app server, bỏ nginx
        if service_keyword not in name:
            continue

        cpu_val = float(cpu.replace("%", "")) / 100

        mem_used = mem.split("/")[0].strip()
        if "MiB" in mem_used:
            ram_mb = float(mem_used.replace("MiB", ""))
        elif "GiB" in mem_used:
            ram_mb = float(mem_used.replace("GiB", "")) * 1024
        else:
            ram_mb = 0.0

        servers.append({
            "server_name": name,
            "cpu": cpu_val,
            "ram": ram_mb
        })

    return servers


def collect_server_history(duration=60, step_seconds=5):
    history = []
    steps = duration // step_seconds

    for _ in range(steps):
        snapshot = get_server_stats()
        history.append(snapshot)
        time.sleep(step_seconds)

    return {
        "step_seconds": step_seconds,
        "history": history
    }


if __name__ == "__main__":
    data = collect_server_history()

    with open("metrics/server_history.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Saved server_history.json (CPU + RAM per app server)")
