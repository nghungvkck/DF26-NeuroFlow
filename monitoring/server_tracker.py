import subprocess
import time
import json
import os

FLAG_FILE = "/home/nghung/Learn/DeepLearning/server-cluster/metrics/load.flag"

def wait_for_load_start():
    print("⏳ Waiting for load test to start...")
    while True:
        if os.path.exists(FLAG_FILE):
            with open(FLAG_FILE) as f:
                if f.read().strip() == "START":
                    print("🚀 Load test detected. Start tracking.")
                    return
        time.sleep(1)


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


def collect_server_history(step_seconds=5):
    history = []

    while True:
        if os.path.exists(FLAG_FILE):
            with open(FLAG_FILE) as f:
                if f.read().strip() == "END":
                    print("⏹️ Load test ended. Stop tracking.")
                    break

        snapshot = get_server_stats()
        history.append(snapshot)
        time.sleep(step_seconds)

    return {
        "step_seconds": step_seconds,
        "history": history
    }


if __name__ == "__main__":
    wait_for_load_start()

    data = collect_server_history(step_seconds=5)

    with open("/home/nghung/Learn/DeepLearning/server-cluster/metrics/server_history.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Saved server_history.json (synced with load test)")
