import subprocess
import threading
import time
import matplotlib.pyplot as plt

running = True

cpu_history = {}   # {server_name: [cpu1, cpu2, ...]}
ram_history = {}   # {server_name: [ram1, ram2, ...]}


def collect_stats(service_name="app", interval=1):
    global running

    while running:
        result = subprocess.check_output(
            [
                "docker", "stats", "--no-stream",
                "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"
            ]
        ).decode().strip().splitlines()

        for line in result:
            name, cpu, mem = line.split(maxsplit=2)

            if service_name in name:
                cpu_val = float(cpu.replace("%", ""))

                mem_used = mem.split("/")[0].strip()
                if "MiB" in mem_used:
                    ram_val = float(mem_used.replace("MiB", ""))
                elif "GiB" in mem_used:
                    ram_val = float(mem_used.replace("GiB", "")) * 1024
                else:
                    ram_val = 0.0

                cpu_history.setdefault(name, []).append(cpu_val)
                ram_history.setdefault(name, []).append(ram_val)

        time.sleep(interval)


def run_wrk():
    subprocess.run([
        "wrk",
        "-t4",
        "-c200",
        "-d30s",
        "http://localhost:8080"
    ])


if __name__ == "__main__":
    t_stats = threading.Thread(target=collect_stats)
    t_wrk = threading.Thread(target=run_wrk)

    t_stats.start()
    t_wrk.start()

    t_wrk.join()
    running = False
    t_stats.join()

    # ========= VẼ BIỂU ĐỒ =========
    plt.figure(figsize=(12, 8))

    # ----- CPU -----
    plt.subplot(2, 1, 1)
    for server, values in cpu_history.items():
        plt.plot(values, label=server)
    plt.title("CPU usage of servers during load test")
    plt.ylabel("CPU (%)")
    plt.legend()
    plt.grid(True)

    # ----- RAM -----
    plt.subplot(2, 1, 2)
    for server, values in ram_history.items():
        plt.plot(values, label=server)
    plt.title("RAM usage of servers during load test")
    plt.xlabel("Load test progress")
    plt.ylabel("RAM (MB)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
