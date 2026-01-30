import subprocess
import time
import csv
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)

CSV_FILE = "logs/metrics.csv"

# ghi header nếu chưa tồn tại
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "container", "cpu", "memory"])

while True:
    output = subprocess.check_output(
        ["docker", "stats", "--no-stream",
         "--format", "{{.Name}},{{.CPUPerc}},{{.MemPerc}}"]
    ).decode().splitlines()

    now = datetime.now().strftime("%H:%M:%S")

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for line in output:
            name, cpu, mem = line.split(",")
            writer.writerow([
                now,
                name,
                float(cpu.replace("%", "")),
                float(mem.replace("%", ""))
            ])

    time.sleep(5)
