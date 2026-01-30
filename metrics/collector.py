import subprocess
import time
from datetime import datetime

while True:
    stats = subprocess.check_output(
        ["docker", "stats", "--no-stream"]
    ).decode()

    with open("logs/metrics.log", "a") as f:
        f.write(f"\n[{datetime.now()}]\n")
        f.write(stats)

    time.sleep(10)
