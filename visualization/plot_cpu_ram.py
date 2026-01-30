import pandas as pd
import matplotlib.pyplot as plt
import os

DATA = "../metrics/logs/metrics.csv"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# chỉ lấy app server (bỏ nginx)
df = df[df["container"].str.contains("app")]

# CPU trung bình theo thời gian
cpu_avg = df.groupby("time")["cpu"].mean()
ram_avg = df.groupby("time")["memory"].mean()

# VẼ CPU
plt.figure()
cpu_avg.plot()
plt.xlabel("Time")
plt.ylabel("CPU (%)")
plt.title("Average CPU Usage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/cpu.png")
plt.close()

# VẼ RAM
plt.figure()
ram_avg.plot()
plt.xlabel("Time")
plt.ylabel("Memory (%)")
plt.title("Average RAM Usage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/ram.png")
plt.close()

print("✅ Saved cpu.png and ram.png")
