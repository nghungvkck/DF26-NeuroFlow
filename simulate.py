import pandas as pd
import numpy as np
from data.load_data import load_csv
from anomaly.simulate_anomaly import inject_ddos
from anomaly.anomaly_detection import zscore_detection
from forecast.arima_forecaster import ARIMAForecaster
from autoscaling.reactive import ReactiveAutoscaler
from autoscaling.predictive import PredictiveAutoscaler
from cost.cost_model import compute_cost
from cost.metrics import sla_violation_rate

# =====================
# Load base traffic
# =====================
df = load_csv("data/train_5m_autoscaling.csv")
ts = df["requests_count"]  # requests/min

# =====================
# Inject DDoS
# =====================
ts_ddos, anomaly = inject_ddos(
    ts.values,
    start=800,
    duration=30,
    intensity=6
)

# =====================
# Forecast
# =====================
forecaster = ARIMAForecaster()
forecaster.fit(pd.Series(ts_ddos[:700]))

# =====================
# Autoscalers
# =====================
reactive = ReactiveAutoscaler(capacity_per_server=500)
predictive = PredictiveAutoscaler(capacity_per_server=500)

N_r, N_p = 5, 5
records = []

for t in range(700, len(ts_ddos)):
    forecast = forecaster.predict(1).iloc[0]

    N_r, _, _ = reactive.step(N_r, ts_ddos[t])
    N_p, action = predictive.step(
    current_servers=N_p,
    forecast_requests=forecast,
    current_requests=ts[t]
)


    records.append({
        "time": t,
        "requests": ts_ddos[t],
        "forecast": forecast,
        "servers_reactive": N_r,
        "servers_predictive": N_p,
        "anomaly": anomaly[t],
        "sla_reactive": ts_ddos[t] > N_r*500,
        "sla_predictive": ts_ddos[t] > N_p*500
    })

df = pd.DataFrame(records)

# =====================
# Cost
# =====================
df["reactive_cost"] = (
    df["servers_reactive"].cumsum() * 0.05 / 60
)
df["predictive_cost"] = (
    df["servers_predictive"].cumsum() * 0.05 / 60
)

df.to_csv("simulation_result.csv", index=False)
