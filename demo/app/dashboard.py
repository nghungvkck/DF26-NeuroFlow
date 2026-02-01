from __future__ import annotations

import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from utils.forecast import forecast_next, discover_models, load_model_metrics
from utils.load_data import load_traffic_data
from utils.scaling import decide_scaling
from forecast_tab_simple import render_forecast_tab
from api_demo_tab import render_api_demo_tab



st.set_page_config(page_title="DataFlow Autoscaling Demo", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
    }
    .action-scale-up {
        background-color: #FF4B4B;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .action-scale-down {
        background-color: #FFA500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .action-hold {
        background-color: #00CC00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("DataFlow Autoscaling Dashboard")

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")


def _calculate_hysteresis_action(predictions: np.ndarray, upper_threshold: float, lower_threshold: float, window: int = 3) -> tuple[str, str]:
    """
    Calculate scaling action with hysteresis to avoid flapping.
    Only recommend scale if threshold is breached consistently.
    """
    if len(predictions) < window:
        window = len(predictions)
    
    recent_predictions = predictions[:window]
    
    # Count how many predictions exceed thresholds
    exceed_upper = np.sum(recent_predictions > upper_threshold)
    exceed_lower = np.sum(recent_predictions < lower_threshold)
    
    # Require majority of window to exceed threshold
    threshold_count = int(np.ceil(window * 0.6))  # 60% of window
    
    if exceed_upper >= threshold_count:
        reason = f"🔺 {exceed_upper}/{window} predictions exceed upper threshold ({upper_threshold:.0f})"
        return "scale_up", reason
    elif exceed_lower >= threshold_count:
        reason = f"🔻 {exceed_lower}/{window} predictions below lower threshold ({lower_threshold:.0f})"
        return "scale_down", reason
    else:
        reason = f"✅ Load within acceptable range ({lower_threshold:.0f} - {upper_threshold:.0f})"
        return "hold", reason


data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
selected_file = st.selectbox("Dataset", data_files, key="dataset_selector")

df = load_traffic_data(selected_file, data_dir=data_dir)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast", "Autoscaling", "Cost Analysis", "API Demo"])

with tab1:
    st.subheader("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_load = float(df["requests_count"].mean())
        st.metric("Avg Load", f"{avg_load:.0f}", delta="requests/5m")
    with col2:
        peak_load = float(df["requests_count"].max())
        st.metric("Peak Load", f"{peak_load:.0f}", delta="max")
    with col3:
        burst_count = int(df["is_burst"].sum()) if "is_burst" in df.columns else 0
        burst_pct = (burst_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Burst Time", f"{burst_pct:.1f}%", delta="of time")
    with col4:
        event_count = int(df["is_event"].sum()) if "is_event" in df.columns else 0
        st.metric("Events", f"{event_count}", delta="detected")
    
    st.subheader("Requests Over Time")
    burst_df = df[df["is_burst"] == 1] if "is_burst" in df.columns else pd.DataFrame()
    event_df = df[df["is_event"] == 1] if "is_event" in df.columns else pd.DataFrame()
    
    line = (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(x="timestamp:T", y="requests_count:Q", tooltip=["timestamp:T", "requests_count:Q"])
    )
    
    bursts = (
        alt.Chart(burst_df)
        .mark_point(color="red", size=80)
        .encode(x="timestamp:T", y="requests_count:Q", tooltip=["timestamp:T", "requests_count:Q"])
    )
    
    events = (
        alt.Chart(event_df)
        .mark_point(color="orange", size=80, shape="cross")
        .encode(x="timestamp:T", y="requests_count:Q", tooltip=["timestamp:T", "requests_count:Q"])
    )
    
    chart = line + bursts + events
    st.altair_chart(chart, use_container_width=True)
    
    st.caption("🔴 Red dots = Burst periods | 🟠 Orange X = Special events")

with tab2:
    # Simple forecast tab with Actual vs Predicted
    render_forecast_tab(df, forecast_next, model_dir)

with tab3:
    st.subheader("Autoscaling Decision")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type_scaling = st.selectbox("Model", ["hybrid", "xgboost", "lightgbm"], key="model_scaling")
    with col2:
        timeframe_scaling = st.selectbox("Timeframe", ["1m", "5m", "15m"], key="timeframe_scaling")
    with col3:
        forecast_horizon_scaling = st.slider("Horizon", min_value=1, max_value=60, value=12, key="horizon_scaling")
    
    forecast_df_scaling, _ = forecast_next(df, forecast_horizon_scaling, model_type=model_type_scaling, timeframe=timeframe_scaling, model_dir=model_dir)
    
    current_requests = float(df["requests_count"].iloc[-1])
    predicted_requests = float(forecast_df_scaling["yhat"].iloc[-1]) if len(forecast_df_scaling) > 0 else current_requests
    
    upper_default = float(df["requests_count"].quantile(0.85))
    lower_default = float(df["requests_count"].quantile(0.15))
    
    col1, col2 = st.columns(2)
    with col1:
        upper_threshold = st.number_input("Upper threshold", value=upper_default, min_value=0.0)
    with col2:
        lower_threshold = st.number_input("Lower threshold", value=lower_default, min_value=0.0)
    
    action, reason = decide_scaling(
        current_requests=current_requests,
        predicted_requests=predicted_requests,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if action == "scale_up":
            st.success(f"🔺 {action.upper()}")
        elif action == "scale_down":
            st.info(f"🔻 {action.upper()}")
        else:
            st.warning(f"⏸️ {action.upper()}")
    
    with col2:
        st.write(f"**Predicted load:** {predicted_requests:.0f} requests")
        st.caption(reason)

with tab4:
    st.subheader("Cost Analysis")
    
    hourly_cost = st.number_input("Cost per server/hour ($)", value=0.50, min_value=0.01, step=0.01)
    
    df_copy = df.copy()
    df_copy["reactive_servers"] = (df_copy["requests_count"] / 1000).apply(np.ceil).astype(int)
    df_copy["reactive_servers"] = df_copy["reactive_servers"].clip(lower=1)
    
    forecast_df_full, _ = forecast_next(df_copy, len(df_copy), model_type="hybrid", timeframe="5m", model_dir=model_dir)
    df_copy["predictive_servers"] = (forecast_df_full["yhat"].iloc[:len(df_copy)] / 1000).apply(np.ceil).astype(int)
    df_copy["predictive_servers"] = df_copy["predictive_servers"].clip(lower=1)
    
    reactive_cost_per_5m = (df_copy["reactive_servers"].mean() * hourly_cost) / 12
    predictive_cost_per_5m = (df_copy["predictive_servers"].mean() * hourly_cost) / 12
    
    total_reactive = reactive_cost_per_5m * len(df_copy)
    total_predictive = predictive_cost_per_5m * len(df_copy)
    savings = total_reactive - total_predictive
    savings_pct = (savings / total_reactive * 100) if total_reactive > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Reactive Cost", f"${total_reactive:.2f}")
    with col2:
        st.metric("Predictive Cost", f"${total_predictive:.2f}")
    with col3:
        st.metric("Savings", f"${savings:.2f}")
    with col4:
        st.metric("Efficiency Gain", f"{savings_pct:.1f}%")
    
    st.subheader("Server Allocation Over Time")
    cost_df = pd.DataFrame({
        "timestamp": df_copy["timestamp"],
        "Reactive": df_copy["reactive_servers"],
        "Predictive": df_copy["predictive_servers"]
    })
    
    cost_chart = alt.Chart(cost_df).mark_line().encode(
        x="timestamp:T",
        y=alt.Y("value:Q", title="Servers"),
        color="variable:N"
    ).transform_fold(
        ["Reactive", "Predictive"],
        as_=["variable", "value"]
    )
    st.altair_chart(cost_chart, use_container_width=True)

with tab5:
    render_api_demo_tab()