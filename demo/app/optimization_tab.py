from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimization import CloudCostModel, HybridAutoscaler, MetricsCollector
from optimization.reactive_scaler import ReactiveOnlyScaler
from utils.load_data import load_traffic_data


def _infer_timeframe_from_name(file_name: str) -> Optional[str]:
    lowered = file_name.lower()
    if "_1m" in lowered:
        return "1m"
    if "_5m" in lowered:
        return "5m"
    if "_15m" in lowered:
        return "15m"
    return None


def _infer_step_minutes(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return 5.0
    deltas = df["timestamp"].diff().dropna()
    if deltas.empty:
        return 5.0
    return float(deltas.median().total_seconds() / 60.0)


def _load_predictions(
    pred_dir: str,
    model_type: str,
    timeframe: str,
    df: pd.DataFrame,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    pred_path = os.path.join(pred_dir, f"{model_type}_{timeframe}_predictions.csv")
    if not os.path.exists(pred_path):
        return None, None

    pred_df = pd.read_csv(pred_path)
    if "split" in pred_df.columns:
        pred_df = pred_df[pred_df["split"] == "test"]

    preferred_cols = ["y_pred", "hybrid_predicted", "predicted", "yhat"]
    pred_col = next((col for col in preferred_cols if col in pred_df.columns), None)
    if pred_col is None:
        return None, None

    if "timestamp" in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    if "timestamp" in pred_df.columns and "y_true" in pred_df.columns:
        override_df = pred_df[["timestamp", "y_true"]].rename(columns={"y_true": "requests_count"})
        override_df = override_df.sort_values("timestamp").reset_index(drop=True)
        return pred_df[pred_col].astype(float).values, override_df

    aligned_df = None
    if "timestamp" in pred_df.columns and "timestamp" in df.columns:
        aligned_df = df.merge(pred_df[["timestamp", pred_col]], on="timestamp", how="inner")
        if not aligned_df.empty:
            return aligned_df[pred_col].astype(float).values, aligned_df

    return pred_df[pred_col].astype(float).values, None


def render_optimization_tab(data_dir: str, predictions_dir: str) -> None:
    st.subheader("Optimization")

    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv") and f.lower().startswith("test_")]
    )
    if not data_files:
        st.warning("No datasets found.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_file = st.selectbox("Dataset", data_files, key="opt_dataset")
    inferred_timeframe = _infer_timeframe_from_name(selected_file) or "5m"
    with col2:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=["1m", "5m", "15m"].index(inferred_timeframe), key="opt_timeframe")
    with col3:
        model_type = st.selectbox("Forecast Model", ["hybrid", "xgboost", "lightgbm"], key="opt_model")
    with col4:
        use_forecast = st.checkbox("Use Forecast", value=True, key="opt_use_forecast")

    df = load_traffic_data(selected_file, data_dir=data_dir)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        capacity_per_server = st.number_input("Capacity per server", value=500, min_value=10, step=10)
    with col2:
        min_servers = st.number_input("Min servers", value=2, min_value=1, step=1)
    with col3:
        max_servers = st.number_input("Max servers", value=20, min_value=2, step=1)
    with col4:
        enable_anomaly = st.checkbox("Anomaly layer", value=True)

    col1, col2 = st.columns(2)
    with col1:
        fixed_servers_input = st.number_input("Fixed servers", value=int(min_servers), min_value=1, step=1)
    with col2:
        st.caption("Fixed servers is used for cost comparison only.")

    run_opt = st.button("Run Optimization", use_container_width=True)

    if not run_opt:
        st.info("Click 'Run Optimization' to simulate autoscaling.")
        return

    predictions = None
    if use_forecast:
        predictions, aligned_df = _load_predictions(predictions_dir, model_type, timeframe, df)
        if aligned_df is not None and not aligned_df.empty:
            df = aligned_df
        if predictions is None:
            st.warning("No forecast predictions found. Running without forecast layer.")

    step_minutes = _infer_step_minutes(df)

    requests = df["requests_count"].astype(float).values
    if predictions is not None:
        min_len = min(len(requests), len(predictions))
        requests = requests[:min_len]
        predictions = predictions[:min_len]

    autoscaler = HybridAutoscaler(
        capacity_per_server=capacity_per_server,
        min_servers=int(min_servers),
        max_servers=int(max_servers),
        enable_anomaly=enable_anomaly,
    )
    cost_model = CloudCostModel(reserved_capacity=int(min_servers))
    metrics_collector = MetricsCollector(capacity_per_server=int(capacity_per_server), step_minutes=step_minutes)

    reactive_scaler = ReactiveOnlyScaler(
        capacity_per_server=float(capacity_per_server),
        min_servers=int(min_servers),
        max_servers=int(max_servers),
        upper_threshold=0.75,
        lower_threshold=0.35,
        consecutive_steps=2,
        cooldown_steps=2,
        initial_servers=int(min_servers),
    )
    fixed_servers = int(fixed_servers_input)

    results = []
    current_servers = int(min_servers)

    reactive_servers = []
    fixed_servers_series = []
    reactive_cost_steps = []
    fixed_cost_steps = []

    for i, req in enumerate(requests):
        forecast_req = float(predictions[i]) if predictions is not None else None
        reactive_count = reactive_scaler.step(float(req))
        reactive_step_cost, _ = cost_model.compute_step_cost(reactive_count, step_minutes / 60.0)
        fixed_step_cost, _ = cost_model.compute_step_cost(fixed_servers, step_minutes / 60.0)
        new_servers, action, metrics = autoscaler.step(
            current_servers=current_servers,
            requests=float(req),
            forecast_requests=forecast_req,
        )
        step_cost, breakdown = cost_model.compute_step_cost(new_servers, step_minutes / 60.0)

        reactive_servers.append(reactive_count)
        fixed_servers_series.append(fixed_servers)
        reactive_cost_steps.append(reactive_step_cost)
        fixed_cost_steps.append(fixed_step_cost)
        
        scaling_action = 0
        if new_servers > current_servers:
            scaling_action = 1
        elif new_servers < current_servers:
            scaling_action = -1
        
        metrics_collector.record(
            t=i,
            servers=new_servers,
            requests=float(req),
            cost=step_cost,
            cost_breakdown=breakdown,
            scaling_action=scaling_action,
        )
        
        results.append(
            {
                "time": i,
                "requests": float(req),
                "forecast": float(forecast_req) if forecast_req is not None else 0.0,
                "fixed_servers": fixed_servers,
                "reactive_servers": reactive_count,
                "fixed_cost": fixed_step_cost,
                "reactive_cost": reactive_step_cost,
                "servers_before": current_servers,
                "servers_after": new_servers,
                "action": action,
                "cpu": metrics["cpu"],
                "cost": step_cost,
                "cost_reserved": breakdown.get("reserved", 0.0),
                "cost_spot": breakdown.get("spot", 0.0),
                "cost_ondemand": breakdown.get("on_demand", 0.0),
                "scale_direction": metrics.get("scale_direction", "NONE"),
            }
        )
        current_servers = new_servers

    results_df = pd.DataFrame(results)
    aggregate_metrics = metrics_collector.compute_aggregate_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost", f"${aggregate_metrics['total_cost']:.2f}")
    with col2:
        st.metric("Scaling Events", f"{aggregate_metrics['scaling_events']}")
    with col3:
        st.metric("Avg Servers", f"{aggregate_metrics['avg_servers']:.1f}")
    with col4:
        st.metric("SLA Violations", f"{aggregate_metrics['sla_violations']}")

    st.subheader("Load & Servers")
    
    fig_requests = go.Figure()
    fig_requests.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["requests"],
            name="Requests",
            line=dict(color="#00D9FF", width=2),
        )
    )
    if results_df["forecast"].sum() > 0:
        fig_requests.add_trace(
            go.Scatter(
                x=results_df["time"],
                y=results_df["forecast"],
                name="Forecast",
                line=dict(color="#FFD700", width=2, dash="dash"),
            )
        )
    
    fig_requests.update_layout(
        height=350,
        xaxis_title="Step",
        yaxis_title="Requests",
        hovermode="x unified",
        template="plotly_dark",
        showlegend=True,
    )
    st.plotly_chart(fig_requests, use_container_width=True)
    
    st.subheader("Servers Over Time")
    fig_servers = go.Figure()
    fig_servers.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["servers_after"],
            name="Hybrid Predictive",
            line=dict(color="#FF6B6B", width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)',
        )
    )
    fig_servers.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["reactive_servers"],
            name="Reactive",
            line=dict(color="#FFA500", width=2, dash="dot"),
        )
    )
    fig_servers.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["fixed_servers"],
            name="Fixed",
            line=dict(color="#A0A0A0", width=2, dash="dash"),
        )
    )
    fig_servers.update_layout(
        height=300,
        xaxis_title="Step",
        yaxis_title="Server Count",
        hovermode="x unified",
        template="plotly_dark",
        showlegend=True,
    )
    st.plotly_chart(fig_servers, use_container_width=True)

    st.subheader("Cost Over Time")
    results_df["cost_cumulative"] = results_df["cost"].cumsum()
    results_df["fixed_cost_cumulative"] = results_df["fixed_cost"].cumsum()
    results_df["reactive_cost_cumulative"] = results_df["reactive_cost"].cumsum()
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["cost_cumulative"],
            name="Hybrid Predictive",
            line=dict(color="#7DFFB3", width=2),
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["reactive_cost_cumulative"],
            name="Reactive",
            line=dict(color="#FFA500", width=2, dash="dot"),
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=results_df["time"],
            y=results_df["fixed_cost_cumulative"],
            name="Fixed",
            line=dict(color="#A0A0A0", width=2, dash="dash"),
        )
    )
    fig_cost.update_layout(
        height=320,
        xaxis_title="Step",
        yaxis_title="Cost ($)",
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    with st.expander("📊 Detailed Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cost Reserved", f"${aggregate_metrics['cost_reserved']:.2f}")
            st.metric("Cost Spot", f"${aggregate_metrics['cost_spot']:.2f}")
            st.metric("Cost On-Demand", f"${aggregate_metrics['cost_ondemand']:.2f}")
        with col2:
            st.metric("Scale Ups", f"{aggregate_metrics['scale_ups']}")
            st.metric("Scale Downs", f"{aggregate_metrics['scale_downs']}")
        with col3:
            st.metric("Avg CPU", f"{aggregate_metrics['avg_cpu']:.1f}%")
        
        st.divider()
        
