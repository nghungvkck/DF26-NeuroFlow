"""
AUTOSCALING PIPELINE DASHBOARD
=============================
Interactive visualization of autoscaling results using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Autoscaling Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Autoscaling Pipeline Dashboard")
st.markdown("""
**PHASE A:** Model Evaluation - Real data forecast accuracy
**PHASE B:** Autoscaling Tests - Synthetic scenario strategy testing
""")

# =====================================================================
# SIDEBAR - MODE SELECTION
# =====================================================================

st.sidebar.header("📊 Visualization Mode")
visualization_mode = st.sidebar.radio(
    "Choose what to visualize",
    ["Autoscaling Tests", "Model Evaluation"],
)

results_dir = Path("results")

if not results_dir.exists():
    st.error("⚠️ No results found. Run `python run_pipeline.py` first.")
    st.stop()

# =====================================================================
# MODE 1: AUTOSCALING TESTS
# =====================================================================

if visualization_mode == "Autoscaling Tests":
    csv_path = results_dir / "simulation_results.csv"
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("⚠️ Results file not found.")
        st.stop()
    
    scenarios = sorted(df['scenario'].unique())
    strategies = sorted(df['strategy'].unique())
    
    st.sidebar.header("🔧 Filters")
    selected_scenario = st.sidebar.selectbox("Select Scenario", scenarios)
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies", strategies, default=strategies
    )
    
    filtered_df = df[
        (df['scenario'] == selected_scenario) &
        (df['strategy'].isin(selected_strategies))
    ].copy()
    
    if filtered_df.empty:
        st.error("No data available for selected filters.")
        st.stop()
    
    st.markdown(f"**Scenario:** {selected_scenario}")
    
    # Tabs for Autoscaling Tests
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Load & Forecast",
        "📈 Pod Timeline",
        "💰 Cost Analysis",
        "🚨 SLA Violations",
        "📋 Metrics Comparison"
    ])
    
    # =====================================================================
    # TAB 1: Load vs Forecast
    # =====================================================================
    
    with tab1:
        st.subheader("Load Pattern vs Forecast Accuracy")
        
        first_strategy = selected_strategies[0]
        scenario_data = filtered_df[filtered_df['strategy'] == first_strategy].sort_values('time')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scenario_data['time'],
            y=scenario_data['actual_requests'],
            name='Actual Load',
            mode='lines',
            line=dict(color='blue', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=scenario_data['time'],
            y=scenario_data['forecast_requests'],
            name='Forecast',
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
        ))
        fig.update_layout(
            title=f"Load Pattern - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Request Rate (req/s)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mean_error = np.mean(np.abs(scenario_data['forecast_requests'] - scenario_data['actual_requests']))
            st.metric("Mean Forecast Error", f"{mean_error:.1f} req/s")
        with col2:
            max_error = np.max(np.abs(scenario_data['forecast_requests'] - scenario_data['actual_requests']))
            st.metric("Max Forecast Error", f"{max_error:.1f} req/s")
        with col3:
            rmse = np.sqrt(np.mean((scenario_data['forecast_requests'] - scenario_data['actual_requests'])**2))
            st.metric("RMSE", f"{rmse:.1f} req/s")
        with col4:
            anomalies = scenario_data['z_anomaly'].sum()
            st.metric("Anomalies Detected", int(anomalies))
    
    # =====================================================================
    # TAB 2: Pod Timeline
    # =====================================================================
    
    with tab2:
        st.subheader("Pod Count Over Time")
        
        fig = go.Figure()
        colors = {'REACTIVE': '#1f77b4', 'PREDICTIVE': '#ff7f0e',
                  'CPU_BASED': '#2ca02c', 'HYBRID': '#d62728'}
        
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=strategy_data['pods_after'],
                name=strategy,
                mode='lines+markers',
                line=dict(color=colors.get(strategy, 'gray'), width=2),
            ))
        
        fig.update_layout(
            title=f"Pod Timeline - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Number of Pods",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        for i, strategy in enumerate(selected_strategies[:3]):
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            events = (strategy_data['scaling_action'] != 0).sum()
            with [col1, col2, col3][i]:
                st.metric(f"{strategy} Scaling Events", int(events))
    
    # =====================================================================
    # TAB 3: Cost Analysis
    # =====================================================================
    
    with tab3:
        st.subheader("Cost Comparison")
        
        fig = go.Figure()
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            cost_per_step = strategy_data['pods_after'] * 0.05 * (5 / 60)
            cumulative_cost = cost_per_step.cumsum()
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=cumulative_cost,
                name=strategy,
                mode='lines',
                line=dict(color=colors.get(strategy, 'gray'), width=2),
            ))
        
        fig.update_layout(
            title=f"Cumulative Cost - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Total Cost ($)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cost Summary")
        cost_summary = []
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            total_cost = (strategy_data['pods_after'] * 0.05 * (5 / 60)).sum()
            avg_pods = strategy_data['pods_after'].mean()
            cost_summary.append({
                'Strategy': strategy,
                'Total Cost ($)': f"{total_cost:.2f}",
                'Avg Pods': f"{avg_pods:.1f}",
                'Cost/Pod': f"{total_cost / len(strategy_data):.4f}"
            })
        st.dataframe(pd.DataFrame(cost_summary), use_container_width=True)
    
    # =====================================================================
    # TAB 4: SLA Violations
    # =====================================================================
    
    with tab4:
        st.subheader("SLA Violations")
        
        fig = go.Figure()
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=strategy_data['sla_breached_before_scaling'].astype(int),
                name=strategy,
                mode='lines',
                fill='tozeroy',
                line=dict(color=colors.get(strategy, 'gray'), width=1),
                opacity=0.6,
            ))
        
        fig.update_layout(
            title=f"SLA Violation Timeline - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="SLA Breached (1=Yes)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("SLA Statistics")
        col1, col2, col3, col4 = st.columns(4)
        for i, strategy in enumerate(selected_strategies[:4]):
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            violations = strategy_data['sla_breached_before_scaling'].sum()
            violation_rate = violations / len(strategy_data) * 100
            with [col1, col2, col3, col4][i]:
                st.metric(f"{strategy}", f"{violations} ({violation_rate:.1f}%)")
    
    # =====================================================================
    # TAB 5: Metrics Comparison
    # =====================================================================
    
    with tab5:
        st.subheader("Comprehensive Metrics Comparison")
        
        metrics_data = []
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            
            total_cost = (strategy_data['pods_after'] * 0.05 * (5 / 60)).sum()
            avg_pods = strategy_data['pods_after'].mean()
            sla_violations = strategy_data['sla_breached_before_scaling'].sum()
            violation_rate = sla_violations / len(strategy_data) * 100
            scaling_events = (strategy_data['scaling_action'] != 0).sum()
            
            utilization = strategy_data['actual_requests'] / (strategy_data['pods_after'] * 500)
            mean_util = utilization.mean() * 100
            max_util = utilization.max() * 100
            
            metrics_data.append({
                'Strategy': strategy,
                'Total Cost': f"${total_cost:.2f}",
                'Avg Pods': f"{avg_pods:.1f}",
                'SLA Violations': f"{sla_violations}",
                'Violation Rate': f"{violation_rate:.1f}%",
                'Scaling Events': f"{scaling_events}",
                'Mean Utilization': f"{mean_util:.1f}%",
                'Max Utilization': f"{max_util:.1f}%",
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

# =====================================================================
# MODE 2: MODEL EVALUATION
# =====================================================================

else:
    model_eval_path = results_dir / "model_evaluation.json"
    
    try:
        with open(model_eval_path, 'r') as f:
            model_evaluation = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model evaluation results not found.")
        st.stop()
    
    st.markdown("**Analysis:** Model Forecast Accuracy on Real Historical Data")
    st.subheader("📈 Model Performance Metrics")
    
    if "best_model_per_timeframe" not in model_evaluation:
        st.error("Invalid model evaluation data format.")
        st.stop()
    
    best_per_tf = model_evaluation.get("best_model_per_timeframe", {})
    metrics_by_model = model_evaluation.get("metrics_by_model", {})
    
    # Display best models
    cols = st.columns(3)
    timeframes = ["1m", "5m", "15m"]
    for i, tf in enumerate(timeframes):
        with cols[i]:
            if tf in best_per_tf:
                best = best_per_tf[tf]
                st.metric(
                    f"Best Model ({tf})",
                    best["model"],
                    delta=f"MAPE: {best['mape']:.2%}"
                )
    
    st.markdown("---")
    
    st.subheader("📊 Detailed Metrics by Model")
    
    for model_name in sorted(metrics_by_model.keys()):
        with st.expander(f"**{model_name.upper()}** Model Performance"):
            model_metrics = metrics_by_model[model_name]
            
            rows = []
            for tf in ["1m", "5m", "15m"]:
                if tf in model_metrics:
                    m = model_metrics[tf]
                    rows.append({
                        "Timeframe": tf.upper(),
                        "MAE": f"{m.get('mae', 0):.2f}",
                        "RMSE": f"{m.get('rmse', 0):.2f}",
                        "MAPE": f"{m.get('mape', 0):.2%}",
                        "Files": int(m.get('files', 0))
                    })
            
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

