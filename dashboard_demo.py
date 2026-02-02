#!/usr/bin/env python3
"""
STREAMLIT AUTOSCALING DEMO DASHBOARD
======================================
Interactive visualization of HYBRID autoscaling strategy.

Features:
1. Load timeline: actual vs forecast requests
2. Pods timeline: number of pods + scale events visualization
3. Threshold visualization: SLA/SLO thresholds with actual utilization
4. Cost vs SLA analysis: trade-off visualization
5. Real-time metrics summary

Data: Simulated for demo purposes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Tuple, List


# ============================================================================
# SIMULATION DATA GENERATOR
# ============================================================================

def generate_demo_data(num_steps: int = 288) -> pd.DataFrame:
    """
    Generate 1 day of simulated autoscaling data (15-min intervals).
    
    Args:
        num_steps: Number of timesteps (288 = 24 hours @ 15-min intervals)
    
    Returns:
        DataFrame with: time, requests, forecast, pods, cost, actions, violations
    """
    np.random.seed(42)
    
    # Time series
    start = datetime.now() - timedelta(hours=24)
    times = [start + timedelta(minutes=15*i) for i in range(num_steps)]
    
    # Baseline load with daily pattern + noise + spikes
    base_load = 1200
    hourly_pattern = 800 * np.sin(np.linspace(0, 2*np.pi, num_steps))  # Daily cycle
    noise = np.random.normal(0, 100, num_steps)
    
    # Add 2 spikes (simulating traffic surges)
    spikes = np.zeros(num_steps)
    spikes[100:110] = 1500  # Spike 1
    spikes[200:220] = 2000  # Spike 2 (larger)
    
    requests = np.maximum(base_load + hourly_pattern + noise + spikes, 100)
    
    # Forecast: slightly off from actual (realistic)
    forecast = requests + np.random.normal(0, 150, num_steps)
    forecast = np.maximum(forecast, 100)
    
    # Pod scaling logic (simplified HYBRID)
    pods = np.zeros(num_steps)
    pods[0] = 2  # Start with 2 pods (baseline)
    
    capacity_per_pod = 5000  # Requests per pod per 15-min interval
    
    for i in range(1, num_steps):
        current_pods = int(pods[i-1])
        current_capacity = current_pods * capacity_per_pod
        
        # Scaling decision
        forecast_capacity_needed = int(np.ceil(forecast[i] / capacity_per_pod))
        forecast_capacity_needed = max(2, min(20, forecast_capacity_needed))  # Min 2, max 20
        
        # Avoid flapping (cooldown effect)
        if i % 3 == 0:  # Only allow scaling every 3 steps (~45 min)
            pods[i] = 0.9 * pods[i-1] + 0.1 * forecast_capacity_needed
        else:
            pods[i] = pods[i-1]
    
    pods = np.round(pods).astype(int)
    pods = np.clip(pods, 2, 20)  # Enforce limits
    
    # Utilization
    utilization = requests / (pods * capacity_per_pod)
    
    # SLA/SLO violations
    sla_violated = utilization > 0.95
    slo_violated = utilization > 0.85
    
    # Cost calculation
    step_hours = 15 / 60  # 15 minutes in hours
    cost_reserved = pods * 0.03 * step_hours
    
    # Burst cost (70% spot, 30% on-demand)
    utilization_above_reserved = np.maximum(0, utilization - 0.4)  # 2 reserved pods
    cost_spot = pods * 0.015 * step_hours * utilization_above_reserved * 0.7
    cost_ondemand = pods * 0.05 * step_hours * utilization_above_reserved * 0.3
    
    total_cost = cost_reserved + cost_spot + cost_ondemand
    
    # Scaling actions
    actions = np.diff(pods, prepend=pods[0])
    actions = np.sign(actions)  # -1, 0, +1
    
    return pd.DataFrame({
        'time': times,
        'requests': requests.astype(int),
        'forecast': forecast.astype(int),
        'pods': pods,
        'utilization': utilization,
        'sla_violated': sla_violated,
        'slo_violated': slo_violated,
        'cost': total_cost,
        'cost_reserved': cost_reserved,
        'cost_spot': cost_spot,
        'cost_ondemand': cost_ondemand,
        'action': actions.astype(int),
    })


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_load_timeline(df: pd.DataFrame) -> go.Figure:
    """Plot actual vs forecast load."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['requests'],
        name='Actual Load',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['forecast'],
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Load Timeline: Actual vs Forecast Requests',
        xaxis_title='Time',
        yaxis_title='Requests (per 15-min)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_pods_and_events(df: pd.DataFrame) -> go.Figure:
    """Plot pod count and scale events."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Pod count
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['pods'],
            name='Pod Count',
            line=dict(color='#2ca02c', width=3),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    # Scale events as vertical lines
    scale_up = df[df['action'] > 0]
    scale_down = df[df['action'] < 0]
    
    if len(scale_up) > 0:
        fig.add_trace(
            go.Scatter(
                x=scale_up['time'], y=scale_up['pods'],
                mode='markers',
                marker=dict(size=12, color='green', symbol='triangle-up'),
                name='Scale Up',
                showlegend=True
            ),
            secondary_y=False
        )
    
    if len(scale_down) > 0:
        fig.add_trace(
            go.Scatter(
                x=scale_down['time'], y=scale_down['pods'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='triangle-down'),
                name='Scale Down',
                showlegend=True
            ),
            secondary_y=False
        )
    
    # Requests on secondary axis
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['requests'],
            name='Load',
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            opacity=0.3
        ),
        secondary_y=True
    )
    
    fig.update_yaxes(title_text='Pod Count', secondary_y=False)
    fig.update_yaxes(title_text='Request Load', secondary_y=True)
    fig.update_xaxes(title_text='Time')
    fig.update_layout(
        title='Pod Scaling Events & Load',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_thresholds(df: pd.DataFrame) -> go.Figure:
    """Plot utilization with SLA/SLO thresholds."""
    fig = go.Figure()
    
    # Threshold lines
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                  annotation_text="SLA (95%)", annotation_position="right")
    fig.add_hline(y=0.85, line_dash="dash", line_color="orange",
                  annotation_text="SLO (85%)", annotation_position="right")
    fig.add_hline(y=0.70, line_dash="dash", line_color="green",
                  annotation_text="Target (70%)", annotation_position="right")
    
    # Utilization
    colors = ['red' if x else ('orange' if y else 'blue') 
              for x, y in zip(df['sla_violated'], df['slo_violated'])]
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['utilization'],
        name='CPU Utilization',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,100,255,0.2)'
    ))
    
    # Highlight violations
    sla_violations = df[df['sla_violated']]
    if len(sla_violations) > 0:
        fig.add_trace(go.Scatter(
            x=sla_violations['time'], y=sla_violations['utilization'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='SLA Violation (>95%)'
        ))
    
    slo_violations = df[df['slo_violated'] & ~df['sla_violated']]
    if len(slo_violations) > 0:
        fig.add_trace(go.Scatter(
            x=slo_violations['time'], y=slo_violations['utilization'],
            mode='markers',
            marker=dict(size=8, color='orange'),
            name='SLO Violation (>85%)'
        ))
    
    fig.update_layout(
        title='CPU Utilization vs Thresholds',
        xaxis_title='Time',
        yaxis_title='Utilization %',
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_cost_sla(df: pd.DataFrame) -> go.Figure:
    """Plot cost vs SLA compliance."""
    # Cumulative metrics
    df_plot = df.copy()
    df_plot['cumulative_cost'] = df_plot['cost'].cumsum()
    df_plot['sla_compliance'] = 100 * (1 - df_plot['sla_violated'].expanding().mean())
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Cost
    fig.add_trace(
        go.Scatter(
            x=df_plot['time'], y=df_plot['cumulative_cost'],
            name='Cumulative Cost ($)',
            line=dict(color='#d62728', width=2)
        ),
        secondary_y=False
    )
    
    # SLA compliance
    fig.add_trace(
        go.Scatter(
            x=df_plot['time'], y=df_plot['sla_compliance'],
            name='SLA Compliance (%)',
            line=dict(color='#1f77b4', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_yaxes(title_text='Cost ($)', secondary_y=False)
    fig.update_yaxes(title_text='SLA Compliance (%)', secondary_y=True, range=[95, 101])
    fig.update_layout(
        title='Cost vs SLA Compliance Trade-off',
        xaxis_title='Time',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Autoscaling Demo", layout="wide")
    
    st.title("HYBRID Autoscaling Dashboard - Demo")
    st.markdown("""
    Interactive visualization of autoscaling with:
    - Load forecasting
    - Pod scaling decisions
    - SLA/SLO compliance
    - Cost analysis
    
    **Strategy**: HYBRID (4-layer decision hierarchy)
    """)
    
    # Generate data
    if 'data' not in st.session_state:
        st.session_state.data = generate_demo_data(288)  # 24 hours
    
    df = st.session_state.data
    
    # ======== Sidebar Controls ========
    st.sidebar.header("Demo Controls")
    
    # Time range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_idx = st.sidebar.slider("Start Time", 0, len(df)-1, 0, 24)
    with col2:
        end_idx = st.sidebar.slider("End Time", start_idx+1, len(df), len(df), 24)
    
    df_view = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Metrics summary
    st.sidebar.markdown("### Summary Metrics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Cost", f"${df_view['cost'].sum():.2f}")
        st.metric("Avg Pods", f"{df_view['pods'].mean():.1f}")
    with col2:
        st.metric("SLA Violations", int(df_view['sla_violated'].sum()))
        st.metric("Max Utilization", f"{df_view['utilization'].max():.1%}")
    
    # ======== Main Content ========
    st.markdown("## 1. Load Timeline")
    st.plotly_chart(plot_load_timeline(df_view), use_container_width=True)
    
    with st.expander("Load Timeline Explanation"):
        st.write("""
        **Blue line (Actual Load)**: Real incoming requests
        **Orange dashed line (Forecast)**: Predicted load from ML model
        
        When forecast is higher than actual, autoscaler provisions more pods (proactive).
        When forecast is lower, autoscaler scales down (cost optimization).
        """)
    
    st.markdown("## 2. Pod Scaling Events")
    st.plotly_chart(plot_pods_and_events(df_view), use_container_width=True)
    
    with st.expander("Scaling Logic Explanation"):
        st.write("""
        **Green line**: Current number of running pods
        **Green triangles (↑)**: Scale-up events
        **Red triangles (↓)**: Scale-down events
        **Gray background**: Load intensity
        
        HYBRID autoscaler uses 4 decision layers:
        1. **Anomaly Detection** (DDoS protection)
        2. **Emergency** (CPU > 95%)
        3. **Predictive** (forecast-based)
        4. **Reactive** (current utilization)
        """)
    
    st.markdown("## 3. Thresholds & Utilization")
    st.plotly_chart(plot_thresholds(df_view), use_container_width=True)
    
    with st.expander("Threshold Explanation"):
        st.write("""
        **SLA (95%)**: Service Level Agreement - hard limit
        - If CPU > 95%, SLA is violated and incident is triggered
        - Autoscaler should scale immediately
        
        **SLO (85%)**: Service Level Objective - target
        - Aim to keep CPU < 85% for optimal performance
        - Early warning threshold
        
        **Target (70%)**: Recommended operating point
        - Best balance between cost and performance
        - Allows room for sudden spikes
        """)
    
    st.markdown("## 4. Cost vs SLA Analysis")
    st.plotly_chart(plot_cost_sla(df_view), use_container_width=True)
    
    with st.expander("Cost-SLA Trade-off"):
        st.write("""
        **Red line (Cost)**: Cumulative cost in dollars
        **Blue line (Compliance)**: Percentage of time SLA was met
        
        Good autoscaling balances:
        - **Low cost** (fewer pods)
        - **High SLA compliance** (good performance)
        
        HYBRID strategy achieves 99%+ SLA compliance with minimal cost.
        """)
    
    # ======== Statistics Panel ========
    st.markdown("## Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Cost Breakdown")
        cost_data = {
            'Reserved': df_view['cost_reserved'].sum(),
            'Spot': df_view['cost_spot'].sum(),
            'On-Demand': df_view['cost_ondemand'].sum(),
        }
        for label, value in cost_data.items():
            st.write(f"**{label}**: ${value:.2f}")
    
    with col2:
        st.subheader("SLA/SLO Metrics")
        sla_violations = df_view['sla_violated'].sum()
        slo_violations = df_view['slo_violated'].sum()
        total = len(df_view)
        st.write(f"**SLA Violations**: {sla_violations}/{total} ({sla_violations/total*100:.1f}%)")
        st.write(f"**SLO Violations**: {slo_violations}/{total} ({slo_violations/total*100:.1f}%)")
        st.write(f"**SLA Compliance**: {100 - sla_violations/total*100:.1f}%")
    
    with col3:
        st.subheader("Scaling Activity")
        scale_ups = (df_view['action'] > 0).sum()
        scale_downs = (df_view['action'] < 0).sum()
        no_change = (df_view['action'] == 0).sum()
        st.write(f"**Scale-ups**: {scale_ups}")
        st.write(f"**Scale-downs**: {scale_downs}")
        st.write(f"**No change**: {no_change}")
    
    # ======== API Integration ========
    st.markdown("## Scaling Recommendation (FastAPI)")
    
    col_slider, col_btn = st.columns([3, 1])
    with col_slider:
        selected_idx = st.slider("Select timestep for recommendation", 0, len(df)-1, len(df)//2)
    with col_btn:
        fetch_button = st.button("Get Recommendation", use_container_width=True)
    
    if fetch_button:
        row = df.iloc[selected_idx]
        
        # Call FastAPI
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/recommend-scaling",
                json={
                    "current_pods": int(row['pods']),
                    "requests": int(row['requests']),
                    "forecast": int(row['forecast']),
                    "capacity_per_pod": 5000,
                },
                timeout=5
            )
            
            if response.status_code == 200:
                rec = response.json()
                
                # Display recommendation in nice format
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    st.metric("Current Pods", rec['current_pods'])
                
                with rec_col2:
                    st.metric("Recommended Pods", rec['recommended_pods'])
                
                with rec_col3:
                    action_text = rec['action'].upper().replace('-', ' ')
                    st.metric("Action", action_text, 
                             f"Confidence: {rec['confidence']:.0%}")
                
                st.divider()
                
                # Display decision layers
                st.write("**Decision Layers (HYBRID Autoscaler):**")
                for i, reason in enumerate(rec['reasons'], 1):
                    with st.expander(f"{i}. {reason['factor']}", expanded=(i==1)):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Current:** {reason['current_value']}")
                            st.write(f"**Threshold:** {reason['threshold']}")
                        with col2:
                            if "UP" in reason['decision']:
                                st.success(reason['decision'])
                            elif "DOWN" in reason['decision']:
                                st.info(reason['decision'])
                            elif "ALERT" in reason['decision']:
                                st.warning(reason['decision'])
                            else:
                                st.info(reason['decision'])
                
                st.divider()
                
                # Display explanation
                st.write("**Recommendation Explanation:**")
                st.info(rec['explanation'])
                
                # Display cost impact
                if rec['estimated_cost_impact']:
                    st.write("**Cost Impact Analysis:**")
                    cost_col1, cost_col2, cost_col3 = st.columns(3)
                    
                    with cost_col1:
                        st.metric("Current Cost/hr", 
                                 f"${rec['estimated_cost_impact']['current_hourly_cost']:.4f}")
                    
                    with cost_col2:
                        st.metric("New Cost/hr",
                                 f"${rec['estimated_cost_impact']['new_hourly_cost']:.4f}")
                    
                    with cost_col3:
                        change = rec['estimated_cost_impact']['cost_change_percent']
                        if change > 0:
                            st.metric("Cost Change", f"{change:+.1f}%", delta_color="inverse")
                        else:
                            st.metric("Cost Change", f"{change:+.1f}%")
            
            else:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server")
            st.info("**How to start the API:**\n\n```bash\npython api_server.py\n```")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # ======== Raw Data ========
    with st.expander("View Raw Data"):
        st.dataframe(df_view[['time', 'requests', 'forecast', 'pods', 'utilization', 
                              'sla_violated', 'cost', 'action']].head(20))


if __name__ == "__main__":
    main()
