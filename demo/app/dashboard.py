from __future__ import annotations

import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

demo_dir = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(demo_dir)
sys.path.insert(0, demo_dir)
sys.path.insert(0, project_root)

from utils.forecast import forecast_next
from utils.load_data import load_traffic_data
from app.forecast_tab_simple import render_forecast_tab
from app.api_demo_tab import render_api_demo_tab
from app.optimization_tab import render_optimization_tab



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

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "forecasting", "models")
predictions_dir = os.path.join(os.path.dirname(__file__), "..", "..", "forecasting", "artifacts", "predictions")


data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
# Default to test_5m_autoscaling.csv
selected_file = "test_5m_autoscaling.csv"

df = load_traffic_data(selected_file, data_dir=data_dir)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Forecast", "Optimization", "API Demo"])

with tab1:
    st.subheader("Overview")
    
    # Dataset selector
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    selected_file = st.selectbox("Select Dataset", data_files, key="tab1_dataset_selector")
    
    # Load selected dataset
    df_tab1 = load_traffic_data(selected_file, data_dir=data_dir)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_load = float(df_tab1["requests_count"].mean())
        st.metric("Avg Load", f"{avg_load:.0f}", delta="requests/5m")
    with col2:
        peak_load = float(df_tab1["requests_count"].max())
        st.metric("Peak Load", f"{peak_load:.0f}", delta="max")
    with col3:
        burst_count = int(df_tab1["is_burst"].sum()) if "is_burst" in df_tab1.columns else 0
        burst_pct = (burst_count / len(df_tab1) * 100) if len(df_tab1) > 0 else 0
        st.metric("Burst Time", f"{burst_pct:.1f}%", delta="of time")
    with col4:
        event_count = int(df_tab1["is_event"].sum()) if "is_event" in df_tab1.columns else 0
        st.metric("Events", f"{event_count}", delta="detected")
    
    st.subheader("Requests Over Time")
    burst_df = df_tab1[df_tab1["is_burst"] == 1] if "is_burst" in df_tab1.columns else pd.DataFrame()
    event_df = df_tab1[df_tab1["is_event"] == 1] if "is_event" in df_tab1.columns else pd.DataFrame()
    
    line = (
        alt.Chart(df_tab1)
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
    
    st.caption("Red dots = Burst periods | Orange X = Special events")
    
    # === TRENDS & CORRELATION ===
    st.divider()
    
    col_trends, col_corr = st.columns(2)
    
    # Trend Analysis
    with col_trends:
        st.subheader("Trend Analysis")
        
        # Calculate rolling average to show trend
        df_trend = df_tab1.copy()
        df_trend["rolling_mean_12"] = df_trend["requests_count"].rolling(window=12).mean()
        df_trend["rolling_mean_48"] = df_trend["requests_count"].rolling(window=48).mean()
        
        trend_chart = alt.Chart(df_trend).mark_line().encode(
            x="timestamp:T",
            y="value:Q",
            color="metric:N"
        ).transform_fold(
            ["requests_count", "rolling_mean_12", "rolling_mean_48"],
            as_=["metric", "value"]
        ).transform_filter(
            alt.datum.value != None
        ).properties(height=300)
        
        st.altair_chart(trend_chart, use_container_width=True)
        st.caption("Original | 12-period MA | 48-period MA")
    
    # Correlation Matrix
    with col_corr:
        st.subheader("Parameter Correlations")
        
        # Select numeric columns, exclude derived features
        numeric_cols = df_tab1.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                        if col != 'is_event'
                        and not col.startswith('rolling_')
                        and not col.startswith('lag_')
                        and not col.startswith('log_')]
        
        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df_tab1[numeric_cols].corr()
            
            # Create heatmap using Plotly
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>'
            ))
            
            fig_corr.update_layout(
                title="Correlation Heatmap",
                height=400,
                xaxis_title="",
                yaxis_title="",
                font=dict(size=10),
                margin=dict(l=100, r=50, t=50, b=100)
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Top Correlations with Load
    st.subheader("Correlations with Load")
    numeric_cols = df_tab1.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'is_event']
    
    if 'requests_count' in numeric_cols and len(numeric_cols) > 1:
        corr_matrix = df_tab1[numeric_cols].corr()
        correlations = corr_matrix['requests_count'].sort_values(ascending=False)
        
        # Filter out rolling/lag/log features to show real parameters only
        real_params = [col for col in correlations.index 
                       if not col.startswith('rolling_') 
                       and not col.startswith('lag_')
                       and not col.startswith('log_')
                       and col != 'requests_count']
        
        col_corr1, col_corr2, col_corr3 = st.columns(3)
        
        # Top positive correlation (real parameters only)
        if len(real_params) > 0:
            top_pos_name = real_params[0]
            top_pos_val = float(correlations[top_pos_name])
            
            with col_corr1:
                st.metric(
                    "Strongest Positive",
                    f"{top_pos_name}",
                    delta=f"{top_pos_val:.3f}"
                )
        
        # Top negative correlation
        negative_corrs = correlations[correlations < 0]
        negative_real = [col for col in negative_corrs.index 
                         if not col.startswith('rolling_') 
                         and not col.startswith('lag_')
                         and not col.startswith('log_')]
        
        if len(negative_real) > 0:
            top_neg_idx = negative_real[0]
            top_neg_val = float(correlations[top_neg_idx])
            
            with col_corr2:
                st.metric(
                    "Strongest Negative",
                    f"{top_neg_idx}",
                    delta=f"{top_neg_val:.3f}"
                )
        
        with col_corr3:
            st.metric(
                "Total Parameters",
                f"{len(numeric_cols)}",
                help="Number of numeric parameters"
            )
    
    st.divider()
    st.subheader("Temporal Feature Analysis")
    
    sample_size = 50
    t = np.arange(sample_size)
    original = 100 + 20*np.sin(t/10) + np.random.normal(0, 5, sample_size)
    lagged_1 = np.concatenate([[0], original[:-1]])
    
    viz_df = pd.DataFrame({
        'Time': t,
        'Current Load': original,
        'Previous Period': lagged_1
    })
    
    viz_chart = alt.Chart(viz_df).mark_line().encode(
        x='Time:Q',
        y=alt.Y('value:Q', title='Requests'),
        color='metric:N'
    ).transform_fold(
        ['Current Load', 'Previous Period'],
        as_=['metric', 'value']
    ).properties(height=300)
    
    st.altair_chart(viz_chart, use_container_width=True)

with tab2:
    render_forecast_tab(df, forecast_next, model_dir)

with tab3:
    render_optimization_tab(data_dir, predictions_dir)

with tab4:
    st.write("API Demo is initializing...")
    try:
        render_api_demo_tab()
    except Exception as e:
        st.error("API Demo failed to render.")
        st.exception(e)