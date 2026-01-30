"""
Enhanced Forecast Tab with Plotly visualization, hysteresis logic, and autoscaling recommendations
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def calculate_hysteresis_action(predictions: np.ndarray, upper_threshold: float, lower_threshold: float, window: int = 3) -> tuple[str, str]:
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


def render_forecast_tab_plotly(df, forecast_next, model_dir, project_root):
    """
    Render the enhanced forecast tab with Plotly visualization
    """
    st.subheader("📈 Time-Series Forecast & Autoscaling")
    
    # Mode selector
    backtest_mode = st.checkbox("🔬 Backtest Mode (simulate past forecast for evaluation)", value=False, key="backtest_mode")
    
    st.divider()
    
    # Controls in two rows
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], key="timeframe", help="Data resolution")
    with col2:
        forecast_horizon = st.slider("Horizon (steps)", min_value=1, max_value=60, value=15, key="horizon", help="Steps into future")
    with col3:
        hysteresis_window = st.slider("Hysteresis Window", min_value=2, max_value=10, value=3, key="hysteresis", help="Consecutive steps to confirm scaling")
    with col4:
        focus_hours_before = st.slider("Focus: Hours Before", min_value=1, max_value=5, value=2, key="hours_before", help="Hours to show before now")
    
    # Backtest mode controls
    backtest_index = None
    if backtest_mode:
        st.markdown("**🎯 Backtest Settings**")
        col1, col2 = st.columns(2)
        with col1:
            max_backtest_idx = len(df) - forecast_horizon - 1
            if max_backtest_idx > 0:
                backtest_index = st.slider(
                    "Backtest point (index)", 
                    min_value=forecast_horizon + 10, 
                    max_value=max_backtest_idx, 
                    value=min(max_backtest_idx // 2, max_backtest_idx),
                    key="backtest_idx",
                    help="Select a point in time to treat as 'now' for simulation"
                )
                backtest_timestamp = pd.to_datetime(df.iloc[backtest_index]["timestamp"])
                st.caption(f"📍 Pseudo-now: {backtest_timestamp}")
            else:
                st.error("Dataset too small for backtesting")
                backtest_index = None
        with col2:
            st.info("ℹ️ Backtest uses only data BEFORE the selected point, then compares forecast with actual future.")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        show_lstm = st.checkbox("Show LSTM (Hybrid)", value=True, key="show_lstm")
    with col2:
        show_xgboost = st.checkbox("Show XGBoost", value=True, key="show_xgboost")
    
    st.divider()
    
    # Determine data split based on mode
    if backtest_mode and backtest_index is not None:
        df_input = df.iloc[:backtest_index + 1].copy()
        df_actual_future = df.iloc[backtest_index + 1:backtest_index + 1 + forecast_horizon].copy()
        pseudo_now_timestamp = pd.to_datetime(df_input.iloc[-1]["timestamp"])
        pseudo_now_idx = backtest_index
    else:
        df_input = df.copy()
        df_actual_future = None
        pseudo_now_timestamp = pd.to_datetime(df_input.iloc[-1]["timestamp"])
        pseudo_now_idx = len(df_input) - 1
    
    # Generate forecasts
    lstm_forecast_df = None
    xgboost_forecast_df = None
    lstm_status = ""
    xgboost_status = ""
    lstm_rmse = 15.0  # Default RMSE for confidence interval
    
    if show_lstm:
        lstm_forecast_df, lstm_status = forecast_next(
            df_input, 
            forecast_horizon, 
            model_type="hybrid",
            timeframe=timeframe, 
            model_dir=model_dir
        )
        # Try to get LSTM RMSE from metrics
        lstm_metrics_path = os.path.join(project_root, "lstm_metrics_summary.txt")
        if os.path.exists(lstm_metrics_path):
            try:
                with open(lstm_metrics_path, 'r') as f:
                    content = f.read()
                    if "RMSE" in content:
                        import re
                        match = re.search(r'RMSE[:\s]+([0-9.]+)', content)
                        if match:
                            lstm_rmse = float(match.group(1))
            except:
                pass
    
    if show_xgboost:
        xgboost_forecast_df, xgboost_status = forecast_next(
            df_input,
            forecast_horizon,
            model_type="xgboost",
            timeframe=timeframe,
            model_dir=model_dir
        )
    
    # Calculate focus window
    points_per_hour = {"1m": 60, "5m": 12, "15m": 4}
    points_before = int(focus_hours_before * points_per_hour.get(timeframe, 12))
    points_after = forecast_horizon
    
    focus_start_idx = max(0, pseudo_now_idx - points_before)
    historical_focus = df.iloc[focus_start_idx:pseudo_now_idx + 1].copy()
    
    # === METRIC CARDS ===
    st.markdown("### 📊 Autoscaling Metrics")
    
    # Calculate thresholds
    upper_default = float(df["requests_count"].quantile(0.85))
    lower_default = float(df["requests_count"].quantile(0.15))
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Thresholds**")
        upper_threshold = st.number_input("Upper (Scale-out)", value=upper_default, min_value=0.0, key="upper_thresh")
        lower_threshold = st.number_input("Lower (Scale-in)", value=lower_default, min_value=0.0, key="lower_thresh")
    
    # Calculate metrics
    current_load = float(df_input["requests_count"].iloc[-1])
    
    # Use best available forecast for decision making
    primary_forecast = lstm_forecast_df if lstm_forecast_df is not None and not lstm_forecast_df.empty else xgboost_forecast_df
    
    predicted_max = current_load
    scaling_action = "hold"
    scaling_reason = "No forecast available"
    
    if primary_forecast is not None and not primary_forecast.empty:
        predicted_values = primary_forecast["yhat"].values
        predicted_max = float(np.max(predicted_values))
        
        # Apply hysteresis logic
        scaling_action, scaling_reason = calculate_hysteresis_action(
            predicted_values, 
            upper_threshold, 
            lower_threshold, 
            hysteresis_window
        )
    
    with col1:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Load", f"{current_load:.0f}", delta="requests")
            st.caption("Latest observation")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            delta_pct = ((predicted_max - current_load) / current_load * 100) if current_load > 0 else 0
            st.metric("Predicted Max", f"{predicted_max:.0f}", delta=f"{delta_pct:+.1f}%")
            st.caption(f"Within {forecast_horizon} steps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Recommended Action**")
            if scaling_action == "scale_up":
                st.markdown('<div class="action-scale-up">🔺 SCALE UP</div>', unsafe_allow_html=True)
            elif scaling_action == "scale_down":
                st.markdown('<div class="action-scale-down">🔻 SCALE DOWN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="action-hold">✅ HOLD</div>', unsafe_allow_html=True)
            st.caption(scaling_reason)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # === PLOTLY CHART ===
    fig = go.Figure()
    
    # Add actual historical data (green)
    fig.add_trace(go.Scatter(
        x=historical_focus["timestamp"],
        y=historical_focus["requests_count"],
        mode='lines',
        name='Actual',
        line=dict(color='#00FF00', width=2),
        hovertemplate='<b>Actual</b><br>Time: %{x}<br>Load: %{y:.0f}<extra></extra>'
    ))
    
    # Add actual future (if backtest mode)
    if backtest_mode and df_actual_future is not None and not df_actual_future.empty:
        fig.add_trace(go.Scatter(
            x=df_actual_future["timestamp"],
            y=df_actual_future["requests_count"],
            mode='lines',
            name='Actual (Future)',
            line=dict(color='#00FF00', width=2, dash='dot'),
            hovertemplate='<b>Actual Future</b><br>Time: %{x}<br>Load: %{y:.0f}<extra></extra>'
        ))
    
    # Add LSTM forecast with confidence interval
    if lstm_forecast_df is not None and not lstm_forecast_df.empty:
        lstm_times = lstm_forecast_df["timestamp"]
        lstm_values = lstm_forecast_df["yhat"]
        
        # Confidence interval based on RMSE
        upper_ci = lstm_values + 1.96 * lstm_rmse
        lower_ci = lstm_values - 1.96 * lstm_rmse
        
        # Add confidence interval (shaded area)
        fig.add_trace(go.Scatter(
            x=pd.concat([lstm_times, lstm_times[::-1]]),
            y=pd.concat([upper_ci, lower_ci[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='LSTM 95% CI',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add LSTM forecast line
        fig.add_trace(go.Scatter(
            x=lstm_times,
            y=lstm_values,
            mode='lines',
            name='LSTM Forecast',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>LSTM</b><br>Time: %{x}<br>Predicted: %{y:.0f}<extra></extra>'
        ))
    
    # Add XGBoost forecast
    if xgboost_forecast_df is not None and not xgboost_forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=xgboost_forecast_df["timestamp"],
            y=xgboost_forecast_df["yhat"],
            mode='lines',
            name='XGBoost Forecast',
            line=dict(color='#ff7f0e', width=3),
            hovertemplate='<b>XGBoost</b><br>Time: %{x}<br>Predicted: %{y:.0f}<extra></extra>'
        ))
    
    # Add vertical line at pseudo-now
    fig.add_vline(
        x=pseudo_now_timestamp.timestamp() * 1000,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Now" if not backtest_mode else "Pseudo-Now",
        annotation_position="top"
    )
    
    # Add threshold lines
    fig.add_hline(
        y=upper_threshold,
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text="Upper Threshold",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=lower_threshold,
        line_dash="dash",
        line_color="yellow",
        line_width=1,
        annotation_text="Lower Threshold",
        annotation_position="right"
    )
    
    # Update layout for dark theme
    mode_label = "BACKTEST" if backtest_mode else "LIVE FORECAST"
    fig.update_layout(
        title=f"{mode_label}: Traffic Forecast & Autoscaling Decision",
        xaxis_title="Time",
        yaxis_title="Requests Count",
        template="plotly_dark",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if backtest_mode:
        st.caption("🔴 Red vertical = Pseudo-now | 🟢 Green = Actual | 🔵 Blue = LSTM | 🟠 Orange = XGBoost | Red/Yellow horizontal = Thresholds")
    else:
        st.caption("🔴 Red vertical = Now | 🟢 Green = History | 🔵 Blue = LSTM | 🟠 Orange = XGBoost | Shaded = 95% CI")
    
    # === FORECAST SUMMARY ===
    st.divider()
    st.markdown("### 📋 Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    if lstm_forecast_df is not None and not lstm_forecast_df.empty:
        with col1:
            st.markdown("**LSTM Forecast**")
            st.write(f"Min: {float(lstm_forecast_df['yhat'].min()):.0f}")
            st.write(f"Mean: {float(lstm_forecast_df['yhat'].mean()):.0f}")
            st.write(f"Max: {float(lstm_forecast_df['yhat'].max()):.0f}")
            status_icon = "✅" if "heuristic" not in lstm_status else "⚠️"
            st.caption(f"{status_icon} {lstm_status}")
    
    if xgboost_forecast_df is not None and not xgboost_forecast_df.empty:
        with col2:
            st.markdown("**XGBoost Forecast**")
            st.write(f"Min: {float(xgboost_forecast_df['yhat'].min()):.0f}")
            st.write(f"Mean: {float(xgboost_forecast_df['yhat'].mean()):.0f}")
            st.write(f"Max: {float(xgboost_forecast_df['yhat'].max()):.0f}")
            status_icon = "✅" if "heuristic" not in xgboost_status else "⚠️"
            st.caption(f"{status_icon} {xgboost_status}")
    
    with col3:
        st.markdown("**Current State**")
        st.write(f"Value: {current_load:.0f}")
        st.write(f"Upper: {upper_threshold:.0f}")
        st.write(f"Lower: {lower_threshold:.0f}")
        st.caption("Latest" if not backtest_mode else "At pseudo-now")
    
    # === BACKTEST METRICS ===
    if backtest_mode and df_actual_future is not None and not df_actual_future.empty:
        st.divider()
        st.markdown("### 📊 Model Performance (Backtest)")
        
        actual_values = df_actual_future["requests_count"].values
        
        # LSTM metrics
        lstm_metrics = None
        if lstm_forecast_df is not None and not lstm_forecast_df.empty:
            predicted = lstm_forecast_df["yhat"].values[:len(actual_values)]
            if len(predicted) > 0:
                mae = float(np.mean(np.abs(actual_values - predicted)))
                rmse = float(np.sqrt(np.mean((actual_values - predicted) ** 2)))
                mape = float(np.mean(np.abs((actual_values - predicted) / (np.abs(actual_values) + 1e-8)))) * 100
                lstm_metrics = {"mae": mae, "rmse": rmse, "mape": mape}
        
        # XGBoost metrics
        xgboost_metrics = None
        if xgboost_forecast_df is not None and not xgboost_forecast_df.empty:
            predicted = xgboost_forecast_df["yhat"].values[:len(actual_values)]
            if len(predicted) > 0:
                mae = float(np.mean(np.abs(actual_values - predicted)))
                rmse = float(np.sqrt(np.mean((actual_values - predicted) ** 2)))
                mape = float(np.mean(np.abs((actual_values - predicted) / (np.abs(actual_values) + 1e-8)))) * 100
                xgboost_metrics = {"mae": mae, "rmse": rmse, "mape": mape}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if lstm_metrics:
                st.markdown("**🔵 LSTM**")
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.metric("MAE", f"{lstm_metrics['mae']:.2f}")
                with subcol2:
                    st.metric("RMSE", f"{lstm_metrics['rmse']:.2f}")
                with subcol3:
                    st.metric("MAPE", f"{lstm_metrics['mape']:.1f}%")
        
        with col2:
            if xgboost_metrics:
                st.markdown("**🟠 XGBoost**")
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.metric("MAE", f"{xgboost_metrics['mae']:.2f}")
                with subcol2:
                    st.metric("RMSE", f"{xgboost_metrics['rmse']:.2f}")
                with subcol3:
                    st.metric("MAPE", f"{xgboost_metrics['mape']:.1f}%")
        
        # Model comparison
        if lstm_metrics and xgboost_metrics:
            st.divider()
            st.markdown("**🏆 Winner**")
            better = "LSTM" if lstm_metrics['mae'] < xgboost_metrics['mae'] else "XGBoost"
            st.success(f"Best overall: **{better}** (lowest MAE)")
    
    elif not backtest_mode:
        st.info("💡 Enable **Backtest Mode** to see model performance metrics.")
    
    # Warnings
    if "heuristic" in lstm_status or "heuristic" in xgboost_status:
        st.warning("⚠️ One or more models using fallback. Check if trained models are available.")
