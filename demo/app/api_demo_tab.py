"""
Metrics-based Forecast API Demo for Autoscaling
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import altair as alt
import os


@st.cache_data
def load_sample_metrics(filename=None):
    """Load and cache sample monitoring metrics"""
    try:
        if filename is None:
            # Try multiple paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "forecasting", "artifacts", "predictions", "lightgbm_5m_predictions.csv"),
                os.path.join(os.path.dirname(__file__), "..", "..", "forecasting", "artifacts", "predictions", "xgboost_5m_predictions.csv"),
                os.path.join(os.path.dirname(__file__), "..", "..", "forecasting", "artifacts", "predictions", "hybrid_5m_predictions.csv"),
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_5m_autoscaling.csv")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    filename = path
                    break
            else:
                return None
        
        df = pd.read_csv(filename)
        if 'timestamp' not in df.columns and 'ds' in df.columns:
            df = df.rename(columns={'ds': 'timestamp'})
        if 'requests_count' not in df.columns and 'y_true' in df.columns:
            df = df.rename(columns={'y_true': 'requests_count'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'requests_count']].copy()
        df = df.rename(columns={'requests_count': 'requests'})
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None


def _open_scaling_test():
    st.session_state.scaling_test_expanded = True


def render_api_demo_tab():
    """Render the API demo tab"""
    
    st.header("📡 API Demo - Metrics Forecast")
    
    st.markdown("""
    Forecast request volumes using the production metrics API.
    
    **Input:** Timestamps + request counts  
    **Output:** Predicted requests for future periods  
    **Features:** Automated server-side (no ML knowledge needed)
    """)

    if "api_demo_ready" not in st.session_state:
        st.session_state.api_demo_ready = True
    
    # API Configuration
    st.subheader("API Configuration")
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        api_url = st.text_input(
            "API URL",
            value="http://localhost:8000",
            help="URL of forecast API server"
        )
    
    with col2:
        if st.button("Test Connection"):
            try:
                r = requests.get(f"{api_url}/health", timeout=3)
                if r.status_code == 200:
                    st.success("✅ Connected")
                else:
                    st.error(f"Error {r.status_code}")
            except Exception as e:
                st.error(f"Failed: {str(e)[:50]}")

    with col3:
        request_timeout = st.number_input(
            "Timeout (s)",
            min_value=3,
            max_value=60,
            value=15,
            step=1,
            help="Max wait time for the forecast API"
        )
    
    st.divider()
    
    # Data Input
    st.subheader("Input Data")
    
    mode = st.radio(
        "Choose data source:",
        ["Manual Input", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    df_history = st.session_state.get("api_demo_history")
    
    if mode == "Upload CSV":
        uploaded = st.file_uploader("CSV file (timestamp, requests)", type=['csv'])
        if uploaded:
            try:
                df_history = pd.read_csv(uploaded)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                if 'requests_count' in df_history.columns:
                    df_history = df_history.rename(columns={'requests_count': 'requests'})
                st.session_state["api_demo_history"] = df_history
                st.success(f"✅ Loaded {len(df_history)} rows")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # Manual
        n = st.number_input("Number of points:", 6, 100, 12)
        now = datetime.now().replace(second=0, microsecond=0)
        
        with st.form("manual_input_form"):
            rows = []
            for i in range(n):
                col1, col2 = st.columns([2, 1])
                ts = now - timedelta(minutes=5*(n-1-i))
                
                with col1:
                    ts_input = st.text_input(
                        "ts", ts.strftime("%Y-%m-%d %H:%M"),
                        key=f"ts_{i}", label_visibility="collapsed"
                    )
                with col2:
                    val = st.number_input(
                        "val",
                        min_value=0.0,
                        value=float(300 + i * 20),
                        step=1.0,
                        key=f"val_{i}",
                        label_visibility="collapsed"
                    )
                
                rows.append({"timestamp": ts_input, "requests": val})
            
            submitted = st.form_submit_button("Confirm")
        
        if submitted:
            df_history = pd.DataFrame(rows)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            st.session_state["api_demo_history"] = df_history
            st.success("Ready!")
    
    st.divider()
    
    # Forecast
    if df_history is not None and len(df_history) >= 6:
        st.subheader("Forecast")
        
        if "api_demo_forecast" not in st.session_state:
            st.session_state.api_demo_forecast = None
        if "api_demo_error" not in st.session_state:
            st.session_state.api_demo_error = None
        
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Forecast horizon:", 1, 48, 12, key="api_demo_horizon")
        with col2:
            st.metric("Input points", len(df_history))
        
        if st.button("🚀 Forecast", type="primary", use_container_width=True):
            with st.status("Calling forecast API...", expanded=False) as status:
                status.write("Preparing request payload")
                try:
                    history = [{
                        "timestamp": row['timestamp'].isoformat(),
                        "requests": float(row['requests'])
                    } for _, row in df_history.iterrows()]
                    
                    payload = {
                        "timeframe": "5m",
                        "history": history,
                        "horizon_steps": horizon
                    }

                    status.write("Sending request")
                    
                    r = requests.post(
                        f"{api_url}/forecast/metrics",
                        json=payload,
                        timeout=request_timeout
                    )
                    
                    if r.status_code != 200:
                        st.session_state.api_demo_error = f"API error {r.status_code}"
                        st.session_state.api_demo_forecast = None
                        status.update(label="Request failed", state="error")
                    else:
                        result = r.json()
                        if not result.get('success'):
                            st.session_state.api_demo_error = result.get('message', 'Forecast failed')
                            st.session_state.api_demo_forecast = None
                            status.update(label="Forecast failed", state="error")
                        else:
                            st.session_state.api_demo_error = None
                            st.session_state.api_demo_forecast = result
                            status.update(label="Forecast complete", state="complete")
                except requests.exceptions.ConnectionError:
                    st.session_state.api_demo_error = f"Cannot connect to {api_url}"
                    st.session_state.api_demo_forecast = None
                    status.update(label="Connection error", state="error")
                except requests.exceptions.Timeout:
                    st.session_state.api_demo_error = "API timeout"
                    st.session_state.api_demo_forecast = None
                    status.update(label="Request timed out", state="error")
                except Exception as e:
                    st.session_state.api_demo_error = f"Error: {str(e)}"
                    st.session_state.api_demo_forecast = None
                    status.update(label="Unexpected error", state="error")
        
        if st.session_state.api_demo_error:
            st.error(st.session_state.api_demo_error)
        
        if st.session_state.api_demo_forecast:
            result = st.session_state.api_demo_forecast
            st.success("✅ Forecast complete")
            forecast_data = result['forecast']
            
            tab1, tab2, tab3 = st.tabs(["📊 Chart", "📋 Data", "📄 JSON"])
            
            with tab1:
                hist_df = df_history.copy()
                hist_df['type'] = 'Historical'
                hist_df = hist_df.rename(columns={'requests': 'value'})
                
                last_ts = df_history['timestamp'].iloc[-1]
                forecast_ts = [
                    last_ts + timedelta(minutes=5*(i+1))
                    for i in range(len(forecast_data))
                ]
                
                forecast_df = pd.DataFrame({
                    'timestamp': forecast_ts,
                    'value': [f['predicted_requests'] for f in forecast_data],
                    'type': 'Forecast'
                })
                
                chart_data = pd.concat(
                    [hist_df[['timestamp', 'value', 'type']], forecast_df],
                    ignore_index=True
                )
                
                chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X('timestamp:T', title='Time'),
                    y=alt.Y('value:Q', title='Requests'),
                    color=alt.Color(
                        'type:N',
                        scale=alt.Scale(
                            domain=['Historical', 'Forecast'],
                            range=['#1f77b4', '#ff7f0e']
                        )
                    ),
                    strokeDash=alt.StrokeDash(
                        'type:N',
                        scale=alt.Scale(
                            domain=['Historical', 'Forecast'],
                            range=[[0], [5, 5]]
                        )
                    )
                ).properties(width=700, height=400, title="Forecast Results").interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                hist_vals = df_history['requests'].values
                pred_vals = [f['predicted_requests'] for f in forecast_data]
                
                with col1:
                    st.metric("Hist Mean", f"{hist_vals.mean():.0f}")
                with col2:
                    st.metric("Hist Peak", f"{hist_vals.max():.0f}")
                with col3:
                    st.metric("Pred Mean", f"{sum(pred_vals)/len(pred_vals):.0f}")
                with col4:
                    st.metric("Pred Peak", f"{max(pred_vals):.0f}")
            
            with tab2:
                forecast_df = pd.DataFrame(forecast_data)
                forecast_df['step_minutes'] = forecast_df['step'] * 5
                st.dataframe(
                    forecast_df[['step', 'step_minutes', 'predicted_requests']],
                    use_container_width=True, hide_index=True
                )
                
                csv = forecast_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "forecast.csv", "text/csv")
            
            with tab3:
                st.json(result)
    
    else:
        st.info("⬆️ Load or enter data (min 6 points) to forecast")
    
    # Documentation
    st.divider()
    st.subheader("Scaling Recommendation API Test")
    
    if "scaling_test_expanded" not in st.session_state:
        st.session_state.scaling_test_expanded = False

    with st.expander(
        "🧪 Test Scaling Recommendation Endpoint",
        expanded=st.session_state.scaling_test_expanded,
    ):
        st.markdown("Test the `/recommend-scaling` endpoint with different scenarios")
        
        test_scenario = st.selectbox(
            "Test Scenario",
            [
                "High Load - Scale Up",
                "Low Load - Scale Down",
                "Anomaly/Spike Detection",
                "Stable State",
                "Custom"
            ]
        )
        
        # Preset scenarios
        scenarios = {
            "High Load - Scale Up": {"current_servers": 3, "requests": 1400, "forecast": 1600, "capacity_per_server": 500},
            "Low Load - Scale Down": {"current_servers": 10, "requests": 600, "forecast": 700, "capacity_per_server": 500},
            "Anomaly/Spike Detection": {"current_servers": 3, "requests": 2000, "forecast": 800, "capacity_per_server": 500},
            "Stable State": {"current_servers": 4, "requests": 1100, "forecast": 1150, "capacity_per_server": 500},
        }
        
        if test_scenario != "Custom":
            params = scenarios[test_scenario]
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                params = {"current_servers": st.number_input("Current Servers", value=3, min_value=1, max_value=20)}
            with col2:
                params["requests"] = st.number_input("Current Requests", value=1000, min_value=0)
            with col3:
                params["forecast"] = st.number_input("Forecast Requests", value=1200, min_value=0)
            with col4:
                params["capacity_per_server"] = st.number_input("Capacity/Server", value=500, min_value=10)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Servers", params["current_servers"])
        with col2:
            st.metric("Current Requests", f"{params['requests']:.0f}")
        with col3:
            st.metric("Forecast", f"{params['forecast']:.0f}")
        with col4:
            util = (params['requests'] / (params['current_servers'] * params['capacity_per_server'])) * 100
            st.metric("Current CPU", f"{util:.1f}%")
        
        if st.button(
            "🚀 Test Scaling Recommendation",
            key="test_scaling",
            on_click=_open_scaling_test,
        ):
            try:
                with st.spinner("Calling /recommend-scaling..."):
                    r = requests.post(
                        f"{api_url}/recommend-scaling",
                        json=params,
                        timeout=5
                    )
                    
                    if r.status_code == 200:
                        rec = r.json()
                        st.success("✅ Recommendation received")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Recommended Servers",
                                rec['recommended_servers'],
                                delta=rec['recommended_servers'] - rec['current_servers']
                            )
                        with col2:
                            st.metric("Action", rec['action'].upper())
                        with col3:
                            st.metric("Confidence", f"{rec['confidence']:.0%}")
                        
                        st.markdown("**Decision Layers:**")
                        for i, reason in enumerate(rec['reasons'], 1):
                            st.markdown(f"**{i}. {reason['factor']}**")
                            st.write(f"   • Current: {reason['current_value']}")
                            st.write(f"   • Threshold: {reason['threshold']}")
                            st.write(f"   • Decision: {reason['decision']}")
                        
                        st.markdown("**Cost Impact:**")
                        cost = rec['estimated_cost_impact']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Cost", f"${cost['current_hourly_cost']:.4f}/hr")
                        with col2:
                            st.metric("New Cost", f"${cost['new_hourly_cost']:.4f}/hr")
                        with col3:
                            st.metric("Change", f"{cost['cost_change_percent']:+.1f}%")
                        
                        with st.expander("Full Explanation"):
                            st.text(rec['explanation'])
                    else:
                        st.error(f"❌ API Error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to {api_url}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    st.subheader("API Documentation")
    
    with st.expander("Forecast Endpoint Format"):
        st.markdown("""
        **POST /forecast/metrics**
        
        Request:
        ```json
        {
          "timeframe": "5m",
          "history": [
            {"timestamp": "2024-01-01T12:00", "requests": 410},
            {"timestamp": "2024-01-01T12:05", "requests": 435}
          ],
          "horizon_steps": 3
        }
        ```
        
        Response:
        ```json
        {
          "success": true,
          "model_used": "LightGBM_5m",
          "forecast": [
            {"step": 1, "predicted_requests": 450},
            {"step": 2, "predicted_requests": 465}
          ]
        }
        ```
        """)
    
    with st.expander("Scaling Recommendation Endpoint"):
        st.markdown("""
        **POST /recommend-scaling**
        
        Request:
        ```json
        {
          "current_servers": 3,
          "requests": 1400,
          "forecast": 1600,
          "capacity_per_server": 500
        }
        ```
        
        Response:
        ```json
        {
          "current_servers": 3,
          "recommended_servers": 4,
          "action": "scale-up",
          "confidence": 0.8,
          "reasons": [
            {
              "factor": "Predictive Scaling",
              "current_value": "Forecast: 1600 requests",
              "threshold": "Current capacity: 1500",
              "decision": "Proactive: Scale UP 1 server"
            }
          ],
          "explanation": "...",
          "estimated_cost_impact": {
            "current_hourly_cost": 0.0315,
            "new_hourly_cost": 0.0405,
            "cost_difference": 0.009,
            "cost_change_percent": 28.6
          }
        }
        ```
        """)
    
    with st.expander("Features"):
        st.markdown("""
        **Automatic Feature Engineering:**
        - Time: hour_of_day, day_of_week, cyclical encoding
        - Lags: 5m, 15m, 6h, 1d
        - Rolling: mean/max over 1h window
        - Burst: detection + ratio metrics
        
        All handled server-side. Just send timestamps + requests!
        """)
