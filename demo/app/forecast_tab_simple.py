"""
Forecast tab: Load test data (future) and show Actual vs Predicted
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_forecast_tab(df, forecast_next, model_dir):
    """
    Forecast visualization: Load test data (future) and show Actual vs Predicted
    """
    st.subheader("📈 Forecast: Actual vs Predicted (Test Set = Future)")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], key="forecast_timeframe")
    with col2:
        model_type = st.selectbox("Model", ["hybrid", "xgboost", "lightgbm"], key="forecast_model")
    
    st.divider()
    
    # Load test data and predictions directly from artifacts
    try:
        results_dir = os.path.join(model_dir, "..", "artifacts", "predictions")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        
        pred_file = f"{model_type}_{timeframe}_predictions.csv"
        pred_path = os.path.join(results_dir, pred_file)
        
        if not os.path.exists(pred_path):
            st.warning(f"⚠️ Predictions not found: {pred_file}")
            return
        
        pred_df = pd.read_csv(pred_path)
        if "split" in pred_df.columns:
            pred_df = pred_df[pred_df["split"] == "test"].copy()
        
        preferred_cols = ["y_pred", "hybrid_predicted", "predicted", "yhat"]
        pred_col = next((col for col in preferred_cols if col in pred_df.columns), None)
        if pred_col is None:
            st.warning(f"⚠️ No prediction column found in {pred_file}")
            return
        
        status = f"{model_type}_csv_{timeframe}"
        predictions = pred_df[pred_col].astype(float).values
        
        test_df = None
        if "timestamp" in pred_df.columns and "y_true" in pred_df.columns:
            test_df = pred_df[["timestamp", "y_true"]].rename(columns={"y_true": "requests_count"})
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
        else:
            test_path = os.path.join(data_dir, f"test_{timeframe}_autoscaling.csv")
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
                if "timestamp" in pred_df.columns:
                    pred_df = pred_df.copy()
                    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                    aligned = test_df.merge(pred_df[["timestamp", pred_col]], on="timestamp", how="inner")
                    if not aligned.empty:
                        test_df = aligned[["timestamp", "requests_count"]]
                        predictions = aligned[pred_col].astype(float).values
        
        if test_df is None:
            st.error(f"❌ Test data not found for {timeframe}")
            return
        
        min_len = min(len(test_df), len(predictions))
        test_df = test_df.iloc[:min_len].copy()
        predictions = predictions[:min_len]
        
        # Prepare data for plotting
        plot_data = []
        
        # Add actual data from test set
        for idx, row in test_df.iterrows():
            plot_data.append({
                'timestamp': pd.to_datetime(row['timestamp']),
                'value': float(row['requests_count']),
                'type': 'Actual'
            })
        
        # Add predictions
        for idx, (ts, pred_val) in enumerate(zip(test_df['timestamp'], predictions)):
            plot_data.append({
                'timestamp': pd.to_datetime(ts),
                'value': float(pred_val),
                'type': 'Predicted'
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add Actual trace - cyan/light blue for better visibility
        actual_data = plot_df[plot_df['type'] == 'Actual']
        fig.add_trace(go.Scatter(
            x=actual_data['timestamp'],
            y=actual_data['value'],
            name='Actual',
            mode='lines+markers',
            line=dict(color='#00D9FF', width=2),  # Cyan - easy to see
            marker=dict(size=5, color='#00D9FF'),
            hovertemplate='<b>Actual</b><br>Time: %{x}<br>Value: %{y:.0f}<extra></extra>'
        ))
        
        # Add Predicted trace - yellow dashed
        predicted_data = plot_df[plot_df['type'] == 'Predicted']
        fig.add_trace(go.Scatter(
            x=predicted_data['timestamp'],
            y=predicted_data['value'],
            name='Predicted',
            mode='lines+markers',
            line=dict(color='#FFD700', width=2, dash='dash'),
            marker=dict(size=5, color='#FFD700'),
            hovertemplate='<b>Predicted</b><br>Time: %{x}<br>Value: %{y:.0f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{model_type.upper()} Forecast on Test Set ({timeframe}) - {status}',
            xaxis_title='Time',
            yaxis_title='Requests Count',
            hovermode='x unified',
            height=500,
            template='plotly_dark',
            font=dict(size=12),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate metrics
        actual_values = test_df['requests_count'].values[:min_len]
        mae = np.mean(np.abs(actual_values - predictions))
        rmse = np.sqrt(np.mean((actual_values - predictions) ** 2))
        
        # sMAPE (Symmetric Mean Absolute Percentage Error) 
        denominator = (np.abs(actual_values) + np.abs(predictions)) / 2
        smape = np.mean(np.abs(actual_values - predictions) / (denominator + 1e-6)) * 100
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Test Samples",
                f"{min_len}",
                help="Number of test data points"
            )
        
        with col2:
            st.metric(
                "MAE",
                f"{mae:.2f}",
                help="Mean Absolute Error"
            )
        
        with col3:
            st.metric(
                "RMSE",
                f"{rmse:.2f}",
                help="Root Mean Squared Error"
            )
        
        with col4:
            st.metric(
                "sMAPE",
                f"{smape:.2f}%",
                help="Symmetric Mean Absolute Percentage Error"
            )
        
        # Model status
        st.caption(f"✅ Status: {status}")
        
    except Exception as e:
        st.error(f"❌ Forecast error: {e}")
        import traceback
        st.write(traceback.format_exc())


