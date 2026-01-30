import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_forecast_tab(df, forecast_next, model_dir, project_root):
    # Forecast visualization - Load test data and show Actual vs Predicted
    st.subheader("📈 Forecast: Actual vs Predicted (Test Set = Future)")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], key="forecast_timeframe")
    with col2:
        model_type = st.selectbox("Model", ["hybrid", "xgboost", "lightgbm"], key="forecast_model")
    
    st.divider()
    
    # Load test data and predictions
    try:
        # Load test data
        data_dir = os.path.join(model_dir, "..", "data")
        test_file = f"test_{timeframe}_autoscaling.csv"
        test_path = os.path.join(data_dir, test_file)
        
        test_df = None
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
        # Load predictions based on model type
        status = "unknown"
        predictions = None
        
        if model_type == "xgboost":
            # Load XGBoost predictions from CSV
            pred_file = f"xgboost_{timeframe}_predictions.csv"
            pred_path = os.path.join(model_dir, pred_file)
            
            if os.path.exists(pred_path):
                pred_df = pd.read_csv(pred_path)
                if 'predicted' in pred_df.columns:
                    predictions = pred_df['predicted'].values
                    status = f"xgboost_csv_{timeframe}"
        
        elif model_type == "lightgbm":
            # Load LightGBM predictions from CSV
            pred_file = f"lightgbm_{timeframe}_predictions.csv"
            pred_path = os.path.join(model_dir, pred_file)
            
            if os.path.exists(pred_path):
                pred_df = pd.read_csv(pred_path)
                if 'predicted' in pred_df.columns:
                    predictions = pred_df['predicted'].values
                    status = f"lightgbm_csv_{timeframe}"
        
        elif model_type == "hybrid":
            # Load hybrid predictions: Prophet baseline + LSTM residuals
            # First try demo/models/, then project root
            pred_file = f"hybrid_{timeframe}_predictions.csv"
            pred_path = os.path.join(model_dir, pred_file)
            
            if os.path.exists(pred_path):
                pred_df = pd.read_csv(pred_path)
                if 'predicted' in pred_df.columns:
                    predictions = pred_df['predicted'].values
                    status = f"hybrid_{timeframe}"
            else:
                # Try to construct hybrid from Prophet + LSTM residuals (project root)
                prophet_file = os.path.join(project_root, f"prophet_{timeframe}_all_predictions.csv")
                lstm_file = os.path.join(project_root, f"lstm_{timeframe}_residual_predictions.csv")
                
                if os.path.exists(prophet_file) and os.path.exists(lstm_file):
                    try:
                        prophet_df = pd.read_csv(prophet_file)
                        lstm_df = pd.read_csv(lstm_file)
                        
                        # Filter test split
                        prophet_test = prophet_df[prophet_df['split'] == 'test'].copy()
                        lstm_test = lstm_df[lstm_df['split'] == 'test'].copy()
                        
                        # Combine: hybrid = prophet_baseline + lstm_residual
                        if len(prophet_test) > 0 and len(lstm_test) > 0:
                            min_len_hybrid = min(len(prophet_test), len(lstm_test))
                            prophet_baseline = prophet_test['predicted'].values[:min_len_hybrid]
                            lstm_residual = lstm_test['residual_predicted'].values[:min_len_hybrid]
                            predictions = prophet_baseline + lstm_residual
                            
                            # Use test_df from prophet (has timestamp and actual)
                            test_df = prophet_test.iloc[:min_len_hybrid].copy()
                            test_df = test_df.rename(columns={'actual': 'requests_count'})
                            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
                            
                            status = f"hybrid_composed_{timeframe}"
                    except Exception as e:
                        st.warning(f"Failed to compose hybrid predictions: {e}")
        
        
        if predictions is None:
            st.warning(f"⚠️ No predictions found for {model_type} model on {timeframe}")
            return
        
        if test_df is None:
            st.error(f"❌ Test data not found for {timeframe}")
            return
        
        # Align predictions with test data
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
        
        # sMAPE (Symmetric Mean Absolute Percentage Error) - better than MAPE
        # sMAPE = 100 * mean(|actual - predicted| / ((|actual| + |predicted|) / 2))
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


