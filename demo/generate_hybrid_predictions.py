"""
Generate hybrid predictions CSV files for demo dashboard
Hybrid = Prophet baseline + LSTM residual correction
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from utils.forecast import load_lstm_model, _load_hybrid_scaler_and_window

def generate_predictions_for_timeframe(timeframe: str):
    """Generate hybrid predictions for a specific timeframe"""
    print(f"\n{'='*60}")
    print(f"Generating hybrid predictions for {timeframe}")
    print(f"{'='*60}")
    
    # Paths
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    model_path = os.path.join(model_dir, f"lstm_{timeframe}_best.keras")
    test_path = os.path.join(data_dir, f"test_{timeframe}_autoscaling.csv")
    train_path = os.path.join(data_dir, f"train_{timeframe}_autoscaling.csv")
    output_path = os.path.join(model_dir, f"hybrid_{timeframe}_predictions.csv")
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ Test data not found: {test_path}")
        return False
    
    if not os.path.exists(train_path):
        print(f"❌ Train data not found: {train_path}")
        return False
    
    # Load data
    print(f"Loading data...")
    train_df = pd.read_csv(train_path, parse_dates=['timestamp'])
    test_df = pd.read_csv(test_path, parse_dates=['timestamp'])
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Load model and scaler
    print(f"Loading LSTM model...")
    model = load_lstm_model(model_path)
    if model is None:
        print(f"❌ Failed to load model")
        return False
    
    package, window_size = _load_hybrid_scaler_and_window()
    scaler = None
    if package is not None:
        scaler = package.get("lstm_models", {}).get(timeframe, {}).get("scaler")
        print(f"Scaler loaded: {scaler is not None}, Window: {window_size}")
    
    # Limit predictions for speed
    max_predictions = min(1000, len(test_df))
    print(f"Generating {max_predictions} predictions (limited for speed)...")
    
    # Generate LSTM residual predictions
    lstm_residuals = []
    
    try:
        lookback = min(window_size or 24, len(train_df))
        
        if scaler is not None:
            # Scaled prediction for residuals
            last_values = train_df["requests_count"].iloc[-lookback:].values.astype(np.float32)
            scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()
            X = scaled.reshape(1, lookback, 1)
            
            for i in range(max_predictions):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{max_predictions}")
                pred_scaled = model.predict(X, verbose=0)
                pred_value_scaled = float(pred_scaled[0, 0])
                lstm_residuals.append(pred_value_scaled)
                X = np.append(X[:, 1:, :], [[[pred_value_scaled]]], axis=1)
            
            # Inverse transform residuals
            lstm_residuals = np.array(lstm_residuals).reshape(-1, 1)
            lstm_residuals = scaler.inverse_transform(lstm_residuals).flatten()
        else:
            # Unscaled prediction
            last_sequence = train_df["requests_count"].iloc[-lookback:].values.astype(np.float32)
            X = last_sequence.reshape(1, lookback, 1)
            
            for i in range(max_predictions):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{max_predictions}")
                pred = model.predict(X, verbose=0)
                pred_value = float(pred[0, 0])
                lstm_residuals.append(pred_value)
                X = np.append(X[:, 1:, :], [[[pred_value]]], axis=1)
            
            lstm_residuals = np.array(lstm_residuals)
        
        print(f"Generated {len(lstm_residuals)} LSTM residuals")
        
        # Load Prophet baseline from parent directory (dataflow root)
        print(f"Loading Prophet baseline...")
        parent_dir = os.path.dirname(os.path.dirname(__file__))  # Go up 1 level to dataflow/
        prophet_file = os.path.join(parent_dir, f"prophet_{timeframe}_all_predictions.csv")
        
        prophet_baseline = None
        if os.path.exists(prophet_file):
            try:
                prophet_df = pd.read_csv(prophet_file)
                # Filter test split if available
                if 'split' in prophet_df.columns:
                    prophet_test = prophet_df[prophet_df['split'] == 'test'].copy()
                    if not prophet_test.empty:
                        prophet_baseline = prophet_test['predicted'].values[:max_predictions]
                elif 'yhat' in prophet_df.columns:
                    prophet_baseline = prophet_df['yhat'].values[:max_predictions]
                elif 'predicted' in prophet_df.columns:
                    prophet_baseline = prophet_df['predicted'].values[:max_predictions]
                
                if prophet_baseline is not None:
                    print(f"✅ Loaded Prophet from {prophet_file}")
                    print(f"   Using {len(prophet_baseline)} Prophet predictions")
            except Exception as e:
                print(f"⚠️  Error loading Prophet: {e}")
                prophet_baseline = None
        else:
            print(f"⚠️  Prophet file not found: {prophet_file}")
        
        # Fallback: Generate simple heuristic if Prophet not available
        if prophet_baseline is None:
            print(f"Generating Prophet baseline (heuristic)...")
            baseline = float(train_df["requests_count"].mean())
            trend = 0
            if len(train_df) > 100:
                recent = train_df["requests_count"].tail(100).values
                trend = (recent[-1] - recent[0]) / len(recent)
            
            prophet_baseline = []
            current = baseline
            for i in range(max_predictions):
                current = current + trend
                prophet_baseline.append(current)
            
            prophet_baseline = np.array(prophet_baseline)
        
        prophet_baseline = np.array(prophet_baseline)
        
        # Combine: Hybrid = Prophet + LSTM residual
        hybrid_predictions = prophet_baseline + lstm_residuals
        hybrid_predictions = np.maximum(0, hybrid_predictions)  # No negative values
        
        print(f"Combined Prophet baseline + LSTM residuals")
        
        # Calculate metrics
        actual = test_df['requests_count'].values[:max_predictions]
        
        mae = np.mean(np.abs(actual - hybrid_predictions))
        rmse = np.sqrt(np.mean((actual - hybrid_predictions) ** 2))
        mape = np.mean(np.abs((actual - hybrid_predictions) / (actual + 1e-6))) * 100
        
        print(f"\nMetrics:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Save predictions
        results_df = pd.DataFrame({
            'timestamp': test_df['timestamp'].values[:max_predictions],
            'actual': actual,
            'predicted': hybrid_predictions,
            'prophet_baseline': prophet_baseline,
            'lstm_residual': lstm_residuals,
            'error': actual - hybrid_predictions,
            'abs_error': np.abs(actual - hybrid_predictions)
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        print(f"   Columns: {list(results_df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    timeframes = ['1m', '5m', '15m']
    
    print("Generating hybrid predictions for all timeframes...")
    print("Hybrid = Prophet baseline + LSTM residual correction\n")
    
    success_count = 0
    for tf in timeframes:
        if generate_predictions_for_timeframe(tf):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(timeframes)} timeframes")
    print(f"{'='*60}")
