from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
from io import StringIO
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.forecast import (
    forecast_next,
    forecast_on_data,
    load_model_metrics,
    discover_models
)
from utils.metrics_forecast import forecast_metrics_lightgbm

app = FastAPI(
    title="Time Series Forecasting API",
    description="REST API for traffic prediction using Hybrid, XGBoost, and LightGBM models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = str(PROJECT_ROOT / "forecasting" / "models")
RESULTS_DIR = str(PROJECT_ROOT / "forecasting" / "artifacts" / "predictions")
METRICS_DIR = str(PROJECT_ROOT / "forecasting" / "artifacts" / "metrics")


class MetricDataPoint(BaseModel):
    timestamp: str = Field(..., description="Timestamp in ISO format or any parseable format (e.g., '12:00', '2024-01-01 12:05')")
    requests: float = Field(..., description="Number of requests in this period", ge=0)


class MetricsForecastRequest(BaseModel):
    timeframe: str = Field("5m", description="Time interval (currently only '5m' supported)")
    history: List[MetricDataPoint] = Field(..., description="Historical metrics data (minimum 6 points)")
    horizon_steps: int = Field(..., description="Number of future periods to forecast", ge=1, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "5m",
                "history": [
                    {"timestamp": "12:00", "requests": 410},
                    {"timestamp": "12:05", "requests": 435},
                    {"timestamp": "12:10", "requests": 460},
                    {"timestamp": "12:15", "requests": 490},
                    {"timestamp": "12:20", "requests": 520},
                    {"timestamp": "12:25", "requests": 550}
                ],
                "horizon_steps": 3
            }
        }


class ForecastDataPoint(BaseModel):
    step: int = Field(..., description="Forecast step (1-based)")
    predicted_requests: float = Field(..., description="Predicted requests for this step")


class MetricsForecastResponse(BaseModel):
    success: bool
    message: str
    model_used: Optional[str] = None
    forecast: Optional[List[ForecastDataPoint]] = None


class ScalingRequest(BaseModel):
    current_servers: int = Field(..., description="Current number of servers", ge=1)
    requests: float = Field(..., description="Current incoming requests", ge=0)
    forecast: float = Field(..., description="Forecasted requests (from ML model)", ge=0)
    capacity_per_server: int = Field(500, description="Requests per server per time unit")


class ScalingReason(BaseModel):
    factor: str
    current_value: str
    threshold: str
    decision: str


class ScalingRecommendation(BaseModel):
    current_servers: int
    recommended_servers: int
    action: str
    reasons: List[ScalingReason]
    confidence: float
    explanation: str
    estimated_cost_impact: Dict[str, float]


class HybridAutoscalerAnalyzer:
    def __init__(self):
        self.min_servers = 2
        self.max_servers = 20
        self.capacity_per_server = 500
        self.sla_threshold = 0.95
        self.slo_threshold = 0.85
        self.target_utilization = 0.70
        self.scale_up_margin = 0.80
        self.scale_down_margin = 0.30
        self.scale_up_step = 1

    def recommend(self, current_servers: int, requests: float, forecast: float, capacity_per_server: int) -> ScalingRecommendation:
        self.capacity_per_server = capacity_per_server
        anomaly_detected = self._detect_anomaly(requests, forecast)
        current_utilization = requests / (current_servers * capacity_per_server)
        sla_breach = current_utilization > self.sla_threshold
        forecast_servers_needed = self._servers_for_load(forecast, capacity_per_server)
        
        reasons = []
        recommended_servers = current_servers
        confidence = 0.5
        
        if anomaly_detected:
            reasons.append(ScalingReason(
                factor="Anomaly Detection",
                current_value=f"{requests:.0f} requests",
                threshold=f"±{int(forecast*0.2):.0f} requests from forecast",
                decision="ALERT: Spike detected, prepare to scale"
            ))
            confidence += 0.1
        
        if sla_breach:
            scale_up_amount = max(1, current_servers // 2)
            recommended_servers = min(self.max_servers, current_servers + scale_up_amount)
            reasons.append(ScalingReason(
                factor="Emergency (SLA Breach)",
                current_value=f"{current_utilization:.1%} utilization",
                threshold=f"< {self.sla_threshold:.0%}",
                decision=f"CRITICAL: Scale UP {scale_up_amount} servers immediately"
            ))
            confidence = 0.95
        elif forecast_servers_needed > current_servers + 1:
            scale_up_amount = forecast_servers_needed - current_servers
            recommended_servers = min(self.max_servers, current_servers + scale_up_amount)
            reasons.append(ScalingReason(
                factor="Predictive Scaling",
                current_value=f"Forecast: {forecast:.0f} requests",
                threshold=f"Current capacity: {current_servers * capacity_per_server}",
                decision=f"Proactive: Scale UP {scale_up_amount} servers (prepare for spike)"
            ))
            confidence = 0.8
        else:
            current_util = requests / (current_servers * capacity_per_server)
            if current_util > self.scale_up_margin:
                scale_up_amount = 1
                recommended_servers = min(self.max_servers, current_servers + scale_up_amount)
                reasons.append(ScalingReason(
                    factor="Reactive (High Load)",
                    current_value=f"{current_util:.1%} utilization",
                    threshold=f"> {self.scale_up_margin:.0%}",
                    decision=f"Scale UP {scale_up_amount} server (responding to load)"
                ))
                confidence = 0.7
            elif current_util < self.scale_down_margin and current_servers > self.min_servers:
                scale_down_amount = 1
                recommended_servers = max(self.min_servers, current_servers - scale_down_amount)
                reasons.append(ScalingReason(
                    factor="Reactive (Low Load)",
                    current_value=f"{current_util:.1%} utilization",
                    threshold=f"< {self.scale_down_margin:.0%}",
                    decision=f"Scale DOWN {scale_down_amount} server (save cost)"
                ))
                confidence = 0.75
            else:
                reasons.append(ScalingReason(
                    factor="Stable State",
                    current_value=f"{current_util:.1%} utilization",
                    threshold=f"{self.scale_down_margin:.0%} - {self.scale_up_margin:.0%}",
                    decision="No scaling needed (system stable)"
                ))
                confidence = 0.85
        
        if recommended_servers > current_servers:
            action = "scale-up"
        elif recommended_servers < current_servers:
            action = "scale-down"
        else:
            action = "no-change"
        
        current_hourly_cost = self._calculate_hourly_cost(current_servers, requests)
        new_hourly_cost = self._calculate_hourly_cost(recommended_servers, requests)
        cost_impact = {
            'current_hourly_cost': float(current_hourly_cost),
            'new_hourly_cost': float(new_hourly_cost),
            'cost_difference': float(new_hourly_cost - current_hourly_cost),
            'cost_change_percent': float((new_hourly_cost - current_hourly_cost) / current_hourly_cost * 100) if current_hourly_cost > 0 else 0.0,
        }
        
        explanation = self._generate_explanation(action, current_servers, recommended_servers, current_utilization, reasons)
        
        return ScalingRecommendation(
            current_servers=current_servers,
            recommended_servers=recommended_servers,
            action=action,
            reasons=reasons,
            confidence=confidence,
            explanation=explanation,
            estimated_cost_impact=cost_impact
        )
    
    def _detect_anomaly(self, requests: float, forecast: float) -> bool:
        if forecast == 0:
            return False
        deviation = abs(requests - forecast) / forecast
        return deviation > 0.5
    
    def _servers_for_load(self, load: float, capacity: int) -> int:
        servers_needed = load / capacity
        return max(self.min_servers, min(self.max_servers, int(servers_needed) + 1))
    
    def _calculate_hourly_cost(self, servers: int, current_requests: float) -> float:
        step_hours = 1.0
        reserved_servers = min(servers, 2)
        cost_reserved = reserved_servers * 0.03 * step_hours
        if servers > 2:
            burst_servers = servers - 2
            cost_spot = burst_servers * 0.7 * 0.015 * step_hours
            cost_ondemand = burst_servers * 0.3 * 0.05 * step_hours
            return cost_reserved + cost_spot + cost_ondemand
        return cost_reserved
    
    def _generate_explanation(self, action: str, current_servers: int, recommended_servers: int, current_util: float, reasons: List[ScalingReason]) -> str:
        if action == "scale-up":
            reason_texts = [r.decision for r in reasons if "UP" in r.decision]
            reason_text = reason_texts[0] if reason_texts else "Load increasing"
            return f"SCALE UP from {current_servers} to {recommended_servers} servers. {reason_text}. Current utilization: {current_util:.1%}"
        elif action == "scale-down":
            reason_texts = [r.decision for r in reasons if "DOWN" in r.decision]
            reason_text = reason_texts[0] if reason_texts else "Load decreasing"
            return f"SCALE DOWN from {current_servers} to {recommended_servers} servers. {reason_text}. Current utilization: {current_util:.1%}"
        else:
            return f"NO CHANGE - Keep {current_servers} servers. System is operating within optimal range. Current utilization: {current_util:.1%}"


analyzer = HybridAutoscalerAnalyzer()


class ForecastRequest(BaseModel):
    data: List[Dict] = Field(..., description="Historical data with 'ds' and 'y' columns")
    horizon: int = Field(..., description="Number of steps to forecast", ge=1, le=100)
    model_type: str = Field(..., description="Model type: 'hybrid', 'xgboost', or 'lightgbm'")
    timeframe: str = Field(..., description="Time window: '1m', '5m', or '15m'")
    model_dir: Optional[str] = Field(None, description="Custom model directory path")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"ds": "1995-08-01 00:00:00", "y": 245},
                    {"ds": "1995-08-01 00:05:00", "y": 312}
                ],
                "horizon": 12,
                "model_type": "xgboost",
                "timeframe": "5m"
            }
        }


class BacktestRequest(BaseModel):
    data: List[Dict] = Field(..., description="Historical data with 'ds' and 'y' columns")
    step: int = Field(..., description="Step size for rolling predictions", ge=1, le=50)
    model_type: str = Field(..., description="Model type: 'hybrid', 'xgboost', or 'lightgbm'")
    timeframe: str = Field(..., description="Time window: '1m', '5m', or '15m'")
    model_dir: Optional[str] = Field(None, description="Custom model directory path")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"ds": "1995-08-01 00:00:00", "y": 245},
                    {"ds": "1995-08-01 00:05:00", "y": 312}
                ],
                "step": 1,
                "model_type": "hybrid",
                "timeframe": "5m"
            }
        }


class ForecastResponse(BaseModel):
    success: bool
    message: str
    predictions: Optional[List[Dict]] = None
    metrics: Optional[Dict] = None


class MetricsResponse(BaseModel):
    model_type: str
    timeframe: str
    metrics: Optional[Dict]
    success: bool
    message: str


class ModelsResponse(BaseModel):
    available_models: Dict
    success: bool


@app.get("/", tags=["Health"])
async def root():
    return {
        "api": "Time Series Forecasting API",
        "version": "2.0.0",
        "status": "running",
        "description": "System metrics forecasting for autoscaling decisions",
        "endpoints": {
            "POST /forecast/metrics": "Forecast from system metrics (recommended for production)",
            "POST /forecast/predict": "Legacy: Forward forecasting with raw data",
            "POST /forecast/backtest": "Legacy: Historical backtesting",
            "GET /metrics/{model_type}/{timeframe}": "Get model performance metrics",
            "GET /models": "Discover available models",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_dir": MODEL_DIR,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "results_dir": RESULTS_DIR,
        "results_dir_exists": os.path.exists(RESULTS_DIR)
    }


@app.post("/recommend-scaling", response_model=ScalingRecommendation, tags=["Autoscaling"])
async def recommend_scaling(request: ScalingRequest) -> ScalingRecommendation:
    try:
        recommendation = analyzer.recommend(
            current_servers=request.current_servers,
            requests=request.requests,
            forecast=request.forecast,
            capacity_per_server=request.capacity_per_server
        )
        return recommendation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/forecast/metrics", response_model=MetricsForecastResponse, tags=["Forecasting"])
async def forecast_from_metrics(request: MetricsForecastRequest):
    """
    Forecast autoscaling metrics from system monitoring data.
    
    This is the production-ready endpoint for autoscaling forecasting.
    It accepts aggregated system metrics (timestamps + request counts)
    and returns predicted request volumes for future periods.
    
    The API handles all feature engineering internally:
    - Lag features (1h, 1d lookback)
    - Rolling statistics (5m, 1h windows)
    - Time-based features (hour, day cyclical encoding)
    - Burst detection
    
    No ML knowledge required from the caller.
    """
    try:
        # Validate timeframe
        if request.timeframe != "5m":
            raise HTTPException(
                status_code=400,
                detail="Currently only timeframe='5m' is supported"
            )
        
        # Convert Pydantic objects to dicts for processing
        history = [
            {"timestamp": point.timestamp, "requests": point.requests}
            for point in request.history
        ]
        
        # Call forecasting function
        model_path = str(Path(__file__).resolve().parents[1] / "forecasting" / "artifacts" / "models")
        success, message, forecast = forecast_metrics_lightgbm(
            history=history,
            horizon_steps=request.horizon_steps,
            model_dir=model_path
        )
        
        if not success:
            print(f"[ERROR] forecast_metrics_lightgbm failed: {message}")
            raise HTTPException(
                status_code=400,
                detail=message
            )
        
        # Convert forecast list to Pydantic objects
        forecast_points = [
            ForecastDataPoint(step=f["step"], predicted_requests=f["predicted_requests"])
            for f in forecast
        ]
        
        return MetricsForecastResponse(
            success=True,
            message=message,
            model_used="LightGBM_5m",
            forecast=forecast_points
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast error: {str(e)}"
        )



async def predict(request: ForecastRequest):
    try:
        df = pd.DataFrame(request.data)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Data must contain 'ds' (timestamp) and 'y' (value) columns"
            )
        
        df['ds'] = pd.to_datetime(df['ds'])
        model_dir = request.model_dir if request.model_dir else MODEL_DIR
        
        result_df, status_msg = forecast_next(
            df=df,
            forecast_horizon=request.horizon,
            model_type=request.model_type,
            timeframe=request.timeframe,
            model_dir=model_dir
        )
        
        if result_df is None:
            return ForecastResponse(
                success=False,
                message=status_msg,
                predictions=None
            )
        
        # Convert result to list of dicts
        predictions = result_df.to_dict('records')
        
        # Convert timestamps to strings for JSON serialization
        for pred in predictions:
            if 'ds' in pred:
                pred['ds'] = str(pred['ds'])
        
        return ForecastResponse(
            success=True,
            message=status_msg,
            predictions=predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/forecast/backtest", response_model=ForecastResponse, tags=["Forecasting"])
async def backtest(request: BacktestRequest):
    try:
        df = pd.DataFrame(request.data)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Data must contain 'ds' (timestamp) and 'y' (value) columns"
            )
        
        df['ds'] = pd.to_datetime(df['ds'])
        model_dir = request.model_dir if request.model_dir else MODEL_DIR
        
        result_df, status_msg = forecast_on_data(
            df=df,
            step=request.step,
            model_type=request.model_type,
            timeframe=request.timeframe,
            model_dir=model_dir
        )
        
        if result_df is None:
            return ForecastResponse(
                success=False,
                message=status_msg,
                predictions=None
            )
        
        # Convert result to list of dicts
        predictions = result_df.to_dict('records')
        
        # Convert timestamps to strings for JSON serialization
        for pred in predictions:
            if 'ds' in pred:
                pred['ds'] = str(pred['ds'])
        
        metrics = None
        if 'y' in result_df.columns and 'yhat' in result_df.columns:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np
            
            y_true = result_df['y'].values
            y_pred = result_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape)
            }
        
        return ForecastResponse(
            success=True,
            message=status_msg,
            predictions=predictions,
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@app.get("/metrics/{model_type}/{timeframe}", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(model_type: str, timeframe: str, model_dir: Optional[str] = None):
    try:
        valid_models = ['hybrid', 'xgboost', 'lightgbm']
        valid_timeframes = ['1m', '5m', '15m']
        
        if model_type not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Must be one of: {valid_models}"
            )
        
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Must be one of: {valid_timeframes}"
            )
        
        dir_path = model_dir if model_dir else RESULTS_DIR
        metrics = load_model_metrics(
            model_type=model_type,
            timeframe=timeframe,
            model_dir=dir_path
        )
        
        if metrics is None:
            return MetricsResponse(
                model_type=model_type,
                timeframe=timeframe,
                metrics=None,
                success=False,
                message=f"Metrics not found for {model_type} {timeframe}"
            )
        
        return MetricsResponse(
            model_type=model_type,
            timeframe=timeframe,
            metrics=metrics,
            success=True,
            message="Metrics loaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models(model_dir: Optional[str] = None):
    try:
        dir_path = model_dir if model_dir else MODEL_DIR
        available_models = discover_models(model_dir=dir_path)
        
        return ModelsResponse(
            available_models=available_models,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error discovering models: {str(e)}")


@app.post("/forecast/predict/csv", tags=["Forecasting"])
async def predict_from_csv(
    file: UploadFile = File(...),
    horizon: int = 12,
    model_type: str = "xgboost",
    timeframe: str = "5m"
):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'ds' (timestamp) and 'y' (value) columns"
            )
        
        df['ds'] = pd.to_datetime(df['ds'])
        result_df, status_msg = forecast_next(
            df=df,
            forecast_horizon=horizon,
            model_type=model_type,
            timeframe=timeframe,
            model_dir=MODEL_DIR
        )
        
        if result_df is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": status_msg}
            )
        
        predictions = result_df.to_dict('records')
        for pred in predictions:
            if 'ds' in pred:
                pred['ds'] = str(pred['ds'])
        
        return {
            "success": True,
            "message": status_msg,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
