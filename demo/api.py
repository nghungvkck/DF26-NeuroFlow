from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
from io import StringIO
import os

from utils.forecast import (
    forecast_next,
    forecast_on_data,
    load_model_metrics,
    discover_models
)

app = FastAPI(
    title="Time Series Forecasting API",
    description="REST API for traffic prediction using Hybrid, XGBoost, and LightGBM models",
    version="1.0.0"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


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
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /forecast/predict": "Forward forecasting",
            "POST /forecast/backtest": "Historical backtesting",
            "GET /metrics/{model_type}/{timeframe}": "Get model metrics",
            "GET /models": "Discover available models",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_dir": MODEL_DIR,
        "model_dir_exists": os.path.exists(MODEL_DIR)
    }


@app.post("/forecast/predict", response_model=ForecastResponse, tags=["Forecasting"])
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
            horizon=request.horizon,
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
        
        dir_path = model_dir if model_dir else MODEL_DIR
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
            horizon=horizon,
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
