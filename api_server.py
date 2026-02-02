#!/usr/bin/env python3
"""
FASTAPI AUTOSCALING RECOMMENDATION SERVER
===========================================
Provides /recommend-scaling endpoint with detailed explanations.

Features:
- Analyzes current load vs capacity
- Makes scaling recommendations
- Explains reasoning in detail
- Returns confidence score
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import json
from datetime import datetime


# ============================================================================
# DATA MODELS
# ============================================================================

class ScalingRequest(BaseModel):
    """Request for scaling recommendation."""
    current_pods: int
    requests: int
    forecast: int
    capacity_per_pod: int = 250


class ScalingReason(BaseModel):
    """Individual reason for scaling."""
    factor: str
    current_value: str
    threshold: str
    decision: str  # positive, negative, or neutral


class ScalingRecommendation(BaseModel):
    """Recommendation response."""
    current_pods: int
    recommended_pods: int
    action: str  # scale-up, scale-down, no-change
    reasons: List[ScalingReason]
    confidence: float  # 0-1
    explanation: str
    estimated_cost_impact: Dict[str, float]


# ============================================================================
# HYBRID AUTOSCALING LOGIC
# ============================================================================

class HybridAutoscalerAnalyzer:
    """Analyzes autoscaling decisions with detailed explanations."""
    
    def __init__(self):
        self.min_pods = 2
        self.max_pods = 20
        self.capacity_per_pod = 250
        
        # Thresholds
        self.sla_threshold = 0.95        # 95% utilization = SLA breach
        self.slo_threshold = 0.85        # 85% utilization = SLO warning
        self.target_utilization = 0.70   # 70% = optimal
        
        # Scaling parameters
        self.scale_up_margin = 0.80      # Scale up if utilization > 80%
        self.scale_down_margin = 0.30    # Scale down if utilization < 30%
        self.scale_up_step = 1            # Step size for scaling
    
    def recommend(self, 
                  current_pods: int, 
                  requests: int, 
                  forecast: int, 
                  capacity_per_pod: int) -> ScalingRecommendation:
        """
        Generate scaling recommendation with full explanation.
        
        Args:
            current_pods: Current number of pods
            requests: Current incoming requests
            forecast: Forecasted requests (from ML model)
            capacity_per_pod: Requests per pod per time unit
        
        Returns:
            ScalingRecommendation with detailed reasoning
        """
        self.capacity_per_pod = capacity_per_pod
        
        # ========== ANALYSIS LAYER 0: ANOMALY DETECTION ==========
        anomaly_detected = self._detect_anomaly(requests, forecast)
        
        # ========== ANALYSIS LAYER 1: EMERGENCY DETECTION ==========
        current_utilization = requests / (current_pods * capacity_per_pod)
        sla_breach = current_utilization > self.sla_threshold
        
        # ========== ANALYSIS LAYER 2: PREDICTIVE SCALING ==========
        # Use forecast to predict future utilization
        forecast_pods_needed = self._pods_for_load(forecast, capacity_per_pod)
        
        # ========== ANALYSIS LAYER 3: REACTIVE SCALING ==========
        # Simple rule: scale if utilization out of band
        
        # ========== DECISION LOGIC ==========
        reasons = []
        recommended_pods = current_pods
        confidence = 0.5
        
        # Layer 0: Anomaly
        if anomaly_detected:
            reasons.append(ScalingReason(
                factor="Anomaly Detection",
                current_value=f"{requests} requests",
                threshold=f"±{int(forecast*0.2)} requests from forecast",
                decision="ALERT: Spike detected, prepare to scale"
            ))
            confidence += 0.1
        
        # Layer 1: Emergency
        if sla_breach:
            scale_up_amount = max(1, current_pods // 2)
            recommended_pods = min(self.max_pods, current_pods + scale_up_amount)
            
            reasons.append(ScalingReason(
                factor="Emergency (SLA Breach)",
                current_value=f"{current_utilization:.1%} utilization",
                threshold=f"< {self.sla_threshold:.0%}",
                decision=f"CRITICAL: Scale UP {scale_up_amount} pods immediately"
            ))
            confidence = 0.95
        
        # Layer 2: Predictive
        elif forecast_pods_needed > current_pods + 1:
            scale_up_amount = forecast_pods_needed - current_pods
            recommended_pods = min(self.max_pods, current_pods + scale_up_amount)
            
            reasons.append(ScalingReason(
                factor="Predictive Scaling",
                current_value=f"Forecast: {forecast} requests",
                threshold=f"Current capacity: {current_pods * capacity_per_pod}",
                decision=f"Proactive: Scale UP {scale_up_amount} pods (prepare for spike)"
            ))
            confidence = 0.8
        
        # Layer 3: Reactive
        else:
            current_util = requests / (current_pods * capacity_per_pod)
            
            if current_util > self.scale_up_margin:
                scale_up_amount = 1
                recommended_pods = min(self.max_pods, current_pods + scale_up_amount)
                
                reasons.append(ScalingReason(
                    factor="Reactive (High Load)",
                    current_value=f"{current_util:.1%} utilization",
                    threshold=f"> {self.scale_up_margin:.0%}",
                    decision=f"Scale UP {scale_up_amount} pod (responding to load)"
                ))
                confidence = 0.7
            
            elif current_util < self.scale_down_margin and current_pods > self.min_pods:
                scale_down_amount = 1
                recommended_pods = max(self.min_pods, current_pods - scale_down_amount)
                
                reasons.append(ScalingReason(
                    factor="Reactive (Low Load)",
                    current_value=f"{current_util:.1%} utilization",
                    threshold=f"< {self.scale_down_margin:.0%}",
                    decision=f"Scale DOWN {scale_down_amount} pod (save cost)"
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
        
        # ========== DETERMINE ACTION ==========
        if recommended_pods > current_pods:
            action = "scale-up"
        elif recommended_pods < current_pods:
            action = "scale-down"
        else:
            action = "no-change"
        
        # ========== CALCULATE COST IMPACT ==========
        current_hourly_cost = self._calculate_hourly_cost(current_pods, requests)
        new_hourly_cost = self._calculate_hourly_cost(recommended_pods, requests)
        
        cost_impact = {
            'current_hourly_cost': float(current_hourly_cost),
            'new_hourly_cost': float(new_hourly_cost),
            'cost_difference': float(new_hourly_cost - current_hourly_cost),
            'cost_change_percent': float((new_hourly_cost - current_hourly_cost) / current_hourly_cost * 100),
        }
        
        # ========== GENERATE EXPLANATION ==========
        explanation = self._generate_explanation(
            action, current_pods, recommended_pods, 
            current_utilization, reasons
        )
        
        return ScalingRecommendation(
            current_pods=current_pods,
            recommended_pods=recommended_pods,
            action=action,
            reasons=reasons,
            confidence=confidence,
            explanation=explanation,
            estimated_cost_impact=cost_impact
        )
    
    def _detect_anomaly(self, requests: int, forecast: int) -> bool:
        """Detect if requests are anomalous compared to forecast."""
        if forecast == 0:
            return False
        
        deviation = abs(requests - forecast) / forecast
        return deviation > 0.5  # >50% deviation is anomaly
    
    def _pods_for_load(self, load: int, capacity: int) -> int:
        """Calculate pods needed for given load."""
        pods_needed = load / capacity
        return max(self.min_pods, min(self.max_pods, int(pods_needed) + 1))
    
    def _calculate_hourly_cost(self, pods: int, current_requests: int) -> float:
        """Calculate hourly cost for pod configuration."""
        # Simplified cost model
        # Reserved baseline (2 pods @ $0.03/hour) + burst cost
        
        step_hours = 1.0  # 1 hour
        
        # Reserved capacity (2 pods)
        reserved_pods = min(pods, 2)
        cost_reserved = reserved_pods * 0.03 * step_hours
        
        # Burst capacity (>2 pods)
        if pods > 2:
            burst_pods = pods - 2
            # 70% spot + 30% on-demand
            cost_spot = burst_pods * 0.7 * 0.015 * step_hours
            cost_ondemand = burst_pods * 0.3 * 0.05 * step_hours
            return cost_reserved + cost_spot + cost_ondemand
        
        return cost_reserved
    
    def _generate_explanation(self, 
                             action: str, 
                             current_pods: int, 
                             recommended_pods: int,
                             current_util: float,
                             reasons: List[ScalingReason]) -> str:
        """Generate human-readable explanation."""
        
        if action == "scale-up":
            reason_texts = [r.decision for r in reasons if "UP" in r.decision]
            reason_text = reason_texts[0] if reason_texts else "Load increasing"
            
            return f"""
RECOMMENDATION: SCALE UP from {current_pods} to {recommended_pods} pods

REASON: {reason_text}

CURRENT STATE:
- Current pods: {current_pods}
- Requests: {current_util:.1%} of capacity
- Utilization: {current_util:.1%}

EXPECTED OUTCOME:
- New capacity: {recommended_pods * self.capacity_per_pod:,} requests
- New utilization: {current_util * current_pods / recommended_pods:.1%}
- Better SLA compliance and lower latency
- Slightly higher cost but improved user experience

ACTION: Add {recommended_pods - current_pods} pod(s) and monitor
            """
        
        elif action == "scale-down":
            reason_texts = [r.decision for r in reasons if "DOWN" in r.decision]
            reason_text = reason_texts[0] if reason_texts else "Load decreasing"
            
            return f"""
RECOMMENDATION: SCALE DOWN from {current_pods} to {recommended_pods} pods

REASON: {reason_text}

CURRENT STATE:
- Current pods: {current_pods}
- Requests: {current_util:.1%} of capacity
- Utilization: Very low (cost optimization opportunity)

EXPECTED OUTCOME:
- Lower infrastructure cost
- Remaining {recommended_pods} pods still sufficient
- Minimal impact on SLA (utilization still healthy)

ACTION: Remove {current_pods - recommended_pods} pod(s) to save cost
            """
        
        else:  # no-change
            return f"""
RECOMMENDATION: NO CHANGE - Keep {current_pods} pods

REASON: System is operating within optimal range

CURRENT STATE:
- Current pods: {current_pods}
- Utilization: {current_util:.1%}
- Status: Stable and efficient

EXPECTED OUTCOME:
- Maintain current performance
- Optimal cost-efficiency
- No action needed at this time

ACTION: Continue monitoring
            """


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Autoscaling Recommendation API",
    description="HYBRID autoscaler decision engine with explanations",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = HybridAutoscalerAnalyzer()


@app.get("/")
async def root():
    """API documentation."""
    return {
        "service": "Autoscaling Recommendation Engine",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/recommend-scaling",
                "method": "POST",
                "description": "Get scaling recommendation with detailed explanation"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check"
            }
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/recommend-scaling", response_model=ScalingRecommendation)
async def recommend_scaling(request: ScalingRequest) -> ScalingRecommendation:
    """
    Get autoscaling recommendation with detailed explanation.
    
    Args:
        request: ScalingRequest with current state
    
    Returns:
        ScalingRecommendation with reasons and explanation
    
    Example:
        POST /recommend-scaling
        {
            "current_pods": 3,
            "requests": 15000,
            "forecast": 18000,
            "capacity_per_pod": 5000
        }
    """
    try:
        recommendation = analyzer.recommend(
            current_pods=request.current_pods,
            requests=request.requests,
            forecast=request.forecast,
            capacity_per_pod=request.capacity_per_pod
        )
        
        return recommendation
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  AUTOSCALING RECOMMENDATION API                           ║
    ║  Listening on http://localhost:8000                       ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Endpoints:
    - GET  http://localhost:8000/              (Documentation)
    - GET  http://localhost:8000/health        (Health check)
    - POST http://localhost:8000/recommend-scaling (Main endpoint)
    
    Try with Streamlit dashboard:
    - streamlit run dashboard_demo.py
    
    Or test with curl:
    - curl -X POST http://localhost:8000/recommend-scaling \\
           -H "Content-Type: application/json" \\
           -d '{"current_pods": 3, "requests": 15000, "forecast": 18000}'
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
