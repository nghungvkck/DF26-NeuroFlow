# Issues Found & Fixes

## 🔴 CRITICAL ISSUES

### Issue 1: Inconsistent SLA Calculation Logic
**Severity:** HIGH  
**Location:** `simulate.py` (line 155-160), `cost/metrics.py`  
**Problem:**
```python
# Current logic:
sla_breached_before_scaling = actual_requests > current_pods * capacity_per_pod
# This checks BEFORE scaling but:
# - Metric later recorded with pods AFTER scaling
# - SLA violation calculated on POST-scaled pods
# - Inconsistency in what "SLA before" means

# Example:
# t=0: requests=300, pods=2, capacity=100
#   - SLA breach BEFORE: 300 > 200 ✓ YES
#   - Action: scale to 4 pods
#   - SLA breach AFTER: 300 > 400 ✗ NO
# → Record says "sla_breached_before=True" but "pods=4"
```

**Impact:** Metrics may be confusing - storing which SLA state?

**Fix:** ✅ Already partial fix exists with `sla_before_scaling` flag, but needs clarification in comments

---

### Issue 2: Forecast Error Rate Calculation Error
**Severity:** HIGH  
**Location:** `autoscaling/hybrid.py` (line 95)  
**Problem:**
```python
# Current code:
errors = np.array(self.recent_errors[-self.forecast_error_window:])
mape = np.mean(np.abs(errors))  # ← ERROR!
# mape should be: np.mean(np.abs(errors / actual))
# Not just absolute errors!

# Example:
# forecast_error = [10, -5, 8, -12, 5]
# Current MAPE = mean([10, 5, 8, 12, 5]) = 8.0 (nonsensical)
# Correct MAPE = mean([10/100, 5/95, 8/108, ...]) = ~0.08 (8%)

# This causes wrong reliability scoring!
```

**Impact:** Forecast reliability scored incorrectly → wrong strategy selection in Hybrid

**Fix:** Calculate actual MAPE (Mean Absolute Percentage Error):
```python
def _forecast_reliability(self):
    if len(self.recent_errors) < 5:
        return 0.5
    
    errors = np.array(self.recent_errors[-self.forecast_error_window:])
    traffic = np.array(self.recent_traffic[-self.forecast_error_window:])
    
    # Avoid division by zero
    valid_idx = traffic > 0
    if not np.any(valid_idx):
        return 0.5
    
    # Calculate percentage errors only where traffic > 0
    pct_errors = np.abs(errors[valid_idx] / traffic[valid_idx])
    mape = np.mean(pct_errors)
    
    # Convert MAPE to reliability (0.2 = 20% error → 60% reliable)
    reliability = max(0, min(1, 1 - mape))  # NOT: 1 - mape * 2
    return reliability
```

---

### Issue 3: Missing Forecast Error Storage
**Severity:** HIGH  
**Location:** `simulate.py`, `autoscaling/hybrid.py`  
**Problem:**
```python
# In simulate.py, we calculate forecast_error:
forecast_error = actual_forecast - actual_requests

# But we NEVER pass it to the hybrid autoscaler!
# So hybrid.py never updates recent_errors

# Current code in simulate.py:
new_pods, action, reason = autoscaler.step(
    current_servers=current_pods,
    requests=actual_requests,
    forecast_requests=actual_forecast,  # ← Only forecast, no error
)

# Hybrid.step() doesn't receive forecast_error
# So recent_errors stays empty → reliability always 0.5 → no learning!
```

**Impact:** Hybrid policy can't learn if forecasts are improving → always uses default reliability

**Fix:** Pass forecast error and update hybrid state:
```python
# In simulate.py:
forecast_error = actual_forecast - actual_requests
new_pods, action, reason = autoscaler.step(
    current_servers=current_pods,
    requests=actual_requests,
    forecast_requests=actual_forecast,
    forecast_error=forecast_error  # ← ADD THIS
)

# In autoscaling/hybrid.py step():
def step(self, current_servers, requests, forecast_requests=None, forecast_error=None):
    # Update forecast error tracking
    if forecast_error is not None:
        self.recent_errors.append(forecast_error)
        if len(self.recent_errors) > self.forecast_error_window:
            self.recent_errors.pop(0)
    
    # ... rest of method
```

---

### Issue 4: Capacity Per Pod May Be Unrealistic
**Severity:** MEDIUM  
**Location:** `simulate.py` (line 26)  
**Problem:**
```python
capacity_per_pod=100,  # requests/s per pod
# But test data max load is ~300 requests
# So minimum pods needed = 300/100 = 3 pods

# Real data from data/real/*.csv likely has different distribution
# If actual max load is 1000 requests but capacity is 100:
# → Need 10 pods minimum
# → SLA violations will be frequent

# Current hardcoding doesn't match data characteristics!
```

**Impact:** SLA metrics may be unrealistic if capacity doesn't match actual data range

**Fix:** Auto-calibrate capacity based on data:
```python
def infer_capacity_from_data(load_series, target_utilization=0.85):
    """Infer reasonable capacity_per_pod from data."""
    max_load = load_series['requests_count'].max()
    p99_load = load_series['requests_count'].quantile(0.99)
    
    # Use p99 as target capacity to avoid frequent SLA
    # With 5 pods baseline:
    suggested_capacity = p99_load / 5 / target_utilization
    return suggested_capacity

# Then use it:
suggested_capacity = infer_capacity_from_data(load_series)
# Validate against min/max reasonable values
capacity_per_pod = max(50, min(200, suggested_capacity))
```

---

### Issue 5: Anomaly Detection Threshold Too High
**Severity:** MEDIUM  
**Location:** `simulate.py` (line 53), `autoscaling/hysteresis.py`  
**Problem:**
```python
anomaly_detector = AnomalyDetector(
    window_size=50,
    zscore_threshold=3.0,  # ← Standard 3σ = 99.7% confidence
    iqr_multiplier=1.5,
    rate_threshold=0.5
)

# With zscore_threshold=3.0:
# - Requires change > 3 standard deviations
# - Only catches extreme anomalies
# - Misses moderate attacks (e.g., 30% DDoS)

# Better: use 2.0 for sensitivity or adaptive
```

**Impact:** Anomaly detection misses moderate issues

**Fix:** Make threshold adaptive:
```python
class AnomalyDetector:
    def __init__(self, window_size=50, zscore_threshold=2.5, ...):
        # Use 2.5σ instead of 3.0σ for better detection
        # 2.5σ = 98.8% confidence (99% false positive rate → 1% FP)
        self.zscore_threshold = zscore_threshold
        
    def detect(self, value, adaptive=True):
        if adaptive and len(self.history) > 10:
            # Adjust threshold based on recent volatility
            recent_std = np.std(self.history[-20:])
            dynamic_threshold = 2.0 * recent_std  # 2σ
        else:
            dynamic_threshold = self.zscore_threshold
        
        # ... rest
```

---

### Issue 6: Cooldown Logic in Reactive vs Hybrid
**Severity:** MEDIUM  
**Location:** `autoscaling/reactive.py`, `autoscaling/hybrid.py`, `autoscaling/predictive.py`  
**Problem:**
```python
# Reactive:
if self.cooldown_timer > 0:
    self.cooldown_timer -= 1
    return current_servers, utilization, action=0

# Hybrid:
if self.cooldown_timer > 0:
    self.cooldown_timer -= 1
    return current_servers, action=0, reason

# Both decrement cooldown BEFORE checking actions
# But should decrement AFTER!
# This means cooldown is effectively OFF by 1 step

# Example:
# t=0: scale action → cooldown_timer = 5
# t=1: cooldown_timer=4, but check happens → action allowed! ✗
# (Should block until cooldown_timer reaches 0)
```

**Impact:** Cooldown is 1 step shorter than intended

**Fix:** Decrement at END of step:
```python
def step(self, current_servers, requests):
    utilization = requests / (current_servers * self.C)
    action = 0
    
    # Check first (before decrement)
    if self.cooldown_timer > 0:
        self.cooldown_timer -= 1
        return current_servers, utilization, action=0
    
    # ... make decision
    
    # Reset cooldown AFTER decision
    if action != 0:
        self.cooldown_timer = self.cooldown
    
    return current_servers, utilization, action
```

---

### Issue 7: Predictive Safety Margin Not Applied Consistently
**Severity:** LOW  
**Location:** `autoscaling/predictive.py`, `autoscaling/hybrid.py`  
**Problem:**
```python
# Predictive:
def required_servers(self, forecast_requests):
    return int(np.ceil(forecast_requests / (self.C * self.alpha)))
    # Uses alpha = safety_margin (default 0.8)

# Hybrid:
required_servers = int(np.ceil(
    forecast_requests / (self.C * self.forecast_safety_margin)
))
# Uses forecast_safety_margin (default 0.85)

# INCONSISTENCY!
# Predictive: 0.8 → 20% headroom
# Hybrid: 0.85 → 15% headroom
# Different risk profiles!
```

**Impact:** Inconsistent scaling behavior between strategies

**Fix:** Use same safety margin:
```python
# Define constant:
FORECAST_SAFETY_MARGIN = 0.80  # Standard 20% headroom

# Both use:
required = int(np.ceil(forecast / (capacity * FORECAST_SAFETY_MARGIN)))
```

---

## 🟡 MEDIUM ISSUES

### Issue 8: Missing Validation of Forecast Result
**Severity:** LOW  
**Location:** `simulate.py` (line 143-151)  
**Problem:**
```python
try:
    forecast_result = forecaster.predict(history, horizon=forecast_horizon)
    actual_forecast = forecast_result.yhat[0] if forecast_result.yhat else actual_requests
except Exception as e:
    print(f"  [Warning: Forecast failed at t={t}: {e}]")
    actual_forecast = actual_requests

# Issues:
# 1. forecast_result.yhat could be empty list → AttributeError
# 2. No validation that yhat[0] is a number
# 3. Silent fallback hides forecast failures
# 4. Using actual_requests as fallback is not conservative!
```

**Impact:** May use invalid forecasts if model returns garbage

**Fix:**
```python
def get_forecast(forecaster, history, horizon=1, fallback_conservative=True):
    try:
        result = forecaster.predict(history, horizon=horizon)
        if result and result.yhat and len(result.yhat) > 0:
            forecast = float(result.yhat[0])
            if forecast > 0:  # Sanity check
                return forecast
    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
    
    # Fallback: use conservative estimate (higher → scale up)
    if fallback_conservative:
        return history['requests_count'].iloc[-1] * 1.1  # +10% margin
    return history['requests_count'].iloc[-1]
```

---

### Issue 9: Min/Max Servers Not Validated Against Load
**Severity:** LOW  
**Location:** All autoscalers (`reactive.py`, `predictive.py`, etc.)  
**Problem:**
```python
def __init__(self, capacity_per_server, min_servers=2, max_servers=20):
    # What if:
    # - max_load = 250 requests
    # - capacity = 100 per pod
    # - max_servers = 2
    # → Can never handle load! (2×100 < 250)

    # Or:
    # - min_load = 10 requests
    # - capacity = 100 per pod
    # - min_servers = 5
    # → Always over-provisioned!
```

**Impact:** Fixed min/max may not match actual load distribution

**Fix:** Validate in constructor:
```python
def __init__(self, capacity_per_server, min_servers=2, max_servers=20,
             expected_min_load=None, expected_max_load=None):
    # Validate min_servers can handle expected max
    if expected_max_load:
        required_for_max = np.ceil(expected_max_load / capacity_per_server)
        if max_servers < required_for_max:
            logger.warning(f"max_servers={max_servers} insufficient for "
                         f"max_load={expected_max_load}")
```

---

## 🟢 MINOR ISSUES

### Issue 10: No Logging of Strategy Changes
**Severity:** VERY LOW  
**Location:** `simulate.py`, `autoscaling/hybrid.py`  
**Problem:**
```python
# When strategies make decisions, reasons are logged but:
# - No systematic logging
# - Reasons stored in local variable
# - Hard to debug strategy behavior

# Example: Hybrid might switch between Emergency/Predictive/Reactive
# But these switches aren't tracked
```

**Impact:** Difficult to understand strategy evolution

**Fix:** Add decision logging:
```python
def step(self, ...):
    decision_log = {
        'timestamp': t,
        'strategy': 'HYBRID',
        'layer_triggered': None,  # EMERGENCY, PREDICTIVE, REACTIVE
        'reason': '',
        'pods_before': current_pods,
        'pods_after': new_pods,
        'confidence': {}
    }
    
    # Track which layer made decision
    if emergency_triggered:
        decision_log['layer_triggered'] = 'EMERGENCY'
    elif predictive_triggered:
        decision_log['layer_triggered'] = 'PREDICTIVE'
        decision_log['confidence'] = {'reliability': reliability}
    
    return new_pods, action, decision_log
```

---

## 📋 Summary of Fixes Needed

| Issue | Severity | Fix Type | Complexity |
|-------|----------|----------|-----------|
| SLA Calculation Inconsistency | HIGH | Clarify Logic | Low |
| MAPE Calculation Error | **CRITICAL** | Math Fix | Low |
| Missing Forecast Error Tracking | **CRITICAL** | Add Parameter | Low |
| Capacity Calibration | MEDIUM | Add Function | Medium |
| Anomaly Threshold | MEDIUM | Add Config | Low |
| Cooldown Off-by-One | MEDIUM | Logic Fix | Low |
| Safety Margin Inconsistency | LOW | Standardize | Low |
| Forecast Validation | LOW | Add Checks | Low |
| Min/Max Validation | LOW | Add Checks | Low |
| Decision Logging | VERY LOW | Add Tracking | Low |

---

## 🛠️ Recommended Fix Order

1. **FIRST:** Issue 3 (Missing Forecast Error) - breaks learning
2. **SECOND:** Issue 2 (MAPE Calculation) - wrong reliability scoring
3. **THIRD:** Issue 6 (Cooldown Logic) - off-by-one error
4. **FOURTH:** Issue 4 (Capacity Calibration) - metrics realism
5. **FIFTH:** Issue 5 (Anomaly Threshold) - detection sensitivity
6. **OTHERS:** Lower priority issues

---

**Last Updated:** February 2, 2026
