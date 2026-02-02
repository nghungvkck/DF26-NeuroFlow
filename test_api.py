#!/usr/bin/env python3
"""
Test script for FastAPI autoscaling recommendation endpoint.
Run this AFTER starting the API server with: python api_server.py
"""

import requests
import json
from typing import Dict


def test_api(endpoint: str = "http://localhost:8000") -> None:
    """Test the autoscaling recommendation API."""
    
    print("\n" + "="*70)
    print("AUTOSCALING RECOMMENDATION API - TEST SUITE")
    print("="*70)
    
    # Test 1: Health check
    print("\n[TEST 1] Health Check")
    print("-" * 70)
    try:
        response = requests.get(f"{endpoint}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Start it with: python api_server.py")
        return
    
    # Test 2: Scale-up scenario (high load)
    print("\n[TEST 2] Scale-Up Scenario (High Load)")
    print("-" * 70)
    request_data = {
        "current_pods": 3,
        "requests": 15000,
        "forecast": 18000,
        "capacity_per_pod": 5000
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{endpoint}/recommend-scaling",
            json=request_data,
            timeout=5
        )
        
        if response.status_code == 200:
            rec = response.json()
            print(f"\n✅ Recommendation received")
            print(f"   Current pods: {rec['current_pods']}")
            print(f"   Recommended pods: {rec['recommended_pods']}")
            print(f"   Action: {rec['action'].upper()}")
            print(f"   Confidence: {rec['confidence']:.0%}")
            print(f"\n   Decision Layers:")
            for i, reason in enumerate(rec['reasons'], 1):
                print(f"   {i}. {reason['factor']}")
                print(f"      → {reason['decision']}")
            print(f"\n   Cost Impact:")
            cost = rec['estimated_cost_impact']
            print(f"   Current: ${cost['current_hourly_cost']:.4f}/hr")
            print(f"   New: ${cost['new_hourly_cost']:.4f}/hr")
            print(f"   Change: {cost['cost_change_percent']:+.1f}%")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 3: Scale-down scenario (low load)
    print("\n[TEST 3] Scale-Down Scenario (Low Load)")
    print("-" * 70)
    request_data = {
        "current_pods": 10,
        "requests": 3000,
        "forecast": 3500,
        "capacity_per_pod": 5000
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{endpoint}/recommend-scaling",
            json=request_data,
            timeout=5
        )
        
        if response.status_code == 200:
            rec = response.json()
            print(f"\n✅ Recommendation received")
            print(f"   Current pods: {rec['current_pods']}")
            print(f"   Recommended pods: {rec['recommended_pods']}")
            print(f"   Action: {rec['action'].upper()}")
            print(f"   Confidence: {rec['confidence']:.0%}")
            print(f"\n   Decision Layers:")
            for i, reason in enumerate(rec['reasons'], 1):
                print(f"   {i}. {reason['factor']}")
                print(f"      → {reason['decision']}")
            print(f"\n   Cost Impact:")
            cost = rec['estimated_cost_impact']
            print(f"   Current: ${cost['current_hourly_cost']:.4f}/hr")
            print(f"   New: ${cost['new_hourly_cost']:.4f}/hr")
            print(f"   Change: {cost['cost_change_percent']:+.1f}%")
        else:
            print(f"❌ Failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 4: Anomaly/Spike scenario
    print("\n[TEST 4] Anomaly Detection (Spike)")
    print("-" * 70)
    request_data = {
        "current_pods": 3,
        "requests": 10000,
        "forecast": 4000,  # Huge deviation = anomaly
        "capacity_per_pod": 5000
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{endpoint}/recommend-scaling",
            json=request_data,
            timeout=5
        )
        
        if response.status_code == 200:
            rec = response.json()
            print(f"\n✅ Recommendation received")
            print(f"   Current pods: {rec['current_pods']}")
            print(f"   Recommended pods: {rec['recommended_pods']}")
            print(f"   Action: {rec['action'].upper()}")
            print(f"   Confidence: {rec['confidence']:.0%}")
            print(f"\n   Decision Layers:")
            for i, reason in enumerate(rec['reasons'], 1):
                print(f"   {i}. {reason['factor']}")
                print(f"      → {reason['decision']}")
            print(f"\n   Cost Impact:")
            cost = rec['estimated_cost_impact']
            print(f"   Current: ${cost['current_hourly_cost']:.4f}/hr")
            print(f"   New: ${cost['new_hourly_cost']:.4f}/hr")
            print(f"   Change: {cost['cost_change_percent']:+.1f}%")
        else:
            print(f"❌ Failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 5: Stable/No-change scenario
    print("\n[TEST 5] Stable State (No Change)")
    print("-" * 70)
    request_data = {
        "current_pods": 4,
        "requests": 12000,
        "forecast": 12500,
        "capacity_per_pod": 5000
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{endpoint}/recommend-scaling",
            json=request_data,
            timeout=5
        )
        
        if response.status_code == 200:
            rec = response.json()
            print(f"\n✅ Recommendation received")
            print(f"   Current pods: {rec['current_pods']}")
            print(f"   Recommended pods: {rec['recommended_pods']}")
            print(f"   Action: {rec['action'].upper()}")
            print(f"   Confidence: {rec['confidence']:.0%}")
            print(f"\n   Decision Layers:")
            for i, reason in enumerate(rec['reasons'], 1):
                print(f"   {i}. {reason['factor']}")
                print(f"      → {reason['decision']}")
            print(f"\n   Cost Impact:")
            cost = rec['estimated_cost_impact']
            print(f"   Current: ${cost['current_hourly_cost']:.4f}/hr")
            print(f"   New: ${cost['new_hourly_cost']:.4f}/hr")
            print(f"   Change: {cost['cost_change_percent']:+.1f}%")
        else:
            print(f"❌ Failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    print("\nTo run the dashboard, use:")
    print("  streamlit run dashboard_demo.py")
    print("\nFor more info, see: DEMO_QUICKSTART.md")
    print()


if __name__ == "__main__":
    import sys
    
    # Optional: custom endpoint
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    test_api(endpoint)
