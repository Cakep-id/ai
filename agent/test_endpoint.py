#!/usr/bin/env python3
"""
Test script untuk test endpoint pipeline inspections
"""

import requests
import json

def test_pipeline_endpoint():
    """Test pipeline inspections endpoint"""
    print("🔧 Testing pipeline inspections endpoint...")
    
    try:
        # Test endpoint
        url = "http://localhost:8000/api/pipeline/inspections?limit=3"
        print(f"Making request to: {url}")
        
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2)[:500]}...")
        else:
            print(f"❌ Error Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_pipeline_endpoint()
