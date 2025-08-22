#!/usr/bin/env python3
"""
Quick test untuk user form API
"""
import requests
import json

def test_user_form_simple():
    """Test user form dengan data minimal"""
    url = "http://localhost:8000/api/user/report-damage"
    
    try:
        # Test dengan data sederhana tanpa file upload dulu
        response = requests.get("http://localhost:8000/api/user/assets")
        print(f"Assets endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Assets endpoint works!")
            print(f"Response: {response.json()}")
        else:
            print("❌ Assets endpoint failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_user_form_simple()
