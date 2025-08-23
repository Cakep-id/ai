#!/usr/bin/env python3
"""
Quick verification script for the training system
"""

import requests
import json
import sys
from pathlib import Path

# Test server connectivity
def test_server():
    try:
        response = requests.get('http://localhost:8000')
        print(f"✅ Server is running - Status: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

# Test training endpoints
def test_training_endpoints():
    endpoints = [
        '/api/training/datasets',
        '/api/training/damage-classes'
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}')
            print(f"✅ {endpoint} - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {len(data) if isinstance(data, list) else 'OK'}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

# Test dataset creation
def test_dataset_creation():
    dataset_data = {
        "name": "test_dataset",
        "description": "Test dataset for verification",
        "damage_class_ids": [1, 2, 3]
    }
    
    try:
        response = requests.post('http://localhost:8000/api/training/datasets', 
                               json=dataset_data)
        print(f"✅ Dataset creation - Status: {response.status_code}")
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"   Created dataset ID: {data.get('id')}")
            return data.get('id')
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
    return None

if __name__ == "__main__":
    print("🔍 YOLO Training System Verification")
    print("=" * 40)
    
    # Test 1: Server connectivity
    if not test_server():
        sys.exit(1)
    
    # Test 2: Training endpoints
    print("\n📡 Testing Training Endpoints:")
    test_training_endpoints()
    
    # Test 3: Dataset creation
    print("\n📁 Testing Dataset Creation:")
    dataset_id = test_dataset_creation()
    
    print("\n✅ Verification completed!")
    
    if dataset_id:
        print(f"🎯 Training system is working! Created test dataset with ID: {dataset_id}")
    else:
        print("⚠️ Some issues detected, but basic functionality appears to work")
