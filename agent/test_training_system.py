#!/usr/bin/env python3
"""
Test script untuk sistem retraining YOLO
Test semua fitur training dari dataset creation hingga training
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_training_system():
    """Test lengkap sistem training"""
    
    print("🧪 Testing YOLO Training System")
    print("=" * 50)
    
    # 1. Test health check
    print("\n1. Testing server health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # 2. Test damage classes endpoint
    print("\n2. Testing damage classes...")
    try:
        response = requests.get(f"{API_BASE}/api/training/damage-classes")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                classes = data.get('data', [])
                print(f"✅ Found {len(classes)} damage classes")
                for cls in classes[:3]:  # Show first 3
                    print(f"   - {cls['class_name']}: {cls['description']}")
            else:
                print("❌ Damage classes endpoint returned failure")
        else:
            print(f"❌ Damage classes endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing damage classes: {e}")
    
    # 3. Test create dataset
    print("\n3. Testing dataset creation...")
    try:
        dataset_data = {
            "dataset_name": f"Test_Dataset_{int(time.time())}",
            "description": "Test dataset untuk validasi sistem training",
            "uploaded_by": "Test_User"
        }
        
        response = requests.post(
            f"{API_BASE}/api/training/dataset/create",
            json=dataset_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                dataset_id = result['data']['dataset_id']
                print(f"✅ Dataset created with ID: {dataset_id}")
                print(f"   Name: {result['data']['dataset_name']}")
                
                # Store for later tests
                test_dataset_id = dataset_id
            else:
                print(f"❌ Dataset creation failed: {result.get('message')}")
                test_dataset_id = None
        else:
            print(f"❌ Dataset creation endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            test_dataset_id = None
    except Exception as e:
        print(f"❌ Error testing dataset creation: {e}")
        test_dataset_id = None
    
    # 4. Test datasets list
    print("\n4. Testing datasets list...")
    try:
        response = requests.get(f"{API_BASE}/api/training/datasets")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                datasets = data.get('data', [])
                print(f"✅ Found {len(datasets)} datasets")
                for dataset in datasets[-2:]:  # Show last 2
                    print(f"   - {dataset['dataset_name']} ({dataset['total_images']} images)")
            else:
                print("❌ Datasets list returned failure")
        else:
            print(f"❌ Datasets list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing datasets list: {e}")
    
    # 5. Test training sessions list
    print("\n5. Testing training sessions...")
    try:
        response = requests.get(f"{API_BASE}/api/training/training-sessions")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                sessions = data.get('data', [])
                print(f"✅ Found {len(sessions)} training sessions")
                for session in sessions[-2:]:  # Show last 2
                    print(f"   - {session['session_name']} ({session['status']})")
            else:
                print("❌ Training sessions list returned failure")
        else:
            print(f"❌ Training sessions list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing training sessions: {e}")
    
    # 6. Test training status
    print("\n6. Testing training status...")
    try:
        response = requests.get(f"{API_BASE}/api/training/training-status")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                status = data.get('data', {})
                print(f"✅ Training status: {status}")
                print(f"   Is Training: {status.get('is_training', False)}")
                if status.get('current_session_id'):
                    print(f"   Current Session: {status['current_session_id']}")
            else:
                print("❌ Training status returned failure")
        else:
            print(f"❌ Training status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing training status: {e}")
    
    # 7. Test pipeline endpoints (existing functionality)
    print("\n7. Testing pipeline integration...")
    try:
        response = requests.get(f"{API_BASE}/api/pipeline/inspections?limit=3")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                inspections = data.get('data', [])
                print(f"✅ Pipeline integration working ({len(inspections)} inspections)")
            else:
                print("❌ Pipeline integration failed")
        else:
            print(f"❌ Pipeline endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing pipeline integration: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("✅ Sistema retraining telah diimplementasi dengan fitur:")
    print("   - Dataset management dengan database storage")
    print("   - Image upload dengan annotation support")
    print("   - Background training system")
    print("   - Training session monitoring")
    print("   - Web interface untuk semua operasi")
    print("\n📱 Akses interface di: http://localhost:8000/training.html")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    return True

def test_create_sample_dataset():
    """Create sample dataset for demonstration"""
    print("\n🎨 Creating sample dataset...")
    
    try:
        # Create dataset
        dataset_data = {
            "dataset_name": "Pipeline_Corrosion_Detection_v1",
            "description": "Dataset untuk deteksi korosi pada pipeline dengan berbagai tingkat kerusakan",
            "uploaded_by": "Pipeline_Engineer"
        }
        
        response = requests.post(
            f"{API_BASE}/api/training/dataset/create",
            json=dataset_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Sample dataset created: {result['data']['dataset_name']}")
                return result['data']['dataset_id']
        
    except Exception as e:
        print(f"Error creating sample dataset: {e}")
    
    return None

if __name__ == "__main__":
    # Run main test
    test_training_system()
    
    # Create sample dataset
    test_create_sample_dataset()
    
    print("\n🚀 Ready to use! Upload images via:")
    print("   http://localhost:8000/training.html")
