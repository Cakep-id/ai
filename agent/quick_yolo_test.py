#!/usr/bin/env python3
"""
Test dynamic YOLO detection
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from services.yolo_service import YOLOService
    
    print("Creating YOLO service...")
    yolo = YOLOService()
    
    print("Testing with different image names:")
    
    # Test different images to verify dynamic behavior
    test_cases = [
        "sample_low.jpg",
        "sample_medium.jpg", 
        "sample_high.jpg",
        "clean_pipe.jpg"
    ]
    
    for test_image in test_cases:
        result = yolo.detect(test_image)
        severity = result.get('severity', 'UNKNOWN')
        detections = len(result.get('detections', []))
        asset_type = result.get('asset_type', 'Unknown')
        
        print(f"{test_image}: {severity} severity, {detections} detections, asset: {asset_type}")
        
        if detections > 0:
            for det in result.get('detections', []):
                confidence = det.get('confidence', 0)
                name = det.get('name', 'unknown')
                bbox = det.get('bbox', {})
                area = bbox.get('width', 0) * bbox.get('height', 0)
                print(f"  - {name}: {confidence:.3f} confidence, area: {area} pixels")
    
    print("\nDynamic detection test completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
