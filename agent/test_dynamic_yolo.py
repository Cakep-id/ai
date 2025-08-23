#!/usr/bin/env python3
"""
Test script untuk verifikasi dynamic YOLO mock detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.yolo_service import YOLOService

def test_dynamic_detection():
    """Test that YOLO mock detection returns different results for different images"""
    
    yolo_service = YOLOService()
    
    # Test with different "image paths" to see if we get different results
    test_images = [
        "test_low_sample.jpg",
        "test_medium_sample.jpg", 
        "test_high_sample.jpg",
        "test_clean_sample.jpg",
        "test_pipeline_1.jpg",
        "test_pipeline_2.jpg",
        "test_pipeline_3.jpg",
        "test_equipment_1.jpg",
        "test_equipment_2.jpg",
        "test_machinery_1.jpg"
    ]
    
    print("=== Testing Dynamic YOLO Mock Detection ===\n")
    
    results = []
    for image_path in test_images:
        print(f"Testing: {image_path}")
        result = yolo_service.detect(image_path)
        
        if result['success']:
            detections = result.get('detections', [])
            print(f"  Asset Type: {result.get('asset_type', 'Unknown')}")
            print(f"  Severity: {result.get('severity', 'Unknown')}")
            print(f"  Detections: {len(detections)}")
            
            for i, detection in enumerate(detections):
                print(f"    [{i+1}] {detection['name']} - Confidence: {detection['confidence']:.3f}")
                bbox = detection['bbox']
                print(f"        Location: ({bbox['x']}, {bbox['y']}) Size: {bbox['width']}x{bbox['height']}")
            
            # Calculate total area for this "image"
            total_area = sum(d['bbox']['width'] * d['bbox']['height'] for d in detections)
            img_width = result['image_info']['width']
            img_height = result['image_info']['height']
            area_percent = (total_area / (img_width * img_height)) * 100
            print(f"  Total Area: {total_area} pixels ({area_percent:.2f}% of image)")
            
            results.append({
                'image': image_path,
                'detections': len(detections),
                'severity': result.get('severity'),
                'area_percent': area_percent,
                'asset_type': result.get('asset_type')
            })
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
            
        print("-" * 50)
    
    # Summary
    print("\n=== SUMMARY ===")
    print("Image\t\t\tDetections\tSeverity\tArea%\t\tAsset Type")
    print("-" * 80)
    for r in results:
        print(f"{r['image']:<20}\t{r['detections']}\t\t{r['severity']:<8}\t{r['area_percent']:.1f}%\t\t{r['asset_type']}")
    
    # Check for variety
    severities = set(r['severity'] for r in results)
    detection_counts = set(r['detections'] for r in results)
    asset_types = set(r['asset_type'] for r in results)
    
    print(f"\nVariety Check:")
    print(f"- Different severities: {len(severities)} ({list(severities)})")
    print(f"- Different detection counts: {len(detection_counts)} ({list(detection_counts)})")
    print(f"- Different asset types: {len(asset_types)} types")
    
    if len(severities) > 1 and len(detection_counts) > 1:
        print("✅ SUCCESS: Dynamic detection is working - getting varied results!")
    else:
        print("❌ ISSUE: Still getting too similar results")

if __name__ == "__main__":
    test_dynamic_detection()
