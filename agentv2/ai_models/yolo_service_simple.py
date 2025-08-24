"""
Simple YOLO Service for testing without heavy dependencies
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

class YOLOService:
    """Simple YOLO service for damage detection"""
    
    def __init__(self):
        """Initialize YOLO service"""
        self.model = None
        self.class_names = [
            "crack", "corrosion", "deformation", "hole", 
            "paint_loss", "rust", "scratch", "wear"
        ]
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Try to load model (mock for now)
        self.model_loaded = self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLO model"""
        try:
            # For testing, we'll mock the model
            print("Mock YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            return False
    
    async def detect_damage(self, image_path: str) -> Dict[str, Any]:
        """Detect damage in image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Mock detections for testing
            detections = self._mock_detections(width, height)
            
            # Create annotated image (simple for now)
            annotated_image = self._create_annotated_image(image, detections)
            
            # Save annotated image
            output_path = image_path.replace('foto_mentah', 'foto_yolo').replace('.', '_yolo.')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            
            return {
                "success": True,
                "detections": detections,
                "annotated_image_path": output_path,
                "image_dimensions": {"width": width, "height": height},
                "processing_time": "0.1s",
                "model_version": "mock_v1.0"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }
    
    def _mock_detections(self, width: int, height: int) -> List[Dict[str, Any]]:
        """Generate mock detections for testing"""
        detections = []
        
        # Add a few mock detections
        mock_boxes = [
            [50, 50, 150, 120, "crack", 0.85],
            [200, 100, 280, 180, "corrosion", 0.72],
            [300, 200, 380, 250, "rust", 0.68]
        ]
        
        for i, (x1, y1, x2, y2, class_name, conf) in enumerate(mock_boxes):
            # Ensure coordinates are within image bounds
            x1 = min(max(0, x1), width)
            y1 = min(max(0, y1), height)
            x2 = min(max(0, x2), width)
            y2 = min(max(0, y2), height)
            
            if x2 > x1 and y2 > y1:  # Valid box
                area = (x2 - x1) * (y2 - y1)
                area_percentage = (area / (width * height)) * 100
                
                detection = {
                    "id": i,
                    "class_id": self.class_names.index(class_name) if class_name in self.class_names else 0,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                    "area_percentage": round(area_percentage, 2),
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)
        
        return detections
    
    def _create_annotated_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Create annotated image with bounding boxes"""
        annotated = image.copy()
        
        colors = {
            "crack": (0, 0, 255),      # Red
            "corrosion": (0, 165, 255), # Orange
            "deformation": (255, 0, 255), # Magenta
            "hole": (0, 0, 0),         # Black
            "paint_loss": (0, 255, 255), # Yellow
            "rust": (19, 69, 139),     # Brown
            "scratch": (255, 255, 0),   # Cyan
            "wear": (128, 128, 128)     # Gray
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_loaded": self.model_loaded,
            "model_type": "YOLOv8 (Mock)",
            "classes": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }

# For testing
if __name__ == "__main__":
    service = YOLOService()
    print("YOLO Service initialized")
    print(json.dumps(service.get_model_info(), indent=2))
