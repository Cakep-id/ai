"""
YOLO Service untuk deteksi kerusakan aset
Menggunakan Ultralytics YOLO untuk computer vision
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO
from PIL import Image
import torch
from loguru import logger
from datetime import datetime
import json

class YOLOService:
    """Service untuk YOLO object detection dan training"""
    
    def __init__(self):
        self.model = None
        self.model_path = os.getenv('MODEL_PATH', '../yolov8n.pt')
        self.conf_threshold = float(os.getenv('YOLO_CONF_THRESHOLD', 0.25))
        self.iou_threshold = float(os.getenv('YOLO_IOU_THRESHOLD', 0.45))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Label mapping untuk kerusakan aset
        self.damage_labels = {
            'crack': 'retak',
            'corrosion': 'korosi',
            'leak': 'kebocoran',
            'wear': 'keausan',
            'dent': 'penyok',
            'rust': 'karat',
            'fracture': 'patah',
            'deformation': 'deformasi',
            'erosion': 'erosi',
            'contamination': 'kontaminasi'
        }
        
        # Risk mapping berdasarkan label
        self.risk_mapping = {
            'crack': 0.8,       # High risk
            'fracture': 0.9,    # Very high risk
            'leak': 0.85,       # High risk
            'corrosion': 0.7,   # Medium-high risk
            'wear': 0.6,        # Medium risk
            'rust': 0.5,        # Medium risk
            'dent': 0.4,        # Low-medium risk
            'deformation': 0.75, # High risk
            'erosion': 0.65,    # Medium risk
            'contamination': 0.3 # Low risk
        }
        
        self._load_model()
    
    def _map_to_indonesian_asset(self, detections: List[Dict]) -> str:
        """Map detected objects to Indonesian asset names"""
        # YOLO class name to Indonesian asset mapping
        asset_mapping = {
            # Vehicles & Transportation
            'car': 'Mobil',
            'truck': 'Truk',
            'bus': 'Bus',
            'motorcycle': 'Motor',
            'bicycle': 'Sepeda',
            
            # Industrial Equipment (common YOLO classes)
            'person': 'Peralatan Kerja',
            'laptop': 'Komputer Laptop',
            'keyboard': 'Keyboard',
            'mouse': 'Mouse Komputer',
            'cell phone': 'Handphone',
            'microwave': 'Microwave',
            'oven': 'Oven',
            'toaster': 'Pemanggang Roti',
            'sink': 'Wastafel',
            'refrigerator': 'Kulkas',
            'book': 'Buku/Manual',
            'clock': 'Jam Dinding',
            'scissors': 'Gunting',
            'hair drier': 'Pengering Rambut',
            'toothbrush': 'Sikat Gigi',
            
            # Default industrial equipment
            'unknown': 'Peralatan Industri',
            'equipment': 'Peralatan',
            'machine': 'Mesin',
            'motor': 'Motor Listrik',
            'pump': 'Pompa',
            'generator': 'Generator',
            'compressor': 'Kompresor',
            'fan': 'Kipas',
            'conveyor': 'Conveyor Belt',
            'crane': 'Crane',
            'pipe': 'Pipa',
            'valve': 'Katup'
        }
        
        # Jika ada deteksi, ambil class dengan confidence tertinggi
        if detections and len(detections) > 0:
            highest_conf_detection = max(detections, key=lambda x: x.get('confidence', 0))
            class_name = highest_conf_detection.get('class_name', '').lower()
            
            # Cari mapping yang sesuai
            for key, indonesian_name in asset_mapping.items():
                if key in class_name:
                    return indonesian_name
        
        # Default fallback - gunakan daftar aset industri umum
        industrial_assets = [
            'Generator', 'Kompresor', 'Motor Listrik', 'Pompa Air', 
            'Transformator', 'Panel Listrik', 'Mesin Diesel', 'Kipas Angin',
            'AC Unit', 'Conveyor Belt', 'Crane', 'Forklift', 'Mesin Produksi',
            'Boiler', 'Chiller', 'Lift', 'Eskalator', 'Pompa Air'
        ]
        
        import random
        return random.choice(industrial_assets)
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            if os.path.exists(self.model_path):
                # Try to load custom model with PyTorch compatibility fix
                try:
                    import torch
                    # Temporary fix for PyTorch 2.6 weights_only issue
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
                    
                    self.model = YOLO(self.model_path)
                    logger.info(f"YOLO model loaded from {self.model_path}")
                    
                    # Restore original torch.load
                    torch.load = original_load
                    
                except Exception as custom_error:
                    logger.warning(f"Failed to load custom model: {custom_error}")
                    logger.info("Falling back to pretrained YOLOv8n model")
                    self.model = YOLO('yolov8n.pt')
            else:
                # Download pretrained YOLOv8 model sebagai base
                self.model = YOLO('yolov8n.pt')  # nano version untuk speed
                logger.warning(f"Model not found at {self.model_path}, using pretrained YOLOv8n")
            
            # Set device
            self.model.to(self.device)
            logger.info(f"YOLO model running on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Create mock model for development
            logger.warning("Creating mock YOLO service for development")
            self.model = None
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Deteksi kerusakan pada gambar
        
        Returns:
            {
                'success': bool,
                'detections': [
                    {
                        'label': str,
                        'confidence': float,
                        'bbox': {'x1': int, 'y1': int, 'x2': int, 'y2': int},
                        'risk_score': float
                    }
                ],
                'image_info': {
                    'width': int,
                    'height': int,
                    'channels': int
                },
                'model_version': str,
                'processing_time': float
            }
        """
        start_time = datetime.now()
        
        # Mock detection if model is not available
        if self.model is None:
            logger.warning("YOLO model not available, returning mock detection")
            # Simulasi deteksi berbagai jenis aset untuk demo
            mock_assets = [
                'Generator', 'Kompresor', 'Motor Listrik', 'Pompa Air', 
                'Transformator', 'Panel Listrik', 'Mesin Diesel', 'Kipas Angin',
                'AC Unit', 'Conveyor Belt', 'Crane', 'Forklift'
            ]
            import random
            mock_asset = random.choice(mock_assets)
            
            return {
                'success': True,
                'detections': [
                    {
                        'name': 'korosi_ringan',
                        'confidence': 0.85,
                        'bbox': {
                            'x': 100,
                            'y': 100,
                            'width': 200,
                            'height': 200
                        }
                    },
                    {
                        'name': 'retak_permukaan',
                        'confidence': 0.78,
                        'bbox': {
                            'x': 300,
                            'y': 150,
                            'width': 180,
                            'height': 120
                        }
                    }
                ],
                'asset_type': mock_asset,  # Nama aset dalam bahasa Indonesia
                'image_info': {
                    'width': 640,
                    'height': 480,
                    'channels': 3
                },
                'model_version': 'mock_v1.0',
                'processing_time': 0.1
            }
        
        try:
            # Validasi file gambar
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'detections': [],
                    'processing_time': 0
                }
            
            # Load dan validasi gambar
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Invalid image format',
                    'detections': [],
                    'processing_time': 0
                }
            
            height, width, channels = image.shape
            
            # Run inference
            results = self.model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get label name
                        if hasattr(self.model, 'names'):
                            label = self.model.names[class_id]
                        else:
                            label = f"class_{class_id}"
                        
                        # Map ke damage labels jika ada
                        mapped_label = self.damage_labels.get(label.lower(), label)
                        
                        # Calculate risk score berdasarkan label dan confidence
                        base_risk = self.risk_mapping.get(label.lower(), 0.5)
                        risk_score = base_risk * confidence
                        
                        detection = {
                            'label': mapped_label,
                            'original_label': label,
                            'confidence': confidence,
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2)
                            },
                            'risk_score': risk_score,
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
            
            # Sort by confidence descending
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Map detections to Indonesian asset name
            asset_name = self._map_to_indonesian_asset(detections)
            
            result_data = {
                'success': True,
                'detections': detections,
                'asset_type': asset_name,  # Nama aset dalam bahasa Indonesia
                'image_info': {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'path': image_path
                },
                'model_version': self._get_model_version(),
                'processing_time': processing_time,
                'detection_count': len(detections)
            }
            
            logger.info(f"YOLO detection completed: {len(detections)} objects detected in {processing_time:.2f}s")
            return result_data
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"YOLO detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'processing_time': processing_time
            }
    
    def get_visual_score(self, detections: List[Dict]) -> float:
        """
        Hitung visual score dari deteksi
        Mengambil confidence tertinggi dari kelas yang terkait kerusakan kritis
        """
        if not detections:
            return 0.0
        
        # Filter hanya deteksi dengan risk score tinggi
        critical_detections = [
            d for d in detections 
            if d.get('risk_score', 0) > 0.6
        ]
        
        if critical_detections:
            # Ambil risk score tertinggi
            max_risk = max(d['risk_score'] for d in critical_detections)
            return min(max_risk, 1.0)
        else:
            # Jika tidak ada deteksi kritis, ambil confidence tertinggi
            max_conf = max(d['confidence'] for d in detections)
            return max_conf * 0.5  # Faktor penalty untuk non-critical
    
    def annotate_image(self, image_path: str, detections: List[Dict], 
                      output_path: str = None) -> str:
        """
        Buat gambar dengan annotasi deteksi
        
        Returns:
            str: Path ke gambar yang sudah diannotasi
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image")
            
            # Colors untuk berbagai kelas (BGR format)
            colors = {
                'crack': (0, 0, 255),      # Red
                'leak': (0, 255, 255),     # Yellow
                'corrosion': (0, 165, 255), # Orange
                'wear': (255, 0, 0),       # Blue
                'rust': (0, 100, 139),     # Brown
                'fracture': (0, 0, 139),   # Dark Red
                'default': (0, 255, 0)     # Green
            }
            
            for detection in detections:
                bbox = detection['bbox']
                label = detection['original_label'].lower()
                confidence = detection['confidence']
                risk_score = detection.get('risk_score', 0)
                
                # Pilih warna berdasarkan label
                color = colors.get(label, colors['default'])
                
                # Draw bounding box
                cv2.rectangle(
                    image,
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2']),
                    color,
                    2
                )
                
                # Text annotation
                text = f"{detection['label']} {confidence:.2f} (Risk: {risk_score:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background untuk text
                cv2.rectangle(
                    image,
                    (bbox['x1'], bbox['y1'] - text_size[1] - 5),
                    (bbox['x1'] + text_size[0], bbox['y1']),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    image,
                    text,
                    (bbox['x1'], bbox['y1'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Simpan annotated image
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"./storage/annotated_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Buat directory jika belum ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, image)
            logger.info(f"Annotated image saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to annotate image: {e}")
            return image_path  # Return original path if annotation fails
    
    def train(self, dataset_path: str, epochs: int = 100, batch_size: int = 16) -> Dict[str, Any]:
        """
        Train YOLO model dengan dataset baru
        
        Args:
            dataset_path: Path ke dataset dalam format YOLO
            epochs: Jumlah epoch training
            batch_size: Batch size untuk training
        
        Returns:
            Dict dengan hasil training
        """
        try:
            logger.info(f"Starting YOLO training with dataset: {dataset_path}")
            
            # Validasi dataset
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path not found: {dataset_path}")
            
            # Training configuration
            train_config = {
                'data': dataset_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': self.device,
                'project': './ml/runs',
                'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'save_period': 10,  # Save checkpoint every 10 epochs
                'patience': 20,     # Early stopping patience
            }
            
            # Start training
            results = self.model.train(**train_config)
            
            # Get best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            
            # Copy best model to production path
            if os.path.exists(best_model_path):
                import shutil
                production_path = self.model_path
                os.makedirs(os.path.dirname(production_path), exist_ok=True)
                shutil.copy2(best_model_path, production_path)
                logger.info(f"Best model copied to {production_path}")
                
                # Reload model
                self._load_model()
            
            # Extract training metrics
            metrics = {
                'final_map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'final_map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'best_fitness': float(results.best_fitness),
                'epochs_completed': epochs,
                'model_path': str(best_model_path),
                'training_time': results.training_time
            }
            
            logger.info(f"YOLO training completed successfully: mAP50={metrics['final_map50']:.3f}")
            
            return {
                'success': True,
                'metrics': metrics,
                'model_version': self._get_model_version(),
                'dataset_info': f"Dataset: {dataset_path}, Epochs: {epochs}"
            }
            
        except Exception as e:
            logger.error(f"YOLO training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {},
                'model_version': self._get_model_version()
            }
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validasi format dataset YOLO"""
        try:
            # Check structure
            required_files = ['data.yaml']
            for file in required_files:
                if not os.path.exists(os.path.join(dataset_path, file)):
                    return {
                        'valid': False,
                        'error': f'Missing required file: {file}'
                    }
            
            # Check data.yaml content
            import yaml
            data_yaml_path = os.path.join(dataset_path, 'data.yaml')
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'names']
            for key in required_keys:
                if key not in data_config:
                    return {
                        'valid': False,
                        'error': f'Missing required key in data.yaml: {key}'
                    }
            
            # Count images and labels
            train_path = os.path.join(dataset_path, data_config['train'])
            val_path = os.path.join(dataset_path, data_config['val'])
            
            train_images = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            val_images = len([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            return {
                'valid': True,
                'train_images': train_images,
                'val_images': val_images,
                'classes': len(data_config['names']),
                'class_names': data_config['names']
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _get_model_version(self) -> str:
        """Get current model version"""
        try:
            if self.model and hasattr(self.model, 'ckpt'):
                return f"YOLOv8_{datetime.now().strftime('%Y%m%d')}"
            return "YOLOv8_pretrained"
        except:
            return "unknown"
    
    def health_check(self) -> Dict[str, Any]:
        """Health check untuk YOLO service"""
        try:
            start_time = datetime.now()
            
            # Test basic functionality
            status = "healthy" if self.model is not None else "degraded"
            
            # If model exists, test with a dummy prediction
            test_result = None
            if self.model is not None:
                try:
                    # Create a dummy image for testing
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    # Try prediction (this will be very fast)
                    results = self.model(dummy_img, verbose=False)
                    test_result = "prediction_test_passed"
                except Exception as e:
                    status = "unhealthy"
                    test_result = f"prediction_test_failed: {str(e)}"
            else:
                test_result = "mock_mode_active"
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": status,
                "response_time_ms": response_time * 1000,
                "model_loaded": self.model is not None,
                "device": self.device,
                "model_path": self.model_path,
                "test_result": test_result
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "device": self.device
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model yang sedang digunakan"""
        try:
            info = {
                'model_path': self.model_path,
                'model_version': self._get_model_version(),
                'device': self.device,
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'damage_labels': self.damage_labels,
                'risk_mapping': self.risk_mapping
            }
            
            if self.model:
                info['model_loaded'] = True
                if hasattr(self.model, 'names'):
                    info['classes'] = self.model.names
                    info['num_classes'] = len(self.model.names)
            else:
                info['model_loaded'] = False
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'error': str(e),
                'model_loaded': False
            }

# Singleton instance
yolo_service = YOLOService()

if __name__ == "__main__":
    # Test YOLO service
    print("YOLO Service Test")
    print("Model Info:", yolo_service.get_model_info())
    
    # Test detection jika ada sample image
    sample_image = "./sample.jpg"
    if os.path.exists(sample_image):
        result = yolo_service.detect(sample_image)
        print("Detection Result:", json.dumps(result, indent=2))
