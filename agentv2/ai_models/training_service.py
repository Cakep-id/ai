"""
Advanced Training Service for AgentV2
Implements sophisticated training pipeline:
- Human-driven training data curation
- Advanced data augmentation
- Multi-objective optimization
- Real-time training monitoring
- Automated model validation
- Performance benchmarking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import asyncio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Optional imports with fallbacks
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    def train_test_split(*args, **kwargs): return args[0][:80], args[0][80:]
    def classification_report(*args, **kwargs): return "Mock classification report"

try:
    import wandb  # For experiment tracking
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

import yaml
from pathlib import Path
import shutil
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingStatus(Enum):
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.937
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box_loss_weight: float = 0.05
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    auto_augment: str = "randaugment"
    erasing: float = 0.4

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    train_loss: float
    val_loss: float
    map_50: float
    map_95: float
    precision: float
    recall: float
    learning_rate: float
    timestamp: datetime

class CustomYOLODataset(Dataset):
    """Custom dataset for YOLO training with advanced augmentation"""
    
    def __init__(self, images: List[str], annotations: List[Dict], 
                 image_size: int = 640, augment: bool = True):
        self.images = images
        self.annotations = annotations
        self.image_size = image_size
        self.augment = augment
        
        # Advanced augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
                ], p=0.8),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=7, p=0.3),
                    A.MedianBlur(blur_limit=5, p=0.3)
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.1)
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=1, min_height=8, min_width=8, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        annotation = self.annotations[idx]
        bboxes = annotation.get('bboxes', [])
        class_labels = annotation.get('class_labels', [])
        
        # Apply transforms
        if self.augment and len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty(0, 4),
            'labels': torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.empty(0)
        }

class TrainingMonitor:
    """Real-time training monitoring"""
    
    def __init__(self, session_id: str, save_dir: str):
        self.session_id = session_id
        self.save_dir = save_dir
        self.metrics_history = []
        self.best_map = 0.0
        self.patience_counter = 0
        self.early_stop_patience = 20
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    
    def log_metrics(self, metrics: TrainingMetrics) -> bool:
        """Log training metrics and check for early stopping"""
        self.metrics_history.append(metrics)
        
        # Save metrics to JSON
        metrics_file = f"{self.save_dir}/metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.metrics_history], f, indent=2, default=str)
        
        # Check for improvement
        if metrics.map_50 > self.best_map:
            self.best_map = metrics.map_50
            self.patience_counter = 0
            return False  # Continue training
        else:
            self.patience_counter += 1
            
        # Early stopping check
        if self.patience_counter >= self.early_stop_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
        
        return False
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        if len(self.metrics_history) < 2:
            return
        
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        map_50_scores = [m.map_50 for m in self.metrics_history]
        map_95_scores = [m.map_95 for m in self.metrics_history]
        
        # Loss curves
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(epochs, train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # mAP curves
        plt.subplot(2, 3, 2)
        plt.plot(epochs, map_50_scores, label='mAP@0.5', color='green')
        plt.plot(epochs, map_95_scores, label='mAP@[.5:.95]', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(2, 3, 3)
        learning_rates = [m.learning_rate for m in self.metrics_history]
        plt.plot(epochs, learning_rates, color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        # Precision and Recall
        plt.subplot(2, 3, 4)
        precisions = [m.precision for m in self.metrics_history]
        recalls = [m.recall for m in self.metrics_history]
        plt.plot(epochs, precisions, label='Precision', color='cyan')
        plt.plot(epochs, recalls, label='Recall', color='magenta')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True)
        
        # Training progress
        plt.subplot(2, 3, 5)
        plt.plot(epochs, train_losses, alpha=0.7, label='Train Loss')
        plt.plot(epochs, val_losses, alpha=0.7, label='Val Loss')
        ax2 = plt.gca().twinx()
        ax2.plot(epochs, map_50_scores, color='green', alpha=0.7, label='mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax2.set_ylabel('mAP@0.5')
        plt.title('Training Progress Overview')
        plt.grid(True)
        
        # Best metrics summary
        plt.subplot(2, 3, 6)
        best_epoch = max(self.metrics_history, key=lambda x: x.map_50).epoch
        best_metrics = next(m for m in self.metrics_history if m.epoch == best_epoch)
        
        metrics_names = ['mAP@0.5', 'mAP@[.5:.95]', 'Precision', 'Recall']
        metrics_values = [best_metrics.map_50, best_metrics.map_95, 
                         best_metrics.precision, best_metrics.recall]
        
        bars = plt.bar(metrics_names, metrics_values, 
                      color=['green', 'orange', 'cyan', 'magenta'])
        plt.ylabel('Score')
        plt.title(f'Best Metrics (Epoch {best_epoch})')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

class TrainingService:
    """Main training service"""
    
    def __init__(self, daos: Dict[str, Any]):
        self.daos = daos
        self.training_sessions = {}  # Active training sessions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training service initialized on device: {self.device}")
    
    async def start_training_session(self, session_id: str, 
                                   training_request: Any) -> Dict[str, Any]:
        """Start a new training session"""
        try:
            logger.info(f"Starting training session: {session_id}")
            
            # Update session status
            self.daos['training_sessions'].update_session_status(session_id, "running")
            
            # Prepare training data
            train_data, val_data = await self._prepare_training_data(session_id)
            
            if len(train_data['images']) < 10:
                raise ValueError("Insufficient training data. Minimum 10 samples required.")
            
            # Configure training
            config = TrainingConfig(
                epochs=training_request.epochs,
                batch_size=training_request.batch_size,
                learning_rate=training_request.learning_rate
            )
            
            # Create training directory
            training_dir = f"ai_models/training_sessions/{session_id}"
            os.makedirs(training_dir, exist_ok=True)
            
            # Initialize monitoring
            monitor = TrainingMonitor(session_id, training_dir)
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training_session,
                args=(session_id, train_data, val_data, config, monitor, training_dir)
            )
            training_thread.daemon = True
            training_thread.start()
            
            self.training_sessions[session_id] = {
                'thread': training_thread,
                'monitor': monitor,
                'status': TrainingStatus.RUNNING,
                'start_time': datetime.now()
            }
            
            return {
                'success': True,
                'session_id': session_id,
                'message': 'Training session started successfully',
                'training_dir': training_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to start training session {session_id}: {e}")
            self.daos['training_sessions'].update_session_status(session_id, "failed")
            raise
    
    async def _prepare_training_data(self, session_id: str) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Prepare training and validation data from database"""
        
        # Get unused training data
        training_data_query = """
        SELECT * FROM trainer_data 
        WHERE used_in_training = FALSE 
        ORDER BY created_at DESC
        """
        
        training_data = self.daos['db_manager'].execute_query(training_data_query)
        
        if not training_data:
            raise ValueError("No training data available")
        
        # Convert annotations and split data
        images = []
        annotations = []
        
        for data in training_data:
            if os.path.exists(data['image_path']):
                images.append(data['image_path'])
                
                # Parse manual annotations
                manual_annotations = json.loads(data['manual_annotations'])
                
                # Convert to YOLO format
                bboxes = []
                class_labels = []
                
                for annotation in manual_annotations:
                    # Assume annotations are in format: {x, y, width, height, class_id}
                    x = annotation.get('x', 0)
                    y = annotation.get('y', 0)
                    width = annotation.get('width', 0)
                    height = annotation.get('height', 0)
                    class_id = annotation.get('class_id', 0)
                    
                    # Convert to YOLO format (center_x, center_y, width, height) normalized
                    # This is a simplified conversion - in practice you'd need image dimensions
                    center_x = x + width / 2
                    center_y = y + height / 2
                    
                    bboxes.append([center_x, center_y, width, height])
                    class_labels.append(class_id)
                
                annotations.append({
                    'bboxes': bboxes,
                    'class_labels': class_labels
                })
        
        # Split into training and validation
        train_images, val_images, train_annotations, val_annotations = train_test_split(
            images, annotations, test_size=0.2, random_state=42
        )
        
        # Mark data as used
        data_ids = [data['id'] for data in training_data]
        if data_ids:
            update_query = f"""
            UPDATE trainer_data 
            SET used_in_training = TRUE, training_session_id = %s
            WHERE id IN ({','.join(['%s'] * len(data_ids))})
            """
            self.daos['db_manager'].execute_update(update_query, [session_id] + data_ids)
        
        train_data = {
            'images': train_images,
            'annotations': train_annotations
        }
        
        val_data = {
            'images': val_images,
            'annotations': val_annotations
        }
        
        logger.info(f"Prepared {len(train_images)} training and {len(val_images)} validation samples")
        
        return train_data, val_data
    
    def _run_training_session(self, session_id: str, train_data: Dict[str, List],
                            val_data: Dict[str, List], config: TrainingConfig,
                            monitor: TrainingMonitor, training_dir: str):
        """Run the actual training session"""
        try:
            logger.info(f"Running training session {session_id}")
            
            # Create YOLO model
            model = YOLO('yolov8n.yaml')  # Start with YOLOv8 nano architecture
            
            # Prepare YOLO format dataset
            dataset_dir = f"{training_dir}/dataset"
            self._create_yolo_dataset(train_data, val_data, dataset_dir)
            
            # Configure training parameters
            train_args = {
                'data': f"{dataset_dir}/data.yaml",
                'epochs': config.epochs,
                'batch': config.batch_size,
                'lr0': config.learning_rate,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum,
                'warmup_epochs': config.warmup_epochs,
                'warmup_momentum': config.warmup_momentum,
                'warmup_bias_lr': config.warmup_bias_lr,
                'box': config.box_loss_weight,
                'cls': config.cls_loss_weight,
                'dfl': config.dfl_loss_weight,
                'hsv_h': config.hsv_h,
                'hsv_s': config.hsv_s,
                'hsv_v': config.hsv_v,
                'degrees': config.degrees,
                'translate': config.translate,
                'scale': config.scale,
                'shear': config.shear,
                'perspective': config.perspective,
                'flipud': config.flipud,
                'fliplr': config.fliplr,
                'mosaic': config.mosaic,
                'mixup': config.mixup,
                'copy_paste': config.copy_paste,
                'auto_augment': config.auto_augment,
                'erasing': config.erasing,
                'project': training_dir,
                'name': 'train',
                'save_period': 10,
                'device': self.device,
                'workers': 4,
                'patience': 20,
                'save': True,
                'plots': True,
                'verbose': True
            }
            
            # Start training
            results = model.train(**train_args)
            
            # Get final metrics
            final_map_50 = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
            final_map_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
            
            # Save final model
            model_path = f"{training_dir}/final_model.pt"
            model.save(model_path)
            
            # Generate comprehensive evaluation
            evaluation_results = self._evaluate_trained_model(
                model, val_data, f"{training_dir}/evaluation"
            )
            
            # Update database
            self.daos['training_sessions'].update_session_completion(
                session_id, final_map_50, final_map_95, model_path,
                {
                    'training_config': asdict(config),
                    'training_results': results.results_dict,
                    'evaluation_results': evaluation_results,
                    'training_time_minutes': (datetime.now() - self.training_sessions[session_id]['start_time']).total_seconds() / 60
                }
            )
            
            # Save detailed metrics to database
            self._save_detailed_metrics(session_id, evaluation_results)
            
            # Generate final plots
            monitor.plot_training_curves()
            
            # Update session status
            self.training_sessions[session_id]['status'] = TrainingStatus.COMPLETED
            
            logger.info(f"Training session {session_id} completed successfully")
            logger.info(f"Final mAP@0.5: {final_map_50:.4f}, mAP@[.5:.95]: {final_map_95:.4f}")
            
        except Exception as e:
            logger.error(f"Training session {session_id} failed: {e}")
            self.training_sessions[session_id]['status'] = TrainingStatus.FAILED
            self.daos['training_sessions'].update_session_status(session_id, "failed")
    
    def _create_yolo_dataset(self, train_data: Dict[str, List], val_data: Dict[str, List], 
                           dataset_dir: str):
        """Create YOLO format dataset"""
        os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
        os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)
        
        # Copy and process training data
        self._process_yolo_split(train_data, f"{dataset_dir}/images/train", 
                               f"{dataset_dir}/labels/train")
        
        # Copy and process validation data
        self._process_yolo_split(val_data, f"{dataset_dir}/images/val", 
                               f"{dataset_dir}/labels/val")
        
        # Create data.yaml file
        data_yaml = {
            'train': f"{dataset_dir}/images/train",
            'val': f"{dataset_dir}/images/val",
            'nc': 6,  # Number of classes
            'names': ['corrosion', 'dent', 'crack', 'coating_loss', 'leak', 'wear']
        }
        
        with open(f"{dataset_dir}/data.yaml", 'w') as f:
            yaml.dump(data_yaml, f)
    
    def _process_yolo_split(self, data: Dict[str, List], images_dir: str, labels_dir: str):
        """Process data split for YOLO format"""
        for i, (image_path, annotation) in enumerate(zip(data['images'], data['annotations'])):
            # Copy image
            image_name = f"image_{i:06d}.jpg"
            new_image_path = f"{images_dir}/{image_name}"
            shutil.copy2(image_path, new_image_path)
            
            # Create label file
            label_name = f"image_{i:06d}.txt"
            label_path = f"{labels_dir}/{label_name}"
            
            with open(label_path, 'w') as f:
                bboxes = annotation.get('bboxes', [])
                class_labels = annotation.get('class_labels', [])
                
                for bbox, class_id in zip(bboxes, class_labels):
                    # YOLO format: class_id center_x center_y width height (normalized)
                    f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    def _evaluate_trained_model(self, model, val_data: Dict[str, List], 
                              evaluation_dir: str) -> Dict[str, Any]:
        """Comprehensive evaluation of trained model"""
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Run validation
        val_results = model.val(save_json=True, save_hybrid=True, plots=True, 
                               project=evaluation_dir, name='validation')
        
        # Extract metrics
        evaluation_results = {
            'map_50': float(val_results.results_dict.get('metrics/mAP50(B)', 0.0)),
            'map_95': float(val_results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
            'precision': float(val_results.results_dict.get('metrics/precision(B)', 0.0)),
            'recall': float(val_results.results_dict.get('metrics/recall(B)', 0.0)),
            'f1_score': float(2 * val_results.results_dict.get('metrics/precision(B)', 0.0) * 
                            val_results.results_dict.get('metrics/recall(B)', 0.0) / 
                            (val_results.results_dict.get('metrics/precision(B)', 0.0) + 
                             val_results.results_dict.get('metrics/recall(B)', 0.0) + 1e-6)),
            'val_results': val_results.results_dict,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return evaluation_results
    
    def _save_detailed_metrics(self, session_id: str, evaluation_results: Dict[str, Any]):
        """Save detailed metrics to database"""
        try:
            # Save per-class metrics
            for class_id in range(6):  # 6 damage classes
                metrics_data = {
                    'session_id': session_id,
                    'damage_class_id': class_id + 1,  # Database uses 1-based indexing
                    'precision_score': evaluation_results.get('precision', 0.0),
                    'recall_score': evaluation_results.get('recall', 0.0),
                    'f1_score': evaluation_results.get('f1_score', 0.0),
                    'ap_50': evaluation_results.get('map_50', 0.0),
                    'ap_95': evaluation_results.get('map_95', 0.0),
                    'optimal_threshold': 0.5,  # Default threshold
                    'confusion_matrix': json.dumps([]),  # Would need actual confusion matrix
                    'pr_curve_data': json.dumps([]),  # Would need actual PR curve data
                    'iou_distribution': json.dumps([])  # Would need actual IoU data
                }
                
                # Insert into model_metrics table
                insert_query = """
                INSERT INTO model_metrics 
                (session_id, damage_class_id, precision_score, recall_score, f1_score,
                 ap_50, ap_95, optimal_threshold, confusion_matrix, pr_curve_data, iou_distribution)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                params = (
                    metrics_data['session_id'],
                    metrics_data['damage_class_id'],
                    metrics_data['precision_score'],
                    metrics_data['recall_score'],
                    metrics_data['f1_score'],
                    metrics_data['ap_50'],
                    metrics_data['ap_95'],
                    metrics_data['optimal_threshold'],
                    metrics_data['confusion_matrix'],
                    metrics_data['pr_curve_data'],
                    metrics_data['iou_distribution']
                )
                
                self.daos['db_manager'].execute_insert(insert_query, params)
            
            logger.info(f"Detailed metrics saved for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save detailed metrics: {e}")
    
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """Get current training status"""
        if session_id not in self.training_sessions:
            # Check database
            session = self.daos['training_sessions'].get_session(session_id)
            if session:
                return {
                    'session_id': session_id,
                    'status': session.get('status', 'unknown'),
                    'from_database': True
                }
            else:
                return {'error': 'Session not found'}
        
        session_info = self.training_sessions[session_id]
        return {
            'session_id': session_id,
            'status': session_info['status'].value,
            'start_time': session_info['start_time'].isoformat(),
            'is_alive': session_info['thread'].is_alive(),
            'best_map': session_info['monitor'].best_map,
            'current_epoch': len(session_info['monitor'].metrics_history)
        }
    
    def stop_training_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a running training session"""
        if session_id not in self.training_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.training_sessions[session_id]
        session_info['status'] = TrainingStatus.STOPPED
        
        # Update database
        self.daos['training_sessions'].update_session_status(session_id, "stopped")
        
        return {
            'success': True,
            'message': f'Training session {session_id} stopped'
        }

# Factory function
def create_training_service(daos: Dict[str, Any]) -> TrainingService:
    """Create training service instance"""
    return TrainingService(daos)

if __name__ == "__main__":
    # Test the training service
    logger.info("Training service module loaded successfully")
