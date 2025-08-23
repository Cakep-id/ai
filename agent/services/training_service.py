"""
YOLO Training Service untuk Background Training Process
Mengelola training YOLO model dengan dataset dari database
"""

import os
import json
import shutil
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from services.db import db_service
from services.dataset_service import dataset_service


class YOLOTrainingService:
    """Service untuk background training YOLO model"""
    
    def __init__(self):
        self.training_dir = Path("ml/training")
        self.models_dir = Path("ml/models")
        self.is_training = False
        self.current_session_id = None
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Setup folder structure untuk training"""
        dirs = [
            self.training_dir / "datasets",
            self.training_dir / "runs",
            self.training_dir / "configs",
            self.models_dir / "trained",
            self.models_dir / "checkpoints"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def start_training_session(self, session_id: int) -> Dict:
        """
        Mulai training session secara asynchronous
        
        Args:
            session_id: ID training session
            
        Returns:
            Dict dengan status training
        """
        try:
            if self.is_training:
                return {
                    'success': False,
                    'error': 'Training sedang berjalan, tunggu hingga selesai'
                }
            
            # Get session info
            session = db_service.fetch_one(
                """
                SELECT s.*, d.dataset_name, d.dataset_id
                FROM yolo_training_sessions s
                JOIN yolo_training_datasets d ON s.dataset_id = d.dataset_id
                WHERE s.session_id = %s
                """,
                (session_id,)
            )
            
            if not session:
                return {
                    'success': False,
                    'error': 'Training session tidak ditemukan'
                }
            
            self.current_session_id = session_id
            self.is_training = True
            
            # Update status to training
            db_service.execute_query(
                "UPDATE yolo_training_sessions SET status = 'training' WHERE session_id = %s",
                (session_id,)
            )
            
            # Start training in background
            asyncio.create_task(self._run_training_process(session))
            
            logger.info(f"Training session {session_id} started")
            
            return {
                'success': True,
                'message': 'Training dimulai dalam background',
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error starting training session: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _run_training_process(self, session_info):
        """
        Proses training utama yang berjalan di background
        """
        session_id = session_info[0]
        dataset_id = session_info[1]
        session_name = session_info[2]
        epochs = session_info[6]
        batch_size = session_info[7]
        learning_rate = session_info[8]
        model_architecture = session_info[9]
        
        try:
            logger.info(f"Starting training process for session {session_id}")
            
            # 1. Prepare dataset
            logger.info("Preparing dataset...")
            dataset_result = await self._prepare_dataset_for_training(dataset_id, session_id)
            if not dataset_result['success']:
                raise Exception(f"Dataset preparation failed: {dataset_result['error']}")
            
            # 2. Create YOLO config
            logger.info("Creating YOLO config...")
            config_result = await self._create_yolo_config(dataset_result['dataset_path'], model_architecture)
            if not config_result['success']:
                raise Exception(f"Config creation failed: {config_result['error']}")
            
            # 3. Start YOLO training (simulated for now)
            logger.info("Starting YOLO training...")
            training_result = await self._simulate_yolo_training(
                config_result['config_path'],
                epochs,
                batch_size,
                learning_rate,
                session_id
            )
            
            if training_result['success']:
                # Update session with results
                db_service.execute_query(
                    """
                    UPDATE yolo_training_sessions 
                    SET status = 'completed', end_time = %s, 
                        training_accuracy = %s, validation_accuracy = %s,
                        model_save_path = %s
                    WHERE session_id = %s
                    """,
                    (
                        datetime.now(),
                        training_result['training_accuracy'],
                        training_result['validation_accuracy'],
                        training_result['model_path'],
                        session_id
                    )
                )
                
                logger.info(f"Training session {session_id} completed successfully")
            else:
                raise Exception(training_result.get('error', 'Training failed'))
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training session {session_id} failed: {error_msg}")
            
            # Update session with error
            db_service.execute_query(
                """
                UPDATE yolo_training_sessions 
                SET status = 'failed', end_time = %s, error_message = %s
                WHERE session_id = %s
                """,
                (datetime.now(), error_msg, session_id)
            )
        
        finally:
            self.is_training = False
            self.current_session_id = None
    
    async def _prepare_dataset_for_training(self, dataset_id: int, session_id: int) -> Dict:
        """
        Prepare dataset dari database ke format YOLO
        """
        try:
            # Create session-specific dataset folder
            session_dataset_dir = self.training_dir / "datasets" / f"session_{session_id}"
            images_dir = session_dataset_dir / "images"
            labels_dir = session_dataset_dir / "labels"
            
            # Create directories
            for dir_path in [images_dir, labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Get images from database
            images = db_service.fetch_all(
                """
                SELECT image_id, image_path, image_filename, damage_type, damage_severity
                FROM yolo_training_images 
                WHERE dataset_id = %s
                """,
                (dataset_id,)
            )
            
            if len(images) < 10:
                return {
                    'success': False,
                    'error': 'Dataset minimal harus memiliki 10 gambar'
                }
            
            # Copy images and create labels
            processed_count = 0
            for image_id, image_path, filename, damage_type, damage_severity in images:
                
                # Copy image file
                source_path = Path(image_path)
                if source_path.exists():
                    dest_image_path = images_dir / filename
                    shutil.copy2(source_path, dest_image_path)
                    
                    # Create label file
                    label_filename = Path(filename).stem + ".txt"
                    dest_label_path = labels_dir / label_filename
                    
                    # Get annotations for this image
                    annotations = db_service.fetch_all(
                        """
                        SELECT class_label_id, x_center, y_center, width, height
                        FROM yolo_annotations 
                        WHERE image_id = %s
                        """,
                        (image_id,)
                    )
                    
                    # Write YOLO format annotations
                    with open(dest_label_path, 'w') as f:
                        for class_id, x_center, y_center, width, height in annotations:
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    processed_count += 1
            
            # Create data.yaml file
            yaml_content = f"""
train: {images_dir}
val: {images_dir}  # Using same folder for validation in this demo

nc: 10  # number of classes
names:
  0: korosi_ringan
  1: korosi_sedang
  2: korosi_parah
  3: retak_permukaan
  4: retak_struktural
  5: kebocoran
  6: keausan
  7: deformasi
  8: kontaminasi
  9: patah
"""
            
            yaml_path = session_dataset_dir / "data.yaml"
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            
            logger.info(f"Dataset prepared: {processed_count} images")
            
            return {
                'success': True,
                'dataset_path': str(session_dataset_dir),
                'yaml_path': str(yaml_path),
                'processed_images': processed_count
            }
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_yolo_config(self, dataset_path: str, model_architecture: str) -> Dict:
        """
        Buat konfigurasi untuk YOLO training
        """
        try:
            config_content = f"""
# YOLO Training Configuration
# Generated for dataset: {dataset_path}

# Model
model: {model_architecture}.pt

# Dataset
data: {dataset_path}/data.yaml

# Training parameters
epochs: 100
patience: 50
batch: 16
imgsz: 640
save: true
save_period: 10
cache: false
device: 0  # Use GPU if available
workers: 8
project: {self.training_dir}/runs
name: training
exist_ok: true

# Hyperparameters
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""
            
            config_path = self.training_dir / "configs" / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            return {
                'success': True,
                'config_path': str(config_path)
            }
            
        except Exception as e:
            logger.error(f"Error creating YOLO config: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _simulate_yolo_training(self, config_path: str, epochs: int, batch_size: int, 
                                     learning_rate: float, session_id: int) -> Dict:
        """
        Simulasi training YOLO (untuk demo purposes)
        Dalam implementasi nyata, ini akan memanggil ultralytics YOLO training
        """
        try:
            # Simulate training progress
            for epoch in range(1, epochs + 1):
                # Simulate training time
                await asyncio.sleep(0.1)  # Simulate training time
                
                # Simulate metrics
                train_loss = max(0.1, 1.0 - (epoch / epochs) * 0.8)
                val_loss = max(0.15, 1.1 - (epoch / epochs) * 0.75)
                
                # Update progress occasionally
                if epoch % 10 == 0:
                    progress_log = f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    logger.info(f"Training progress: {progress_log}")
                    
                    # Update database with progress
                    db_service.execute_query(
                        """
                        UPDATE yolo_training_sessions 
                        SET training_logs = CONCAT(COALESCE(training_logs, ''), %s, '\n')
                        WHERE session_id = %s
                        """,
                        (progress_log, session_id)
                    )
            
            # Simulate final model save
            model_save_path = self.models_dir / "trained" / f"model_session_{session_id}.pt"
            
            # Create a dummy model file for demo
            with open(model_save_path, 'w') as f:
                f.write(f"# Trained YOLO model for session {session_id}\n")
                f.write(f"# Training completed at: {datetime.now()}\n")
            
            # Final metrics
            training_accuracy = 0.85 + np.random.uniform(-0.1, 0.1)  # Simulate 80-90% accuracy
            validation_accuracy = 0.80 + np.random.uniform(-0.1, 0.1)  # Simulate 75-85% accuracy
            
            logger.info(f"Training completed - Acc: {training_accuracy:.3f}, Val Acc: {validation_accuracy:.3f}")
            
            return {
                'success': True,
                'model_path': str(model_save_path),
                'training_accuracy': training_accuracy,
                'validation_accuracy': validation_accuracy,
                'epochs_completed': epochs
            }
            
        except Exception as e:
            logger.error(f"Error in YOLO training simulation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_session_id': self.current_session_id
        }
    
    async def stop_training(self) -> Dict:
        """Stop current training (if running)"""
        if not self.is_training:
            return {
                'success': False,
                'error': 'Tidak ada training yang sedang berjalan'
            }
        
        try:
            # Update session status
            if self.current_session_id:
                db_service.execute_query(
                    """
                    UPDATE yolo_training_sessions 
                    SET status = 'cancelled', end_time = %s
                    WHERE session_id = %s
                    """,
                    (datetime.now(), self.current_session_id)
                )
            
            self.is_training = False
            self.current_session_id = None
            
            logger.info("Training stopped by user")
            
            return {
                'success': True,
                'message': 'Training berhasil dihentikan'
            }
            
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Create service instance
training_service = YOLOTrainingService()
