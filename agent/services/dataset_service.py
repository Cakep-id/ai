"""
Dataset Training Service untuk YOLO Model Retraining
Mengelola upload, validasi, dan training dataset
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from services.db import db_service


class DatasetTrainingService:
    """Service untuk mengelola dataset training YOLO"""
    
    def __init__(self):
        self.base_dataset_dir = Path("ml/datasets")
        self.base_training_dir = Path("ml/training")
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB per file
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Setup folder structure untuk dataset training"""
        dirs = [
            self.base_dataset_dir / "images",
            self.base_dataset_dir / "labels", 
            self.base_dataset_dir / "temp",
            self.base_training_dir / "runs",
            self.base_training_dir / "models",
            self.base_training_dir / "configs"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, dataset_name: str, description: str, uploaded_by: str) -> Dict:
        """
        Buat dataset baru untuk training
        
        Args:
            dataset_name: Nama dataset
            description: Deskripsi dataset
            uploaded_by: Nama uploader
            
        Returns:
            Dict dengan dataset_id dan info
        """
        try:
            # Validasi nama dataset
            if not dataset_name or len(dataset_name.strip()) < 3:
                return {
                    'success': False,
                    'error': 'Nama dataset minimal 3 karakter'
                }
            
            # Check duplikasi nama
            existing = db_service.fetch_one(
                "SELECT dataset_id FROM yolo_training_datasets WHERE dataset_name = %s AND is_active = 1",
                (dataset_name.strip(),)
            )
            
            if existing:
                return {
                    'success': False,
                    'error': 'Nama dataset sudah digunakan'
                }
            
            # Insert ke database
            insert_query = """
            INSERT INTO yolo_training_datasets 
            (dataset_name, description, uploaded_by, status) 
            VALUES (%s, %s, %s, 'uploading')
            """
            
            dataset_id = db_service.execute_query(
                insert_query, 
                (dataset_name.strip(), description.strip(), uploaded_by.strip())
            )
            
            if not dataset_id:
                return {
                    'success': False,
                    'error': 'Gagal membuat dataset di database'
                }
            
            # Buat folder dataset
            dataset_folder = self.base_dataset_dir / f"dataset_{dataset_id}"
            dataset_folder.mkdir(exist_ok=True)
            (dataset_folder / "images").mkdir(exist_ok=True)
            (dataset_folder / "labels").mkdir(exist_ok=True)
            
            logger.info(f"Dataset created: {dataset_name} (ID: {dataset_id})")
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'folder_path': str(dataset_folder)
            }
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def upload_training_image(
        self,
        dataset_id: int,
        image_file_path: str,
        damage_type: str,
        damage_severity: str,
        damage_description: str = "",
        annotations: List[Dict] = None
    ) -> Dict:
        """
        Upload gambar training dengan annotation
        
        Args:
            dataset_id: ID dataset
            image_file_path: Path ke file gambar
            damage_type: Jenis kerusakan
            damage_severity: Tingkat severity (LOW/MEDIUM/HIGH/CRITICAL)
            damage_description: Deskripsi kerusakan
            annotations: List annotations bounding box
            
        Returns:
            Dict dengan status upload
        """
        try:
            # Validasi dataset exists
            dataset = db_service.fetch_one(
                "SELECT * FROM yolo_training_datasets WHERE dataset_id = %s AND is_active = 1",
                (dataset_id,)
            )
            
            if not dataset:
                return {
                    'success': False,
                    'error': 'Dataset tidak ditemukan'
                }
            
            # Validasi file gambar
            if not os.path.exists(image_file_path):
                return {
                    'success': False,
                    'error': 'File gambar tidak ditemukan'
                }
            
            # Check file size
            file_size = os.path.getsize(image_file_path)
            if file_size > self.max_file_size:
                return {
                    'success': False,
                    'error': f'File terlalu besar. Maksimal {self.max_file_size // (1024*1024)}MB'
                }
            
            # Validasi format gambar
            file_ext = Path(image_file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Format tidak didukung. Gunakan: {", ".join(self.supported_formats)}'
                }
            
            # Load dan validasi gambar
            try:
                image = cv2.imread(image_file_path)
                if image is None:
                    return {
                        'success': False,
                        'error': 'File gambar corrupt atau tidak valid'
                    }
                height, width, channels = image.shape
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Error membaca gambar: {e}'
                }
            
            # Generate unique filename
            original_filename = Path(image_file_path).name
            file_hash = hashlib.md5(open(image_file_path, 'rb').read()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{file_hash}_{original_filename}"
            
            # Setup paths
            dataset_folder = self.base_dataset_dir / f"dataset_{dataset_id}"
            dest_image_path = dataset_folder / "images" / new_filename
            dest_label_path = dataset_folder / "labels" / f"{Path(new_filename).stem}.txt"
            
            # Copy image file
            shutil.copy2(image_file_path, dest_image_path)
            
            # Insert ke database
            insert_query = """
            INSERT INTO yolo_training_images 
            (dataset_id, image_filename, image_path, image_width, image_height, 
             damage_type, damage_severity, damage_description) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            image_id = db_service.execute_query(
                insert_query,
                (dataset_id, new_filename, str(dest_image_path), width, height,
                 damage_type, damage_severity, damage_description)
            )
            
            if not image_id:
                # Cleanup file if database insert failed
                if dest_image_path.exists():
                    dest_image_path.unlink()
                return {
                    'success': False,
                    'error': 'Gagal menyimpan data gambar ke database'
                }
            
            # Process annotations jika ada
            label_lines = []
            if annotations:
                for ann in annotations:
                    result = self._save_annotation(image_id, ann, width, height)
                    if result['success']:
                        label_lines.append(result['yolo_line'])
            
            # Buat file label YOLO format
            if label_lines:
                with open(dest_label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
            
            # Update total images di dataset
            self._update_dataset_count(dataset_id)
            
            logger.info(f"Image uploaded: {new_filename} to dataset {dataset_id}")
            
            return {
                'success': True,
                'image_id': image_id,
                'filename': new_filename,
                'image_path': str(dest_image_path),
                'annotations_count': len(label_lines)
            }
            
        except Exception as e:
            logger.error(f"Error uploading training image: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_annotation(self, image_id: int, annotation: Dict, img_width: int, img_height: int) -> Dict:
        """
        Simpan annotation bounding box ke database
        
        Args:
            image_id: ID gambar
            annotation: Dict dengan x, y, width, height, class_name
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Dict dengan status dan YOLO format line
        """
        try:
            # Get class info
            class_info = db_service.fetch_one(
                "SELECT class_label_id FROM yolo_damage_classes WHERE class_name = %s",
                (annotation['class_name'],)
            )
            
            if not class_info:
                return {
                    'success': False,
                    'error': f"Class tidak ditemukan: {annotation['class_name']}"
                }
            
            class_id = class_info[0]
            
            # Convert to normalized YOLO format
            x = annotation['x']
            y = annotation['y'] 
            w = annotation['width']
            h = annotation['height']
            
            # Calculate center points and normalize
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Validasi range
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                   0 < norm_width <= 1 and 0 < norm_height <= 1):
                return {
                    'success': False,
                    'error': 'Koordinat annotation tidak valid'
                }
            
            # Insert ke database
            insert_query = """
            INSERT INTO yolo_annotations 
            (image_id, class_id, class_name, x_center, y_center, width, height, confidence) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            annotation_id = db_service.execute_query(
                insert_query,
                (image_id, class_id, annotation['class_name'], 
                 x_center, y_center, norm_width, norm_height,
                 annotation.get('confidence', 1.0))
            )
            
            if not annotation_id:
                return {
                    'success': False,
                    'error': 'Gagal menyimpan annotation'
                }
            
            # Format YOLO line: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            
            return {
                'success': True,
                'annotation_id': annotation_id,
                'yolo_line': yolo_line
            }
            
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_dataset_count(self, dataset_id: int):
        """Update jumlah total gambar di dataset"""
        try:
            count_query = """
            SELECT COUNT(*) FROM yolo_training_images 
            WHERE dataset_id = %s
            """
            
            result = db_service.fetch_one(count_query, (dataset_id,))
            total_images = result[0] if result else 0
            
            update_query = """
            UPDATE yolo_training_datasets 
            SET total_images = %s 
            WHERE dataset_id = %s
            """
            
            db_service.execute_query(update_query, (total_images, dataset_id))
            
        except Exception as e:
            logger.error(f"Error updating dataset count: {e}")
    
    def get_datasets(self, limit: int = 50) -> List[Dict]:
        """Ambil daftar dataset"""
        try:
            query = """
            SELECT dataset_id, dataset_name, description, uploaded_by, 
                   upload_date, total_images, status, dataset_version
            FROM yolo_training_datasets 
            WHERE is_active = 1 
            ORDER BY upload_date DESC 
            LIMIT %s
            """
            
            results = db_service.fetch_all(query, (limit,))
            
            datasets = []
            for row in results:
                datasets.append({
                    'dataset_id': row[0],
                    'dataset_name': row[1],
                    'description': row[2],
                    'uploaded_by': row[3],
                    'upload_date': row[4].isoformat() if row[4] else None,
                    'total_images': row[5],
                    'status': row[6],
                    'dataset_version': row[7]
                })
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error fetching datasets: {e}")
            return []
    
    def get_dataset_images(self, dataset_id: int, limit: int = 100) -> List[Dict]:
        """Ambil daftar gambar dalam dataset"""
        try:
            query = """
            SELECT image_id, image_filename, damage_type, damage_severity, 
                   damage_description, uploaded_at, is_validated
            FROM yolo_training_images 
            WHERE dataset_id = %s 
            ORDER BY uploaded_at DESC 
            LIMIT %s
            """
            
            results = db_service.fetch_all(query, (dataset_id, limit))
            
            images = []
            for row in results:
                images.append({
                    'image_id': row[0],
                    'filename': row[1],
                    'damage_type': row[2],
                    'damage_severity': row[3],
                    'damage_description': row[4],
                    'uploaded_at': row[5].isoformat() if row[5] else None,
                    'is_validated': bool(row[6])
                })
            
            return images
            
        except Exception as e:
            logger.error(f"Error fetching dataset images: {e}")
            return []


# Create service instance
dataset_service = DatasetTrainingService()
