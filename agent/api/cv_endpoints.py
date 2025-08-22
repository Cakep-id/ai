"""
Computer Vision API Endpoints
Endpoints untuk YOLO object detection dan training
"""

import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from services import yolo_service, db_service

router = APIRouter()

# Pydantic models
class DetectionResponse(BaseModel):
    success: bool
    detections: List[dict]
    image_info: dict
    model_version: str
    processing_time: float
    error: Optional[str] = None

class TrainingRequest(BaseModel):
    dataset_path: str
    epochs: int = 100
    batch_size: int = 16

class TrainingResponse(BaseModel):
    success: bool
    metrics: dict
    model_version: str
    dataset_info: str
    error: Optional[str] = None

# Helper functions
def validate_image_file(file: UploadFile) -> bool:
    """Validasi file gambar"""
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']
    return file.content_type in allowed_types

def save_uploaded_file(file: UploadFile, destination: str) -> str:
    """Simpan uploaded file"""
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {destination}")
        return destination
        
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@router.post("/detect", response_model=DetectionResponse)
async def detect_damage(
    file: UploadFile = File(..., description="Image file untuk deteksi"),
    report_id: Optional[int] = Form(None, description="Report ID untuk logging"),
    save_result: bool = Form(True, description="Simpan hasil ke database")
):
    """
    Deteksi kerusakan pada gambar menggunakan YOLO
    
    - **file**: File gambar (JPEG, PNG, BMP, TIFF)
    - **report_id**: ID report untuk logging (opsional)
    - **save_result**: Simpan hasil ke database (default: True)
    """
    try:
        # Validasi file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: JPEG, PNG, BMP, TIFF. Got: {file.content_type}"
            )
        
        # Check file size (max 50MB)
        max_size = int(os.getenv('MAX_FILE_SIZE', '52428800'))  # 50MB default
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {max_size / 1024 / 1024:.1f}MB"
            )
        
        # Simpan file temporary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"detect_{timestamp}_{file.filename}"
        temp_path = os.path.join("./storage/temp", temp_filename)
        
        saved_path = save_uploaded_file(file, temp_path)
        
        try:
            # Run YOLO detection
            logger.info(f"Running YOLO detection on {saved_path}")
            detection_result = yolo_service.detect(saved_path)
            
            if not detection_result['success']:
                raise HTTPException(
                    status_code=500,
                    detail=f"Detection failed: {detection_result.get('error', 'Unknown error')}"
                )
            
            # Simpan hasil ke database jika diminta dan ada report_id
            if save_result and report_id:
                try:
                    detections = detection_result['detections']
                    model_version = detection_result['model_version']
                    
                    for detection in detections:
                        db_service.save_detection(
                            report_id=report_id,
                            label=detection['label'],
                            confidence=detection['confidence'],
                            bbox=detection['bbox'],
                            model_ver=model_version
                        )
                    
                    logger.info(f"Saved {len(detections)} detections to database for report {report_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to save detection results: {e}")
                    # Don't fail the request, just log the error
            
            # Generate annotated image
            try:
                annotated_path = yolo_service.annotate_image(
                    saved_path, 
                    detection_result['detections']
                )
                detection_result['annotated_image_path'] = annotated_path
            except Exception as e:
                logger.warning(f"Failed to generate annotated image: {e}")
            
            # Clean up temporary file
            try:
                os.remove(saved_path)
            except:
                pass
            
            return DetectionResponse(**detection_result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Detection processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Detection processing failed: {str(e)}")
        
        finally:
            # Cleanup temp file
            try:
                if os.path.exists(saved_path):
                    os.remove(saved_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detect endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/retrain", response_model=TrainingResponse)
async def retrain_model(
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """
    Retrain YOLO model dengan dataset baru
    
    - **dataset_path**: Path ke dataset dalam format YOLO
    - **epochs**: Jumlah epoch training (default: 100)
    - **batch_size**: Batch size untuk training (default: 16)
    """
    try:
        # Validasi dataset path
        if not os.path.exists(request.dataset_path):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset path not found: {request.dataset_path}"
            )
        
        # Validasi dataset format
        validation_result = yolo_service.validate_dataset(request.dataset_path)
        if not validation_result['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset format: {validation_result['error']}"
            )
        
        logger.info(f"Starting YOLO retraining with {validation_result['train_images']} train images")
        
        # Run training in background untuk request yang cepat
        def run_training():
            try:
                training_result = yolo_service.train(
                    dataset_path=request.dataset_path,
                    epochs=request.epochs,
                    batch_size=request.batch_size
                )
                
                # Save training log ke database
                if training_result['success']:
                    dataset_info = f"Dataset: {request.dataset_path}, Images: {validation_result['train_images']}"
                    metrics = training_result['metrics']
                    
                    db_service.save_training_log(
                        model_type='YOLO',
                        dataset_info=dataset_info,
                        model_ver=training_result['model_version'],
                        accuracy=metrics.get('final_map50'),
                        loss=metrics.get('best_fitness')
                    )
                    
                    logger.info("YOLO training completed and logged")
                else:
                    logger.error(f"YOLO training failed: {training_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Background training failed: {e}")
        
        # Add background task
        background_tasks.add_task(run_training)
        
        # Return immediate response
        return TrainingResponse(
            success=True,
            metrics={
                'status': 'training_started',
                'dataset_images': validation_result['train_images'],
                'validation_images': validation_result['val_images'],
                'classes': validation_result['classes']
            },
            model_version=yolo_service._get_model_version(),
            dataset_info=f"Dataset: {request.dataset_path}, Epochs: {request.epochs}",
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Training initiation failed: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """Get informasi model YOLO yang sedang digunakan"""
    try:
        model_info = yolo_service.get_model_info()
        
        # Tambahkan training history dari database
        try:
            training_logs = db_service.get_training_logs(model_type='YOLO', limit=10)
            model_info['training_history'] = training_logs
        except Exception as e:
            logger.warning(f"Failed to get training history: {e}")
            model_info['training_history'] = []
        
        return {
            "success": True,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get model info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/model/stats")
async def get_detection_stats():
    """Get statistik deteksi dari database"""
    try:
        # Query detection statistics
        stats_query = """
        SELECT 
            label,
            COUNT(*) as detection_count,
            AVG(confidence) as avg_confidence,
            MAX(confidence) as max_confidence,
            MIN(confidence) as min_confidence
        FROM detections 
        WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY label
        ORDER BY detection_count DESC
        """
        
        detection_stats = db_service.execute_query(stats_query)
        
        # Query recent detections
        recent_query = """
        SELECT d.*, r.asset_id, r.description
        FROM detections d
        JOIN reports r ON d.report_id = r.report_id
        ORDER BY d.detected_at DESC
        LIMIT 20
        """
        
        recent_detections = db_service.execute_query(recent_query)
        
        return {
            "success": True,
            "stats": {
                "detection_summary": detection_stats,
                "recent_detections": recent_detections,
                "total_detections": sum(stat['detection_count'] for stat in detection_stats),
                "unique_labels": len(detection_stats)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get detection stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection stats: {str(e)}")

@router.post("/validate-dataset")
async def validate_dataset_format(dataset_path: str):
    """Validasi format dataset YOLO"""
    try:
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset path not found: {dataset_path}"
            )
        
        validation_result = yolo_service.validate_dataset(dataset_path)
        
        return {
            "success": True,
            "validation": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset validation failed: {str(e)}")

@router.get("/labels")
async def get_damage_labels():
    """Get daftar label kerusakan yang didukung"""
    try:
        model_info = yolo_service.get_model_info()
        
        damage_labels = model_info.get('damage_labels', {})
        risk_mapping = model_info.get('risk_mapping', {})
        
        labels_info = []
        for en_label, id_label in damage_labels.items():
            labels_info.append({
                'english': en_label,
                'indonesian': id_label,
                'risk_score': risk_mapping.get(en_label, 0.5)
            })
        
        return {
            "success": True,
            "labels": labels_info,
            "total_labels": len(labels_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get labels failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get labels: {str(e)}")
