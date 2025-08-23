"""
API Endpoints untuk Dataset Training Management
"""

import os
import tempfile
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.dataset_service import dataset_service
from services.training_service import training_service
from services.db import db_service
from loguru import logger


router = APIRouter(prefix="/api/training", tags=["Dataset Training"])


class DatasetCreate(BaseModel):
    dataset_name: str
    description: str = ""
    uploaded_by: str


class ImageUpload(BaseModel):
    damage_type: str
    damage_severity: str
    damage_description: str = ""


class AnnotationData(BaseModel):
    x: float
    y: float  
    width: float
    height: float
    class_name: str
    confidence: float = 1.0


@router.post("/dataset/create")
async def create_dataset(dataset_data: DatasetCreate):
    """
    Buat dataset baru untuk training
    """
    try:
        result = dataset_service.create_dataset(
            dataset_name=dataset_data.dataset_name,
            description=dataset_data.description,
            uploaded_by=dataset_data.uploaded_by
        )
        
        if result['success']:
            return JSONResponse({
                "success": True,
                "message": "Dataset berhasil dibuat",
                "data": {
                    "dataset_id": result['dataset_id'],
                    "dataset_name": result['dataset_name'],
                    "folder_path": result['folder_path']
                }
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result['error']
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def get_datasets(limit: int = 50):
    """
    Ambil daftar semua dataset
    """
    try:
        datasets = dataset_service.get_datasets(limit=limit)
        
        return JSONResponse({
            "success": True,
            "data": datasets,
            "total": len(datasets)
        })
        
    except Exception as e:
        logger.error(f"Error fetching datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/{dataset_id}/images")
async def get_dataset_images(dataset_id: int, limit: int = 100):
    """
    Ambil daftar gambar dalam dataset
    """
    try:
        images = dataset_service.get_dataset_images(dataset_id, limit=limit)
        
        return JSONResponse({
            "success": True,
            "dataset_id": dataset_id,
            "data": images,
            "total": len(images)
        })
        
    except Exception as e:
        logger.error(f"Error fetching dataset images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/{dataset_id}/upload-image")
async def upload_training_image(
    dataset_id: int,
    image_file: UploadFile = File(...),
    damage_type: str = Form(...),
    damage_severity: str = Form(...),
    damage_description: str = Form(""),
    annotations: str = Form("[]")  # JSON string of annotations
):
    """
    Upload gambar training dengan annotation
    """
    try:
        # Validasi file upload
        if not image_file.filename:
            return JSONResponse({
                "success": False,
                "message": "File gambar harus diupload"
            }, status_code=400)
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = Path(image_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return JSONResponse({
                "success": False,
                "message": f"Format file tidak didukung. Gunakan: {', '.join(allowed_extensions)}"
            }, status_code=400)
        
        # Validasi damage severity
        valid_severities = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
        if damage_severity.upper() not in valid_severities:
            return JSONResponse({
                "success": False,
                "message": f"Severity harus salah satu dari: {', '.join(valid_severities)}"
            }, status_code=400)
        
        # Save uploaded file to temp
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        try:
            content = await image_file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Parse annotations
            import json
            try:
                annotation_list = json.loads(annotations) if annotations else []
            except json.JSONDecodeError:
                annotation_list = []
            
            # Upload gambar
            result = dataset_service.upload_training_image(
                dataset_id=dataset_id,
                image_file_path=temp_file.name,
                damage_type=damage_type,
                damage_severity=damage_severity.upper(),
                damage_description=damage_description,
                annotations=annotation_list
            )
            
            if result['success']:
                return JSONResponse({
                    "success": True,
                    "message": "Gambar berhasil diupload",
                    "data": {
                        "image_id": result['image_id'],
                        "filename": result['filename'],
                        "annotations_count": result['annotations_count']
                    }
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": result['error']
                }, status_code=400)
                
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
    except Exception as e:
        logger.error(f"Error uploading training image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/damage-classes")
async def get_damage_classes():
    """
    Ambil daftar class kerusakan yang tersedia
    """
    try:
        query = """
        SELECT class_name, class_label_id, description, risk_weight, color_hex
        FROM yolo_damage_classes 
        WHERE is_active = 1 
        ORDER BY class_label_id
        """
        
        results = db_service.fetch_all(query)
        
        classes = []
        for row in results:
            classes.append({
                'class_name': row[0],
                'class_id': row[1],
                'description': row[2],
                'risk_weight': float(row[3]),
                'color': row[4]
            })
        
        return JSONResponse({
            "success": True,
            "data": classes,
            "total": len(classes)
        })
        
    except Exception as e:
        logger.error(f"Error fetching damage classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/{dataset_id}/start-training")
async def start_training_session(
    dataset_id: int,
    session_name: str = Form(...),
    started_by: str = Form(...),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.001),
    model_architecture: str = Form("yolov8n")
):
    """
    Mulai sesi training untuk dataset
    """
    try:
        # Validasi dataset
        dataset = db_service.fetch_one(
            "SELECT dataset_name, total_images FROM yolo_training_datasets WHERE dataset_id = %s AND is_active = 1",
            (dataset_id,)
        )
        
        if not dataset:
            return JSONResponse({
                "success": False,
                "message": "Dataset tidak ditemukan"
            }, status_code=404)
        
        dataset_name, total_images = dataset
        
        # Minimal images check
        if total_images < 10:
            return JSONResponse({
                "success": False,
                "message": f"Dataset minimal harus memiliki 10 gambar. Saat ini: {total_images}"
            }, status_code=400)
        
        # Insert training session
        insert_query = """
        INSERT INTO yolo_training_sessions 
        (dataset_id, session_name, started_by, epochs, batch_size, 
         learning_rate, model_architecture, status) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'preparing')
        """
        
        session_id = db_service.execute_query(
            insert_query,
            (dataset_id, session_name, started_by, epochs, batch_size,
             learning_rate, model_architecture)
        )
        
        if not session_id:
            return JSONResponse({
                "success": False,
                "message": "Gagal membuat sesi training"
            }, status_code=500)
        
        # Update dataset status
        db_service.execute_query(
            "UPDATE yolo_training_datasets SET status = 'training' WHERE dataset_id = %s",
            (dataset_id,)
        )
        
        # TODO: Start background training task
        # Start the actual training process
        training_result = await training_service.start_training_session(session_id)
        
        if training_result['success']:
            logger.info(f"Training session started: {session_name} for dataset {dataset_id}")
            
            return JSONResponse({
                "success": True,
                "message": "Sesi training berhasil dimulai dan sedang berjalan di background",
                "data": {
                    "session_id": session_id,
                    "dataset_name": dataset_name,
                    "total_images": total_images,
                    "status": "training"
                }
            })
        else:
            # Update status to failed if training start failed
            db_service.execute_query(
                "UPDATE yolo_training_sessions SET status = 'failed', error_message = %s WHERE session_id = %s",
                (training_result['error'], session_id)
            )
            
            return JSONResponse({
                "success": False,
                "message": f"Gagal memulai training: {training_result['error']}"
            }, status_code=500)
        
    except Exception as e:
        logger.error(f"Error starting training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-sessions")
async def get_training_sessions(limit: int = 50):
    """
    Ambil daftar sesi training
    """
    try:
        query = """
        SELECT s.session_id, s.session_name, s.started_by, s.start_time, 
               s.end_time, s.status, s.epochs, s.training_accuracy, 
               s.validation_accuracy, d.dataset_name
        FROM yolo_training_sessions s
        JOIN yolo_training_datasets d ON s.dataset_id = d.dataset_id
        ORDER BY s.start_time DESC
        LIMIT %s
        """
        
        results = db_service.fetch_all(query, (limit,))
        
        sessions = []
        for row in results:
            sessions.append({
                'session_id': row[0],
                'session_name': row[1],
                'started_by': row[2],
                'start_time': row[3].isoformat() if row[3] else None,
                'end_time': row[4].isoformat() if row[4] else None,
                'status': row[5],
                'epochs': row[6],
                'training_accuracy': float(row[7]) if row[7] else None,
                'validation_accuracy': float(row[8]) if row[8] else None,
                'dataset_name': row[9]
            })
        
        return JSONResponse({
            "success": True,
            "data": sessions,
            "total": len(sessions)
        })
        
    except Exception as e:
        logger.error(f"Error fetching training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """
    Hapus dataset (soft delete)
    """
    try:
        # Check if dataset exists
        dataset = db_service.fetch_one(
            "SELECT dataset_name FROM yolo_training_datasets WHERE dataset_id = %s AND is_active = 1",
            (dataset_id,)
        )
        
        if not dataset:
            return JSONResponse({
                "success": False,
                "message": "Dataset tidak ditemukan"
            }, status_code=404)
        
        # Soft delete
        db_service.execute_query(
            "UPDATE yolo_training_datasets SET is_active = 0 WHERE dataset_id = %s",
            (dataset_id,)
        )
        
        logger.info(f"Dataset {dataset_id} deleted")
        
        return JSONResponse({
            "success": True,
            "message": "Dataset berhasil dihapus"
        })
        
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-status")
async def get_training_status():
    """
    Get current training status
    """
    try:
        status = training_service.get_training_status()
        
        return JSONResponse({
            "success": True,
            "data": status
        })
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-training")
async def stop_current_training():
    """
    Stop current training session
    """
    try:
        result = await training_service.stop_training()
        
        if result['success']:
            return JSONResponse({
                "success": True,
                "message": result['message']
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result['error']
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-session/{session_id}/logs")
async def get_training_logs(session_id: int):
    """
    Get training logs for a session
    """
    try:
        logs = db_service.fetch_one(
            "SELECT training_logs, status, error_message FROM yolo_training_sessions WHERE session_id = %s",
            (session_id,)
        )
        
        if not logs:
            return JSONResponse({
                "success": False,
                "message": "Training session tidak ditemukan"
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "data": {
                "logs": logs[0] or "",
                "status": logs[1],
                "error_message": logs[2]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting training logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
