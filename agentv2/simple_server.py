#!/usr/bin/env python3
"""
Simple test server to check if endpoints are working
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import datetime
import os

app = FastAPI(title="AgentV2 Test Server", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# ============== HEALTH & CONFIG ENDPOINTS ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "mock",
            "yolo": "mock", 
            "risk_engine": "mock"
        }
    }

@app.get("/api/system-config/{key}")
async def get_system_config(key: str):
    """Get system configuration value"""
    default_config = {
        "default_confidence_threshold": 0.7,
        "pixel_to_mm_ratio": 0.5,
        "max_file_size_mb": 50,
        "max_upload_files": 10,
        "supported_formats": ["jpg", "jpeg", "png"],
        "training_epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001
    }
    
    if key in default_config:
        return {"key": key, "value": default_config[key]}
    
    raise HTTPException(status_code=404, detail=f"Configuration key '{key}' not found")

# ============== ADMIN ENDPOINTS ==============

@app.get("/api/admin/dashboard")
async def get_admin_dashboard():
    """Get admin dashboard data"""
    return {
        "total_inspections": 1245,
        "pending_validations": 23,
        "critical_assets": 5,
        "system_health": "good",
        "recent_inspections": [
            {
                "id": "INS-001",
                "asset_id": "PIPE-A-001",
                "timestamp": "2025-08-24T10:00:00Z",
                "risk_level": "medium",
                "confidence": 0.85
            }
        ]
    }

@app.get("/api/admin/reports/pending")
async def get_pending_reports():
    """Get pending validation reports"""
    return {
        "pending_reports": [
            {
                "id": "RPT-001",
                "asset_id": "PIPE-A-001",
                "detected_damages": ["crack", "corrosion"],
                "confidence": 0.78,
                "submitted_at": "2025-08-24T09:30:00Z"
            }
        ]
    }

# ============== TRAINER ENDPOINTS ==============

@app.get("/api/trainer/dashboard")
async def get_trainer_dashboard():
    """Get trainer dashboard data"""
    return {
        "recent_sessions": [
            {
                "session_id": "train-session-1",
                "started_at": "2025-08-24T08:00:00Z",
                "status": "completed",
                "epochs_completed": 100,
                "best_accuracy": 0.87
            }
        ],
        "data_stats": {
            "total_images": 1250,
            "training_images": 1000,
            "validation_images": 250
        },
        "performance_stats": {
            "avg_precision": 0.82,
            "avg_recall": 0.79,
            "avg_f1": 0.805,
            "avg_accuracy": 0.83
        },
        "class_distribution": [
            {"class_name": "crack", "image_count": 420},
            {"class_name": "corrosion", "image_count": 380},
            {"class_name": "dent", "image_count": 280},
            {"class_name": "normal", "image_count": 170}
        ]
    }

@app.post("/api/trainer/upload")
async def upload_training_images():
    """Mock upload endpoint"""
    return {
        "message": "Upload functionality available",
        "status": "success"
    }

@app.get("/api/trainer/images/unannotated")
async def get_unannotated_images():
    """Get unannotated images for training"""
    return {
        "unannotated_images": [
            {
                "id": "img_001",
                "file_path": "uploads/training_data/sample_001.jpg",
                "original_filename": "pipeline_crack_001.jpg",
                "damage_class_id": 1,
                "uploaded_at": "2025-08-24T10:00:00Z"
            },
            {
                "id": "img_002", 
                "file_path": "uploads/training_data/sample_002.jpg",
                "original_filename": "corrosion_sample_002.jpg",
                "damage_class_id": 2,
                "uploaded_at": "2025-08-24T09:30:00Z"
            }
        ]
    }

@app.get("/api/trainer/sessions/active")
async def get_active_training_sessions():
    """Get currently active training sessions"""
    return {
        "active_sessions": [
            {
                "session_id": "active_session_001",
                "status": "running",
                "started_at": "2025-08-24T08:00:00Z",
                "current_epoch": 45,
                "total_epochs": 100,
                "current_loss": 0.234,
                "best_accuracy": 0.82
            }
        ]
    }

@app.get("/api/trainer/evaluation/latest")
async def get_latest_evaluation():
    """Get latest model evaluation results"""
    return {
        "latest_evaluation": {
            "evaluation_id": "eval_001",
            "model_version": "v2.1",
            "evaluation_date": "2025-08-24T12:00:00Z",
            "overall_accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1_score": 0.87,
            "confusion_matrix": {
                "crack": {"tp": 95, "fp": 5, "fn": 8, "tn": 892},
                "corrosion": {"tp": 88, "fp": 12, "fn": 7, "tn": 893},
                "dent": {"tp": 76, "fp": 8, "fn": 15, "tn": 901},
                "normal": {"tp": 920, "fp": 15, "fn": 12, "tn": 53}
            }
        }
    }

@app.get("/api/trainer/sessions")
async def get_all_training_sessions():
    """Get all training sessions with pagination"""
    return {
        "sessions": [
            {
                "session_id": "session_001",
                "status": "completed",
                "started_at": "2025-08-24T06:00:00Z",
                "completed_at": "2025-08-24T08:30:00Z",
                "epochs_completed": 100,
                "total_epochs": 100,
                "best_accuracy": 0.87,
                "trainer_identifier": "trainer_001"
            },
            {
                "session_id": "session_002",
                "status": "failed",
                "started_at": "2025-08-23T14:00:00Z",
                "completed_at": "2025-08-23T15:15:00Z",
                "epochs_completed": 25,
                "total_epochs": 100,
                "best_accuracy": 0.45,
                "trainer_identifier": "trainer_002"
            }
        ]
    }

# ============== USER ENDPOINTS ==============

@app.get("/api/user/reports")
async def get_user_reports():
    """Get user reports"""
    return {
        "reports": [
            {
                "id": "RPT-001",
                "asset_id": "PIPE-A-001",
                "status": "validated",
                "risk_level": "medium",
                "created_at": "2025-08-24T09:00:00Z"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
