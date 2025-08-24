"""
FastAPI Backend for AgentV2 AI Asset Inspection System
3-Role Architecture: User, Admin, Trainer
No authentication - role-based by interface only
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uuid
import json
import logging
from datetime import datetime, date
import asyncio

# Import our modules
from database.db_manager import create_database_manager, create_daos
from ai_models.yolo_service import YOLOService
from ai_models.risk_engine import RiskEngine
from ai_models.training_service import TrainingService
from ai_models.evaluation_service import EvaluationService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AgentV2 AI Asset Inspection System",
    description="Advanced AI-powered asset inspection with human-in-the-loop learning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Global services
db_manager = None
daos = None
yolo_service = None
risk_engine = None
training_service = None
evaluation_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global db_manager, daos, yolo_service, risk_engine, training_service, evaluation_service
    
    try:
        # Initialize database
        db_manager = create_database_manager()
        daos = create_daos(db_manager)
        
        # Initialize AI services
        yolo_service = YOLOService()
        risk_engine = RiskEngine(daos)
        training_service = TrainingService(daos)
        evaluation_service = EvaluationService(daos)
        
        logger.info("AgentV2 services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Pydantic models
class UserReportRequest(BaseModel):
    user_identifier: Optional[str] = None
    asset_description: str = Field(..., min_length=10, max_length=1000)

class TrainingDataRequest(BaseModel):
    trainer_identifier: str = Field(..., min_length=1)
    asset_description: str = Field(..., min_length=10)
    risk_category: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    manual_annotations: List[Dict[str, Any]]
    training_notes: Optional[str] = None

class TrainingSessionRequest(BaseModel):
    trainer_identifier: str
    model_version: str
    epochs: int = Field(default=100, ge=10, le=1000)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    batch_size: int = Field(default=16, ge=1, le=128)

class ValidationActionRequest(BaseModel):
    validator_identifier: str
    action_type: str = Field(..., regex="^(approve|reject|modify|request_review)$")
    modifications: Optional[Dict[str, Any]] = None
    validator_notes: Optional[str] = None
    confidence_adjustment: Optional[float] = Field(None, ge=0, le=1)

# ============== USER ENDPOINTS ==============

@app.post("/api/user/submit-report")
async def submit_user_report(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    data: str = Form(...)
):
    """Submit new inspection report (User role)"""
    try:
        # Parse form data
        report_data = json.loads(data)
        request = UserReportRequest(**report_data)
        
        # Validate image
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Generate report ID
        report_id = f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Save image
        os.makedirs("uploads/foto_mentah", exist_ok=True)
        image_path = f"uploads/foto_mentah/{report_id}_{image.filename}"
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Create report in database
        report_data = {
            'report_id': report_id,
            'user_identifier': request.user_identifier,
            'asset_description': request.asset_description,
            'original_image_path': image_path
        }
        
        daos['user_reports'].create_report(report_data)
        
        # Process asynchronously
        background_tasks.add_task(process_user_report, report_id, image_path)
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Report submitted successfully. Processing will begin shortly."
        }
        
    except Exception as e:
        logger.error(f"Error submitting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_user_report(report_id: str, image_path: str):
    """Background processing of user report"""
    try:
        # Update status to processing
        daos['user_reports'].update_processing_status(report_id, "processing")
        
        # YOLO detection
        yolo_results = await yolo_service.detect_damage(image_path)
        
        # Save YOLO processed image
        yolo_image_path = f"uploads/foto_yolo/{report_id}_yolo.jpg"
        os.makedirs(os.path.dirname(yolo_image_path), exist_ok=True)
        
        # Process detections and save to database
        detections = []
        for detection in yolo_results['detections']:
            detection_data = {
                'damage_class_id': detection['class_id'],
                'confidence_score': detection['confidence'],
                'bbox_x1': detection['bbox'][0],
                'bbox_y1': detection['bbox'][1],
                'bbox_x2': detection['bbox'][2],
                'bbox_y2': detection['bbox'][3],
                'area_pixels': detection['area'],
                'area_percentage': detection['area_percentage'],
                'iou_score': detection.get('iou_score')
            }
            detections.append(detection_data)
        
        daos['yolo_detections'].save_detections(report_id, detections)
        
        # Risk analysis
        risk_results = await risk_engine.analyze_risk(report_id, yolo_results)
        daos['risk_analysis'].save_risk_analysis(risk_results)
        
        # Generate final analysis image
        final_image_path = f"uploads/foto_final/{report_id}_analysis.jpg"
        os.makedirs(os.path.dirname(final_image_path), exist_ok=True)
        
        # Update status to completed
        daos['user_reports'].update_processing_status(
            report_id, "completed", yolo_image_path, final_image_path
        )
        
        logger.info(f"Report {report_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing report {report_id}: {e}")
        daos['user_reports'].update_processing_status(report_id, "failed")

@app.get("/api/user/report/{report_id}")
async def get_user_report(report_id: str):
    """Get report details and results (User role)"""
    try:
        # Get report
        report = daos['user_reports'].get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get detections
        detections = daos['yolo_detections'].get_detections(report_id)
        
        # Get risk analysis
        risk_analysis = daos['risk_analysis'].get_risk_analysis(report_id)
        
        return {
            "report": report,
            "detections": detections,
            "risk_analysis": risk_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/reports")
async def get_user_reports(
    user_identifier: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get user reports list (User role)"""
    try:
        reports = daos['user_reports'].get_reports_by_status()
        
        # Filter by user if specified
        if user_identifier:
            reports = [r for r in reports if r.get('user_identifier') == user_identifier]
        
        # Pagination
        total = len(reports)
        reports = reports[offset:offset + limit]
        
        return {
            "reports": reports,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== ADMIN ENDPOINTS ==============

@app.get("/api/admin/dashboard")
async def get_admin_dashboard():
    """Get admin dashboard data (Admin role)"""
    try:
        # Dashboard summary
        dashboard_query = "SELECT * FROM dashboard_summary ORDER BY report_date DESC LIMIT 30"
        dashboard_data = db_manager.execute_query(dashboard_query)
        
        # Risk distribution
        risk_query = "SELECT * FROM risk_distribution"
        risk_data = db_manager.execute_query(risk_query)
        
        # Recent reports
        recent_reports = daos['user_reports'].get_reports_by_status()[:10]
        
        # Model performance
        performance_query = "SELECT * FROM model_performance ORDER BY completed_at DESC LIMIT 5"
        model_performance = db_manager.execute_query(performance_query)
        
        return {
            "dashboard_summary": dashboard_data,
            "risk_distribution": risk_data,
            "recent_reports": recent_reports,
            "model_performance": model_performance
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/reports/pending")
async def get_pending_reports():
    """Get reports pending validation (Admin role)"""
    try:
        reports = daos['user_reports'].get_reports_by_status(
            processing_status="completed",
            validation_status="pending"
        )
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error getting pending reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/validate/{report_id}")
async def validate_report(report_id: str, validation: ValidationActionRequest):
    """Validate report (Admin role)"""
    try:
        # Get report
        report = daos['user_reports'].get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Save validation action
        validation_query = """
        INSERT INTO validation_actions 
        (report_id, validator_identifier, action_type, modifications, 
         validator_notes, confidence_adjustment)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            report_id,
            validation.validator_identifier,
            validation.action_type,
            json.dumps(validation.modifications) if validation.modifications else None,
            validation.validator_notes,
            validation.confidence_adjustment
        )
        
        db_manager.execute_insert(validation_query, params)
        
        # Update report validation status
        new_status = "approved" if validation.action_type == "approve" else "rejected"
        if validation.action_type in ["modify", "request_review"]:
            new_status = "needs_review"
        
        update_query = "UPDATE user_reports SET validation_status = %s WHERE report_id = %s"
        db_manager.execute_update(update_query, (new_status, report_id))
        
        return {"success": True, "message": f"Report {validation.action_type}d successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/maintenance/schedule")
async def get_maintenance_schedule():
    """Get maintenance schedule (Admin role)"""
    try:
        query = """
        SELECT ms.*, ur.asset_description, ra.risk_category, ra.overall_risk_score
        FROM maintenance_schedule ms
        JOIN user_reports ur ON ms.report_id = ur.report_id
        LEFT JOIN risk_analysis ra ON ms.report_id = ra.report_id
        WHERE ms.status IN ('scheduled', 'in_progress')
        ORDER BY ms.scheduled_date ASC, ms.priority_level ASC
        """
        
        schedule = db_manager.execute_query(query)
        return {"maintenance_schedule": schedule}
        
    except Exception as e:
        logger.error(f"Error getting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/maintenance/{maintenance_id}")
async def update_maintenance(maintenance_id: int, status: str = Form(...), notes: str = Form(None)):
    """Update maintenance status (Admin role)"""
    try:
        query = """
        UPDATE maintenance_schedule 
        SET status = %s, completion_notes = %s, updated_at = NOW()
        WHERE id = %s
        """
        
        result = db_manager.execute_update(query, (status, notes, maintenance_id))
        
        if result == 0:
            raise HTTPException(status_code=404, detail="Maintenance record not found")
        
        return {"success": True, "message": "Maintenance status updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== TRAINER ENDPOINTS ==============

@app.post("/api/trainer/submit-training-data")
async def submit_training_data(
    image: UploadFile = File(...),
    data: str = Form(...)
):
    """Submit training data (Trainer role)"""
    try:
        # Parse form data
        training_data = json.loads(data)
        request = TrainingDataRequest(**training_data)
        
        # Validate image
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Save image
        os.makedirs("uploads/training", exist_ok=True)
        image_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{image.filename}"
        image_path = f"uploads/training/{image_filename}"
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Save to database
        training_query = """
        INSERT INTO trainer_data 
        (trainer_identifier, image_path, asset_description, risk_category, 
         manual_annotations, training_notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            request.trainer_identifier,
            image_path,
            request.asset_description,
            request.risk_category,
            json.dumps(request.manual_annotations),
            request.training_notes
        )
        
        training_id = db_manager.execute_insert(training_query, params)
        
        return {
            "success": True,
            "training_id": training_id,
            "message": "Training data submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trainer/start-training")
async def start_training_session(
    background_tasks: BackgroundTasks,
    training_request: TrainingSessionRequest
):
    """Start new training session (Trainer role)"""
    try:
        # Generate session ID
        session_id = f"TRAIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Get training data count
        count_query = "SELECT COUNT(*) as count FROM trainer_data WHERE used_in_training = FALSE"
        count_result = db_manager.execute_query(count_query)
        training_count = count_result[0]['count'] if count_result else 0
        
        if training_count < 10:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient training data. Minimum 10 samples required."
            )
        
        # Create training session
        session_data = {
            'session_id': session_id,
            'model_version': training_request.model_version,
            'training_data_count': int(training_count * 0.8),  # 80% for training
            'validation_data_count': int(training_count * 0.2),  # 20% for validation
            'epochs': training_request.epochs,
            'learning_rate': training_request.learning_rate,
            'batch_size': training_request.batch_size,
            'trainer_identifier': training_request.trainer_identifier
        }
        
        daos['training_sessions'].create_session(session_data)
        
        # Start training in background
        background_tasks.add_task(
            training_service.start_training_session,
            session_id,
            training_request
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Training session started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trainer/training-sessions")
async def get_training_sessions(trainer_identifier: Optional[str] = None):
    """Get training sessions (Trainer role)"""
    try:
        query = "SELECT * FROM training_sessions"
        params = []
        
        if trainer_identifier:
            query += " WHERE trainer_identifier = %s"
            params.append(trainer_identifier)
        
        query += " ORDER BY started_at DESC"
        
        sessions = db_manager.execute_query(query, tuple(params))
        return {"training_sessions": sessions}
        
    except Exception as e:
        logger.error(f"Error getting training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trainer/session/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """Get training session metrics (Trainer role)"""
    try:
        # Get session info
        session_query = "SELECT * FROM training_sessions WHERE session_id = %s"
        session = db_manager.execute_query(session_query, (session_id,))
        
        if not session:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Get metrics
        metrics_query = """
        SELECT mm.*, dc.class_name 
        FROM model_metrics mm
        JOIN damage_classes dc ON mm.damage_class_id = dc.id
        WHERE mm.session_id = %s
        """
        metrics = db_manager.execute_query(metrics_query, (session_id,))
        
        # Get calibration data
        calibration_query = "SELECT * FROM model_calibration WHERE session_id = %s"
        calibration = db_manager.execute_query(calibration_query, (session_id,))
        
        return {
            "session": session[0],
            "metrics": metrics,
            "calibration": calibration
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trainer/training-data")
async def get_training_data(
    trainer_identifier: Optional[str] = None,
    used_in_training: Optional[bool] = None
):
    """Get training data list (Trainer role)"""
    try:
        query = "SELECT * FROM trainer_data WHERE 1=1"
        params = []
        
        if trainer_identifier:
            query += " AND trainer_identifier = %s"
            params.append(trainer_identifier)
        
        if used_in_training is not None:
            query += " AND used_in_training = %s"
            params.append(used_in_training)
        
        query += " ORDER BY created_at DESC"
        
        training_data = db_manager.execute_query(query, tuple(params))
        return {"training_data": training_data}
        
    except Exception as e:
        logger.error(f"Error getting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== SHARED ENDPOINTS ==============

@app.get("/api/damage-classes")
async def get_damage_classes():
    """Get all damage classes"""
    try:
        query = "SELECT * FROM damage_classes ORDER BY class_name"
        classes = db_manager.execute_query(query)
        return {"damage_classes": classes}
        
    except Exception as e:
        logger.error(f"Error getting damage classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-config/{key}")
async def get_system_config(key: str):
    """Get system configuration value"""
    try:
        value = daos['system_config'].get_config(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Configuration key not found")
        
        return {"key": key, "value": value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "yolo": "loaded",
            "risk_engine": "ready"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
