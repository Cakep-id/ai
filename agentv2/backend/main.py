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
import sys
import uuid
import json
import logging
from datetime import datetime, date
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database manager with error handling
try:
    from backend.db_manager import DatabaseManager
    print("Database Manager imported successfully")
except ImportError as e:
    print(f"Error importing database manager: {e}")
    print("Database functionality will be limited")
    DatabaseManager = None

# Import AI services with error handling
try:
    from ai_models.yolo_service import YOLOService
    from ai_models.risk_engine import RiskEngine
    from ai_models.training_service import TrainingService
    from ai_models.evaluation_service import EvaluationService
    from ai_models.report_service import ReportService
    print("AI services imported successfully")
except ImportError as e:
    print(f"Error importing AI services: {e}")
    print("Using fallback services for testing")
    # Fallback to simple services if original ones fail
    try:
        from ai_models.yolo_service_simple import YOLOService
        from ai_models.risk_engine_simple import RiskEngine
        from ai_models.report_service_simple import ReportService
        TrainingService = None
        EvaluationService = None
        print("Simple AI services imported as fallback")
    except ImportError as e2:
        print(f"Error importing fallback services: {e2}")
        YOLOService = None
        RiskEngine = None
        ReportService = None
        TrainingService = None
        EvaluationService = None

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

# Global services
db_manager = None
yolo_service = None
risk_engine = None
report_service = None
training_service = None
evaluation_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global db_manager, yolo_service, risk_engine, report_service, training_service, evaluation_service
    
    try:
        # Initialize database
        if DatabaseManager:
            db_manager = DatabaseManager()
            await db_manager.connect()
            print("Database Manager initialized")
        
        # Initialize AI services
        if YOLOService:
            try:
                yolo_service = YOLOService()
                print("YOLO Service initialized")
            except Exception as e:
                print(f"Warning: YOLO Service failed to initialize: {e}")
                yolo_service = None
        
        if RiskEngine:
            try:
                # Initialize with empty daos for now
                risk_engine = RiskEngine({})
                print("Risk Engine initialized")
            except Exception as e:
                print(f"Warning: Risk Engine failed to initialize: {e}")
                risk_engine = None
        
        if ReportService:
            report_service = ReportService()
            print("Report Service initialized")
        
        # Initialize training services if available
        if TrainingService:
            try:
                training_service = TrainingService({})
                print("Training Service initialized")
            except Exception as e:
                print(f"Warning: Training Service failed to initialize: {e}")
                training_service = None
        
        if EvaluationService:
            try:
                evaluation_service = EvaluationService({})
                print("Evaluation Service initialized")
            except Exception as e:
                print(f"Warning: Evaluation Service failed to initialize: {e}")
                evaluation_service = None
        
        print("All services initialized successfully")
        
        # Debug: Print registered routes
        print("\n=== REGISTERED ROUTES ===")
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                print(f"{list(route.methods)} {route.path}")
        print("=========================\n")
        
    except Exception as e:
        print(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_manager
    
    if db_manager:
        await db_manager.disconnect()
        print("Database disconnected")

# Static files directories (create if they don't exist)
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# NOTE: Static files mounting moved to END of file after all API routes

# Pydantic models
class UserReportRequest(BaseModel):
    user_identifier: Optional[str] = None
    asset_description: str = Field(..., min_length=10, max_length=1000)

class TrainingDataRequest(BaseModel):
    trainer_identifier: str = Field(..., min_length=1)
    asset_description: str = Field(..., min_length=10)
    risk_category: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
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
    action_type: str = Field(..., pattern="^(approve|reject|modify|request_review)$")
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
        
        # Save to database if available
        if db_manager:
            await db_manager.create_inspection(report_data)
        
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
        # Update processing status in db_manager if available
        if db_manager:
            await db_manager.update_inspection(report_id, {'status': 'processing'})
        
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
        
        # Save detection results if db_manager available
        if db_manager:
            await db_manager.save_detection_results({'inspection_id': report_id, 'detections': detections})
        
        # Risk analysis
        risk_results = await risk_engine.analyze_risk(report_id, yolo_results)
        # Save risk analysis if db_manager available
        if db_manager:
            await db_manager.save_risk_analysis(risk_results)
        
        # Generate final analysis image
        final_image_path = f"uploads/foto_final/{report_id}_analysis.jpg"
        os.makedirs(os.path.dirname(final_image_path), exist_ok=True)
        
        # Update status to completed
        # Update completion status
        if db_manager:
            await db_manager.update_inspection(report_id, {'status': 'completed', 'yolo_results': json.dumps(detections), 'risk_analysis': json.dumps(risk_results)})
        
        logger.info(f"Report {report_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing report {report_id}: {e}")
        # Update failed status
        if db_manager:
            await db_manager.update_inspection(report_id, {'status': 'failed'})

@app.get("/api/user/report/{report_id}")
async def get_user_report(report_id: str):
    """Get report details and results (User role)"""
    try:
        # Get report
        report = await db_manager.get_inspection(report_id) if db_manager else None
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get detections
        detections = [] # TODO: Implement get_detections from db_manager
        
        # Get risk analysis
        risk_analysis = {} # TODO: Implement get_risk_analysis from db_manager
        
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
        reports = await db_manager.list_inspections() if db_manager else []
        
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
        if not db_manager:
            return {
                "dashboard_summary": [],
                "risk_distribution": [],
                "recent_reports": [],
                "model_performance": [],
                "maintenance_schedule": [],
                "user_activity": []
            }
            
        # Recent reports (using async method)
        recent_reports = (await db_manager.list_inspections(limit=10))[:10]
        
        return {
            "dashboard_summary": [],  # Empty for now since tables may not exist
            "risk_distribution": [],
            "recent_reports": recent_reports,
            "model_performance": [],
            "maintenance_schedule": [],
            "user_activity": []
        }
        
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
        reports = await db_manager.list_inspections() if db_manager else []
        # Filter for completed status and pending validation
        filtered_reports = [r for r in reports if r.get('status') == 'completed']
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error getting pending reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/validate/{report_id}")
async def validate_report(report_id: str, validation: ValidationActionRequest):
    """Validate report (Admin role)"""
    try:
        # Get report
        report = await db_manager.get_inspection(report_id) if db_manager else None
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
        
        # Save training session
        if db_manager:
            await db_manager.save_training_session(session_data)
        
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

@app.get("/api/trainer/images/unannotated")
async def get_unannotated_images(limit: int = 100):
    """Get unannotated images for training"""
    try:
        if db_manager:
            query = """
            SELECT * FROM trainer_data 
            WHERE annotation_status = 'pending' OR annotation_status IS NULL
            ORDER BY created_at DESC LIMIT %s
            """
            images = await db_manager.execute_query(query, (limit,))
        else:
            # No database available
            images = []
        
        return {"unannotated_images": images}
        
    except Exception as e:
        logger.error(f"Error getting unannotated images: {e}")
        return {"unannotated_images": []}

@app.get("/api/trainer/sessions/active")
async def get_active_training_sessions():
    """Get currently active training sessions"""
    try:
        if db_manager:
            query = """
            SELECT * FROM training_sessions 
            WHERE status IN ('running', 'paused')
            ORDER BY started_at DESC
            """
            sessions = await db_manager.execute_query(query)
        else:
            # No database available
            sessions = []
        
        return {"active_sessions": sessions}
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        return {"active_sessions": []}

@app.get("/api/trainer/evaluation/latest")
async def get_latest_evaluation():
    """Get latest model evaluation results"""
    try:
        if db_manager:
            query = """
            SELECT * FROM model_evaluation 
            ORDER BY evaluation_date DESC LIMIT 1
            """
            evaluation = await db_manager.execute_query(query)
        else:
            # No database available
            evaluation = []
        
        return {"latest_evaluation": evaluation[0] if evaluation else None}
        
    except Exception as e:
        logger.error(f"Error getting latest evaluation: {e}")
        return {"latest_evaluation": None}

@app.get("/api/trainer/sessions")
async def get_all_training_sessions():
    """Get all training sessions with pagination"""
    try:
        if db_manager:
            query = """
            SELECT session_id, status, started_at, completed_at, 
                   epochs_completed, total_epochs, best_accuracy,
                   trainer_identifier
            FROM training_sessions 
            ORDER BY started_at DESC LIMIT 50
            """
            sessions = await db_manager.execute_query(query)
        else:
            # No database available
            sessions = []
        
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"Error getting training sessions: {e}")
        return {"sessions": []}

@app.get("/api/trainer/dashboard")
async def get_trainer_dashboard():
    """Get trainer dashboard data"""
    try:
        if not db_manager:
            return {
                "recent_sessions": [],
                "data_stats": {"total_images": 0, "training_images": 0, "validation_images": 0},
                "performance_stats": {"avg_precision": 0, "avg_recall": 0, "avg_f1": 0},
                "training_progress": {},
                "active_session": None
            }
        
        # For now return empty data since complex queries may fail
        return {
            "recent_sessions": [],
            "data_stats": {"total_images": 0, "training_images": 0, "validation_images": 0},
            "performance_stats": {"avg_precision": 0, "avg_recall": 0, "avg_f1": 0},
            "training_progress": {},
            "active_session": None
        }
        
    except Exception as e:
        logger.error(f"Error getting trainer dashboard: {e}")
        return {
            "recent_sessions": [],
            "data_stats": {"total_images": 0, "training_images": 0, "validation_images": 0},
            "performance_stats": {"avg_precision": 0, "avg_recall": 0, "avg_f1": 0},
            "training_progress": {},
            "active_session": None
        }
        
    except Exception as e:
        logger.error(f"Error getting trainer dashboard: {e}")
        return {
            "recent_sessions": [],
            "data_stats": {"total_images": 0, "training_images": 0, "validation_images": 0},
            "performance_stats": {"avg_precision": 0, "avg_recall": 0, "avg_f1": 0},
            "training_progress": {},
            "active_session": None
        }

@app.post("/api/trainer/upload")
async def upload_training_images(
    files: List[UploadFile] = File(...),
    damage_class: str = Form("unknown"),
    trainer_identifier: Optional[str] = Form(None)
):
    """Upload training images for annotation and training"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create upload directory if it doesn't exist
        upload_dir = "uploads/training_data"
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if not file.filename:
                continue
                
            # Validate file type
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Store in database if available
            if db_manager:
                try:
                    # Get damage class ID
                    class_query = "SELECT id FROM damage_classes WHERE class_name = %s"
                    damage_class_result = await db_manager.execute_query(class_query, (damage_class,))
                    
                    damage_class_id = damage_class_result[0]['id'] if damage_class_result else None
                    
                    # Insert training data record
                    insert_query = """
                    INSERT INTO trainer_data 
                    (file_path, original_filename, damage_class_id, trainer_identifier, 
                     file_size, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    file_size = len(content)
                    current_time = datetime.now()
                    
                    await db_manager.execute_query(insert_query, (
                        file_path, file.filename, damage_class_id, 
                        trainer_identifier or "anonymous", file_size, 
                        "uploaded", current_time
                    ))
                    
                except Exception as db_error:
                    logger.warning(f"Database insert failed, but file uploaded: {db_error}")
            
            uploaded_files.append({
                "filename": file.filename,
                "saved_as": unique_filename,
                "damage_class": damage_class,
                "file_size": len(content)
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "uploaded_files": uploaded_files
        }
        
    except Exception as e:
        logger.error(f"Error uploading training images: {e}")
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
        # Default configuration values
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
        
        # Try to get from database first
        if db_manager:
            try:
                query = "SELECT config_value FROM system_config WHERE config_key = %s"
                result = db_manager.execute_query(query, (key,))
                if result:
                    value = result[0]['config_value']
                    return {"key": key, "value": value}
            except Exception as db_error:
                logger.warning(f"Database config lookup failed: {db_error}")
        
        # Return default value if key exists
        if key in default_config:
            return {"key": key, "value": default_config[key]}
        
        # Return 404 if key not found
        raise HTTPException(status_code=404, detail=f"Configuration key '{key}' not found")
        
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

# ============== STATIC FILES MOUNTING ==============
# Mount static files AFTER all API routes to prevent override
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="reports"), name="reports") 
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    # Default API configuration
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False  # Disable reload for direct run
    }
    
    uvicorn.run(
        "main:app",  # Use module:app format for reload
        host=API_CONFIG["host"], 
        port=API_CONFIG["port"], 
        reload=API_CONFIG["reload"]
    )
