"""
Simple FastAPI Backend for AgentV2
Testing version with mock services
"""

import os
import sys
import logging
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database manager with fallback
try:
    from backend.db_manager_simple import DatabaseManager
    print("Simple Database Manager imported successfully")
except ImportError as e:
    print(f"Error importing simple database manager: {e}")
    DatabaseManager = None

# Import AI services with error handling
try:
    from ai_models.yolo_service_simple import YOLOService
    from ai_models.risk_engine_simple import RiskEngine
    from ai_models.report_service_simple import ReportService
    print("Simple AI services imported successfully")
except ImportError as e:
    print(f"Error importing simple AI services: {e}")
    YOLOService = None
    RiskEngine = None
    ReportService = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AgentV2 AI Asset Inspection System (Simple)",
    description="Simple AI-powered asset inspection for testing",
    version="2.0.0-simple"
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

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global db_manager, yolo_service, risk_engine, report_service
    
    try:
        # Initialize database
        if DatabaseManager:
            db_manager = DatabaseManager()
            await db_manager.connect()
            print("Database Manager initialized")
        
        # Initialize AI services
        if YOLOService:
            yolo_service = YOLOService()
            print("YOLO Service initialized")
        
        if RiskEngine:
            risk_engine = RiskEngine()
            print("Risk Engine initialized")
        
        if ReportService:
            report_service = ReportService()
            print("Report Service initialized")
        
        print("All services initialized successfully")
        
    except Exception as e:
        print(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_manager
    
    if db_manager:
        await db_manager.disconnect()
        print("Database disconnected")

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

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AgentV2 AI Asset Inspection System (Simple)",
        "version": "2.0.0-simple",
        "status": "running",
        "mode": "testing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_manager is not None,
            "yolo": yolo_service is not None,
            "risk_engine": risk_engine is not None,
            "report_service": report_service is not None
        }
    }
    
    if db_manager:
        try:
            db_health = await db_manager.health_check()
            health_status["database_status"] = db_health
        except Exception as e:
            health_status["database_error"] = str(e)
    
    return health_status

# User endpoints
@app.post("/api/user/submit-report")
async def submit_user_report(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    asset_description: str = Form(...),
    user_identifier: Optional[str] = Form(None)
):
    """Submit user report with image"""
    try:
        # Generate report ID
        report_id = f"USR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Save uploaded image
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, f"{report_id}_{image.filename}")
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Create report data
        report_data = {
            "report_id": report_id,
            "user_identifier": user_identifier or "anonymous",
            "asset_description": asset_description,
            "image_path": image_path,
            "status": "submitted",
            "submitted_at": datetime.now().isoformat()
        }
        
        # Save to database if available
        if db_manager:
            await db_manager.create_inspection(report_data)
        
        # Process in background
        background_tasks.add_task(process_user_report, report_id, image_path, asset_description)
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Report submitted successfully and is being processed"
        }
        
    except Exception as e:
        logger.error(f"Error submitting user report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_user_report(report_id: str, image_path: str, asset_description: str):
    """Process user report in background"""
    try:
        # Run YOLO detection
        if yolo_service:
            yolo_results = await yolo_service.detect_damage(image_path)
        else:
            yolo_results = {"detections": [], "model_info": "mock"}
        
        # Run risk analysis
        if risk_engine:
            risk_analysis = await risk_engine.analyze_risk(report_id, yolo_results)
        else:
            risk_analysis = {"overall_risk_score": 1.0, "risk_category": "LOW"}
        
        # Generate report
        if report_service:
            report_data = {
                "report_id": report_id,
                "asset_description": asset_description,
                "yolo_results": yolo_results,
                "risk_analysis": risk_analysis
            }
            report_result = await report_service.generate_inspection_report(report_data)
        else:
            report_result = {"success": True, "report": {"status": "completed"}}
        
        # Update status
        if db_manager:
            await db_manager.update_inspection(report_id, {
                "status": "completed",
                "yolo_results": yolo_results,
                "risk_analysis": risk_analysis,
                "completed_at": datetime.now().isoformat()
            })
        
        print(f"Report {report_id} processed successfully")
        
    except Exception as e:
        print(f"Error processing report {report_id}: {e}")
        if db_manager:
            await db_manager.update_inspection(report_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })

@app.get("/api/user/report/{report_id}")
async def get_user_report(report_id: str):
    """Get user report by ID"""
    try:
        if db_manager:
            report = await db_manager.get_inspection(report_id)
            if report:
                return {"success": True, "report": report}
        
        return {"success": False, "error": "Report not found"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/reports")
async def list_user_reports(limit: int = 10):
    """List user reports"""
    try:
        if db_manager:
            reports = await db_manager.list_inspections(limit=limit)
            return {"success": True, "reports": reports}
        
        return {"success": True, "reports": []}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@app.get("/api/admin/dashboard")
async def admin_dashboard():
    """Admin dashboard data"""
    try:
        dashboard_data = {
            "total_reports": 0,
            "pending_reports": 0,
            "completed_reports": 0,
            "failed_reports": 0,
            "system_status": "operational"
        }
        
        if db_manager:
            stats = db_manager.get_stats()
            dashboard_data.update(stats)
        
        return {"success": True, "data": dashboard_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/reports")
async def admin_list_reports(status: Optional[str] = None, limit: int = 50):
    """List all reports for admin"""
    try:
        if db_manager:
            reports = await db_manager.list_inspections(limit=limit)
            if status:
                reports = [r for r in reports if r.get('status') == status]
            return {"success": True, "reports": reports}
        
        return {"success": True, "reports": []}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Trainer endpoints
@app.post("/api/trainer/submit-training-data")
async def submit_training_data(data: TrainingDataRequest):
    """Submit training data"""
    try:
        training_id = f"TRN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        training_data = {
            "training_id": training_id,
            "trainer_identifier": data.trainer_identifier,
            "asset_description": data.asset_description,
            "risk_category": data.risk_category,
            "manual_annotations": data.manual_annotations,
            "training_notes": data.training_notes,
            "submitted_at": datetime.now().isoformat(),
            "status": "submitted"
        }
        
        if db_manager:
            await db_manager.save_training_session(training_data)
        
        return {
            "success": True,
            "training_id": training_id,
            "message": "Training data submitted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trainer/sessions")
async def list_training_sessions(limit: int = 20):
    """List training sessions"""
    try:
        if db_manager:
            sessions = await db_manager.list_training_sessions(limit=limit)
            return {"success": True, "sessions": sessions}
        
        return {"success": True, "sessions": []}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoints
@app.post("/api/test/yolo")
async def test_yolo_detection(image: UploadFile = File(...)):
    """Test YOLO detection"""
    try:
        # Save test image
        test_dir = "temp_uploads"
        os.makedirs(test_dir, exist_ok=True)
        image_path = os.path.join(test_dir, f"test_{image.filename}")
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Run detection
        if yolo_service:
            results = await yolo_service.detect_damage(image_path)
        else:
            results = {"detections": [], "message": "YOLO service not available"}
        
        # Cleanup
        os.remove(image_path)
        
        return {"success": True, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/risk-analysis")
async def test_risk_analysis(yolo_results: Dict[str, Any]):
    """Test risk analysis"""
    try:
        if risk_engine:
            test_report_id = f"TEST_{str(uuid.uuid4())[:8]}"
            results = await risk_engine.analyze_risk(test_report_id, yolo_results)
        else:
            results = {"message": "Risk engine not available"}
        
        return {"success": True, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static files - check if frontend directory exists
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    
    @app.get("/user")
    async def serve_user_interface():
        """Serve user interface"""
        return FileResponse("frontend/user.html")
    
    @app.get("/admin")
    async def serve_admin_interface():
        """Serve admin interface"""
        return FileResponse("frontend/admin.html")
    
    @app.get("/trainer")
    async def serve_trainer_interface():
        """Serve trainer interface"""
        return FileResponse("frontend/trainer.html")
else:
    @app.get("/user")
    async def serve_user_interface():
        """Serve user interface placeholder"""
        return {"message": "User interface not available - frontend directory not found"}
    
    @app.get("/admin")
    async def serve_admin_interface():
        """Serve admin interface placeholder"""
        return {"message": "Admin interface not available - frontend directory not found"}
    
    @app.get("/trainer")
    async def serve_trainer_interface():
        """Serve trainer interface placeholder"""
        return {"message": "Trainer interface not available - frontend directory not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
