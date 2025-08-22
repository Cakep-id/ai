"""
CAKEP.id Early Warning System (EWS) AI Module
FastAPI application untuk damage detection, NLP analysis, risk assessment, dan automated scheduling

Features:
- YOLO-based computer vision untuk damage detection  
- Groq AI untuk NLP analysis
- Risk assessment engine yang menggabungkan CV + NLP
- Automated maintenance scheduling
- Admin interface untuk upload dan retrain
"""

import os
import asyncio
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration (with fallback)
try:
    from config import settings, get_config_summary
except ImportError:
    # Fallback jika config.py tidak tersedia
    class Settings:
        api_host = "0.0.0.0"
        api_port = 8000
        cors_origins_list = ["*"]
        debug = True
    settings = Settings()
    def get_config_summary():
        return {"config": "fallback mode"}

# Import services
from services import (
    db_service, 
    yolo_service, 
    groq_service, 
    risk_engine, 
    scheduler_service
)

# Initialize FastAPI app
app = FastAPI(
    title="CAKEP.id EWS AI Module",
    description="Early Warning System untuk Pemeliharaan Aset dengan AI Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
cors_origins = getattr(settings, 'cors_origins_list', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
from api import cv_endpoints, nlp_endpoints, risk_endpoints, schedule_endpoints, admin_endpoints

app.include_router(cv_endpoints.router, prefix="/api/cv", tags=["Computer Vision"])
app.include_router(nlp_endpoints.router, prefix="/api/nlp", tags=["NLP Analysis"])
app.include_router(risk_endpoints.router, prefix="/api/risk", tags=["Risk Assessment"])
app.include_router(schedule_endpoints.router, prefix="/api/schedule", tags=["Scheduling"])
app.include_router(admin_endpoints.router, prefix="/api/admin", tags=["Administration"])

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("🚀 Starting CAKEP.id EWS AI Module...")
    
    # Setup logging
    log_level = getattr(settings, 'log_level', 'INFO')
    logger.add(
        "logs/ews_ai.log",
        rotation="100 MB",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
    )
    
    # Initialize database
    try:
        db_health = db_service.health_check()
        if db_health['success']:
            logger.info("✅ Database connection established")
        else:
            logger.warning("⚠️ Database connection issues")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    
    # Test services
    logger.info("🔧 Testing services...")
    
    # Test YOLO service
    try:
        yolo_health = yolo_service.health_check()
        if yolo_health['success']:
            logger.info("✅ YOLO service ready")
        else:
            logger.warning("⚠️ YOLO service issues")
    except Exception as e:
        logger.error(f"❌ YOLO service error: {e}")
    
    # Test Groq service
    try:
        groq_health = groq_service.test_connection()
        if groq_health['success']:
            logger.info("✅ Groq AI service ready")
        else:
            logger.warning("⚠️ Groq AI service issues")
    except Exception as e:
        logger.error(f"❌ Groq AI service error: {e}")
    
    # Test Risk Engine
    try:
        test_result = risk_engine.aggregate_risk(0.5, 0.5, 'test', 'test')
        if test_result['success']:
            logger.info("✅ Risk Engine ready")
        else:
            logger.warning("⚠️ Risk Engine issues")
    except Exception as e:
        logger.error(f"❌ Risk Engine error: {e}")
    
    logger.info("🎯 CAKEP.id EWS AI Module startup completed!")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Shutting down CAKEP.id EWS AI Module...")
    
    # Close database connections
    try:
        db_service.close_connections()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")
    
    logger.info("👋 CAKEP.id EWS AI Module shutdown completed!")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint dengan informasi sistem"""
    return {
        "message": "CAKEP.id Early Warning System AI Module",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "config": "/config",
            "cv_api": "/api/cv",
            "nlp_api": "/api/nlp",
            "risk_api": "/api/risk",
            "schedule_api": "/api/schedule",
            "admin_api": "/api/admin"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Database health
        try:
            db_health = db_service.health_check()
            health_status["services"]["database"] = db_health
        except Exception as e:
            health_status["services"]["database"] = {"success": False, "error": str(e)}
        
        # YOLO health
        try:
            yolo_health = yolo_service.health_check()
            health_status["services"]["yolo"] = yolo_health
        except Exception as e:
            health_status["services"]["yolo"] = {"success": False, "error": str(e)}
        
        # Groq health
        try:
            groq_health = groq_service.test_connection()
            health_status["services"]["groq"] = groq_health
        except Exception as e:
            health_status["services"]["groq"] = {"success": False, "error": str(e)}
        
        # Risk engine health
        try:
            risk_test = risk_engine.aggregate_risk(0.5, 0.5, 'test', 'test')
            health_status["services"]["risk_engine"] = {"success": risk_test["success"]}
        except Exception as e:
            health_status["services"]["risk_engine"] = {"success": False, "error": str(e)}
        
        # Overall status
        all_healthy = all(
            service.get("success", False) 
            for service in health_status["services"].values()
        )
        
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Configuration endpoint
@app.get("/config")
async def get_configuration():
    """Get system configuration (non-sensitive info)"""
    try:
        if hasattr(settings, '__dict__'):
            config_summary = get_config_summary()
        else:
            config_summary = {"mode": "fallback"}
        
        return {
            "success": True,
            "config": config_summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get configuration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Run application
if __name__ == "__main__":
    # Get configuration
    host = getattr(settings, 'api_host', '0.0.0.0')
    port = getattr(settings, 'api_port', 8000)
    debug_mode = getattr(settings, 'debug', True)
    
    logger.info(f"Starting server at http://{host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug_mode,
        access_log=True
    )
