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
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration and services
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

from services import (
    db_service, 
    yolo_service, 
    groq_service, 
    risk_engine, 
    scheduler_service
)

# Import services untuk health check
from services import db_service

# Initialize FastAPI app
app = FastAPI(
    title="CAKEP.id EWS API",
    description="Early Warning System untuk Pemeliharaan Aset Industri",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan domain production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files jika ada
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routers
app.include_router(cv_router, prefix="/cv", tags=["Computer Vision"])
app.include_router(nlp_router, prefix="/nlp", tags=["Natural Language Processing"])
app.include_router(risk_router, prefix="/risk", tags=["Risk Assessment"])
app.include_router(schedule_router, prefix="/schedule", tags=["Scheduling"])
app.include_router(admin_router, prefix="/admin", tags=["Administration"])

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting CAKEP.id EWS API...")
    
    # Initialize database
    try:
        from services.db import init_database
        if init_database():
            logger.info("Database initialized successfully")
        else:
            logger.error("Database initialization failed")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    
    # Test services
    logger.info("Testing services...")
    
    # Test YOLO service
    try:
        from services import yolo_service
        model_info = yolo_service.get_model_info()
        logger.info(f"YOLO service ready: {model_info.get('model_loaded', False)}")
    except Exception as e:
        logger.warning(f"YOLO service error: {e}")
    
    # Test Groq service
    try:
        from services import groq_service
        connection_test = groq_service.test_connection()
        logger.info(f"Groq service ready: {connection_test.get('success', False)}")
    except Exception as e:
        logger.warning(f"Groq service error: {e}")
    
    logger.info("CAKEP.id EWS API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down CAKEP.id EWS API...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CAKEP.id Early Warning System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "computer_vision": "/cv/",
            "nlp": "/nlp/", 
            "risk_assessment": "/risk/",
            "scheduling": "/schedule/",
            "admin": "/admin/"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = db_service.test_connection()
        
        # Test services status
        services_status = {}
        
        # YOLO service
        try:
            from services import yolo_service
            model_info = yolo_service.get_model_info()
            services_status['yolo'] = {
                'status': 'ok' if model_info.get('model_loaded', False) else 'warning',
                'model_loaded': model_info.get('model_loaded', False),
                'device': model_info.get('device', 'unknown')
            }
        except Exception as e:
            services_status['yolo'] = {'status': 'error', 'error': str(e)}
        
        # Groq service
        try:
            from services import groq_service
            connection_test = groq_service.test_connection()
            services_status['groq'] = {
                'status': 'ok' if connection_test.get('success', False) else 'error',
                'connected': connection_test.get('success', False)
            }
        except Exception as e:
            services_status['groq'] = {'status': 'error', 'error': str(e)}
        
        # Risk engine
        try:
            from services import risk_engine
            config = risk_engine.get_configuration()
            services_status['risk_engine'] = {
                'status': 'ok',
                'version': config.get('engine_version', 'unknown')
            }
        except Exception as e:
            services_status['risk_engine'] = {'status': 'error', 'error': str(e)}
        
        # Scheduler
        try:
            from services import scheduler_service
            config = scheduler_service.get_configuration()
            services_status['scheduler'] = {
                'status': 'ok',
                'version': config.get('scheduler_version', 'unknown')
            }
        except Exception as e:
            services_status['scheduler'] = {'status': 'error', 'error': str(e)}
        
        # Overall status
        all_services_ok = all(
            service.get('status') == 'ok' 
            for service in services_status.values()
        )
        
        overall_status = 'healthy' if db_status and all_services_ok else 'degraded'
        
        return {
            "status": overall_status,
            "timestamp": str(logger._core.now()),
            "database": {
                "status": "connected" if db_status else "disconnected"
            },
            "services": services_status,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get system configuration"""
    try:
        config = {
            "api_version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "services": {}
        }
        
        # Get services configuration
        try:
            from services import risk_engine, scheduler_service
            config["services"]["risk_engine"] = risk_engine.get_configuration()
            config["services"]["scheduler"] = scheduler_service.get_configuration()
        except Exception as e:
            logger.warning(f"Error getting services config: {e}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

if __name__ == "__main__":
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.add(
        "./logs/cakep_ews.log",
        rotation="1 day",
        retention="7 days",
        level=log_level
    )
    
    logger.info(f"Starting CAKEP.id EWS API on {host}:{port}")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level.lower()
    )
