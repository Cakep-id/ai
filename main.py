from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
from dotenv import load_dotenv

from database import db
from chatbot_engine import chatbot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Cakep.id AI Service",
    description="AI service for chatbot and intelligent responses",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    category: Optional[str] = "faq"

class ChatResponse(BaseModel):
    response: str
    confidence: float
    category: str
    source: str
    matched_question: Optional[str] = None
    data_id: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool
    model_loaded: bool
    knowledge_base_size: dict

class UpdateResponse(BaseModel):
    success: bool
    message: str
    updated_items: int

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("🚀 Starting AI Service...")
        
        # Connect to database
        logger.info("🔄 Connecting to database...")
        if not db.connect():
            logger.error("❌ Failed to connect to database")
            raise Exception("Database connection failed")
        
        # Load initial training data
        logger.info("🔄 Loading training data...")
        await update_knowledge_base()
        
        logger.info("✅ AI Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AI Service: {e}")
        raise e

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down AI Service...")
    db.disconnect()
    logger.info("✅ AI Service shutdown complete")

async def update_knowledge_base():
    """Update chatbot knowledge base from database"""
    try:
        training_data = db.get_training_data()
        chatbot.update_knowledge_base(training_data)
        logger.info(f"📚 Knowledge base updated with {len(training_data)} items")
        return len(training_data)
    except Exception as e:
        logger.error(f"❌ Error updating knowledge base: {e}")
        return 0

# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Cakep.id AI Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        db_connected = db.test_connection()
        model_loaded = chatbot.model is not None
        kb_stats = chatbot.get_stats()
        
        return HealthResponse(
            status="healthy" if db_connected and model_loaded else "unhealthy",
            message="AI Service is running",
            database_connected=db_connected,
            model_loaded=model_loaded,
            knowledge_base_size={
                "total": kb_stats['total_items'],
                "faq": kb_stats['faq_items'],
                "assistant": kb_stats['assistant_items']
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Validate category
        if request.category not in ["faq", "assistant"]:
            request.category = "faq"
        
        # Get response from chatbot
        result = chatbot.find_best_answer(request.message.strip(), request.category)
        
        logger.info(f"💬 Chat request processed: category={request.category}, confidence={result['confidence']:.2f}, source={result['source']}")
        
        return ChatResponse(
            response=result['answer'],
            confidence=result['confidence'],
            category=request.category,
            source=result['source'],
            matched_question=result.get('matched_question'),
            data_id=result.get('data_id')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/update-data", response_model=UpdateResponse)
async def update_data_endpoint():
    """Update knowledge base with latest training data"""
    try:
        updated_items = await update_knowledge_base()
        
        return UpdateResponse(
            success=True,
            message="Knowledge base updated successfully",
            updated_items=updated_items
        )
        
    except Exception as e:
        logger.error(f"❌ Update data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update knowledge base")

@app.get("/stats")
async def get_stats():
    """Get AI service statistics"""
    try:
        kb_stats = chatbot.get_stats()
        db_connected = db.test_connection()
        
        return {
            "ai_service": {
                "status": "running",
                "model_name": kb_stats['model_name'],
                "similarity_threshold": kb_stats['similarity_threshold']
            },
            "database": {
                "connected": db_connected,
                "host": db.host,
                "database": db.database
            },
            "knowledge_base": {
                "total_items": kb_stats['total_items'],
                "faq_items": kb_stats['faq_items'],
                "assistant_items": kb_stats['assistant_items'],
                "embeddings_generated": kb_stats['embeddings_generated']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("AI_HOST", "0.0.0.0")
    port = int(os.getenv("AI_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"🚀 Starting AI Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
