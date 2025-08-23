"""
Feedback Learning Endpoints for AI Training
Handles human feedback collection and AI learning improvements
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from services.db import get_db_connection
from services.training_service import TrainingService
from loguru import logger

router = APIRouter(prefix="/api/training")

# Pydantic models
class HumanFeedback(BaseModel):
    inspection_id: str
    human_assessment: str  # LOW, MEDIUM, HIGH
    feedback_notes: Optional[str] = None
    feedback_date: datetime

class LearningTrigger(BaseModel):
    force_retrain: bool = False

# Initialize services
training_service = TrainingService()

@router.post("/feedback")
async def submit_human_feedback(feedback: HumanFeedback):
    """Submit human feedback for AI learning"""
    try:
        conn = get_db_connection()
        
        # Store feedback in database
        query = """
        INSERT INTO human_feedback 
        (inspection_id, human_assessment, feedback_notes, feedback_date, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        conn.execute(query, (
            feedback.inspection_id,
            feedback.human_assessment,
            feedback.feedback_notes,
            feedback.feedback_date,
            datetime.now()
        ))
        conn.commit()
        
        logger.info(f"Human feedback submitted for inspection {feedback.inspection_id}")
        
        return {
            "message": "Feedback submitted successfully",
            "inspection_id": feedback.inspection_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/pending")
async def get_pending_feedback():
    """Get pending feedback for review"""
    try:
        conn = get_db_connection()
        
        query = """
        SELECT 
            hf.inspection_id,
            hf.human_assessment,
            hf.feedback_notes,
            hf.feedback_date,
            pi.pipeline_id,
            pi.ai_assessment,
            pi.confidence_score
        FROM human_feedback hf
        JOIN pipeline_inspections pi ON hf.inspection_id = pi.inspection_id
        WHERE hf.processed = FALSE
        ORDER BY hf.feedback_date DESC
        LIMIT 20
        """
        
        result = conn.execute(query)
        feedbacks = []
        
        for row in result:
            feedbacks.append({
                "inspection_id": row[0],
                "human_assessment": row[1],
                "feedback_notes": row[2],
                "feedback_date": row[3],
                "pipeline_id": row[4],
                "ai_assessment": row[5],
                "confidence_score": row[6]
            })
        
        return feedbacks
        
    except Exception as e:
        logger.error(f"Error loading pending feedback: {e}")
        return []

@router.get("/learning-progress")
async def get_learning_progress():
    """Get AI learning progress statistics"""
    try:
        conn = get_db_connection()
        
        # Count total feedback
        total_feedback_query = "SELECT COUNT(*) FROM human_feedback"
        total_feedback = conn.execute(total_feedback_query).scalar()
        
        # Calculate accuracy improvement
        accuracy_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN human_assessment = ai_assessment THEN 1 ELSE 0 END) as matches
        FROM human_feedback hf
        JOIN pipeline_inspections pi ON hf.inspection_id = pi.inspection_id
        WHERE hf.feedback_date >= %s
        """
        
        last_month = datetime.now() - timedelta(days=30)
        result = conn.execute(accuracy_query, (last_month,))
        row = result.fetchone()
        
        accuracy = 0
        if row and row[0] > 0:
            accuracy = (row[1] / row[0]) * 100
        
        # Get last training update
        last_update_query = """
        SELECT MAX(completed_at) FROM yolo_training_sessions 
        WHERE status = 'completed'
        """
        last_update = conn.execute(last_update_query).scalar()
        
        return {
            "total_feedback": total_feedback,
            "accuracy_improvement": f"{accuracy:.1f}%",
            "last_update": last_update
        }
        
    except Exception as e:
        logger.error(f"Error loading learning progress: {e}")
        return {
            "total_feedback": 0,
            "accuracy_improvement": "N/A",
            "last_update": None
        }

@router.post("/trigger-learning")
async def trigger_learning_update(trigger: LearningTrigger):
    """Trigger AI learning update based on human feedback"""
    try:
        # Start background learning process
        result = await training_service.trigger_feedback_learning(trigger.force_retrain)
        
        return {
            "message": "Learning update triggered successfully",
            "session_id": result.get("session_id"),
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Error triggering learning update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/statistics")
async def get_feedback_statistics():
    """Get detailed feedback statistics"""
    try:
        conn = get_db_connection()
        
        # Feedback distribution by assessment
        distribution_query = """
        SELECT 
            human_assessment,
            COUNT(*) as count
        FROM human_feedback
        GROUP BY human_assessment
        """
        
        result = conn.execute(distribution_query)
        distribution = {row[0]: row[1] for row in result}
        
        # Accuracy trends over time
        trends_query = """
        SELECT 
            DATE(feedback_date) as date,
            COUNT(*) as total,
            SUM(CASE WHEN hf.human_assessment = pi.ai_assessment THEN 1 ELSE 0 END) as matches
        FROM human_feedback hf
        JOIN pipeline_inspections pi ON hf.inspection_id = pi.inspection_id
        WHERE feedback_date >= %s
        GROUP BY DATE(feedback_date)
        ORDER BY date DESC
        LIMIT 30
        """
        
        last_month = datetime.now() - timedelta(days=30)
        result = conn.execute(trends_query, (last_month,))
        
        trends = []
        for row in result:
            accuracy = (row[2] / row[1] * 100) if row[1] > 0 else 0
            trends.append({
                "date": row[0],
                "total": row[1],
                "accuracy": accuracy
            })
        
        return {
            "distribution": distribution,
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Error loading feedback statistics: {e}")
        return {"distribution": {}, "trends": []}
