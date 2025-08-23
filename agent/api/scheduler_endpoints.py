"""
Scheduler Endpoints for Maintenance Management
Handles maintenance scheduling and pipeline maintenance tasks
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from services.db import get_db_connection
from services.scheduler import SchedulerService
from loguru import logger

router = APIRouter(prefix="/api/scheduler")

# Pydantic models
class MaintenanceTask(BaseModel):
    pipeline_id: str
    maintenance_date: datetime
    maintenance_type: str  # routine, repair, replacement, emergency
    priority: str  # routine, urgent, critical
    description: Optional[str] = None
    assigned_to: Optional[str] = None

class AutoScheduleRequest(BaseModel):
    inspection_id: str
    priority: str
    auto_generated: bool = True
    created_date: datetime

# Initialize services
scheduler_service = SchedulerService()

@router.post("/maintenance")
async def create_maintenance_task(task: MaintenanceTask):
    """Create a new maintenance task"""
    try:
        conn = get_db_connection()
        
        # Insert maintenance task
        query = """
        INSERT INTO maintenance_schedule 
        (pipeline_id, maintenance_date, maintenance_type, priority, description, assigned_to, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        conn.execute(query, (
            task.pipeline_id,
            task.maintenance_date,
            task.maintenance_type,
            task.priority,
            task.description,
            task.assigned_to,
            'scheduled',
            datetime.now()
        ))
        conn.commit()
        
        logger.info(f"Maintenance task created for pipeline {task.pipeline_id}")
        
        return {
            "message": "Maintenance task created successfully",
            "pipeline_id": task.pipeline_id,
            "maintenance_date": task.maintenance_date,
            "status": "scheduled"
        }
        
    except Exception as e:
        logger.error(f"Error creating maintenance task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auto-schedule")
async def auto_schedule_maintenance(request: AutoScheduleRequest):
    """Auto-schedule maintenance based on inspection results"""
    try:
        conn = get_db_connection()
        
        # Get inspection details
        inspection_query = """
        SELECT pipeline_id, risk_level, risk_score, location
        FROM pipeline_inspections 
        WHERE inspection_id = %s
        """
        
        result = conn.execute(inspection_query, (request.inspection_id,))
        inspection = result.fetchone()
        
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        pipeline_id, risk_level, risk_score, location = inspection
        
        # Calculate maintenance date based on risk level
        today = datetime.now()
        if risk_level == 'HIGH':
            maintenance_date = today + timedelta(days=1)  # Tomorrow
            maintenance_type = 'emergency'
        elif risk_level == 'MEDIUM':
            maintenance_date = today + timedelta(days=7)  # Next week
            maintenance_type = 'repair'
        else:
            maintenance_date = today + timedelta(days=30)  # Next month
            maintenance_type = 'routine'
        
        # Create auto-scheduled task
        task_query = """
        INSERT INTO maintenance_schedule 
        (pipeline_id, maintenance_date, maintenance_type, priority, description, status, auto_generated, inspection_id, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        description = f"Auto-scheduled maintenance based on {risk_level} risk inspection (score: {risk_score:.3f})"
        
        conn.execute(task_query, (
            pipeline_id,
            maintenance_date,
            maintenance_type,
            request.priority,
            description,
            'scheduled',
            True,
            request.inspection_id,
            datetime.now()
        ))
        conn.commit()
        
        logger.info(f"Auto-scheduled maintenance for pipeline {pipeline_id} based on inspection {request.inspection_id}")
        
        return {
            "message": "Maintenance auto-scheduled successfully",
            "pipeline_id": pipeline_id,
            "maintenance_date": maintenance_date,
            "maintenance_type": maintenance_type,
            "priority": request.priority
        }
        
    except Exception as e:
        logger.error(f"Error auto-scheduling maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/maintenance")
async def get_maintenance_schedule():
    """Get all maintenance tasks"""
    try:
        conn = get_db_connection()
        
        query = """
        SELECT 
            id, pipeline_id, maintenance_date, maintenance_type, priority, 
            description, assigned_to, status, auto_generated, created_at
        FROM maintenance_schedule
        WHERE status IN ('scheduled', 'in_progress')
        ORDER BY 
            CASE priority 
                WHEN 'critical' THEN 1 
                WHEN 'urgent' THEN 2 
                WHEN 'routine' THEN 3 
            END,
            maintenance_date ASC
        """
        
        result = conn.execute(query)
        schedules = []
        
        for row in result:
            schedules.append({
                "id": row[0],
                "pipeline_id": row[1],
                "maintenance_date": row[2],
                "maintenance_type": row[3],
                "priority": row[4],
                "description": row[5],
                "assigned_to": row[6],
                "status": row[7],
                "auto_generated": row[8],
                "created_at": row[9]
            })
        
        return schedules
        
    except Exception as e:
        logger.error(f"Error loading maintenance schedule: {e}")
        return []

@router.get("/maintenance/upcoming")
async def get_upcoming_maintenance():
    """Get upcoming maintenance tasks (next 7 days)"""
    try:
        conn = get_db_connection()
        
        next_week = datetime.now() + timedelta(days=7)
        
        query = """
        SELECT 
            pipeline_id, maintenance_date, maintenance_type, priority, assigned_to
        FROM maintenance_schedule
        WHERE status = 'scheduled' 
        AND maintenance_date BETWEEN %s AND %s
        ORDER BY maintenance_date ASC
        """
        
        result = conn.execute(query, (datetime.now(), next_week))
        upcoming = []
        
        for row in result:
            upcoming.append({
                "pipeline_id": row[0],
                "maintenance_date": row[1],
                "maintenance_type": row[2],
                "priority": row[3],
                "assigned_to": row[4]
            })
        
        return upcoming
        
    except Exception as e:
        logger.error(f"Error loading upcoming maintenance: {e}")
        return []

@router.put("/maintenance/{task_id}/status")
async def update_maintenance_status(task_id: int, status: str):
    """Update maintenance task status"""
    try:
        conn = get_db_connection()
        
        valid_statuses = ['scheduled', 'in_progress', 'completed', 'cancelled']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        query = """
        UPDATE maintenance_schedule 
        SET status = %s, updated_at = %s
        WHERE id = %s
        """
        
        conn.execute(query, (status, datetime.now(), task_id))
        conn.commit()
        
        return {
            "message": "Maintenance status updated successfully",
            "task_id": task_id,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error updating maintenance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/maintenance/statistics")
async def get_maintenance_statistics():
    """Get maintenance statistics"""
    try:
        conn = get_db_connection()
        
        # Count by status
        status_query = """
        SELECT status, COUNT(*) as count
        FROM maintenance_schedule
        GROUP BY status
        """
        
        result = conn.execute(status_query)
        status_counts = {row[0]: row[1] for row in result}
        
        # Count by priority
        priority_query = """
        SELECT priority, COUNT(*) as count
        FROM maintenance_schedule
        WHERE status IN ('scheduled', 'in_progress')
        GROUP BY priority
        """
        
        result = conn.execute(priority_query)
        priority_counts = {row[0]: row[1] for row in result}
        
        # Overdue tasks
        overdue_query = """
        SELECT COUNT(*) FROM maintenance_schedule
        WHERE status = 'scheduled' AND maintenance_date < %s
        """
        
        overdue_count = conn.execute(overdue_query, (datetime.now(),)).scalar()
        
        return {
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "overdue_tasks": overdue_count
        }
        
    except Exception as e:
        logger.error(f"Error loading maintenance statistics: {e}")
        return {
            "status_distribution": {},
            "priority_distribution": {},
            "overdue_tasks": 0
        }
