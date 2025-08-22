"""
Scheduler Service untuk CAKEP.id EWS - Simplified Version
Auto generate jadwal pemeliharaan berdasarkan risk assessment
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SchedulerService:
    """Service untuk generate dan manage jadwal pemeliharaan"""
    
    def __init__(self):
        # SLA configuration (dalam jam)
        self.sla_hours = {
            'CRITICAL': 1,   # 1 jam
            'HIGH': 4,       # 4 jam
            'MEDIUM': 24,    # 24 jam  
            'LOW': 168       # 168 jam (7 hari)
        }
        
        logger.info("Scheduler Service initialized")
    
    async def create_maintenance_schedule(self, asset_id: int, report_id: int, 
                                        priority: str, maintenance_type: str = 'corrective',
                                        procedures: List[str] = None):
        """
        Create maintenance schedule otomatis untuk laporan dengan risk tinggi
        """
        try:
            from .db import db_service
            
            # Determine scheduled date based on priority
            hours_to_add = self.sla_hours.get(priority, 24)
            scheduled_date = datetime.now() + timedelta(hours=hours_to_add)
            
            # Convert procedures list to text
            procedure_text = "\n".join(procedures) if procedures else "Prosedur perbaikan standar"
            
            # Insert ke database
            schedule_query = """
            INSERT INTO maintenance_schedules (
                asset_id, report_id, maintenance_type, scheduled_date, 
                priority, procedure_steps, status, created_at
            ) VALUES (
                :asset_id, :report_id, :maintenance_type, :scheduled_date,
                :priority, :procedure_steps, 'scheduled', :created_at
            )
            """
            
            schedule_id = db_service.execute_insert(schedule_query, {
                "asset_id": asset_id,
                "report_id": report_id,
                "maintenance_type": maintenance_type,
                "scheduled_date": scheduled_date,
                "priority": priority,
                "procedure_steps": procedure_text,
                "created_at": datetime.now()
            })
            
            logger.info(f"Auto-scheduled maintenance {schedule_id} for asset {asset_id} with {priority} priority")
            
            return {
                "success": True,
                "schedule_id": schedule_id,
                "scheduled_date": scheduled_date.isoformat(),
                "priority": priority
            }
            
        except Exception as e:
            logger.error(f"Error creating maintenance schedule: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_maintenance_priority(self, asset_id: int, report_id: int, new_priority: str):
        """
        Update priority jadwal maintenance berdasarkan koreksi admin
        """
        try:
            from .db import db_service
            
            # Update priority dan reschedule jika perlu
            hours_to_add = self.sla_hours.get(new_priority, 24)
            new_scheduled_date = datetime.now() + timedelta(hours=hours_to_add)
            
            update_query = """
            UPDATE maintenance_schedules SET 
                priority = :priority,
                scheduled_date = :scheduled_date
            WHERE asset_id = :asset_id AND report_id = :report_id AND status = 'scheduled'
            """
            
            db_service.execute_insert(update_query, {
                "priority": new_priority,
                "scheduled_date": new_scheduled_date,
                "asset_id": asset_id,
                "report_id": report_id
            })
            
            logger.info(f"Updated maintenance priority to {new_priority} for asset {asset_id}")
            
            return {
                "success": True,
                "new_priority": new_priority,
                "new_scheduled_date": new_scheduled_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating maintenance priority: {e}")
            return {"success": False, "error": str(e)}
    
    def get_upcoming_schedules(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Get jadwal maintenance yang akan datang
        """
        try:
            from .db import db_service
            
            end_date = datetime.now() + timedelta(days=days_ahead)
            
            query = """
            SELECT 
                ms.*,
                a.asset_name,
                a.location,
                ur.description as report_description
            FROM maintenance_schedules ms
            JOIN assets a ON ms.asset_id = a.asset_id
            LEFT JOIN user_reports ur ON ms.report_id = ur.report_id
            WHERE ms.scheduled_date BETWEEN NOW() AND :end_date
            AND ms.status = 'scheduled'
            ORDER BY ms.scheduled_date ASC, ms.priority DESC
            """
            
            schedules = db_service.execute_query(query, {"end_date": end_date})
            
            return {
                "success": True,
                "schedules": schedules,
                "total": len(schedules)
            }
            
        except Exception as e:
            logger.error(f"Error getting upcoming schedules: {e}")
            return {"success": False, "error": str(e)}
    
    def get_overdue_schedules(self) -> Dict[str, Any]:
        """
        Get jadwal maintenance yang terlambat
        """
        try:
            from .db import db_service
            
            query = """
            SELECT 
                ms.*,
                a.asset_name,
                a.location,
                TIMESTAMPDIFF(HOUR, ms.scheduled_date, NOW()) as hours_overdue
            FROM maintenance_schedules ms
            JOIN assets a ON ms.asset_id = a.asset_id
            WHERE ms.scheduled_date < NOW()
            AND ms.status = 'scheduled'
            ORDER BY hours_overdue DESC
            """
            
            overdue = db_service.execute_query(query)
            
            return {
                "success": True,
                "overdue_schedules": overdue,
                "total": len(overdue)
            }
            
        except Exception as e:
            logger.error(f"Error getting overdue schedules: {e}")
            return {"success": False, "error": str(e)}

# Singleton instance
scheduler = SchedulerService()

if __name__ == "__main__":
    # Test scheduler service
    print("Scheduler Service Test")
    print("SLA Hours:", scheduler.sla_hours)
