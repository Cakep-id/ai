"""
Validation Service untuk CAKEP.id EWS
Menangani validation workflow dan feedback loop untuk AI learning
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .db import db_service

class ValidationService:
    """Service untuk menangani validation workflow"""
    
    def __init__(self):
        self.auto_validation_threshold = 0.9  # Threshold untuk auto-approve
        
    def submit_for_validation(self, item_type: str, item_id: int, 
                            priority: str = 'MEDIUM') -> Dict:
        """
        Submit item untuk validation queue
        
        Args:
            item_type: Type of item (detection, nlp_analysis, risk_assessment, procedure, schedule)
            item_id: ID of the item
            priority: Priority level (HIGH, MEDIUM, LOW)
        """
        try:
            # Check if already in validation queue
            existing = db_service.execute_query(
                "SELECT * FROM validation_queue WHERE item_type = :item_type AND item_id = :item_id AND status != 'completed'",
                {"item_type": item_type, "item_id": item_id}
            )
            
            if existing:
                return {
                    'success': False,
                    'message': 'Item already in validation queue',
                    'queue_id': existing[0]['queue_id']
                }
            
            # Add to validation queue
            queue_id = db_service.execute_insert(
                """INSERT INTO validation_queue (item_type, item_id, priority, status) 
                   VALUES (:item_type, :item_id, :priority, 'pending')""",
                {"item_type": item_type, "item_id": item_id, "priority": priority}
            )
            
            logger.info(f"Added {item_type} {item_id} to validation queue with ID {queue_id}")
            
            return {
                'success': True,
                'queue_id': queue_id,
                'message': 'Item added to validation queue'
            }
            
        except Exception as e:
            logger.error(f"Failed to submit for validation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_validation_queue(self, validator_id: Optional[int] = None, 
                           status: str = 'pending') -> List[Dict]:
        """Get validation queue items"""
        try:
            if validator_id:
                query = """
                SELECT vq.*, 
                       CASE vq.item_type
                           WHEN 'detection' THEN (SELECT JSON_OBJECT('report_id', ad.report_id, 'confidence', ad.confidence, 'damage_score', ad.damage_score) FROM ai_detections ad WHERE ad.detection_id = vq.item_id)
                           WHEN 'nlp_analysis' THEN (SELECT JSON_OBJECT('report_id', na.report_id, 'category', na.category, 'confidence', na.confidence) FROM nlp_analyses na WHERE na.analysis_id = vq.item_id)
                           WHEN 'risk_assessment' THEN (SELECT JSON_OBJECT('report_id', ra.report_id, 'risk_level', ra.risk_level, 'final_risk_score', ra.final_risk_score) FROM risk_assessments ra WHERE ra.risk_id = vq.item_id)
                           WHEN 'procedure' THEN (SELECT JSON_OBJECT('report_id', ap.report_id, 'risk_level', ap.risk_level) FROM ai_procedures ap WHERE ap.procedure_id = vq.item_id)
                           WHEN 'schedule' THEN (SELECT JSON_OBJECT('report_id', s.report_id, 'priority', s.priority, 'scheduled_date', s.scheduled_date) FROM schedules s WHERE s.schedule_id = vq.item_id)
                       END as item_preview
                FROM validation_queue vq
                WHERE (vq.assigned_to = :validator_id OR vq.assigned_to IS NULL) 
                AND vq.status = :status
                ORDER BY 
                    CASE vq.priority WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 END,
                    vq.created_at ASC
                """
                params = {"validator_id": validator_id, "status": status}
            else:
                query = """
                SELECT vq.*,
                       CASE vq.item_type
                           WHEN 'detection' THEN (SELECT JSON_OBJECT('report_id', ad.report_id, 'confidence', ad.confidence) FROM ai_detections ad WHERE ad.detection_id = vq.item_id)
                           WHEN 'nlp_analysis' THEN (SELECT JSON_OBJECT('report_id', na.report_id, 'category', na.category) FROM nlp_analyses na WHERE na.analysis_id = vq.item_id)
                           WHEN 'risk_assessment' THEN (SELECT JSON_OBJECT('report_id', ra.report_id, 'risk_level', ra.risk_level) FROM risk_assessments ra WHERE ra.risk_id = vq.item_id)
                           WHEN 'procedure' THEN (SELECT JSON_OBJECT('report_id', ap.report_id, 'risk_level', ap.risk_level) FROM ai_procedures ap WHERE ap.procedure_id = vq.item_id)
                           WHEN 'schedule' THEN (SELECT JSON_OBJECT('report_id', s.report_id, 'priority', s.priority) FROM schedules s WHERE s.schedule_id = vq.item_id)
                       END as item_preview
                FROM validation_queue vq
                WHERE vq.status = :status
                ORDER BY 
                    CASE vq.priority WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 END,
                    vq.created_at ASC
                """
                params = {"status": status}
            
            items = db_service.execute_query(query, params)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to get validation queue: {e}")
            return []
    
    def validate_detection(self, detection_id: int, validator_id: int, 
                         is_approved: bool, feedback: str = None,
                         corrected_detections: List[Dict] = None) -> Dict:
        """Validate AI detection result"""
        try:
            validation_status = 'approved' if is_approved else 'rejected'
            
            # Update detection validation status
            db_service.execute_query(
                """UPDATE ai_detections 
                   SET validation_status = :validation_status, validated_by = :validator_id, 
                       validated_at = NOW(), validation_feedback = :feedback
                   WHERE detection_id = :detection_id""",
                {"validation_status": validation_status, "validator_id": validator_id, 
                 "feedback": feedback, "detection_id": detection_id}
            )
            
            # Jika approved, tambahkan ke training data
            if is_approved:
                detection = db_service.execute_query(
                    "SELECT * FROM ai_detections WHERE detection_id = :detection_id",
                    {"detection_id": detection_id}
                )[0]
                
                # Get original report untuk context
                report = db_service.execute_query(
                    "SELECT * FROM reports WHERE report_id = :report_id",
                    {"report_id": detection['report_id']}
                )[0]
                
                training_data = {
                    'image_path': report['image_path'],
                    'detections': corrected_detections or detection['detections'],
                    'validation_notes': feedback
                }
                
                # Add to training data
                db_service.execute_query(
                    """INSERT INTO training_data (data_type, source_id, input_data, expected_output, validation_notes, created_by)
                       VALUES ('yolo_detection', :source_id, :input_data, :expected_output, :validation_notes, :created_by)""",
                    {"source_id": detection_id, "input_data": json.dumps({'image_path': report['image_path']}), 
                     "expected_output": json.dumps(training_data['detections']), 
                     "validation_notes": feedback, "created_by": validator_id}
                )
                
                # Update detection as training data
                db_service.execute_query(
                    "UPDATE ai_detections SET is_training_data = TRUE WHERE detection_id = :detection_id",
                    {"detection_id": detection_id}
                )
            
            # Update validation queue
            self._update_validation_queue('detection', detection_id)
            
            logger.info(f"Detection {detection_id} validated as {validation_status}")
            
            return {
                'success': True,
                'validation_status': validation_status,
                'message': f'Detection {validation_status} successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to validate detection: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_nlp_analysis(self, analysis_id: int, validator_id: int,
                            is_approved: bool, feedback: str = None,
                            corrected_category: str = None) -> Dict:
        """Validate NLP analysis result"""
        try:
            validation_status = 'approved' if is_approved else 'rejected'
            
            # Update NLP analysis validation status
            db_service.execute_query(
                """UPDATE nlp_analyses 
                   SET validation_status = :validation_status, validated_by = :validator_id, 
                       validated_at = NOW(), validation_feedback = :feedback
                   WHERE analysis_id = :analysis_id""",
                {"validation_status": validation_status, "validator_id": validator_id, 
                 "feedback": feedback, "analysis_id": analysis_id}
            )
            
            # Jika approved, tambahkan ke training data
            if is_approved:
                analysis = db_service.execute_query(
                    "SELECT * FROM nlp_analyses WHERE analysis_id = :analysis_id",
                    {"analysis_id": analysis_id}
                )[0]
                
                report = db_service.execute_query(
                    "SELECT description FROM reports WHERE report_id = :report_id",
                    {"report_id": analysis['report_id']}
                )[0]
                
                training_data = {
                    'text': report['description'],
                    'category': corrected_category or analysis['category'],
                    'validation_notes': feedback
                }
                
                # Add to training data
                db_service.execute_query(
                    """INSERT INTO training_data (data_type, source_id, input_data, expected_output, validation_notes, created_by)
                       VALUES ('nlp_analysis', :source_id, :input_data, :expected_output, :validation_notes, :created_by)""",
                    {"source_id": analysis_id, "input_data": json.dumps({'text': report['description']}),
                     "expected_output": json.dumps({'category': training_data['category']}), 
                     "validation_notes": feedback, "created_by": validator_id}
                )
                
                # Update analysis as training data
                db_service.execute_query(
                    "UPDATE nlp_analyses SET is_training_data = TRUE WHERE analysis_id = :analysis_id",
                    {"analysis_id": analysis_id}
                )
            
            # Update validation queue
            self._update_validation_queue('nlp_analysis', analysis_id)
            
            logger.info(f"NLP analysis {analysis_id} validated as {validation_status}")
            
            return {
                'success': True,
                'validation_status': validation_status,
                'message': f'NLP analysis {validation_status} successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to validate NLP analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_schedule(self, schedule_id: int, validator_id: int,
                        modifications: Dict = None, feedback: str = None) -> Dict:
        """Validate and potentially modify schedule"""
        try:
            # Get original schedule
            original_schedule = db_service.execute_query(
                "SELECT * FROM schedules WHERE schedule_id = :schedule_id",
                {"schedule_id": schedule_id}
            )[0]
            
            # Store original for learning
            original_data = {
                'scheduled_date': original_schedule['scheduled_date'].isoformat() if original_schedule['scheduled_date'] else None,
                'priority': original_schedule['priority'],
                'estimated_hours': float(original_schedule['estimated_hours']) if original_schedule['estimated_hours'] else 0
            }
            
            update_fields = []
            update_values = []
            
            # Apply modifications if any
            if modifications:
                update_params = {"schedule_id": schedule_id, "validator_id": validator_id}
                update_fields = []
                
                if 'scheduled_date' in modifications:
                    update_fields.append("scheduled_date = :scheduled_date")
                    update_params["scheduled_date"] = modifications['scheduled_date']
                
                if 'priority' in modifications:
                    update_fields.append("priority = :priority")
                    update_params["priority"] = modifications['priority']
                
                if 'estimated_hours' in modifications:
                    update_fields.append("estimated_hours = :estimated_hours")
                    update_params["estimated_hours"] = modifications['estimated_hours']
                
                # Track modifications
                update_fields.extend([
                    "modified_by = :validator_id",
                    "modified_at = NOW()",
                    "original_schedule = :original_schedule",
                    "admin_changes = :admin_changes",
                    "learning_feedback = :learning_feedback"
                ])
                update_params.update({
                    "original_schedule": json.dumps(original_data),
                    "admin_changes": json.dumps(modifications),
                    "learning_feedback": feedback
                })
                
                # Update schedule
                if update_fields:
                    query = f"UPDATE schedules SET {', '.join(update_fields)} WHERE schedule_id = :schedule_id"
                    db_service.execute_query(query, update_params)
                
                # Add to training data for schedule learning
                training_input = {
                    'report_id': original_schedule['report_id'],
                    'original_schedule': original_data
                }
                
                training_output = {
                    'modified_schedule': modifications,
                    'feedback': feedback
                }
                
                db_service.execute_query(
                    """INSERT INTO training_data (data_type, source_id, input_data, expected_output, validation_notes, created_by)
                       VALUES ('scheduling', :source_id, :input_data, :expected_output, :validation_notes, :created_by)""",
                    {"source_id": schedule_id, "input_data": json.dumps(training_input), 
                     "expected_output": json.dumps(training_output), "validation_notes": feedback, 
                     "created_by": validator_id}
                )
            
            # Update validation queue
            self._update_validation_queue('schedule', schedule_id)
            
            logger.info(f"Schedule {schedule_id} validated and modified")
            
            return {
                'success': True,
                'message': 'Schedule validated and updated successfully',
                'modifications': modifications or {}
            }
            
        except Exception as e:
            logger.error(f"Failed to validate schedule: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_data_summary(self) -> Dict:
        """Get summary of available training data"""
        try:
            summary = {}
            
            # Count by data type
            data_types = ['yolo_detection', 'nlp_analysis', 'risk_assessment', 'scheduling']
            
            for data_type in data_types:
                count = db_service.execute_query(
                    "SELECT COUNT(*) as count FROM training_data WHERE data_type = :data_type AND used_for_training = FALSE",
                    {"data_type": data_type}
                )[0]['count']
                
                summary[data_type] = {
                    'available_samples': count,
                    'ready_for_training': count >= 10  # Minimum samples
                }
            
            # Get recent validation activity
            recent_validations = db_service.execute_query(
                """SELECT item_type, COUNT(*) as count 
                   FROM validation_queue 
                   WHERE status = 'completed' AND processed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                   GROUP BY item_type""",
                {}
            )
            
            summary['recent_activity'] = {item['item_type']: item['count'] for item in recent_validations}
            
            return {
                'success': True,
                'training_data_summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get training data summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_validation_queue(self, item_type: str, item_id: int):
        """Update validation queue status to completed"""
        try:
            db_service.execute_query(
                """UPDATE validation_queue 
                   SET status = 'completed', processed_at = NOW() 
                   WHERE item_type = :item_type AND item_id = :item_id""",
                {"item_type": item_type, "item_id": item_id}
            )
        except Exception as e:
            logger.error(f"Failed to update validation queue: {e}")
    
    def check_auto_validation(self, item_type: str, confidence: float) -> bool:
        """Check if item can be auto-validated based on confidence"""
        return confidence >= self.auto_validation_threshold

# Singleton instance
validation_service = ValidationService()

if __name__ == "__main__":
    # Test validation service
    print("Validation Service Test")
    
    # Test get queue
    queue = validation_service.get_validation_queue()
    print(f"Validation Queue Items: {len(queue)}")
    
    # Test training data summary
    summary = validation_service.get_training_data_summary()
    print("Training Data Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
