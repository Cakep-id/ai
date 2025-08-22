"""
Validation API Endpoints
Endpoints untuk validation workflow dan admin approval
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from services import validation_service, db_service

router = APIRouter()

# Pydantic models
class DetectionValidationRequest(BaseModel):
    detection_id: int = Field(..., description="Detection ID to validate")
    is_approved: bool = Field(..., description="Approval status")
    feedback: Optional[str] = Field(None, description="Validation feedback")
    corrected_detections: Optional[List[Dict]] = Field(None, description="Corrected detection results")

class NLPValidationRequest(BaseModel):
    analysis_id: int = Field(..., description="Analysis ID to validate")
    is_approved: bool = Field(..., description="Approval status")
    feedback: Optional[str] = Field(None, description="Validation feedback")
    corrected_category: Optional[str] = Field(None, description="Corrected category")

class ScheduleValidationRequest(BaseModel):
    schedule_id: int = Field(..., description="Schedule ID to validate")
    modifications: Optional[Dict[str, Any]] = Field(None, description="Schedule modifications")
    feedback: Optional[str] = Field(None, description="Validation feedback")

class ValidationResponse(BaseModel):
    success: bool
    validation_status: Optional[str] = None
    message: str
    training_data_added: bool = False
    error: Optional[str] = None

@router.get("/queue")
async def get_validation_queue(
    validator_id: Optional[int] = None,
    status: str = "pending",
    limit: int = 20
):
    """
    Get validation queue items
    
    - **validator_id**: Filter by assigned validator (optional)
    - **status**: Filter by status (pending, in_progress, completed)
    - **limit**: Maximum items to return
    """
    try:
        queue_items = validation_service.get_validation_queue(validator_id, status)
        
        # Limit results
        if len(queue_items) > limit:
            queue_items = queue_items[:limit]
        
        # Enrich dengan detail data
        enriched_items = []
        for item in queue_items:
            try:
                item_detail = await _get_item_detail(item['item_type'], item['item_id'])
                item['detail'] = item_detail
                enriched_items.append(item)
            except Exception as e:
                logger.error(f"Failed to enrich item {item['item_type']} {item['item_id']}: {e}")
                item['detail'] = None
                enriched_items.append(item)
        
        return {
            'success': True,
            'queue_items': enriched_items,
            'total_items': len(enriched_items),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get validation queue failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation queue: {str(e)}")

@router.post("/detection", response_model=ValidationResponse)
async def validate_detection(request: DetectionValidationRequest, validator_id: int = 1):
    """
    Validate AI detection result
    
    - **detection_id**: ID of detection to validate
    - **is_approved**: Whether to approve or reject
    - **feedback**: Validation feedback notes
    - **corrected_detections**: Corrected detection results if needed
    """
    try:
        result = validation_service.validate_detection(
            detection_id=request.detection_id,
            validator_id=validator_id,
            is_approved=request.is_approved,
            feedback=request.feedback,
            corrected_detections=request.corrected_detections
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Validation failed'))
        
        return ValidationResponse(
            success=True,
            validation_status=result['validation_status'],
            message=result['message'],
            training_data_added=request.is_approved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection validation failed: {str(e)}")

@router.post("/nlp", response_model=ValidationResponse)
async def validate_nlp_analysis(request: NLPValidationRequest, validator_id: int = 1):
    """
    Validate NLP analysis result
    
    - **analysis_id**: ID of analysis to validate
    - **is_approved**: Whether to approve or reject
    - **feedback**: Validation feedback notes
    - **corrected_category**: Corrected category if needed
    """
    try:
        result = validation_service.validate_nlp_analysis(
            analysis_id=request.analysis_id,
            validator_id=validator_id,
            is_approved=request.is_approved,
            feedback=request.feedback,
            corrected_category=request.corrected_category
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Validation failed'))
        
        return ValidationResponse(
            success=True,
            validation_status=result['validation_status'],
            message=result['message'],
            training_data_added=request.is_approved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NLP validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"NLP validation failed: {str(e)}")

@router.post("/schedule", response_model=ValidationResponse)
async def validate_schedule(request: ScheduleValidationRequest, validator_id: int = 1):
    """
    Validate and modify schedule
    
    - **schedule_id**: ID of schedule to validate
    - **modifications**: Schedule modifications (optional)
    - **feedback**: Validation feedback notes
    """
    try:
        result = validation_service.validate_schedule(
            schedule_id=request.schedule_id,
            validator_id=validator_id,
            modifications=request.modifications,
            feedback=request.feedback
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Validation failed'))
        
        return ValidationResponse(
            success=True,
            message=result['message'],
            training_data_added=bool(request.modifications)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schedule validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schedule validation failed: {str(e)}")

@router.get("/training-data/summary")
async def get_training_data_summary():
    """Get summary of available training data"""
    try:
        summary = validation_service.get_training_data_summary()
        
        if not summary['success']:
            raise HTTPException(status_code=500, detail=summary.get('error', 'Failed to get summary'))
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get training data summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training data summary: {str(e)}")

@router.post("/training-data/prepare-batch/{data_type}")
async def prepare_training_batch(
    data_type: str,
    batch_size: int = 50,
    batch_name: Optional[str] = None
):
    """
    Prepare training batch from validated data
    
    - **data_type**: Type of data (yolo_detection, nlp_analysis, risk_assessment, scheduling)
    - **batch_size**: Number of samples to include
    - **batch_name**: Custom batch name
    """
    try:
        valid_types = ['yolo_detection', 'nlp_analysis', 'risk_assessment', 'scheduling']
        if data_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Get available training data
        available_data = db_service.execute_query(
            """SELECT * FROM training_data 
               WHERE data_type = %s AND used_for_training = FALSE 
               ORDER BY created_at ASC LIMIT %s""",
            (data_type, batch_size)
        )
        
        if len(available_data) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient training data. Need at least 10 samples, found {len(available_data)}"
            )
        
        # Generate batch name
        if not batch_name:
            batch_name = f"{data_type}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mark data as used for training
        training_ids = [item['training_id'] for item in available_data]
        
        # Update training data
        for training_id in training_ids:
            db_service.execute_query(
                "UPDATE training_data SET used_for_training = TRUE, training_batch = %s WHERE training_id = %s",
                (batch_name, training_id)
            )
        
        # Create training log entry
        training_params = {
            'data_type': data_type,
            'batch_size': len(available_data),
            'batch_name': batch_name
        }
        
        log_id = db_service.execute_query(
            """INSERT INTO training_logs (model_type, training_batch, training_params, data_count, status, started_by)
               VALUES (%s, %s, %s, %s, 'prepared', 1)""",
            (data_type.split('_')[0], batch_name, json.dumps(training_params), len(available_data)),
            return_id=True
        )
        
        logger.info(f"Prepared training batch {batch_name} with {len(available_data)} samples")
        
        return {
            'success': True,
            'batch_name': batch_name,
            'data_count': len(available_data),
            'training_log_id': log_id,
            'message': f'Training batch prepared with {len(available_data)} samples',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prepare training batch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare training batch: {str(e)}")

@router.get("/stats")
async def get_validation_stats():
    """Get validation statistics"""
    try:
        # Validation queue stats
        queue_stats = db_service.execute_query(
            """SELECT 
                item_type,
                status,
                COUNT(*) as count
               FROM validation_queue 
               WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
               GROUP BY item_type, status""",
            ()
        )
        
        # Training data stats
        training_stats = db_service.execute_query(
            """SELECT 
                data_type,
                COUNT(*) as total_samples,
                SUM(CASE WHEN used_for_training = TRUE THEN 1 ELSE 0 END) as used_samples,
                SUM(CASE WHEN used_for_training = FALSE THEN 1 ELSE 0 END) as available_samples
               FROM training_data
               GROUP BY data_type""",
            ()
        )
        
        # Recent validation activity
        recent_activity = db_service.execute_query(
            """SELECT 
                DATE(processed_at) as validation_date,
                item_type,
                COUNT(*) as count
               FROM validation_queue 
               WHERE status = 'completed' AND processed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
               GROUP BY DATE(processed_at), item_type
               ORDER BY validation_date DESC""",
            ()
        )
        
        return {
            'success': True,
            'stats': {
                'queue_stats': queue_stats,
                'training_stats': training_stats,
                'recent_activity': recent_activity
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get validation stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation stats: {str(e)}")

@router.post("/auto-validate/{item_type}/{item_id}")
async def auto_validate_high_confidence(item_type: str, item_id: int, confidence: float):
    """Auto-validate high confidence results"""
    try:
        if validation_service.check_auto_validation(item_type, confidence):
            # Auto-approve high confidence results
            if item_type == 'detection':
                result = validation_service.validate_detection(
                    detection_id=item_id,
                    validator_id=0,  # System auto-validation
                    is_approved=True,
                    feedback=f"Auto-validated (confidence: {confidence:.3f})"
                )
            elif item_type == 'nlp_analysis':
                result = validation_service.validate_nlp_analysis(
                    analysis_id=item_id,
                    validator_id=0,
                    is_approved=True,
                    feedback=f"Auto-validated (confidence: {confidence:.3f})"
                )
            else:
                return {
                    'success': False,
                    'message': f'Auto-validation not supported for {item_type}'
                }
            
            return {
                'success': True,
                'auto_validated': True,
                'confidence': confidence,
                'result': result
            }
        else:
            return {
                'success': True,
                'auto_validated': False,
                'confidence': confidence,
                'message': f'Confidence {confidence:.3f} below auto-validation threshold'
            }
            
    except Exception as e:
        logger.error(f"Auto-validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-validation failed: {str(e)}")

# Helper functions
async def _get_item_detail(item_type: str, item_id: int) -> Dict:
    """Get detailed information for validation item"""
    try:
        if item_type == 'detection':
            detail = db_service.execute_query(
                """SELECT ad.*, r.description, r.image_path, a.asset_name
                   FROM ai_detections ad
                   JOIN reports r ON ad.report_id = r.report_id
                   JOIN assets a ON r.asset_id = a.asset_id
                   WHERE ad.detection_id = %s""",
                (item_id,)
            )
        elif item_type == 'nlp_analysis':
            detail = db_service.execute_query(
                """SELECT na.*, r.description, a.asset_name
                   FROM nlp_analyses na
                   JOIN reports r ON na.report_id = r.report_id
                   JOIN assets a ON r.asset_id = a.asset_id
                   WHERE na.analysis_id = %s""",
                (item_id,)
            )
        elif item_type == 'risk_assessment':
            detail = db_service.execute_query(
                """SELECT ra.*, r.description, a.asset_name
                   FROM risk_assessments ra
                   JOIN reports r ON ra.report_id = r.report_id
                   JOIN assets a ON r.asset_id = a.asset_id
                   WHERE ra.risk_id = %s""",
                (item_id,)
            )
        elif item_type == 'procedure':
            detail = db_service.execute_query(
                """SELECT ap.*, r.description, a.asset_name
                   FROM ai_procedures ap
                   JOIN reports r ON ap.report_id = r.report_id
                   JOIN assets a ON r.asset_id = a.asset_id
                   WHERE ap.procedure_id = %s""",
                (item_id,)
            )
        elif item_type == 'schedule':
            detail = db_service.execute_query(
                """SELECT s.*, r.description, a.asset_name
                   FROM schedules s
                   JOIN reports r ON s.report_id = r.report_id
                   JOIN assets a ON r.asset_id = a.asset_id
                   WHERE s.schedule_id = %s""",
                (item_id,)
            )
        else:
            return {}
        
        return detail[0] if detail else {}
        
    except Exception as e:
        logger.error(f"Failed to get item detail: {e}")
        return {}
