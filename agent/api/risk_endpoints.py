"""
Risk Assessment API Endpoints
Endpoints untuk risk assessment yang menggabungkan CV dan NLP results
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from services import risk_engine, db_service, yolo_service, groq_service

router = APIRouter()

# Pydantic models
class RiskAssessmentRequest(BaseModel):
    report_id: int = Field(..., description="Report ID untuk assessment")
    visual_score: Optional[float] = Field(None, description="Visual damage score (0-1)")
    text_score: Optional[float] = Field(None, description="Text analysis score (0-1)")
    force_new_analysis: bool = Field(False, description="Force new CV/NLP analysis")

class ManualRiskRequest(BaseModel):
    visual_score: float = Field(..., ge=0, le=1, description="Visual damage score (0-1)")
    text_score: float = Field(..., ge=0, le=1, description="Text analysis score (0-1)")
    asset_type: Optional[str] = Field(None, description="Asset type untuk context")
    damage_category: Optional[str] = Field(None, description="Damage category")

class RiskResponse(BaseModel):
    success: bool
    report_id: Optional[int] = None
    final_risk_score: float
    risk_level: str
    visual_score: float
    text_score: float
    confidence: float
    repair_procedures: List[str]
    estimated_cost: Dict[str, float]
    time_estimates: Dict[str, float]
    priority: str
    sla_deadline: str
    impact_assessment: Dict[str, Any]
    risk_factors: List[str]
    error: Optional[str] = None

class BulkRiskRequest(BaseModel):
    report_ids: List[int] = Field(..., description="List of report IDs")
    recalculate: bool = Field(False, description="Recalculate existing assessments")

class RiskThresholdUpdate(BaseModel):
    visual_weight: float = Field(..., ge=0, le=1, description="Weight untuk visual score")
    text_weight: float = Field(..., ge=0, le=1, description="Weight untuk text score")
    risk_thresholds: Dict[str, float] = Field(..., description="Risk level thresholds")

@router.post("/assess", response_model=RiskResponse)
async def assess_risk(request: RiskAssessmentRequest, background_tasks: BackgroundTasks):
    """
    Lakukan risk assessment untuk report dengan menggabungkan CV dan NLP results
    
    - **report_id**: ID report yang akan diassess
    - **visual_score**: Score visual (optional, akan auto-generate jika tidak ada)
    - **text_score**: Score text (optional, akan auto-generate jika tidak ada)
    - **force_new_analysis**: Force new analysis meskipun sudah ada hasil sebelumnya
    """
    try:
        # Validate report exists
        report = db_service.get_report(request.report_id)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"Report {request.report_id} not found"
            )
        
        logger.info(f"Assessing risk for report {request.report_id}")
        
        visual_score = request.visual_score
        text_score = request.text_score
        
        # Get or generate visual score
        if visual_score is None or request.force_new_analysis:
            try:
                # Check existing detection
                detections = db_service.get_detections_by_report(request.report_id)
                if detections and not request.force_new_analysis:
                    # Use existing detection
                    detection = detections[0]
                    visual_score = detection.get('damage_score', 0.0)
                    logger.info(f"Using existing visual score: {visual_score}")
                else:
                    # Need to run detection first
                    if report.get('image_path'):
                        logger.info("Running YOLO detection for visual score")
                        detection_result = yolo_service.detect(report['image_path'])
                        if detection_result['success']:
                            visual_score = detection_result.get('damage_score', 0.0)
                            
                            # Save detection result in background
                            background_tasks.add_task(
                                db_service.save_detection,
                                request.report_id,
                                detection_result['detections'],
                                detection_result['confidence'],
                                detection_result['damage_score'],
                                yolo_service.model_path
                            )
                        else:
                            visual_score = 0.0
                            logger.warning(f"YOLO detection failed: {detection_result.get('error')}")
                    else:
                        visual_score = 0.0
                        logger.warning("No image path found for visual analysis")
            except Exception as e:
                logger.error(f"Visual analysis failed: {e}")
                visual_score = 0.0
        
        # Get or generate text score
        if text_score is None or request.force_new_analysis:
            try:
                # Check existing NLP analysis
                nlp_analyses = db_service.get_nlp_analyses_by_report(request.report_id)
                if nlp_analyses and not request.force_new_analysis:
                    # Calculate text score from existing analysis
                    nlp_analysis = nlp_analyses[0]
                    text_score = groq_service.get_text_score(nlp_analysis)
                    logger.info(f"Using existing text score: {text_score}")
                else:
                    # Need to run NLP analysis
                    if report.get('description'):
                        logger.info("Running Groq analysis for text score")
                        nlp_result = groq_service.analyze(
                            text=report['description'],
                            asset_context=None
                        )
                        if nlp_result['success']:
                            text_score = groq_service.get_text_score(nlp_result)
                            
                            # Save NLP result in background
                            background_tasks.add_task(
                                db_service.save_nlp_analysis,
                                request.report_id,
                                nlp_result['category'],
                                nlp_result['confidence'],
                                nlp_result['keyphrases'],
                                nlp_result['model_version']
                            )
                        else:
                            text_score = 0.0
                            logger.warning(f"NLP analysis failed: {nlp_result.get('error')}")
                    else:
                        text_score = 0.0
                        logger.warning("No description found for text analysis")
            except Exception as e:
                logger.error(f"Text analysis failed: {e}")
                text_score = 0.0
        
        # Run risk assessment
        risk_result = risk_engine.aggregate_risk(
            visual_score=visual_score or 0.0,
            text_score=text_score or 0.0,
            asset_type=report.get('asset_type'),
            damage_category=None  # Will be inferred from scores
        )
        
        if not risk_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Risk assessment failed: {risk_result.get('error')}"
            )
        
        # Save risk assessment to database in background
        background_tasks.add_task(
            db_service.save_risk_assessment,
            request.report_id,
            risk_result['final_risk_score'],
            risk_result['risk_level'],
            visual_score or 0.0,
            text_score or 0.0,
            risk_result['confidence']
        )
        
        # Format response
        response_data = {
            'success': True,
            'report_id': request.report_id,
            'final_risk_score': risk_result['final_risk_score'],
            'risk_level': risk_result['risk_level'],
            'visual_score': visual_score or 0.0,
            'text_score': text_score or 0.0,
            'confidence': risk_result['confidence'],
            'repair_procedures': risk_result['repair_procedures'],
            'estimated_cost': risk_result['estimated_cost'],
            'time_estimates': risk_result['time_estimates'],
            'priority': risk_result['priority'],
            'sla_deadline': risk_result['sla_deadline'],
            'impact_assessment': risk_result['impact_assessment'],
            'risk_factors': risk_result['risk_factors']
        }
        
        return RiskResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.post("/assess/manual", response_model=RiskResponse)
async def assess_manual_risk(request: ManualRiskRequest):
    """
    Manual risk assessment dengan visual dan text scores yang sudah diketahui
    
    - **visual_score**: Score visual damage (0-1)
    - **text_score**: Score text analysis (0-1)
    - **asset_type**: Tipe aset (optional)
    - **damage_category**: Kategori kerusakan (optional)
    """
    try:
        logger.info(f"Manual risk assessment: visual={request.visual_score}, text={request.text_score}")
        
        # Validate scores
        if not (0 <= request.visual_score <= 1):
            raise HTTPException(
                status_code=400,
                detail="Visual score must be between 0 and 1"
            )
        
        if not (0 <= request.text_score <= 1):
            raise HTTPException(
                status_code=400,
                detail="Text score must be between 0 and 1"
            )
        
        # Run risk assessment
        risk_result = risk_engine.aggregate_risk(
            visual_score=request.visual_score,
            text_score=request.text_score,
            asset_type=request.asset_type,
            damage_category=request.damage_category
        )
        
        if not risk_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Risk assessment failed: {risk_result.get('error')}"
            )
        
        # Format response
        response_data = {
            'success': True,
            'report_id': None,
            'final_risk_score': risk_result['final_risk_score'],
            'risk_level': risk_result['risk_level'],
            'visual_score': request.visual_score,
            'text_score': request.text_score,
            'confidence': risk_result['confidence'],
            'repair_procedures': risk_result['repair_procedures'],
            'estimated_cost': risk_result['estimated_cost'],
            'time_estimates': risk_result['time_estimates'],
            'priority': risk_result['priority'],
            'sla_deadline': risk_result['sla_deadline'],
            'impact_assessment': risk_result['impact_assessment'],
            'risk_factors': risk_result['risk_factors']
        }
        
        return RiskResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Manual risk assessment failed: {str(e)}")

@router.post("/assess/bulk")
async def assess_bulk_risk(request: BulkRiskRequest, background_tasks: BackgroundTasks):
    """
    Bulk risk assessment untuk multiple reports
    
    - **report_ids**: List of report IDs untuk assessment
    - **recalculate**: Recalculate existing assessments
    """
    try:
        if not request.report_ids:
            raise HTTPException(
                status_code=400,
                detail="Report IDs list cannot be empty"
            )
        
        if len(request.report_ids) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 reports per bulk request"
            )
        
        logger.info(f"Bulk risk assessment for {len(request.report_ids)} reports")
        
        # Process each report
        results = []
        
        for report_id in request.report_ids:
            try:
                # Check if assessment already exists
                if not request.recalculate:
                    existing = db_service.get_risk_assessment(report_id)
                    if existing:
                        results.append({
                            'report_id': report_id,
                            'status': 'existing',
                            'risk_level': existing['risk_level'],
                            'risk_score': existing['risk_score']
                        })
                        continue
                
                # Queue for background processing
                background_tasks.add_task(
                    _process_single_risk_assessment,
                    report_id
                )
                
                results.append({
                    'report_id': report_id,
                    'status': 'queued',
                    'message': 'Queued for background processing'
                })
                
            except Exception as e:
                logger.error(f"Failed to process report {report_id}: {e}")
                results.append({
                    'report_id': report_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Summary
        summary = {
            'total_reports': len(request.report_ids),
            'queued': len([r for r in results if r['status'] == 'queued']),
            'existing': len([r for r in results if r['status'] == 'existing']),
            'errors': len([r for r in results if r['status'] == 'error'])
        }
        
        return {
            'success': True,
            'results': results,
            'summary': summary,
            'message': 'Bulk assessment initiated. Check individual reports for completion status.',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk assessment failed: {str(e)}")

@router.get("/report/{report_id}")
async def get_risk_assessment(report_id: int):
    """Get existing risk assessment untuk report"""
    try:
        risk_assessment = db_service.get_risk_assessment(report_id)
        
        if not risk_assessment:
            raise HTTPException(
                status_code=404,
                detail=f"No risk assessment found for report {report_id}"
            )
        
        # Get related data
        detections = db_service.get_detections_by_report(report_id)
        nlp_analyses = db_service.get_nlp_analyses_by_report(report_id)
        
        return {
            'success': True,
            'risk_assessment': risk_assessment,
            'detections': detections,
            'nlp_analyses': nlp_analyses,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk assessment: {str(e)}")

@router.get("/stats")
async def get_risk_stats():
    """Get statistik risk assessments"""
    try:
        # Risk level distribution
        risk_level_query = """
        SELECT 
            risk_level,
            COUNT(*) as count,
            AVG(risk_score) as avg_score,
            AVG(visual_score) as avg_visual,
            AVG(text_score) as avg_text
        FROM risk_scores 
        WHERE assessed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY risk_level
        ORDER BY avg_score DESC
        """
        
        risk_level_stats = db_service.execute_query(risk_level_query)
        
        # Time series (last 7 days)
        time_series_query = """
        SELECT 
            DATE(assessed_at) as assessment_date,
            COUNT(*) as daily_count,
            AVG(risk_score) as daily_avg_score,
            risk_level,
            COUNT(*) as level_count
        FROM risk_scores 
        WHERE assessed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY DATE(assessed_at), risk_level
        ORDER BY assessment_date, risk_level
        """
        
        time_series = db_service.execute_query(time_series_query)
        
        # High risk reports
        high_risk_query = """
        SELECT 
            rs.report_id,
            rs.risk_score,
            rs.risk_level,
            r.asset_id,
            r.reported_at,
            a.asset_name
        FROM risk_scores rs
        JOIN reports r ON rs.report_id = r.report_id
        JOIN assets a ON r.asset_id = a.asset_id
        WHERE rs.risk_level IN ('CRITICAL', 'HIGH')
        AND rs.assessed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        ORDER BY rs.risk_score DESC, rs.assessed_at DESC
        LIMIT 20
        """
        
        high_risk_reports = db_service.execute_query(high_risk_query)
        
        # Score distribution
        score_distribution_query = """
        SELECT 
            CASE 
                WHEN risk_score < 0.2 THEN '0.0-0.2'
                WHEN risk_score < 0.4 THEN '0.2-0.4'
                WHEN risk_score < 0.6 THEN '0.4-0.6'
                WHEN risk_score < 0.8 THEN '0.6-0.8'
                ELSE '0.8-1.0'
            END as score_range,
            COUNT(*) as count
        FROM risk_scores 
        WHERE assessed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY score_range
        ORDER BY score_range
        """
        
        score_distribution = db_service.execute_query(score_distribution_query)
        
        return {
            'success': True,
            'stats': {
                'risk_level_distribution': risk_level_stats,
                'time_series': time_series,
                'high_risk_reports': high_risk_reports,
                'score_distribution': score_distribution,
                'total_assessments': sum(stat['count'] for stat in risk_level_stats)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get risk stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk stats: {str(e)}")

@router.get("/config")
async def get_risk_config():
    """Get current risk engine configuration"""
    try:
        config = {
            'visual_weight': risk_engine.visual_weight,
            'text_weight': risk_engine.text_weight,
            'risk_thresholds': risk_engine.risk_thresholds,
            'asset_multipliers': risk_engine.asset_multipliers,
            'sla_hours': risk_engine.sla_hours,
            'cost_estimates': risk_engine.cost_estimates
        }
        
        return {
            'success': True,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get risk config failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk config: {str(e)}")

@router.post("/config/update")
async def update_risk_config(config: RiskThresholdUpdate):
    """Update risk engine configuration"""
    try:
        # Validate weights sum to 1
        if abs(config.visual_weight + config.text_weight - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail="Visual weight + Text weight must equal 1.0"
            )
        
        # Update configuration
        risk_engine.visual_weight = config.visual_weight
        risk_engine.text_weight = config.text_weight
        
        if config.risk_thresholds:
            risk_engine.risk_thresholds.update(config.risk_thresholds)
        
        logger.info(f"Updated risk engine config: visual_weight={config.visual_weight}, text_weight={config.text_weight}")
        
        # Return new configuration
        new_config = {
            'visual_weight': risk_engine.visual_weight,
            'text_weight': risk_engine.text_weight,
            'risk_thresholds': risk_engine.risk_thresholds
        }
        
        return {
            'success': True,
            'message': 'Risk engine configuration updated',
            'new_config': new_config,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update risk config failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update risk config: {str(e)}")

# Background task function
async def _process_single_risk_assessment(report_id: int):
    """Background task untuk single risk assessment"""
    try:
        logger.info(f"Background processing risk assessment for report {report_id}")
        
        # Get report
        report = db_service.get_report(report_id)
        if not report:
            logger.error(f"Report {report_id} not found for background processing")
            return
        
        visual_score = 0.0
        text_score = 0.0
        
        # Get visual score
        try:
            detections = db_service.get_detections_by_report(report_id)
            if detections:
                visual_score = detections[0].get('damage_score', 0.0)
            elif report.get('image_path'):
                detection_result = yolo_service.detect(report['image_path'])
                if detection_result['success']:
                    visual_score = detection_result.get('damage_score', 0.0)
                    db_service.save_detection(
                        report_id,
                        detection_result['detections'],
                        detection_result['confidence'],
                        detection_result['damage_score'],
                        yolo_service.model_path
                    )
        except Exception as e:
            logger.error(f"Visual analysis failed in background: {e}")
        
        # Get text score
        try:
            nlp_analyses = db_service.get_nlp_analyses_by_report(report_id)
            if nlp_analyses:
                text_score = groq_service.get_text_score(nlp_analyses[0])
            elif report.get('description'):
                nlp_result = groq_service.analyze(
                    text=report['description'],
                    asset_context=None
                )
                if nlp_result['success']:
                    text_score = groq_service.get_text_score(nlp_result)
                    db_service.save_nlp_analysis(
                        report_id,
                        nlp_result['category'],
                        nlp_result['confidence'],
                        nlp_result['keyphrases'],
                        nlp_result['model_version']
                    )
        except Exception as e:
            logger.error(f"Text analysis failed in background: {e}")
        
        # Run risk assessment
        risk_result = risk_engine.aggregate_risk(
            visual_score=visual_score,
            text_score=text_score,
            asset_type=report.get('asset_type'),
            damage_category=None
        )
        
        if risk_result['success']:
            # Save to database
            db_service.save_risk_assessment(
                report_id,
                risk_result['final_risk_score'],
                risk_result['risk_level'],
                visual_score,
                text_score,
                risk_result['confidence']
            )
            
            logger.info(f"Background risk assessment completed for report {report_id}: {risk_result['risk_level']}")
        else:
            logger.error(f"Background risk assessment failed for report {report_id}: {risk_result.get('error')}")
        
    except Exception as e:
        logger.error(f"Background risk assessment error for report {report_id}: {e}")
