"""
Schedule API Endpoints
Endpoints untuk automated maintenance scheduling berdasarkan risk assessment
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from services import scheduler_service, db_service, risk_engine

router = APIRouter()

# Pydantic models
class ScheduleGenerationRequest(BaseModel):
    report_id: int = Field(..., description="Report ID untuk scheduling")
    force_reschedule: bool = Field(False, description="Force reschedule existing")
    custom_priority: Optional[str] = Field(None, description="Override priority")

class BulkScheduleRequest(BaseModel):
    report_ids: List[int] = Field(..., description="List report IDs")
    priority_filter: Optional[str] = Field(None, description="Filter by priority")
    risk_level_filter: Optional[str] = Field(None, description="Filter by risk level")

class ScheduleResponse(BaseModel):
    success: bool
    report_id: int
    schedule_id: Optional[int] = None
    priority: str
    scheduled_date: str
    deadline: str
    estimated_hours: float
    resource_requirements: Dict[str, Any]
    work_order: Dict[str, Any]
    sla_compliance: bool
    error: Optional[str] = None

class ScheduleUpdateRequest(BaseModel):
    schedule_id: int = Field(..., description="Schedule ID")
    new_date: Optional[str] = Field(None, description="New scheduled date")
    new_priority: Optional[str] = Field(None, description="New priority")
    status_update: Optional[str] = Field(None, description="Status update")
    completion_notes: Optional[str] = Field(None, description="Completion notes")

class ResourcePlanningRequest(BaseModel):
    start_date: str = Field(..., description="Planning start date")
    end_date: str = Field(..., description="Planning end date")
    include_completed: bool = Field(False, description="Include completed schedules")

@router.post("/generate", response_model=ScheduleResponse)
async def generate_schedule(request: ScheduleGenerationRequest):
    """
    Generate maintenance schedule berdasarkan risk assessment
    
    - **report_id**: Report ID yang akan dijadwalkan
    - **force_reschedule**: Force reschedule jika sudah ada schedule
    - **custom_priority**: Override priority default
    """
    try:
        # Validate report exists
        report = db_service.get_report(request.report_id)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"Report {request.report_id} not found"
            )
        
        logger.info(f"Generating schedule for report {request.report_id}")
        
        # Check existing schedule
        existing_schedule = db_service.get_schedule_by_report(request.report_id)
        if existing_schedule and not request.force_reschedule:
            # Return existing schedule
            return ScheduleResponse(
                success=True,
                report_id=request.report_id,
                schedule_id=existing_schedule['schedule_id'],
                priority=existing_schedule['priority'],
                scheduled_date=existing_schedule['scheduled_date'].isoformat(),
                deadline=existing_schedule['deadline'].isoformat(),
                estimated_hours=existing_schedule['estimated_hours'],
                resource_requirements=existing_schedule.get('resource_requirements', {}),
                work_order=existing_schedule.get('work_order', {}),
                sla_compliance=existing_schedule.get('sla_compliance', True)
            )
        
        # Get risk assessment
        risk_assessment = db_service.get_risk_assessment(request.report_id)
        if not risk_assessment:
            raise HTTPException(
                status_code=400,
                detail=f"No risk assessment found for report {request.report_id}. Run risk assessment first."
            )
        
        # Generate schedule
        schedule_result = scheduler_service.generate_schedule(
            report_id=request.report_id,
            risk_level=risk_assessment['risk_level'],
            risk_score=risk_assessment['risk_score'],
            asset_type=report.get('asset_type'),
            custom_priority=request.custom_priority
        )
        
        if not schedule_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Schedule generation failed: {schedule_result.get('error')}"
            )
        
        # Save schedule to database
        try:
            schedule_data = schedule_result['schedule']
            schedule_id = db_service.save_schedule(
                report_id=request.report_id,
                priority=schedule_data['priority'],
                scheduled_date=schedule_data['scheduled_date'],
                deadline=schedule_data['deadline'],
                estimated_hours=schedule_data['estimated_hours'],
                resource_requirements=schedule_data['resource_requirements']
            )
            
            logger.info(f"Schedule saved with ID {schedule_id}")
            
        except Exception as e:
            logger.error(f"Failed to save schedule: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save schedule: {str(e)}"
            )
        
        # Format response
        schedule_data = schedule_result['schedule']
        response_data = {
            'success': True,
            'report_id': request.report_id,
            'schedule_id': schedule_id,
            'priority': schedule_data['priority'],
            'scheduled_date': schedule_data['scheduled_date'].isoformat(),
            'deadline': schedule_data['deadline'].isoformat(),
            'estimated_hours': schedule_data['estimated_hours'],
            'resource_requirements': schedule_data['resource_requirements'],
            'work_order': schedule_data['work_order'],
            'sla_compliance': schedule_data['sla_compliance']
        }
        
        return ScheduleResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schedule generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {str(e)}")

@router.post("/generate/bulk")
async def generate_bulk_schedules(request: BulkScheduleRequest, background_tasks: BackgroundTasks):
    """
    Generate schedules untuk multiple reports dengan filtering
    
    - **report_ids**: List report IDs
    - **priority_filter**: Filter by priority (CRITICAL, HIGH, MEDIUM, LOW)
    - **risk_level_filter**: Filter by risk level
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
        
        logger.info(f"Bulk schedule generation for {len(request.report_ids)} reports")
        
        # Apply filters if provided
        filtered_reports = []
        
        for report_id in request.report_ids:
            try:
                # Get report and risk assessment
                report = db_service.get_report(report_id)
                if not report:
                    continue
                
                risk_assessment = db_service.get_risk_assessment(report_id)
                if not risk_assessment:
                    continue
                
                # Apply filters
                if request.risk_level_filter and risk_assessment['risk_level'] != request.risk_level_filter:
                    continue
                
                # Priority akan di-generate, jadi skip priority filter untuk sekarang
                # atau bisa pre-calculate priority untuk filtering
                
                filtered_reports.append({
                    'report_id': report_id,
                    'risk_level': risk_assessment['risk_level'],
                    'risk_score': risk_assessment['risk_score'],
                    'asset_type': report.get('asset_type')
                })
                
            except Exception as e:
                logger.error(f"Failed to process report {report_id} for filtering: {e}")
                continue
        
        logger.info(f"Filtered to {len(filtered_reports)} reports for scheduling")
        
        # Process schedules in background
        results = []
        
        for report_info in filtered_reports:
            report_id = report_info['report_id']
            
            # Check existing schedule
            existing = db_service.get_schedule_by_report(report_id)
            if existing:
                results.append({
                    'report_id': report_id,
                    'status': 'existing',
                    'schedule_id': existing['schedule_id'],
                    'priority': existing['priority']
                })
                continue
            
            # Queue for background processing
            background_tasks.add_task(
                _process_single_schedule,
                report_id,
                report_info['risk_level'],
                report_info['risk_score'],
                report_info['asset_type']
            )
            
            results.append({
                'report_id': report_id,
                'status': 'queued',
                'message': 'Queued for background processing'
            })
        
        # Summary
        summary = {
            'total_requested': len(request.report_ids),
            'filtered_reports': len(filtered_reports),
            'queued': len([r for r in results if r['status'] == 'queued']),
            'existing': len([r for r in results if r['status'] == 'existing'])
        }
        
        return {
            'success': True,
            'results': results,
            'summary': summary,
            'message': 'Bulk scheduling initiated. Check individual schedules for completion status.',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk schedule generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk scheduling failed: {str(e)}")

@router.get("/report/{report_id}")
async def get_schedule(report_id: int):
    """Get schedule untuk specific report"""
    try:
        schedule = db_service.get_schedule_by_report(report_id)
        
        if not schedule:
            raise HTTPException(
                status_code=404,
                detail=f"No schedule found for report {report_id}"
            )
        
        # Get related data
        report = db_service.get_report(report_id)
        risk_assessment = db_service.get_risk_assessment(report_id)
        
        return {
            'success': True,
            'schedule': schedule,
            'report': report,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get schedule failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get schedule: {str(e)}")

@router.put("/update")
async def update_schedule(request: ScheduleUpdateRequest):
    """
    Update existing schedule
    
    - **schedule_id**: Schedule ID to update
    - **new_date**: New scheduled date (ISO format)
    - **new_priority**: New priority level
    - **status_update**: Update status (SCHEDULED, IN_PROGRESS, COMPLETED, CANCELLED)
    - **completion_notes**: Notes untuk completion
    """
    try:
        # Validate schedule exists
        schedule = db_service.get_schedule(request.schedule_id)
        if not schedule:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule {request.schedule_id} not found"
            )
        
        logger.info(f"Updating schedule {request.schedule_id}")
        
        update_data = {}
        
        # Parse new date if provided
        if request.new_date:
            try:
                new_date = datetime.fromisoformat(request.new_date.replace('Z', '+00:00'))
                update_data['scheduled_date'] = new_date
                
                # Recalculate deadline if date changed
                if schedule['priority'] == 'CRITICAL':
                    update_data['deadline'] = new_date + timedelta(hours=4)
                elif schedule['priority'] == 'HIGH':
                    update_data['deadline'] = new_date + timedelta(hours=24)
                elif schedule['priority'] == 'MEDIUM':
                    update_data['deadline'] = new_date + timedelta(hours=72)
                else:  # LOW
                    update_data['deadline'] = new_date + timedelta(hours=168)
                    
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                )
        
        if request.new_priority:
            if request.new_priority not in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                raise HTTPException(
                    status_code=400,
                    detail="Priority must be one of: CRITICAL, HIGH, MEDIUM, LOW"
                )
            update_data['priority'] = request.new_priority
        
        if request.status_update:
            valid_statuses = ['SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED']
            if request.status_update not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Status must be one of: {', '.join(valid_statuses)}"
                )
            update_data['status'] = request.status_update
            
            # Set completion timestamp if completed
            if request.status_update == 'COMPLETED':
                update_data['completed_at'] = datetime.now()
        
        if request.completion_notes:
            update_data['completion_notes'] = request.completion_notes
        
        # Update schedule
        success = db_service.update_schedule(request.schedule_id, update_data)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update schedule"
            )
        
        # Get updated schedule
        updated_schedule = db_service.get_schedule(request.schedule_id)
        
        return {
            'success': True,
            'message': 'Schedule updated successfully',
            'updated_schedule': updated_schedule,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update schedule failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {str(e)}")

@router.get("/list")
async def list_schedules(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List schedules dengan filtering dan pagination
    
    - **status**: Filter by status (SCHEDULED, IN_PROGRESS, COMPLETED, CANCELLED)
    - **priority**: Filter by priority (CRITICAL, HIGH, MEDIUM, LOW)
    - **start_date**: Filter scheduled_date >= start_date
    - **end_date**: Filter scheduled_date <= end_date
    - **limit**: Maximum records to return
    - **offset**: Records to skip for pagination
    """
    try:
        # Build query
        where_conditions = []
        params = {}
        
        if status:
            where_conditions.append("s.status = :status")
            params['status'] = status
        
        if priority:
            where_conditions.append("s.priority = :priority")
            params['priority'] = priority
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                where_conditions.append("s.scheduled_date >= :start_date")
                params['start_date'] = start_dt
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid start_date format. Use ISO format"
                )
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                where_conditions.append("s.scheduled_date <= :end_date")
                params['end_date'] = end_dt
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid end_date format. Use ISO format"
                )
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Main query
        query = f"""
        SELECT 
            s.*,
            r.asset_id,
            r.description as report_description,
            a.asset_name,
            rs.risk_level,
            rs.risk_score
        FROM schedules s
        JOIN reports r ON s.report_id = r.report_id
        JOIN assets a ON r.asset_id = a.asset_id
        LEFT JOIN risk_scores rs ON s.report_id = rs.report_id
        WHERE {where_clause}
        ORDER BY s.scheduled_date ASC, s.priority DESC
        LIMIT :limit OFFSET :offset
        """
        
        params.update({'limit': limit, 'offset': offset})
        
        schedules = db_service.execute_query(query, params)
        
        # Count total for pagination
        count_query = f"""
        SELECT COUNT(*) as total
        FROM schedules s
        JOIN reports r ON s.report_id = r.report_id
        WHERE {where_clause}
        """
        
        count_params = {k: v for k, v in params.items() if k not in ['limit', 'offset']}
        total_count = db_service.execute_query(count_query, count_params)[0]['total']
        
        return {
            'success': True,
            'schedules': schedules,
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': total_count > offset + limit
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List schedules failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list schedules: {str(e)}")

@router.get("/calendar")
async def get_calendar_view(
    start_date: str,
    end_date: str,
    view_type: str = "week"
):
    """
    Get calendar view dari schedules
    
    - **start_date**: Calendar start date
    - **end_date**: Calendar end date  
    - **view_type**: View type (day, week, month)
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use ISO format"
            )
        
        # Get schedules in date range
        query = """
        SELECT 
            s.*,
            r.asset_id,
            r.description as report_description,
            a.asset_name,
            a.location,
            rs.risk_level,
            rs.risk_score
        FROM schedules s
        JOIN reports r ON s.report_id = r.report_id
        JOIN assets a ON r.asset_id = a.asset_id
        LEFT JOIN risk_scores rs ON s.report_id = rs.report_id
        WHERE s.scheduled_date BETWEEN :start_date AND :end_date
        ORDER BY s.scheduled_date, s.priority DESC
        """
        
        schedules = db_service.execute_query(query, {
            'start_date': start_dt,
            'end_date': end_dt
        })
        
        # Group by date for calendar view
        calendar_data = {}
        
        for schedule in schedules:
            date_key = schedule['scheduled_date'].strftime('%Y-%m-%d')
            
            if date_key not in calendar_data:
                calendar_data[date_key] = {
                    'date': date_key,
                    'schedules': [],
                    'summary': {
                        'total': 0,
                        'by_priority': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                        'by_status': {}
                    }
                }
            
            calendar_data[date_key]['schedules'].append(schedule)
            calendar_data[date_key]['summary']['total'] += 1
            calendar_data[date_key]['summary']['by_priority'][schedule['priority']] += 1
            
            status = schedule.get('status', 'SCHEDULED')
            if status in calendar_data[date_key]['summary']['by_status']:
                calendar_data[date_key]['summary']['by_status'][status] += 1
            else:
                calendar_data[date_key]['summary']['by_status'][status] = 1
        
        # Convert to list and sort
        calendar_list = list(calendar_data.values())
        calendar_list.sort(key=lambda x: x['date'])
        
        return {
            'success': True,
            'calendar': calendar_list,
            'date_range': {
                'start': start_date,
                'end': end_date,
                'view_type': view_type
            },
            'summary': {
                'total_days': len(calendar_list),
                'total_schedules': sum(day['summary']['total'] for day in calendar_list)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get calendar view failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar view: {str(e)}")

@router.get("/resource-planning")
async def get_resource_planning(request: ResourcePlanningRequest):
    """
    Get resource planning untuk period tertentu
    
    - **start_date**: Planning start date
    - **end_date**: Planning end date
    - **include_completed**: Include completed schedules
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use ISO format"
            )
        
        # Build status filter
        status_condition = ""
        if not request.include_completed:
            status_condition = "AND s.status != 'COMPLETED'"
        
        # Get schedules with resource requirements
        query = f"""
        SELECT 
            s.*,
            r.asset_id,
            a.asset_name,
            a.location,
            rs.risk_level
        FROM schedules s
        JOIN reports r ON s.report_id = r.report_id
        JOIN assets a ON r.asset_id = a.asset_id
        LEFT JOIN risk_scores rs ON s.report_id = rs.report_id
        WHERE s.scheduled_date BETWEEN :start_date AND :end_date
        {status_condition}
        ORDER BY s.scheduled_date
        """
        
        schedules = db_service.execute_query(query, {
            'start_date': start_dt,
            'end_date': end_dt
        })
        
        # Aggregate resource requirements
        resource_planning = scheduler_service.calculate_resource_planning(
            schedules,
            start_dt,
            end_dt
        )
        
        return {
            'success': True,
            'resource_planning': resource_planning,
            'schedules': schedules,
            'period': {
                'start_date': request.start_date,
                'end_date': request.end_date,
                'total_schedules': len(schedules)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resource planning failed: {str(e)}")

@router.get("/stats")
async def get_schedule_stats():
    """Get statistik scheduling"""
    try:
        # Status distribution
        status_query = """
        SELECT 
            status,
            COUNT(*) as count,
            AVG(estimated_hours) as avg_hours
        FROM schedules 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY status
        ORDER BY count DESC
        """
        
        status_stats = db_service.execute_query(status_query)
        
        # Priority distribution
        priority_query = """
        SELECT 
            priority,
            COUNT(*) as count,
            AVG(estimated_hours) as avg_hours,
            AVG(TIMESTAMPDIFF(HOUR, created_at, scheduled_date)) as avg_lead_time
        FROM schedules 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY priority
        ORDER BY 
            CASE priority 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'HIGH' THEN 2 
                WHEN 'MEDIUM' THEN 3 
                WHEN 'LOW' THEN 4 
            END
        """
        
        priority_stats = db_service.execute_query(priority_query)
        
        # SLA compliance
        sla_query = """
        SELECT 
            CASE 
                WHEN completed_at IS NULL THEN 'PENDING'
                WHEN completed_at <= deadline THEN 'ON_TIME'
                ELSE 'DELAYED'
            END as sla_status,
            COUNT(*) as count
        FROM schedules 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY sla_status
        """
        
        sla_stats = db_service.execute_query(sla_query)
        
        # Upcoming critical schedules
        upcoming_query = """
        SELECT 
            s.*,
            r.asset_id,
            a.asset_name,
            rs.risk_level
        FROM schedules s
        JOIN reports r ON s.report_id = r.report_id
        JOIN assets a ON r.asset_id = a.asset_id
        LEFT JOIN risk_scores rs ON s.report_id = rs.report_id
        WHERE s.scheduled_date BETWEEN NOW() AND DATE_ADD(NOW(), INTERVAL 7 DAY)
        AND s.status IN ('SCHEDULED', 'IN_PROGRESS')
        AND s.priority IN ('CRITICAL', 'HIGH')
        ORDER BY s.scheduled_date, s.priority DESC
        LIMIT 10
        """
        
        upcoming_critical = db_service.execute_query(upcoming_query)
        
        return {
            'success': True,
            'stats': {
                'status_distribution': status_stats,
                'priority_distribution': priority_stats,
                'sla_compliance': sla_stats,
                'upcoming_critical': upcoming_critical,
                'total_schedules': sum(stat['count'] for stat in status_stats)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get schedule stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get schedule stats: {str(e)}")

# Background task function
async def _process_single_schedule(report_id: int, risk_level: str, risk_score: float, asset_type: str):
    """Background task untuk single schedule generation"""
    try:
        logger.info(f"Background processing schedule for report {report_id}")
        
        # Generate schedule
        schedule_result = scheduler_service.generate_schedule(
            report_id=report_id,
            risk_level=risk_level,
            risk_score=risk_score,
            asset_type=asset_type
        )
        
        if schedule_result['success']:
            # Save to database
            schedule_data = schedule_result['schedule']
            schedule_id = db_service.save_schedule(
                report_id=report_id,
                priority=schedule_data['priority'],
                scheduled_date=schedule_data['scheduled_date'],
                deadline=schedule_data['deadline'],
                estimated_hours=schedule_data['estimated_hours'],
                resource_requirements=schedule_data['resource_requirements']
            )
            
            logger.info(f"Background schedule generated for report {report_id}: ID {schedule_id}")
        else:
            logger.error(f"Background schedule generation failed for report {report_id}: {schedule_result.get('error')}")
        
    except Exception as e:
        logger.error(f"Background schedule generation error for report {report_id}: {e}")
