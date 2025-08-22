"""
Admin endpoints untuk validation workflow dan training AI
Alur: Admin approve/reject laporan user, upload training data sendiri
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
import uuid
from datetime import datetime, timedelta
import json
from loguru import logger

from services.db import db_service
from services.yolo_service import yolo_service
from services.groq_service import groq_service
from services.scheduler import scheduler_service

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Pydantic models
class RiskLevelUpdate(BaseModel):
    new_risk_level: str
    notes: Optional[str] = ""

@router.get("/validation-queue")
async def get_validation_queue(
    status: str = "pending",
    risk_level: Optional[str] = None,
    limit: int = 50
):
    """
    Get antrian validasi laporan user
    """
    try:
        base_query = """
        SELECT 
            r.*,
            a.asset_name,
            a.location as asset_location,
            a.criticality,
            u.full_name as reporter_name,
            TIMESTAMPDIFF(HOUR, r.reported_at, NOW()) as hours_since_report
        FROM user_reports r
        JOIN assets a ON r.asset_id = a.asset_id
        JOIN users u ON r.reported_by_user_id = u.user_id
        WHERE r.admin_status = :status
        """
        
        params = {"status": status}
        
        if risk_level:
            base_query += " AND r.ai_risk_level = :risk_level"
            params["risk_level"] = risk_level
        
        base_query += " ORDER BY r.ai_risk_level DESC, r.reported_at ASC LIMIT :limit"
        params["limit"] = limit
        
        reports = db_service.execute_query(base_query, params)
        
        # Parse JSON procedures dan tambah urgency info
        for report in reports:
            if report['ai_procedures']:
                try:
                    report['ai_procedures'] = json.loads(report['ai_procedures'])
                except:
                    report['ai_procedures'] = []
            
            # Calculate urgency - Updated untuk 3 level saja
            hours_since = report['hours_since_report']
            if report['ai_risk_level'] == 'HIGH' and hours_since > 2:
                report['urgency_status'] = 'OVERDUE'
            elif report['ai_risk_level'] == 'MEDIUM' and hours_since > 8:
                report['urgency_status'] = 'OVERDUE'
            elif report['ai_risk_level'] == 'HIGH':
                report['urgency_status'] = 'URGENT'
            else:
                report['urgency_status'] = 'NORMAL'
        
        return {
            "success": True,
            "queue": reports,
            "total": len(reports)
        }
        
    except Exception as e:
        logger.error(f"Error getting validation queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-report/{report_id}")
async def validate_report(
    report_id: int,
    action: str = Form(...),  # "approve" or "reject"
    admin_notes: str = Form(""),
    corrected_damage: Optional[str] = Form(None),
    corrected_risk: Optional[str] = Form(None),
    corrected_procedures: Optional[str] = Form(None)
):
    """
    Admin approve atau reject laporan user
    Jika approve: tambah ke training data AI
    Jika reject: tidak digunakan untuk training
    """
    try:
        if action not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Action harus 'approve' atau 'reject'")
        
        # Get laporan
        report_query = "SELECT * FROM user_reports WHERE report_id = :report_id"
        report = db_service.execute_query(report_query, {"report_id": report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")
        
        report_data = report[0]
        
        # Update status validasi
        update_query = """
        UPDATE user_reports SET 
            admin_status = :status,
            validated_by = :admin_id,
            validated_at = :validated_at,
            admin_notes = :admin_notes,
            admin_corrected_damage = :corrected_damage,
            admin_corrected_risk = :corrected_risk,
            admin_corrected_procedures = :corrected_procedures
        WHERE report_id = :report_id
        """
        
        db_service.execute_insert(update_query, {
            "status": "approved" if action == "approve" else "rejected",
            "admin_id": 1,  # Default admin untuk demo
            "validated_at": datetime.now(),
            "admin_notes": admin_notes,
            "corrected_damage": corrected_damage,
            "corrected_risk": corrected_risk,
            "corrected_procedures": corrected_procedures,
            "report_id": report_id
        })
        
        # Jika approve, tambah ke learning data
        if action == "approve":
            # Determine final labels (admin correction atau AI result)
            final_damage = corrected_damage or report_data['ai_detected_damage']
            final_risk = corrected_risk or report_data['ai_risk_level']
            
            # Tambah ke AI learning history
            learning_query = """
            INSERT INTO ai_learning_history (
                source_type, source_id, image_path, damage_label, 
                risk_level, confidence_score, model_version, learned_at
            ) VALUES (
                'user_report', :report_id, :image_path, :damage_label,
                :risk_level, :confidence, 'yolo_v1.0', :learned_at
            )
            """
            
            db_service.execute_insert(learning_query, {
                "report_id": report_id,
                "image_path": report_data['image_path'],
                "damage_label": final_damage,
                "risk_level": final_risk,
                "confidence": report_data['ai_confidence'] or 0.5,
                "learned_at": datetime.now()
            })
            
            # Mark as training data
            db_service.execute_insert(
                "UPDATE user_reports SET is_used_for_training = TRUE, training_added_at = :timestamp WHERE report_id = :report_id",
                {"timestamp": datetime.now(), "report_id": report_id}
            )
            
            # Update maintenance schedule jika ada koreksi
            if corrected_risk and corrected_risk != report_data['ai_risk_level']:
                await scheduler_service.update_maintenance_priority(
                    asset_id=report_data['asset_id'],
                    report_id=report_id,
                    new_priority=corrected_risk
                )
            
            logger.info(f"Report {report_id} approved and added to training data")
            
            return {
                "success": True,
                "message": f"Laporan berhasil di-approve dan ditambahkan ke data training AI",
                "learning_added": True,
                "final_labels": {
                    "damage": final_damage,
                    "risk": final_risk
                }
            }
        else:
            logger.info(f"Report {report_id} rejected, not added to training")
            
            return {
                "success": True,
                "message": "Laporan di-reject, tidak akan digunakan untuk training AI",
                "learning_added": False
            }
        
    except Exception as e:
        logger.error(f"Error validating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-training-data")
async def upload_training_data(
    background_tasks: BackgroundTasks,
    category_id: int = Form(...),
    damage_type_id: int = Form(...),
    damage_description: str = Form(...),
    risk_level: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Admin upload gambar training langsung untuk AI
    """
    try:
        if risk_level not in ['HIGH', 'MEDIUM', 'LOW']:
            raise HTTPException(status_code=400, detail="Risk level tidak valid")
        
        # Validasi category dan damage type
        category_query = "SELECT * FROM asset_categories WHERE category_id = :id"
        category = db_service.execute_query(category_query, {"id": category_id})
        if not category:
            raise HTTPException(status_code=404, detail="Asset category tidak ditemukan")
        
        damage_query = "SELECT * FROM damage_types WHERE damage_type_id = :id"
        damage_type = db_service.execute_query(damage_query, {"id": damage_type_id})
        if not damage_type:
            raise HTTPException(status_code=404, detail="Damage type tidak ditemukan")
        
        uploaded_files = []
        
        for image in images:
            # Validasi file gambar
            if not image.content_type.startswith('image/'):
                continue
            
            # Simpan dengan nama unik
            file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
            unique_filename = f"admin_training_{uuid.uuid4()}.{file_extension}"
            image_path = f"uploads/admin_training/{unique_filename}"
            
            # Buat direktori
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Simpan file
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            
            # Insert ke database
            training_query = """
            INSERT INTO admin_training_data (
                uploaded_by_admin, image_path, asset_category_id, damage_type_id,
                damage_description, risk_level, uploaded_at
            ) VALUES (
                :admin_id, :image_path, :category_id, :damage_type_id,
                :description, :risk_level, :uploaded_at
            )
            """
            
            training_id = db_service.execute_insert(training_query, {
                "admin_id": 1,  # Default admin
                "image_path": image_path,
                "category_id": category_id,
                "damage_type_id": damage_type_id,
                "description": damage_description,
                "risk_level": risk_level,
                "uploaded_at": datetime.now()
            })
            
            # Tambah ke learning history
            if training_id:
                learning_query = """
                INSERT INTO ai_learning_history (
                    source_type, source_id, image_path, damage_label,
                    risk_level, model_version, learned_at
                ) VALUES (
                    'admin_training', :training_id, :image_path, :damage_label,
                    :risk_level, 'yolo_v1.0', :learned_at
                )
                """
                
                db_service.execute_insert(learning_query, {
                    "training_id": training_id,
                    "image_path": image_path,
                    "damage_label": damage_type[0]['damage_name'],
                    "risk_level": risk_level,
                    "learned_at": datetime.now()
                })
                
                uploaded_files.append({
                    "filename": image.filename,
                    "training_id": training_id,
                    "saved_path": image_path
                })
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="Tidak ada gambar valid yang diupload")
        
        # Schedule AI retrain jika sudah cukup data
        background_tasks.add_task(check_and_trigger_retrain)
        
        logger.info(f"Admin uploaded {len(uploaded_files)} training images")
        
        return {
            "success": True,
            "message": f"Berhasil upload {len(uploaded_files)} gambar training",
            "uploaded_files": uploaded_files,
            "training_scheduled": True
        }
        
    except Exception as e:
        logger.error(f"Error uploading training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-data")
async def get_training_data(
    source_type: Optional[str] = None,  # "user_report" atau "admin_training"
    limit: int = 100
):
    """
    Get data training yang sudah dikumpulkan
    """
    try:
        base_query = """
        SELECT 
            ah.*,
            CASE 
                WHEN ah.source_type = 'user_report' THEN ur.description
                WHEN ah.source_type = 'admin_training' THEN atd.damage_description
            END as description,
            CASE 
                WHEN ah.source_type = 'user_report' THEN u.full_name
                WHEN ah.source_type = 'admin_training' THEN ua.full_name
            END as source_user
        FROM ai_learning_history ah
        LEFT JOIN user_reports ur ON ah.source_type = 'user_report' AND ah.source_id = ur.report_id
        LEFT JOIN users u ON ur.reported_by_user_id = u.user_id
        LEFT JOIN admin_training_data atd ON ah.source_type = 'admin_training' AND ah.source_id = atd.training_id
        LEFT JOIN users ua ON atd.uploaded_by_admin = ua.user_id
        WHERE 1=1
        """
        
        params = {}
        
        if source_type:
            base_query += " AND ah.source_type = :source_type"
            params["source_type"] = source_type
        
        base_query += " ORDER BY ah.learned_at DESC LIMIT :limit"
        params["limit"] = limit
        
        training_data = db_service.execute_query(base_query, params)
        
        # Get statistics
        stats_query = """
        SELECT 
            source_type,
            damage_label,
            risk_level,
            COUNT(*) as count
        FROM ai_learning_history 
        GROUP BY source_type, damage_label, risk_level
        ORDER BY count DESC
        """
        
        stats = db_service.execute_query(stats_query)
        
        return {
            "success": True,
            "training_data": training_data,
            "statistics": stats,
            "total_samples": len(training_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard-stats")
async def get_dashboard_stats():
    """
    Get statistik untuk admin dashboard
    """
    try:
        # Pending validations
        pending_count = db_service.execute_query(
            "SELECT COUNT(*) as count FROM user_reports WHERE admin_status = 'pending'"
        )[0]['count']
        
        # Risk level distribution
        risk_stats = db_service.execute_query("""
            SELECT ai_risk_level, COUNT(*) as count 
            FROM user_reports 
            WHERE admin_status = 'pending' 
            GROUP BY ai_risk_level
        """)
        
        # Training data count
        training_count = db_service.execute_query(
            "SELECT COUNT(*) as count FROM ai_learning_history"
        )[0]['count']
        
        # Recent activities
        recent_activities = db_service.execute_query("""
            SELECT 
                'report' as type,
                CONCAT('Laporan kerusakan: ', description) as activity,
                reported_at as timestamp
            FROM user_reports 
            WHERE DATE(reported_at) >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            
            UNION ALL
            
            SELECT 
                'training' as type,
                CONCAT('Training data: ', damage_description) as activity,
                uploaded_at as timestamp
            FROM admin_training_data 
            WHERE DATE(uploaded_at) >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            
            ORDER BY timestamp DESC LIMIT 10
        """)
        
        # Maintenance schedules
        upcoming_maintenance = db_service.execute_query("""
            SELECT 
                ms.*,
                a.asset_name
            FROM maintenance_schedules ms
            JOIN assets a ON ms.asset_id = a.asset_id
            WHERE ms.scheduled_date >= CURDATE() 
            AND ms.status = 'scheduled'
            ORDER BY ms.scheduled_date ASC LIMIT 5
        """)
        
        return {
            "success": True,
            "stats": {
                "pending_validations": pending_count,
                "training_samples": training_count,
                "risk_distribution": risk_stats,
                "recent_activities": recent_activities,
                "upcoming_maintenance": upcoming_maintenance
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_and_trigger_retrain():
    """
    Background task untuk check apakah perlu retrain AI
    """
    try:
        # Check jumlah training data baru
        new_data_count = db_service.execute_query("""
            SELECT COUNT(*) as count 
            FROM ai_learning_history 
            WHERE learned_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
        """)[0]['count']
        
        # Trigger retrain jika ada minimal 20 sample baru
        if new_data_count >= 20:
            logger.info(f"Triggering AI retrain with {new_data_count} new samples")
            # Implementasi retrain logic di sini
            # await yolo_service.retrain_model()
        
    except Exception as e:
        logger.error(f"Error in background retrain check: {e}")

@router.get("/assets")
async def get_asset_categories():
    """Get asset categories dan damage types untuk form"""
    try:
        categories = db_service.execute_query("SELECT * FROM asset_categories ORDER BY category_name")
        damage_types = db_service.execute_query("SELECT * FROM damage_types ORDER BY damage_name")
        
        return {
            "success": True,
            "categories": categories,
            "damage_types": damage_types
        }
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-risk-level/{report_id}")
async def update_risk_level(
    report_id: int,
    request: RiskLevelUpdate
):
    """
    Admin dapat mengubah risk level yang sudah di-generate oleh AI
    Endpoint khusus untuk edit risk level tanpa approve/reject penuh
    """
    try:
        # Validate risk level
        valid_risk_levels = ["LOW", "MEDIUM", "HIGH"]
        if request.new_risk_level not in valid_risk_levels:
            raise HTTPException(
                status_code=400, 
                detail=f"Risk level harus salah satu dari: {', '.join(valid_risk_levels)}"
            )
        
        # Check if report exists
        report_query = "SELECT * FROM user_reports WHERE report_id = :report_id"
        report = db_service.execute_query(report_query, {"report_id": report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")
        
        current_report = report[0]
        old_risk_level = current_report.get('ai_risk_level', 'UNKNOWN')
        
        # Update risk level
        update_query = """
        UPDATE user_reports SET 
            ai_risk_level = :new_risk_level,
            admin_corrected_risk = :new_risk_level,
            admin_notes = CONCAT(
                COALESCE(admin_notes, ''), 
                :new_notes
            ),
            validated_at = :validated_at,
            validated_by = :admin_id
        WHERE report_id = :report_id
        """
        
        timestamp = datetime.now()
        notes_addition = f"\n[{timestamp.strftime('%Y-%m-%d %H:%M')}] Risk level diubah dari {old_risk_level} ke {request.new_risk_level}"
        if request.notes:
            notes_addition += f" - {request.notes}"
        
        db_service.execute_insert(update_query, {
            "new_risk_level": request.new_risk_level,
            "new_notes": notes_addition,
            "validated_at": timestamp,
            "admin_id": 1,  # Default admin for demo
            "report_id": report_id
        })
        
        # Log perubahan
        logger.info(f"Admin mengubah risk level report {report_id} dari {old_risk_level} ke {request.new_risk_level}")
        
        # Update maintenance schedule jika diperlukan untuk HIGH risk
        if request.new_risk_level == 'HIGH' and old_risk_level != 'HIGH':
            # Buat/update schedule maintenance untuk high risk
            await scheduler_service.create_maintenance_schedule(
                asset_name=current_report.get('ai_detected_damage', 'Unknown Asset'),
                asset_location=current_report.get('location_details', 'Unknown Location'),
                report_id=report_id,
                priority=request.new_risk_level,
                maintenance_type='corrective',
                procedures=[]
            )
            logger.info(f"Created urgent maintenance schedule for report {report_id} due to {request.new_risk_level} risk")
        
        return {
            "success": True,
            "message": f"Risk level berhasil diubah dari {old_risk_level} ke {request.new_risk_level}",
            "report_id": report_id,
            "old_risk_level": old_risk_level,
            "new_risk_level": request.new_risk_level,
            "updated_at": timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating risk level for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengubah risk level: {str(e)}")

@router.get("/reports/pending")
async def get_pending_reports(
    risk_level: Optional[str] = None,
    limit: int = 20
):
    """
    Get laporan yang pending untuk admin review/edit
    """
    try:
        base_query = """
        SELECT 
            r.*,
            a.asset_name,
            a.location as asset_location,
            a.criticality,
            u.full_name as reporter_name,
            TIMESTAMPDIFF(HOUR, r.reported_at, NOW()) as hours_since_report
        FROM user_reports r
        JOIN assets a ON r.asset_id = a.asset_id
        JOIN users u ON r.reported_by_user_id = u.user_id
        WHERE r.admin_status IN ('pending', 'under_review')
        """
        
        params = {}
        
        if risk_level:
            base_query += " AND r.ai_risk_level = :risk_level"
            params["risk_level"] = risk_level
        
        base_query += " ORDER BY r.reported_at DESC LIMIT :limit"
        params["limit"] = limit
        
        reports = db_service.execute_query(base_query, params)
        
        # Parse JSON procedures jika ada
        for report in reports:
            if report.get('ai_procedures'):
                try:
                    report['ai_procedures'] = json.loads(report['ai_procedures'])
                except:
                    report['ai_procedures'] = []
        
        return {
            "success": True,
            "reports": reports,
            "total": len(reports)
        }
        
    except Exception as e:
        logger.error(f"Error getting pending reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))
