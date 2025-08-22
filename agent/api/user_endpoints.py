"""
User endpoints untuk laporan kerusakan
Alur: User input gambar + deskripsi -> AI analyze -> Admin validation
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List, Dict, Any
import os
import uuid
from datetime import datetime
import json
from loguru import logger

from services.db import db_service
from services.yolo_service import yolo_service
from services.groq_service import groq_service
from services.risk_engine import risk_engine
from services.scheduler import scheduler_service

router = APIRouter(prefix="/api/user", tags=["user"])

@router.post("/report-damage")
async def report_damage(
    description: str = Form(...),
    image: UploadFile = File(...)
):
    """
    User melaporkan kerusakan dengan gambar dan deskripsi singkat
    AI langsung menganalysis untuk mendeteksi aset dan damage assessment
    """
    try:
        # Validasi file gambar
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
        # Simpan gambar dengan nama unik
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        image_path = f"uploads/user_reports/{unique_filename}"
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Simpan file
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # AI Analysis untuk mendeteksi jenis aset dan kerusakan
        logger.info(f"Starting AI analysis for image {unique_filename}")
        
        # 1. Computer Vision Analysis - detect asset dan damage
        cv_results = yolo_service.detect(image_path)
        
        # 2. NLP Analysis dari deskripsi
        nlp_results = await groq_service.analyze_damage_description(description)
        
        # 3. Tentukan asset berdasarkan AI detection
        detected_asset_name = cv_results.get('asset_type', 'Peralatan Industri')
        detected_location = "Lokasi terdeteksi dari analisis gambar"
        
        # Buat atau cari asset dummy untuk demo (asset_id=1)
        # Atau kita bisa membuat asset baru berdasarkan AI detection
        default_asset_id = 1  # Default asset untuk demo
        
        # Insert laporan ke database sesuai schema yang benar
        report_query = """
        INSERT INTO user_reports (
            asset_id, reported_by_user_id, image_path, 
            description, location_details, reported_at,
            ai_detected_damage, ai_analyzed_at
        ) VALUES (
            :asset_id, :user_id, :image_path, 
            :description, :location_details, :reported_at,
            :ai_detected_damage, :ai_analyzed_at
        )
        """
        
        report_id = db_service.execute_insert(report_query, {
            "asset_id": default_asset_id,
            "user_id": 1,  # Default user untuk demo
            "image_path": image_path,
            "description": description,
            "location_details": detected_location,
            "reported_at": datetime.now(),
            "ai_detected_damage": detected_asset_name,
            "ai_analyzed_at": datetime.now()
        })
        
        if not report_id:
            raise HTTPException(status_code=500, detail="Gagal menyimpan laporan")
        
        # AI Analysis
        logger.info(f"Starting AI analysis for report {report_id}")
        
        # 1. Computer Vision Analysis
        cv_results = yolo_service.detect(image_path)
        
        # 2. NLP Analysis dari deskripsi
        nlp_results = await groq_service.analyze_damage_description(description)
        
        # 4. Risk Assessment berdasarkan AI detection
        risk_assessment = risk_engine.aggregate_risk(
            report_id, cv_results, nlp_results, 'MEDIUM'  # Default criticality
        )
        
        # 5. Generate repair procedures
        procedures = await groq_service.generate_repair_procedures(
            damage_description=description,
            cv_results=cv_results,
            risk_level=risk_assessment['risk_level']
        )
        
        # Update laporan dengan hasil AI
        update_query = """
        UPDATE user_reports SET 
            ai_detected_damage = :damage,
            ai_risk_level = :risk_level,
            ai_confidence = :confidence,
            ai_procedures = :procedures,
            ai_analyzed_at = :analyzed_at
        WHERE report_id = :report_id
        """
        
        db_service.execute_insert(update_query, {
            "damage": risk_assessment.get('detected_damage', 'Unknown'),
            "risk_level": risk_assessment['risk_level'],
            "confidence": risk_assessment['risk_score'],
            "procedures": json.dumps(procedures, ensure_ascii=False),
            "analyzed_at": datetime.now(),
            "report_id": report_id
        })
        
        # Auto-schedule maintenance jika risk tinggi
        if risk_assessment['risk_level'] in ['CRITICAL', 'HIGH']:
            await scheduler_service.create_maintenance_schedule(
                asset_name=detected_asset_name,
                asset_location=detected_location,
                report_id=report_id,
                priority=risk_assessment['risk_level'],
                maintenance_type='corrective',
                procedures=procedures
            )
        
        logger.info(f"Report {report_id} processed successfully with {risk_assessment['risk_level']} risk")
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Laporan berhasil dikirim dan dianalisis AI",
            "ai_analysis": {
                "detected_asset": detected_asset_name,
                "asset_location": detected_location,
                "detected_damage": risk_assessment.get('detected_damage', 'Unknown'),
                "risk_level": risk_assessment['risk_level'],
                "confidence": round(risk_assessment['risk_score'], 2),
                "procedures": procedures
            },
            "status": "Menunggu validasi admin"
        }
        
    except Exception as e:
        logger.error(f"Error in report_damage: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses laporan: {str(e)}")

@router.get("/my-reports")
async def get_my_reports(user_id: int = 1):  # Default untuk demo
    """Get laporan user"""
    try:
        query = """
        SELECT 
            r.*,
            u.full_name as validated_by_name
        FROM user_reports r
        LEFT JOIN users u ON r.validated_by = u.user_id
        WHERE r.reported_by_user_id = :user_id
        ORDER BY r.reported_at DESC
        """
        
        reports = db_service.execute_query(query, {"user_id": user_id})
        
        # Parse JSON procedures
        for report in reports:
            if report['ai_procedures']:
                try:
                    report['ai_procedures'] = json.loads(report['ai_procedures'])
                except:
                    report['ai_procedures'] = []
        
        return {
            "success": True,
            "reports": reports
        }
        
    except Exception as e:
        logger.error(f"Error getting user reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{report_id}")
async def get_report_detail(report_id: int):
    """Get detail laporan"""
    try:
        query = """
        SELECT 
            r.*,
            a.asset_name,
            a.location as asset_location,
            a.criticality,
            u.full_name as validated_by_name,
            ms.scheduled_date as next_maintenance
        FROM user_reports r
        JOIN assets a ON r.asset_id = a.asset_id
        LEFT JOIN users u ON r.validated_by = u.user_id
        LEFT JOIN maintenance_schedules ms ON r.report_id = ms.report_id 
        WHERE r.report_id = :report_id
        """
        
        report = db_service.execute_query(query, {"report_id": report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")
        
        report_data = report[0]
        
        # Parse JSON procedures
        if report_data['ai_procedures']:
            try:
                report_data['ai_procedures'] = json.loads(report_data['ai_procedures'])
            except:
                report_data['ai_procedures'] = []
        
        return {
            "success": True,
            "report": report_data
        }
        
    except Exception as e:
        logger.error(f"Error getting report detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets")
async def get_assets():
    """Get daftar aset untuk form laporan"""
    try:
        query = """
        SELECT 
            a.*,
            ac.category_name
        FROM assets a
        JOIN asset_categories ac ON a.category_id = ac.category_id
        WHERE a.status = 'active'
        ORDER BY a.asset_name
        """
        
        assets = db_service.execute_query(query)
        
        return {
            "success": True,
            "assets": assets
        }
        
    except Exception as e:
        logger.error(f"Error getting assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
