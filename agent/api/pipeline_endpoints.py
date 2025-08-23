"""
Pipeline Inspection API Endpoints
API untuk analisis kerusakan pipa industri migas
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List
import os
import json
from datetime import datetime
from pathlib import Path

from services.pipeline_service import pipeline_service
from services.db import db_service

router = APIRouter()

@router.post("/analyze")
async def analyze_pipeline(
    image: UploadFile = File(...),
    pipeline_id: str = Form(...),
    location: str = Form(...),
    inspector_name: str = Form(...)
):
    """Analyze pipeline image for damage detection and risk assessment"""
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_{pipeline_id}_{timestamp}.{image.filename.split('.')[-1]}"
        file_path = f"uploads/images/{filename}"
        
        # Ensure upload directory exists
        os.makedirs("uploads/images", exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        logger.info(f"Pipeline image uploaded: {filename}")
        
        # Analyze image with YOLO
        yolo_results = yolo_service.detect_objects(file_path)
        
        # Analyze with NLP (if text data available)
        nlp_results = groq_service.analyze_pipeline_condition("Pipeline inspection image analysis")
        
        # Calculate risk assessment
        risk_analysis = risk_engine.aggregate_risk(
            report_id=1,  # This would be dynamic in production
            visual_data=yolo_results,
            text_data=nlp_results
        )
        
        # Generate unique inspection ID
        inspection_id = f"INSP_{pipeline_id}_{timestamp}"
        
        # Store inspection results in database
        try:
            conn = get_db_connection()
            
            # Create table if not exists
            create_table_query = """
            CREATE TABLE IF NOT EXISTS pipeline_inspections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                inspection_id VARCHAR(255) UNIQUE,
                pipeline_id VARCHAR(255),
                location TEXT,
                inspector_name VARCHAR(255),
                image_path VARCHAR(500),
                risk_level VARCHAR(50),
                risk_score DECIMAL(5,3),
                confidence_score DECIMAL(5,3),
                yolo_detections JSON,
                nlp_analysis JSON,
                inspection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(create_table_query)
            
            # Insert inspection record
            insert_query = """
            INSERT INTO pipeline_inspections 
            (inspection_id, pipeline_id, location, inspector_name, image_path, 
             risk_level, risk_score, confidence_score, yolo_detections, nlp_analysis)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            import json
            conn.execute(insert_query, (
                inspection_id,
                pipeline_id,
                location,
                inspector_name,
                file_path,
                risk_analysis['overall_risk'],
                risk_analysis['risk_score'],
                risk_analysis.get('confidence', 0.8),
                json.dumps(yolo_results),
                json.dumps(nlp_results)
            ))
            conn.commit()
            
            logger.info(f"Inspection results stored for {inspection_id}")
            
        except Exception as db_error:
            logger.warning(f"Database storage failed: {db_error}")
            # Continue without failing the analysis
        
        # Prepare response
        response = {
            "inspection_id": inspection_id,
            "pipeline_id": pipeline_id,
            "location": location,
            "inspector_name": inspector_name,
            "image_path": file_path,
            "yolo_detection": yolo_results,
            "nlp_analysis": nlp_results,
            "risk_analysis": risk_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Pipeline analysis completed for {pipeline_id}")
        return response
        
    except Exception as e:
        logger.error(f"Pipeline analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_inspection_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 50
):
    """Get pipeline inspection history with filters"""
    try:
        conn = get_db_connection()
        
        # Build query with filters
        query = """
        SELECT 
            pi.inspection_id,
            pi.pipeline_id,
            pi.location,
            pi.inspector_name,
            pi.risk_level,
            pi.risk_score,
            pi.confidence_score,
            pi.inspection_date,
            CASE WHEN hf.inspection_id IS NOT NULL THEN TRUE ELSE FALSE END as has_feedback
        FROM pipeline_inspections pi
        LEFT JOIN human_feedback hf ON pi.inspection_id = hf.inspection_id
        WHERE 1=1
        """
        
        params = []
        
        if start_date:
            query += " AND pi.inspection_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND pi.inspection_date <= %s"
            params.append(end_date)
        
        if pipeline_id:
            query += " AND pi.pipeline_id LIKE %s"
            params.append(f"%{pipeline_id}%")
        
        if risk_level:
            query += " AND pi.risk_level = %s"
            params.append(risk_level)
        
        query += " ORDER BY pi.inspection_date DESC LIMIT %s"
        params.append(limit)
        
        result = conn.execute(query, params)
        history = []
        
        for row in result:
            history.append({
                "inspection_id": row[0],
                "pipeline_id": row[1],
                "location": row[2],
                "inspector_name": row[3],
                "risk_level": row[4],
                "risk_score": float(row[5]),
                "confidence_score": float(row[6]),
                "inspection_date": row[7],
                "has_feedback": bool(row[8])
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Error loading inspection history: {e}")
        return []

@router.get("/inspection/{inspection_id}")
async def get_inspection_details(inspection_id: str):
    """Get detailed inspection results by ID"""
    try:
        conn = get_db_connection()
        
        query = """
        SELECT 
            inspection_id, pipeline_id, location, inspector_name,
            image_path, risk_level, risk_score, confidence_score,
            yolo_detections, nlp_analysis, inspection_date
        FROM pipeline_inspections
        WHERE inspection_id = %s
        """
        
        result = conn.execute(query, (inspection_id,))
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        import json
        
        return {
            "inspection_id": row[0],
            "pipeline_id": row[1],
            "location": row[2],
            "inspector_name": row[3],
            "image_path": row[4],
            "risk_level": row[5],
            "risk_score": float(row[6]),
            "confidence_score": float(row[7]),
            "yolo_detections": json.loads(row[8]) if row[8] else [],
            "nlp_analysis": json.loads(row[9]) if row[9] else {},
            "inspection_date": row[10]
        }
        
    except Exception as e:
        logger.error(f"Error loading inspection details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/inspections")
async def get_pipeline_inspections(
    limit: int = 50,
    order_by_severity: bool = True,
    level_filter: Optional[str] = None,
    status_filter: Optional[str] = None
):
    """
    Ambil daftar hasil inspeksi pipa
    """
    try:
        # Build query dengan filter - use direct string construction to avoid parameter binding issues
        where_conditions = ["is_active = 1"]
        
        if level_filter:
            # Sanitize input untuk prevent SQL injection
            level_filter = level_filter.replace("'", "").replace(";", "")
            where_conditions.append(f"level_kerusakan = '{level_filter}'")
        
        if status_filter:
            # Sanitize input untuk prevent SQL injection
            status_filter = status_filter.replace("'", "").replace(";", "")
            where_conditions.append(f"status_inspeksi = '{status_filter}'")
        
        where_clause = " AND ".join(where_conditions)
        
        order_clause = "FIELD(level_kerusakan, 'High', 'Medium', 'Low'), tanggal_inspeksi DESC" if order_by_severity else "tanggal_inspeksi DESC"
        
        # Use direct query construction without parameters to avoid SQLAlchemy binding issues
        query = f"""
        SELECT 
            id, nama_pipa, lokasi_pipa, tanggal_inspeksi, inspector_name,
            deskripsi_kerusakan, ukuran_kerusakan_mm, area_kerusakan_percent,
            level_kerusakan, status_inspeksi, rekomendasi_tindakan,
            foto_mentah_path, foto_yolo_path, foto_fix_path
        FROM pipeline_inspections 
        WHERE {where_clause}
        ORDER BY {order_clause}
        LIMIT {int(limit)}
        """
        
        # Call fetch_all without parameters
        results = db_service.fetch_all(query)
        
        inspections = []
        for row in results:
            inspections.append({
                'id': row[0],
                'nama_pipa': row[1],
                'lokasi_pipa': row[2],
                'tanggal_inspeksi': row[3].isoformat() if row[3] else None,
                'inspector_name': row[4],
                'deskripsi_kerusakan': row[5],
                'ukuran_kerusakan_mm': row[6],
                'area_kerusakan_percent': row[7],
                'level_kerusakan': row[8],
                'status_inspeksi': row[9],
                'rekomendasi_tindakan': row[10],
                'foto_paths': {
                    'mentah': row[11],
                    'yolo': row[12],
                    'analyzed': row[13]
                }
            })
        
        return JSONResponse({
            "success": True,
            "data": inspections,
            "total": len(inspections)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/inspection/{inspection_id}")
async def get_inspection_detail(inspection_id: int):
    """
    Ambil detail inspeksi berdasarkan ID
    """
    try:
        query = """
        SELECT * FROM pipeline_inspections 
        WHERE id = %s AND is_active = TRUE
        """
        
        result = db_service.fetch_one(query, (inspection_id,))
        
        if not result:
            raise HTTPException(status_code=404, detail="Inspeksi tidak ditemukan")
        
        # Column mapping
        columns = [
            'id', 'nama_pipa', 'lokasi_pipa', 'tanggal_inspeksi', 'inspector_name',
            'yolo_detections', 'confidence_threshold', 'deskripsi_kerusakan',
            'ukuran_kerusakan_pixel', 'ukuran_kerusakan_mm', 'area_kerusakan_percent',
            'level_kerusakan', 'risk_score', 'folder_output', 'foto_mentah_path',
            'foto_yolo_path', 'foto_fix_path', 'rekomendasi_tindakan',
            'prosedur_perbaikan', 'estimasi_waktu_perbaikan', 'alat_dibutuhkan',
            'status_inspeksi', 'prioritas', 'created_at', 'updated_at', 'is_active'
        ]
        
        inspection_data = dict(zip(columns, result))
        
        # Parse JSON fields
        if inspection_data['yolo_detections']:
            try:
                inspection_data['yolo_detections'] = json.loads(inspection_data['yolo_detections'])
            except:
                inspection_data['yolo_detections'] = []
        
        # Format dates
        for date_field in ['tanggal_inspeksi', 'created_at', 'updated_at']:
            if inspection_data[date_field]:
                inspection_data[date_field] = inspection_data[date_field].isoformat()
        
        return JSONResponse({
            "success": True,
            "data": inspection_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/download/{inspection_id}/{file_type}")
async def download_inspection_file(inspection_id: int, file_type: str):
    """
    Download file hasil inspeksi
    file_type: mentah, yolo, analyzed, report
    """
    try:
        query = """
        SELECT foto_mentah_path, foto_yolo_path, foto_fix_path, folder_output
        FROM pipeline_inspections 
        WHERE id = %s AND is_active = TRUE
        """
        
        result = db_service.fetch_one(query, (inspection_id,))
        
        if not result:
            raise HTTPException(status_code=404, detail="Inspeksi tidak ditemukan")
        
        file_path = None
        if file_type == "mentah":
            file_path = result[0]
        elif file_type == "yolo":
            file_path = result[1]
        elif file_type == "analyzed":
            file_path = result[2]
        elif file_type == "report":
            # Cari file report di folder output
            folder_output = result[3]
            if folder_output:
                report_folder = Path(folder_output) / "reports"
                report_files = list(report_folder.glob("*.txt"))
                if report_files:
                    file_path = str(report_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File tidak ditemukan")
        
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/stats")
async def get_pipeline_statistics():
    """
    Statistik inspeksi pipa
    """
    try:
        # Total inspections
        total_query = "SELECT COUNT(*) FROM pipeline_inspections WHERE is_active = TRUE"
        total_count = db_service.fetch_one(total_query)[0]
        
        # By level
        level_query = """
        SELECT level_kerusakan, COUNT(*) as count
        FROM pipeline_inspections 
        WHERE is_active = TRUE
        GROUP BY level_kerusakan
        """
        level_results = db_service.fetch_all(level_query)
        level_stats = {row[0]: row[1] for row in level_results}
        
        # By status
        status_query = """
        SELECT status_inspeksi, COUNT(*) as count
        FROM pipeline_inspections 
        WHERE is_active = TRUE
        GROUP BY status_inspeksi
        """
        status_results = db_service.fetch_all(status_query)
        status_stats = {row[0]: row[1] for row in status_results}
        
        # Recent high priority
        high_priority_query = """
        SELECT COUNT(*) FROM pipeline_inspections 
        WHERE level_kerusakan = 'High' 
        AND status_inspeksi IN ('pending', 'analyzed')
        AND is_active = TRUE
        """
        high_priority_count = db_service.fetch_one(high_priority_query)[0]
        
        return JSONResponse({
            "success": True,
            "data": {
                "total_inspections": total_count,
                "by_level": level_stats,
                "by_status": status_stats,
                "high_priority_pending": high_priority_count
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/cleanup")
async def cleanup_old_files(keep_days: int = 30):
    """
    Cleanup file lama yang tidak diperlukan
    """
    try:
        pipeline_service.cleanup_old_folders(keep_days)
        
        return JSONResponse({
            "success": True,
            "message": f"Cleanup completed. Files older than {keep_days} days removed."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/pipeline/inspection/{inspection_id}/status")
async def update_inspection_status(
    inspection_id: int,
    new_status: str = Form(...),
    notes: str = Form("")
):
    """
    Update status inspeksi
    """
    try:
        valid_statuses = ['pending', 'analyzed', 'scheduled', 'repaired']
        if new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail="Status tidak valid")
        
        query = """
        UPDATE pipeline_inspections 
        SET status_inspeksi = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s AND is_active = TRUE
        """
        
        db_service.execute_insert(query, (new_status, inspection_id))
        
        return JSONResponse({
            "success": True,
            "message": "Status berhasil diupdate"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
