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

@router.post("/pipeline/analyze")
async def analyze_pipeline_image(
    image: UploadFile = File(...),
    nama_pipa: str = Form(...),
    lokasi_pipa: str = Form(""),
    inspector_name: str = Form("")
):
    """
    Analisis gambar pipa untuk deteksi kerusakan
    """
    try:
        # Validasi file
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            raise HTTPException(status_code=400, detail="Format file tidak didukung")
        
        # Simpan file temporary
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        try:
            # Analisis gambar
            result = pipeline_service.analyze_pipeline_image(
                str(temp_file_path),
                nama_pipa,
                lokasi_pipa,
                inspector_name
            )
            
            # Cleanup temp file
            os.unlink(temp_file_path)
            
            if result['success']:
                return JSONResponse({
                    "success": True,
                    "message": "Analisis berhasil",
                    "data": {
                        "inspection_id": result['inspection_id'],
                        "database_id": result['database_id'],
                        "nama_pipa": result['nama_pipa'],
                        "level_kerusakan": result['level_kerusakan'],
                        "deskripsi_kerusakan": result['deskripsi_kerusakan'],
                        "ukuran_kerusakan_mm": result['ukuran_kerusakan_mm'],
                        "area_percent": result['area_percent'],
                        "rekomendasi": result['rekomendasi'],
                        "output_folder": result['output_folder'],
                        "foto_paths": result['foto_paths']
                    }
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": result.get('message', 'Analisis gagal'),
                    "error": result.get('error')
                }, status_code=400)
                
        except Exception as e:
            # Cleanup temp file on error
            if temp_file_path.exists():
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
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
