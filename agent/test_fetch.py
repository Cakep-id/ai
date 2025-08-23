#!/usr/bin/env python3
"""
Test script untuk test database fetch setelah fix SQLAlchemy issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.db import db_service

def test_fetch_all():
    """Test fetch_all method"""
    print("🔧 Testing database fetch_all method...")
    
    try:
        # Test connection first
        print("1. Testing database connection...")
        if db_service.test_connection():
            print("✅ Database connection OK")
        else:
            print("❌ Database connection failed")
            return
        
        # Test simple SELECT
        print("2. Testing simple SELECT query...")
        query = "SELECT COUNT(*) FROM pipeline_inspections WHERE is_active = 1"
        result = db_service.fetch_all(query)
        print(f"✅ Query result: {result}")
        
        # Test full pipeline inspection fetch
        print("3. Testing pipeline inspection fetch...")
        query = """
        SELECT * FROM pipeline_inspections 
        WHERE is_active = 1 
        ORDER BY tanggal_inspeksi DESC
        LIMIT 3
        """
        results = db_service.fetch_all(query)
        print(f"✅ Found {len(results)} inspection records")
        
        if results:
            print("First record columns:")
            columns = [
                'id', 'nama_pipa', 'lokasi_pipa', 'tanggal_inspeksi', 'inspector_name',
                'yolo_detections', 'confidence_threshold', 'deskripsi_kerusakan',
                'ukuran_kerusakan_pixel', 'ukuran_kerusakan_mm', 'area_kerusakan_percent',
                'level_kerusakan', 'risk_score', 'folder_output', 'foto_mentah_path',
                'foto_yolo_path', 'foto_fix_path', 'rekomendasi_tindakan',
                'prosedur_perbaikan', 'estimasi_waktu_perbaikan', 'alat_dibutuhkan',
                'status_inspeksi', 'prioritas', 'created_at', 'updated_at', 'is_active'
            ]
            
            first_record = dict(zip(columns, results[0]))
            print(f"ID: {first_record['id']}")
            print(f"Nama Pipa: {first_record['nama_pipa']}")
            print(f"Level Kerusakan: {first_record['level_kerusakan']}")
            print(f"Tanggal: {first_record['tanggal_inspeksi']}")
        
        print("✅ All database fetch tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fetch_all()
