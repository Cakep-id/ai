"""
Pipeline Inspection Service untuk Analisis Kerusakan Pipa Industri Migas
Menggunaka            # YOLO detection
            logger.info(f"Running YOLO detection on {image_path}")
            yolo_results = yolo_service.detect(str(image_path))OLO untuk deteksi awal dan AI untuk analisis deskriptif
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from services.yolo_service import yolo_service
from services.groq_service import groq_service
from services.db import db_service


class PipelineAnalysisService:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.pixel_to_mm_ratio = 0.1  # Default ratio, should be calibrated per camera setup
        
        # Damage type mappings dari YOLO labels
        self.damage_types = {
            'corrosion': 'Korosi',
            'crack': 'Retakan', 
            'dent': 'Penyok',
            'rust': 'Karat',
            'leak': 'Kebocoran',
            'erosion': 'Erosi',
            'coating_damage': 'Kerusakan Coating',
            'weld_defect': 'Cacat Las'
        }
        
        # Setup output directories
        self.base_output_dir = Path("pipeline_analysis_output")
        self.setup_directories()
    
    def setup_directories(self):
        """Setup folder structure untuk output"""
        dirs = [
            self.base_output_dir / "foto_mentah",
            self.base_output_dir / "foto_yolo", 
            self.base_output_dir / "foto_fix",
            self.base_output_dir / "reports"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def cleanup_old_folders(self, keep_days: int = 30):
        """Hapus folder lama yang tidak diperlukan"""
        try:
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
            
            for folder in self.base_output_dir.iterdir():
                if folder.is_dir():
                    if folder.stat().st_mtime < cutoff_time:
                        shutil.rmtree(folder)
                        logger.info(f"Cleaned up old folder: {folder}")
        except Exception as e:
            logger.error(f"Error cleaning up folders: {e}")
    
    def analyze_pipeline_image(
        self, 
        image_path: str, 
        nama_pipa: str,
        lokasi_pipa: str = "",
        inspector_name: str = ""
    ) -> Dict:
        """
        Analisis utama untuk gambar pipa
        """
        try:
            # Generate unique ID untuk inspection ini
            inspection_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.base_output_dir / f"inspection_{inspection_id}"
            output_folder.mkdir(exist_ok=True)
            
            # 1. Copy foto mentah
            foto_mentah_path = output_folder / "foto_mentah" / f"{nama_pipa}_original.jpg"
            foto_mentah_path.parent.mkdir(exist_ok=True)
            shutil.copy2(image_path, foto_mentah_path)
            
            # 2. YOLO Detection
            logger.info(f"Running YOLO detection on {image_path}")
            yolo_results = yolo_service.detect(str(image_path))
            
            if not yolo_results.get('success', False):
                raise Exception("YOLO detection failed")
            
            # 3. Filter berdasarkan confidence
            filtered_detections = self._filter_detections(yolo_results.get('detections', []))
            
            if not filtered_detections:
                return {
                    'success': False,
                    'message': 'No significant damage detected',
                    'inspection_id': inspection_id
                }
            
            # 4. Analisis ukuran dan deskripsi
            image = cv2.imread(image_path)
            analysis_results = self._analyze_damage(image, filtered_detections)
            
            # 5. Generate foto YOLO dengan bounding boxes
            foto_yolo_path = self._generate_yolo_image(
                image, filtered_detections, output_folder / "foto_yolo" / f"{nama_pipa}_yolo.jpg"
            )
            
            # 6. Generate deskripsi AI yang rinci
            deskripsi_kerusakan = self._generate_damage_description(analysis_results)
            
            # 7. Klasifikasi level kerusakan
            level_kerusakan = self._classify_damage_level(analysis_results['total_area_percent'])
            
            # 8. Generate rekomendasi
            rekomendasi = self._generate_recommendations(level_kerusakan, analysis_results)
            
            # 9. Get prosedur perbaikan dari Groq API
            prosedur_perbaikan = self._get_repair_procedures(deskripsi_kerusakan, level_kerusakan)
            
            # 10. Generate foto fix dengan anotasi lengkap
            foto_fix_path = self._generate_annotated_image(
                image, analysis_results, deskripsi_kerusakan, level_kerusakan,
                output_folder / "foto_fix" / f"{nama_pipa}_analyzed.jpg"
            )
            
            # 11. Simpan ke database
            db_result = self._save_to_database({
                'nama_pipa': nama_pipa,
                'lokasi_pipa': lokasi_pipa,
                'inspector_name': inspector_name,
                'yolo_detections': json.dumps(filtered_detections),
                'deskripsi_kerusakan': deskripsi_kerusakan,
                'ukuran_kerusakan_pixel': analysis_results['total_area_pixel'],
                'ukuran_kerusakan_mm': analysis_results['total_area_mm'],
                'area_kerusakan_percent': analysis_results['total_area_percent'],
                'level_kerusakan': level_kerusakan,
                'folder_output': str(output_folder),
                'foto_mentah_path': str(foto_mentah_path),
                'foto_yolo_path': str(foto_yolo_path),
                'foto_fix_path': str(foto_fix_path),
                'rekomendasi_tindakan': rekomendasi,
                'prosedur_perbaikan': prosedur_perbaikan
            })
            
            # 12. Generate laporan
            report_path = self._generate_report(
                inspection_id, nama_pipa, analysis_results, 
                deskripsi_kerusakan, level_kerusakan, rekomendasi, 
                prosedur_perbaikan, output_folder
            )
            
            return {
                'success': True,
                'inspection_id': inspection_id,
                'database_id': db_result,
                'nama_pipa': nama_pipa,
                'level_kerusakan': level_kerusakan,
                'deskripsi_kerusakan': deskripsi_kerusakan,
                'ukuran_kerusakan_mm': analysis_results['total_area_mm'],
                'area_percent': analysis_results['total_area_percent'],
                'rekomendasi': rekomendasi,
                'prosedur_perbaikan': prosedur_perbaikan,
                'output_folder': str(output_folder),
                'report_path': str(report_path),
                'foto_paths': {
                    'mentah': str(foto_mentah_path),
                    'yolo': str(foto_yolo_path),
                    'analyzed': str(foto_fix_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'inspection_id': inspection_id if 'inspection_id' in locals() else None
            }
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections berdasarkan confidence threshold"""
        return [
            det for det in detections 
            if det.get('confidence', 0) >= self.confidence_threshold
        ]
    
    def _analyze_damage(self, image: np.ndarray, detections: List[Dict]) -> Dict:
        """Analisis ukuran dan area kerusakan"""
        image_height, image_width = image.shape[:2]
        total_area_pixel = 0
        damage_details = []
        
        for detection in detections:
            bbox = detection.get('bbox', {})
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            area_pixel = width * height
            area_mm = area_pixel * (self.pixel_to_mm_ratio ** 2)
            
            damage_details.append({
                'object_name': detection.get('name', 'unknown'),
                'confidence': detection.get('confidence', 0),
                'area_pixel': area_pixel,
                'area_mm': area_mm,
                'bbox': bbox
            })
            
            total_area_pixel += area_pixel
        
        total_area_mm = total_area_pixel * (self.pixel_to_mm_ratio ** 2)
        total_area_percent = (total_area_pixel / (image_width * image_height)) * 100
        
        return {
            'total_area_pixel': total_area_pixel,
            'total_area_mm': total_area_mm,
            'total_area_percent': total_area_percent,
            'damage_details': damage_details,
            'image_dimensions': (image_width, image_height)
        }
    
    def _generate_damage_description(self, analysis: Dict) -> str:
        """Generate deskripsi kerusakan yang rinci"""
        damage_details = analysis['damage_details']
        
        if not damage_details:
            return "Tidak terdeteksi kerusakan signifikan"
        
        descriptions = []
        
        for detail in damage_details:
            obj_name = self.damage_types.get(detail['object_name'], detail['object_name'])
            area_mm = detail['area_mm']
            confidence = detail['confidence']
            
            if area_mm < 100:
                size_desc = "ringan"
            elif area_mm < 1000:
                size_desc = "sedang"
            else:
                size_desc = "signifikan"
            
            desc = f"{obj_name} {size_desc} dengan area ±{area_mm:.0f} mm² (confidence: {confidence:.2f})"
            descriptions.append(desc)
        
        total_area_mm = analysis['total_area_mm']
        area_percent = analysis['total_area_percent']
        
        main_desc = f"Terdeteksi {len(damage_details)} jenis kerusakan. "
        main_desc += ". ".join(descriptions)
        main_desc += f". Total area kerusakan: {total_area_mm:.0f} mm² ({area_percent:.1f}% dari area inspeksi)."
        
        return main_desc
    
    def _classify_damage_level(self, area_percent: float) -> str:
        """Klasifikasi level kerusakan berdasarkan % area"""
        if area_percent < 5:
            return "Low"
        elif area_percent <= 15:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendations(self, level: str, analysis: Dict) -> str:
        """Generate rekomendasi tindakan"""
        recommendations = {
            'Low': f"Pantau secara berkala. Area kerusakan {analysis['total_area_percent']:.1f}% masih dalam batas aman. Inspeksi ulang dalam 6 bulan.",
            'Medium': f"Jadwalkan inspeksi lanjutan dalam 1-2 bulan. Area kerusakan {analysis['total_area_percent']:.1f}% memerlukan perhatian. Evaluasi penyebab kerusakan.",
            'High': f"Lakukan perbaikan segera! Area kerusakan {analysis['total_area_percent']:.1f}% sudah mencapai level kritis. Hentikan operasi jika diperlukan."
        }
        return recommendations.get(level, "Evaluasi lebih lanjut diperlukan")
    
    def _get_repair_procedures(self, deskripsi: str, level: str) -> str:
        """Get prosedur perbaikan detail dari Groq API"""
        try:
            prompt = f"""
            Sebagai ahli maintenance industri migas, berikan prosedur perbaikan detail untuk:
            
            Kerusakan: {deskripsi}
            Level: {level}
            
            Berikan dalam format:
            1. Langkah Persiapan
            2. Alat dan Material yang Dibutuhkan
            3. Prosedur Perbaikan Step by Step
            4. Estimasi Waktu
            5. Safety Precautions
            6. Quality Check
            
            Jawab dalam bahasa Indonesia, praktis dan actionable.
            """
            
            response = groq_service.analyze(prompt)
            if response.get('success'):
                return response.get('analysis', 'Prosedur perbaikan tidak tersedia')
            else:
                return "Gagal mendapatkan prosedur perbaikan dari AI"
                
        except Exception as e:
            logger.error(f"Error getting repair procedures: {e}")
            return f"Error: {str(e)}"
    
    def _generate_yolo_image(self, image: np.ndarray, detections: List[Dict], output_path: Path) -> str:
        """Generate gambar dengan bounding boxes YOLO"""
        output_path.parent.mkdir(exist_ok=True)
        
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', {})
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Add label
            label = f"{detection.get('name', 'unknown')} ({detection.get('confidence', 0):.2f})"
            cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), result_image)
        return str(output_path)
    
    def _generate_annotated_image(
        self, image: np.ndarray, analysis: Dict, deskripsi: str, 
        level: str, output_path: Path
    ) -> str:
        """Generate gambar final dengan anotasi lengkap"""
        output_path.parent.mkdir(exist_ok=True)
        
        result_image = image.copy()
        
        # Color coding berdasarkan level
        colors = {
            'Low': (0, 255, 0),      # Green
            'Medium': (0, 165, 255), # Orange  
            'High': (0, 0, 255)      # Red
        }
        color = colors.get(level, (255, 255, 255))
        
        # Draw detections dengan color coding
        for detail in analysis['damage_details']:
            bbox = detail['bbox']
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            cv2.rectangle(result_image, (x, y), (x + width, y + height), color, 3)
            
            # Label dengan ukuran
            obj_name = self.damage_types.get(detail['object_name'], detail['object_name'])
            label = f"{obj_name}: {detail['area_mm']:.0f}mm²"
            cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add summary information
        summary_text = [
            f"Level: {level}",
            f"Total Area: {analysis['total_area_mm']:.0f} mm²",
            f"Percentage: {analysis['total_area_percent']:.1f}%"
        ]
        
        y_offset = 30
        for text in summary_text:
            cv2.putText(result_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        cv2.imwrite(str(output_path), result_image)
        return str(output_path)
    
    def _save_to_database(self, data: Dict) -> int:
        """Simpan hasil analisis ke database"""
        query = """
        INSERT INTO pipeline_inspections (
            nama_pipa, lokasi_pipa, inspector_name, yolo_detections,
            deskripsi_kerusakan, ukuran_kerusakan_pixel, ukuran_kerusakan_mm,
            area_kerusakan_percent, level_kerusakan, folder_output,
            foto_mentah_path, foto_yolo_path, foto_fix_path,
            rekomendasi_tindakan, prosedur_perbaikan, status_inspeksi
        ) VALUES (
            :nama_pipa, :lokasi_pipa, :inspector_name, :yolo_detections,
            :deskripsi_kerusakan, :ukuran_kerusakan_pixel, :ukuran_kerusakan_mm,
            :area_kerusakan_percent, :level_kerusakan, :folder_output,
            :foto_mentah_path, :foto_yolo_path, :foto_fix_path,
            :rekomendasi_tindakan, :prosedur_perbaikan, :status_inspeksi
        )
        """
        
        params = {
            'nama_pipa': data['nama_pipa'],
            'lokasi_pipa': data['lokasi_pipa'],
            'inspector_name': data['inspector_name'],
            'yolo_detections': data['yolo_detections'],
            'deskripsi_kerusakan': data['deskripsi_kerusakan'],
            'ukuran_kerusakan_pixel': data['ukuran_kerusakan_pixel'],
            'ukuran_kerusakan_mm': data['ukuran_kerusakan_mm'],
            'area_kerusakan_percent': data['area_kerusakan_percent'],
            'level_kerusakan': data['level_kerusakan'],
            'folder_output': data['folder_output'],
            'foto_mentah_path': data['foto_mentah_path'],
            'foto_yolo_path': data['foto_yolo_path'],
            'foto_fix_path': data['foto_fix_path'],
            'rekomendasi_tindakan': data['rekomendasi_tindakan'],
            'prosedur_perbaikan': data['prosedur_perbaikan'],
            'status_inspeksi': 'analyzed'
        }
        
        return db_service.execute_insert(query, params)
    
    def _generate_report(
        self, inspection_id: str, nama_pipa: str, analysis: Dict,
        deskripsi: str, level: str, rekomendasi: str, prosedur: str,
        output_folder: Path
    ) -> str:
        """Generate laporan lengkap"""
        report_path = output_folder / "reports" / f"{nama_pipa}_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        report_content = f"""
LAPORAN INSPEKSI PIPA - {nama_pipa}
=====================================

ID Inspeksi: {inspection_id}
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nama Pipa: {nama_pipa}

HASIL ANALISIS
--------------
{deskripsi}

TINGKAT KERUSAKAN: {level}
Area Total: {analysis['total_area_mm']:.0f} mm² ({analysis['total_area_percent']:.1f}%)

REKOMENDASI TINDAKAN
-------------------
{rekomendasi}

PROSEDUR PERBAIKAN DETAIL
------------------------
{prosedur}

DETAIL DETEKSI
--------------
"""
        
        for i, detail in enumerate(analysis['damage_details'], 1):
            obj_name = self.damage_types.get(detail['object_name'], detail['object_name'])
            report_content += f"""
{i}. {obj_name}
   - Confidence: {detail['confidence']:.2f}
   - Area: {detail['area_mm']:.0f} mm²
   - Lokasi: x={detail['bbox']['x']}, y={detail['bbox']['y']}
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def get_inspection_results(self, limit: int = 50, order_by_severity: bool = True) -> List[Dict]:
        """Ambil hasil inspeksi dari database dengan urutan severity"""
        order_clause = "FIELD(level_kerusakan, 'High', 'Medium', 'Low'), tanggal_inspeksi DESC" if order_by_severity else "tanggal_inspeksi DESC"
        
        # Use simple direct query without parameters to avoid SQLAlchemy parameter issues
        query = f"""
        SELECT * FROM pipeline_inspections 
        WHERE is_active = 1 
        ORDER BY {order_clause}
        LIMIT {int(limit)}
        """
        
        results = db_service.fetch_all(query)
        
        # Convert to list of dictionaries
        columns = [
            'id', 'nama_pipa', 'lokasi_pipa', 'tanggal_inspeksi', 'inspector_name',
            'yolo_detections', 'confidence_threshold', 'deskripsi_kerusakan',
            'ukuran_kerusakan_pixel', 'ukuran_kerusakan_mm', 'area_kerusakan_percent',
            'level_kerusakan', 'risk_score', 'folder_output', 'foto_mentah_path',
            'foto_yolo_path', 'foto_fix_path', 'rekomendasi_tindakan',
            'prosedur_perbaikan', 'estimasi_waktu_perbaikan', 'alat_dibutuhkan',
            'status_inspeksi', 'prioritas', 'created_at', 'updated_at', 'is_active'
        ]
        
        return [dict(zip(columns, row)) for row in results]


# Global instance
pipeline_service = PipelineAnalysisService()
