-- Pipeline Inspection Database Schema untuk Industri Migas
-- Schema untuk analisis kerusakan pipa dengan AI

USE cakep_ews;

-- Tabel utama untuk inspeksi pipa
CREATE TABLE pipeline_inspections (
    id INT PRIMARY KEY AUTO_INCREMENT,
    nama_pipa VARCHAR(255) NOT NULL,
    lokasi_pipa VARCHAR(500),
    tanggal_inspeksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inspector_name VARCHAR(100),
    
    -- Hasil YOLO Detection
    yolo_detections JSON, -- Raw YOLO results
    confidence_threshold FLOAT DEFAULT 0.3,
    
    -- Analisis AI Kerusakan
    deskripsi_kerusakan TEXT NOT NULL,
    ukuran_kerusakan_pixel INT,
    ukuran_kerusakan_mm FLOAT,
    area_kerusakan_percent FLOAT,
    level_kerusakan ENUM('Low', 'Medium', 'High') NOT NULL,
    risk_score FLOAT,
    
    -- File Management
    folder_output VARCHAR(500),
    foto_mentah_path VARCHAR(500),
    foto_yolo_path VARCHAR(500),
    foto_fix_path VARCHAR(500),
    
    -- Rekomendasi dan Prosedur
    rekomendasi_tindakan TEXT,
    prosedur_perbaikan TEXT, -- Dari Groq API
    estimasi_waktu_perbaikan INT, -- dalam jam
    alat_dibutuhkan TEXT,
    
    -- Status dan Prioritas
    status_inspeksi ENUM('pending', 'analyzed', 'scheduled', 'repaired') DEFAULT 'pending',
    prioritas INT DEFAULT 0, -- untuk sorting
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Tabel detail deteksi per objek (jika ada multiple detections per inspeksi)
CREATE TABLE detection_details (
    detail_id INT PRIMARY KEY AUTO_INCREMENT,
    inspection_id INT NOT NULL,
    object_name VARCHAR(100) NOT NULL, -- Label dari YOLO
    confidence_score FLOAT NOT NULL,
    bbox_x INT,
    bbox_y INT,
    bbox_width INT,
    bbox_height INT,
    mask_area INT,
    damage_type VARCHAR(100),
    severity_level ENUM('Low', 'Medium', 'High'),
    
    FOREIGN KEY (inspection_id) REFERENCES pipeline_inspections(id) ON DELETE CASCADE
);

-- Index untuk performance
CREATE INDEX idx_pipeline_level ON pipeline_inspections(level_kerusakan);
CREATE INDEX idx_pipeline_date ON pipeline_inspections(tanggal_inspeksi);
CREATE INDEX idx_pipeline_status ON pipeline_inspections(status_inspeksi);
CREATE INDEX idx_pipeline_prioritas ON pipeline_inspections(prioritas DESC);

-- Insert sample data
INSERT INTO pipeline_inspections (
    nama_pipa, lokasi_pipa, deskripsi_kerusakan, 
    level_kerusakan, rekomendasi_tindakan
) VALUES 
('Pipeline-001', 'Sector A - Main Distribution', 'Sample inspection data', 'Medium', 'Schedule detailed inspection'),
('Pipeline-002', 'Sector B - Secondary Line', 'Sample inspection data', 'Low', 'Monitor regularly');
