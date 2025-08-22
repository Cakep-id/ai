-- CAKEP.id EWS Database Schema - Optimized untuk Training Mandiri
-- Schema yang dioptimalkan untuk self-learning tanpa dataset eksternal

-- Database creation
CREATE DATABASE IF NOT EXISTS cakep_ews 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE cakep_ews;

-- 1. Users table (admin dan user)
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    full_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 2. Asset Categories (untuk klasifikasi aset)
CREATE TABLE asset_categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    maintenance_frequency_days INT DEFAULT 30,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Assets table
CREATE TABLE assets (
    asset_id INT PRIMARY KEY AUTO_INCREMENT,
    asset_name VARCHAR(100) NOT NULL,
    category_id INT NOT NULL,
    location VARCHAR(200),
    description TEXT,
    installation_date DATE,
    last_maintenance DATE,
    next_maintenance DATE,
    criticality ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') DEFAULT 'MEDIUM',
    status ENUM('active', 'maintenance', 'inactive') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES asset_categories(category_id)
);

-- 4. Damage Types (untuk klasifikasi kerusakan)
CREATE TABLE damage_types (
    damage_type_id INT PRIMARY KEY AUTO_INCREMENT,
    damage_name VARCHAR(100) NOT NULL,
    description TEXT,
    default_risk_level ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') DEFAULT 'MEDIUM',
    repair_urgency_hours INT DEFAULT 24,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. User Reports (laporan dari user)
CREATE TABLE user_reports (
    report_id INT PRIMARY KEY AUTO_INCREMENT,
    asset_id INT NOT NULL,
    reported_by_user_id INT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    location_details VARCHAR(200),
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- AI Analysis Results
    ai_detected_damage VARCHAR(100),
    ai_risk_level ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'),
    ai_confidence FLOAT,
    ai_procedures TEXT,
    ai_analyzed_at TIMESTAMP NULL,
    
    -- Admin Validation
    admin_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
    validated_by INT NULL,
    validated_at TIMESTAMP NULL,
    
    -- Admin Corrections (jika ada edit dari admin)
    admin_corrected_damage VARCHAR(100) NULL,
    admin_corrected_risk ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') NULL,
    admin_corrected_procedures TEXT NULL,
    admin_notes TEXT NULL,
    
    -- Learning Status
    is_used_for_training BOOLEAN DEFAULT FALSE,
    training_added_at TIMESTAMP NULL,
    
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (reported_by_user_id) REFERENCES users(user_id),
    FOREIGN KEY (validated_by) REFERENCES users(user_id)
);

-- 6. Admin Training Data (gambar yang diupload admin untuk training)
CREATE TABLE admin_training_data (
    training_id INT PRIMARY KEY AUTO_INCREMENT,
    uploaded_by_admin INT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    asset_category_id INT NOT NULL,
    damage_type_id INT NOT NULL,
    damage_description TEXT NOT NULL,
    risk_level ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') NOT NULL,
    annotations JSON, -- koordinat bbox jika ada
    is_active BOOLEAN DEFAULT TRUE,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (uploaded_by_admin) REFERENCES users(user_id),
    FOREIGN KEY (asset_category_id) REFERENCES asset_categories(category_id),
    FOREIGN KEY (damage_type_id) REFERENCES damage_types(damage_type_id)
);

-- 7. AI Learning History (track pembelajaran AI)
CREATE TABLE ai_learning_history (
    learning_id INT PRIMARY KEY AUTO_INCREMENT,
    source_type ENUM('user_report', 'admin_training') NOT NULL,
    source_id INT NOT NULL, -- report_id atau training_id
    image_path VARCHAR(500) NOT NULL,
    damage_label VARCHAR(100) NOT NULL,
    risk_level ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') NOT NULL,
    confidence_score FLOAT,
    model_version VARCHAR(50),
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_validated BOOLEAN DEFAULT TRUE
);

-- 8. Maintenance Schedules (jadwal pemeliharaan otomatis)
CREATE TABLE maintenance_schedules (
    schedule_id INT PRIMARY KEY AUTO_INCREMENT,
    asset_id INT NOT NULL,
    report_id INT NULL, -- dari laporan yang memicu schedule
    maintenance_type ENUM('preventive', 'corrective', 'emergency') NOT NULL,
    scheduled_date DATE NOT NULL,
    priority ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') NOT NULL,
    estimated_duration_hours INT DEFAULT 2,
    assigned_technician VARCHAR(100),
    procedure_steps TEXT,
    status ENUM('scheduled', 'in_progress', 'completed', 'cancelled') DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    completion_notes TEXT,
    
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id)
);

-- 9. Model Performance Metrics (track performa model)
CREATE TABLE model_metrics (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    model_version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    total_predictions INT DEFAULT 0,
    correct_predictions INT DEFAULT 0,
    false_positives INT DEFAULT 0,
    false_negatives INT DEFAULT 0,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. System Configurations
CREATE TABLE system_config (
    config_id INT PRIMARY KEY AUTO_INCREMENT,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by INT,
    
    FOREIGN KEY (updated_by) REFERENCES users(user_id)
);

-- Insert default data

-- Default users
INSERT INTO users (username, email, password_hash, role, full_name) VALUES
('admin', 'admin@cakep.id', '$2b$12$example_hash', 'admin', 'Administrator'),
('user_demo', 'user@cakep.id', '$2b$12$example_hash', 'user', 'Demo User');

INSERT INTO asset_categories (category_name, description, maintenance_frequency_days) VALUES
('Pompa', 'Pompa air dan sistem hidrolik', 60),
('Pipa', 'Sistem perpipaan dan distribusi', 90),
('Tangki', 'Tangki penyimpanan fluida', 120),
('Motor', 'Motor listrik dan penggerak', 30),
('Struktur Bangunan', 'Struktur bangunan dan infrastruktur', 90),
('Sistem Ventilasi', 'AC dan sistem ventilasi', 30);

-- Default assets
INSERT INTO assets (asset_name, category_id, location, description, criticality) VALUES
('Pompa Utama', 1, 'Ruang Mesin Lantai 1', 'Pompa utama sistem air', 'HIGH'),
('Pipa Distribusi A', 2, 'Koridor Utama', 'Pipa distribusi air utama', 'MEDIUM'),
('Tangki Air Utama', 3, 'Atap Gedung', 'Tangki penyimpanan air utama', 'CRITICAL'),
('Motor Ventilasi', 4, 'Ruang Mesin', 'Motor sistem ventilasi gedung', 'MEDIUM');

INSERT INTO damage_types (damage_name, description, default_risk_level, repair_urgency_hours) VALUES
('Retak', 'Retak pada struktur atau komponen', 'HIGH', 24),
('Korosi', 'Korosi atau karat pada logam', 'MEDIUM', 72),
('Kebocoran', 'Kebocoran fluida atau gas', 'HIGH', 12),
('Keausan', 'Keausan normal komponen', 'MEDIUM', 168),
('Penyok', 'Deformasi fisik komponen', 'LOW', 168),
('Karat', 'Karat pada permukaan logam', 'MEDIUM', 72),
('Patah', 'Patahan pada komponen', 'CRITICAL', 4),
('Deformasi', 'Perubahan bentuk tidak normal', 'HIGH', 24),
('Erosi', 'Pengikisan material', 'MEDIUM', 72),
('Kontaminasi', 'Kotoran atau kontaminan', 'LOW', 168);

INSERT INTO system_config (config_key, config_value, description) VALUES
('ai_confidence_threshold', '0.75', 'Threshold minimum confidence untuk AI prediction'),
('auto_schedule_enabled', 'true', 'Enable automatic maintenance scheduling'),
('training_batch_size', '20', 'Jumlah data minimum untuk retrain model'),
('high_risk_alert_hours', '4', 'Jam maksimal response untuk high risk'),
('critical_risk_alert_hours', '1', 'Jam maksimal response untuk critical risk');

-- Create indexes untuk performance
CREATE INDEX idx_user_reports_status ON user_reports(admin_status);
CREATE INDEX idx_user_reports_ai_risk ON user_reports(ai_risk_level);
CREATE INDEX idx_user_reports_date ON user_reports(reported_at);
CREATE INDEX idx_maintenance_schedules_date ON maintenance_schedules(scheduled_date);
CREATE INDEX idx_maintenance_schedules_status ON maintenance_schedules(status);
CREATE INDEX idx_ai_learning_source ON ai_learning_history(source_type, source_id);
CREATE INDEX idx_training_data_active ON admin_training_data(is_active);
