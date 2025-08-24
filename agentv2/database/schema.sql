-- Database Schema for AI Asset Inspection System
-- No static dataset - all data from human input (trainer) and user reports

-- Use the AgentV2 database
USE agentv2_db;

-- Asset damage classes/categories
CREATE TABLE damage_classes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    class_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    severity_weight DECIMAL(3,2) DEFAULT 1.0,
    color_code VARCHAR(7) DEFAULT '#FF0000',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert basic damage classes
INSERT INTO damage_classes (class_name, description, severity_weight, color_code) VALUES
('corrosion', 'Korosi pada permukaan aset', 1.5, '#FF4444'),
('dent', 'Penyok atau deformasi fisik', 1.2, '#FFA500'),
('crack', 'Retak atau patahan', 1.8, '#FF0000'),
('coating_loss', 'Kehilangan lapisan pelindung', 1.0, '#FFFF00'),
('leak', 'Kebocoran pada sistem', 2.0, '#8B0000'),
('wear', 'Keausan material', 0.8, '#FFB347');

-- User reports/submissions
CREATE TABLE user_reports (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) UNIQUE NOT NULL,
    user_identifier VARCHAR(100), -- No auth, just identifier
    asset_description TEXT NOT NULL,
    original_image_path VARCHAR(500) NOT NULL,
    yolo_processed_image_path VARCHAR(500),
    final_analysis_image_path VARCHAR(500),
    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    validation_status ENUM('pending', 'approved', 'rejected', 'needs_review') DEFAULT 'pending',
    INDEX idx_report_id (report_id),
    INDEX idx_submission_date (submission_date),
    INDEX idx_status (processing_status, validation_status)
);

-- YOLO detection results
CREATE TABLE yolo_detections (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) NOT NULL,
    damage_class_id INT NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    bbox_x1 DECIMAL(8,4) NOT NULL,
    bbox_y1 DECIMAL(8,4) NOT NULL,
    bbox_x2 DECIMAL(8,4) NOT NULL,
    bbox_y2 DECIMAL(8,4) NOT NULL,
    area_pixels INT NOT NULL,
    area_percentage DECIMAL(5,2) NOT NULL,
    width_mm DECIMAL(8,2),
    height_mm DECIMAL(8,2),
    depth_mm DECIMAL(8,2), -- For NDT data if available
    iou_score DECIMAL(5,4),
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id) ON DELETE CASCADE,
    FOREIGN KEY (damage_class_id) REFERENCES damage_classes(id),
    INDEX idx_report_detection (report_id),
    INDEX idx_confidence (confidence_score),
    INDEX idx_damage_class (damage_class_id)
);

-- Risk analysis results
CREATE TABLE risk_analysis (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) NOT NULL,
    overall_risk_score DECIMAL(5,3) NOT NULL,
    risk_category ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    probability_score DECIMAL(5,4) NOT NULL,
    consequence_score DECIMAL(5,4) NOT NULL,
    severity_calculation JSON, -- Store calculation details
    geometry_based_severity DECIMAL(5,3),
    calibrated_confidence DECIMAL(5,4),
    uncertainty_score DECIMAL(5,4), -- MC-Dropout or ensemble uncertainty
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id) ON DELETE CASCADE,
    INDEX idx_report_risk (report_id),
    INDEX idx_risk_category (risk_category),
    INDEX idx_risk_score (overall_risk_score)
);

-- External API responses (Grok for repair procedures)
CREATE TABLE repair_procedures (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) NOT NULL,
    damage_summary TEXT NOT NULL,
    repair_procedure TEXT NOT NULL,
    estimated_cost_range VARCHAR(100),
    urgency_level ENUM('immediate', 'within_week', 'within_month', 'routine') NOT NULL,
    required_skills TEXT,
    safety_considerations TEXT,
    api_response_raw JSON, -- Store full API response
    generated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id) ON DELETE CASCADE,
    INDEX idx_report_procedure (report_id)
);

-- Maintenance scheduling
CREATE TABLE maintenance_schedule (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) NOT NULL,
    scheduled_date DATE NOT NULL,
    maintenance_type ENUM('inspection', 'repair', 'replacement', 'emergency') NOT NULL,
    priority_level INT NOT NULL DEFAULT 3, -- 1=highest, 5=lowest
    estimated_duration_hours DECIMAL(4,1),
    assigned_technician VARCHAR(200),
    status ENUM('scheduled', 'in_progress', 'completed', 'cancelled') DEFAULT 'scheduled',
    completion_notes TEXT,
    actual_duration_hours DECIMAL(4,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id) ON DELETE CASCADE,
    INDEX idx_scheduled_date (scheduled_date),
    INDEX idx_priority (priority_level),
    INDEX idx_status (status)
);

-- Admin/Validator actions
CREATE TABLE validation_actions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    report_id VARCHAR(100) NOT NULL,
    validator_identifier VARCHAR(100) NOT NULL,
    action_type ENUM('approve', 'reject', 'modify', 'request_review') NOT NULL,
    modifications JSON, -- Store any corrections made
    validator_notes TEXT,
    confidence_adjustment DECIMAL(3,2), -- Validator's confidence in the result
    action_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES user_reports(report_id) ON DELETE CASCADE,
    INDEX idx_report_validation (report_id),
    INDEX idx_validator (validator_identifier),
    INDEX idx_action_type (action_type)
);

-- Trainer input data (manual training data)
CREATE TABLE trainer_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    trainer_identifier VARCHAR(100) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    asset_description TEXT NOT NULL,
    risk_category ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    manual_annotations JSON NOT NULL, -- Bounding box annotations
    training_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_in_training BOOLEAN DEFAULT FALSE,
    training_session_id VARCHAR(100),
    INDEX idx_trainer (trainer_identifier),
    INDEX idx_used_training (used_in_training),
    INDEX idx_training_session (training_session_id)
);

-- Model training sessions
CREATE TABLE training_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    training_data_count INT NOT NULL,
    validation_data_count INT NOT NULL,
    epochs INT NOT NULL,
    learning_rate DECIMAL(8,6) NOT NULL,
    batch_size INT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status ENUM('running', 'completed', 'failed', 'stopped') DEFAULT 'running',
    final_map_50 DECIMAL(5,4), -- mAP@0.5
    final_map_95 DECIMAL(5,4), -- mAP@[.5:.95]
    model_path VARCHAR(500),
    training_logs JSON,
    trainer_identifier VARCHAR(100),
    INDEX idx_session_id (session_id),
    INDEX idx_status (status),
    INDEX idx_trainer (trainer_identifier)
);

-- Model evaluation metrics per class
CREATE TABLE model_metrics (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) NOT NULL,
    damage_class_id INT NOT NULL,
    precision_score DECIMAL(5,4) NOT NULL,
    recall_score DECIMAL(5,4) NOT NULL,
    f1_score DECIMAL(5,4) NOT NULL,
    ap_50 DECIMAL(5,4) NOT NULL, -- Average Precision @0.5
    ap_95 DECIMAL(5,4) NOT NULL, -- Average Precision @[.5:.95]
    optimal_threshold DECIMAL(5,4) NOT NULL,
    confusion_matrix JSON, -- Store confusion matrix data
    pr_curve_data JSON, -- Precision-Recall curve points
    iou_distribution JSON, -- IoU distribution data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (damage_class_id) REFERENCES damage_classes(id),
    INDEX idx_session_metrics (session_id),
    INDEX idx_class_metrics (damage_class_id)
);

-- Model calibration data
CREATE TABLE model_calibration (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) NOT NULL,
    calibration_method ENUM('temperature_scaling', 'platt_scaling', 'isotonic_regression') NOT NULL,
    calibration_parameters JSON NOT NULL,
    reliability_diagram_data JSON,
    ece_score DECIMAL(5,4), -- Expected Calibration Error
    mce_score DECIMAL(5,4), -- Maximum Calibration Error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_calibration (session_id)
);

-- System configuration and thresholds
CREATE TABLE system_config (
    id INT PRIMARY KEY AUTO_INCREMENT,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    data_type ENUM('string', 'number', 'boolean', 'json') NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(100)
);

-- Insert default configurations
INSERT INTO system_config (config_key, config_value, data_type, description) VALUES
('default_confidence_threshold', '0.5', 'number', 'Default confidence threshold for YOLO detections'),
('pixel_to_mm_ratio', '1.0', 'number', 'Conversion ratio from pixels to millimeters'),
('risk_calculation_weights', '{"geometry": 0.4, "confidence": 0.3, "consequence": 0.3}', 'json', 'Weights for risk score calculation'),
('maintenance_schedule_days', '{"LOW": 90, "MEDIUM": 30, "HIGH": 7, "CRITICAL": 1}', 'json', 'Days to schedule maintenance by risk level'),
('grok_api_endpoint', '', 'string', 'Grok API endpoint URL'),
('grok_api_key', '', 'string', 'Grok API authentication key'),
('max_file_size_mb', '10', 'number', 'Maximum upload file size in MB'),
('supported_image_formats', '["jpg", "jpeg", "png", "bmp"]', 'json', 'Supported image file formats');

-- Views for analytics and reporting

-- Summary view for dashboard
CREATE VIEW dashboard_summary AS
SELECT 
    DATE(submission_date) as report_date,
    COUNT(*) as total_reports,
    SUM(CASE WHEN validation_status = 'approved' THEN 1 ELSE 0 END) as approved_reports,
    SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed_reports,
    AVG(ra.overall_risk_score) as avg_risk_score
FROM user_reports ur
LEFT JOIN risk_analysis ra ON ur.report_id = ra.report_id
GROUP BY DATE(submission_date);

-- Model performance view
CREATE VIEW model_performance AS
SELECT 
    ts.session_id,
    ts.model_version,
    ts.completed_at,
    ts.final_map_50,
    ts.final_map_95,
    AVG(mm.f1_score) as avg_f1_score,
    AVG(mm.precision_score) as avg_precision,
    AVG(mm.recall_score) as avg_recall
FROM training_sessions ts
LEFT JOIN model_metrics mm ON ts.session_id = mm.session_id
WHERE ts.status = 'completed'
GROUP BY ts.session_id, ts.model_version, ts.completed_at, ts.final_map_50, ts.final_map_95;

-- Risk distribution view
CREATE VIEW risk_distribution AS
SELECT 
    risk_category,
    COUNT(*) as count,
    AVG(overall_risk_score) as avg_score,
    MIN(overall_risk_score) as min_score,
    MAX(overall_risk_score) as max_score
FROM risk_analysis
GROUP BY risk_category;

-- Damage detection frequency
CREATE VIEW damage_frequency AS
SELECT 
    dc.class_name,
    dc.description,
    COUNT(yd.id) as detection_count,
    AVG(yd.confidence_score) as avg_confidence,
    AVG(yd.area_percentage) as avg_area_percentage
FROM damage_classes dc
LEFT JOIN yolo_detections yd ON dc.id = yd.damage_class_id
GROUP BY dc.id, dc.class_name, dc.description;

-- Maintenance scheduling efficiency
CREATE VIEW maintenance_efficiency AS
SELECT 
    maintenance_type,
    priority_level,
    COUNT(*) as scheduled_count,
    AVG(estimated_duration_hours) as avg_estimated_duration,
    AVG(actual_duration_hours) as avg_actual_duration,
    AVG(DATEDIFF(updated_at, created_at)) as avg_completion_days
FROM maintenance_schedule
WHERE status = 'completed'
GROUP BY maintenance_type, priority_level;

-- Triggers for automatic maintenance scheduling
DELIMITER //
CREATE TRIGGER auto_schedule_maintenance
AFTER INSERT ON risk_analysis
FOR EACH ROW
BEGIN
    DECLARE schedule_days INT;
    DECLARE maintenance_type VARCHAR(20);
    DECLARE priority INT;
    
    -- Determine scheduling based on risk category
    CASE NEW.risk_category
        WHEN 'CRITICAL' THEN 
            SET schedule_days = 1, maintenance_type = 'emergency', priority = 1;
        WHEN 'HIGH' THEN 
            SET schedule_days = 7, maintenance_type = 'repair', priority = 2;
        WHEN 'MEDIUM' THEN 
            SET schedule_days = 30, maintenance_type = 'inspection', priority = 3;
        WHEN 'LOW' THEN 
            SET schedule_days = 90, maintenance_type = 'inspection', priority = 4;
    END CASE;
    
    -- Insert maintenance schedule
    INSERT INTO maintenance_schedule 
    (report_id, scheduled_date, maintenance_type, priority_level, estimated_duration_hours)
    VALUES 
    (NEW.report_id, DATE_ADD(CURDATE(), INTERVAL schedule_days DAY), 
     maintenance_type, priority, 
     CASE NEW.risk_category 
         WHEN 'CRITICAL' THEN 8.0 
         WHEN 'HIGH' THEN 4.0 
         WHEN 'MEDIUM' THEN 2.0 
         ELSE 1.0 
     END);
END//
DELIMITER ;

-- Indexes for performance optimization
CREATE INDEX idx_user_reports_composite ON user_reports(submission_date, processing_status, validation_status);
CREATE INDEX idx_yolo_detections_composite ON yolo_detections(report_id, damage_class_id, confidence_score);
CREATE INDEX idx_risk_analysis_composite ON risk_analysis(risk_category, overall_risk_score, analysis_timestamp);
CREATE INDEX idx_maintenance_composite ON maintenance_schedule(scheduled_date, priority_level, status);
CREATE INDEX idx_training_composite ON training_sessions(status, completed_at, final_map_50);

-- Full-text search indexes for descriptions
CREATE FULLTEXT INDEX idx_asset_description ON user_reports(asset_description);
CREATE FULLTEXT INDEX idx_training_notes ON trainer_data(training_notes, asset_description);
