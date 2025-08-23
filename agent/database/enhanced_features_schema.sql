-- Database schema for enhanced features
-- Human Feedback, Maintenance Schedule, and Pipeline Inspections

-- Create pipeline inspections table
CREATE TABLE IF NOT EXISTS pipeline_inspections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    inspection_id VARCHAR(255) UNIQUE NOT NULL,
    pipeline_id VARCHAR(255) NOT NULL,
    location TEXT NOT NULL,
    inspector_name VARCHAR(255) NOT NULL,
    image_path VARCHAR(500),
    risk_level ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    risk_score DECIMAL(5,3) NOT NULL,
    confidence_score DECIMAL(5,3) DEFAULT 0.8,
    yolo_detections JSON,
    nlp_analysis JSON,
    inspection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_inspection_date (inspection_date),
    INDEX idx_risk_level (risk_level)
);

-- Create human feedback table for AI learning
CREATE TABLE IF NOT EXISTS human_feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    inspection_id VARCHAR(255) NOT NULL,
    human_assessment ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    feedback_notes TEXT,
    feedback_date TIMESTAMP NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES pipeline_inspections(inspection_id) ON DELETE CASCADE,
    INDEX idx_inspection_id (inspection_id),
    INDEX idx_processed (processed),
    INDEX idx_feedback_date (feedback_date)
);

-- Create maintenance schedule table
CREATE TABLE IF NOT EXISTS maintenance_schedule (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pipeline_id VARCHAR(255) NOT NULL,
    maintenance_date TIMESTAMP NOT NULL,
    maintenance_type ENUM('routine', 'repair', 'replacement', 'emergency') NOT NULL,
    priority ENUM('routine', 'urgent', 'critical') NOT NULL,
    description TEXT,
    assigned_to VARCHAR(255),
    status ENUM('scheduled', 'in_progress', 'completed', 'cancelled') DEFAULT 'scheduled',
    auto_generated BOOLEAN DEFAULT FALSE,
    inspection_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_maintenance_date (maintenance_date),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    FOREIGN KEY (inspection_id) REFERENCES pipeline_inspections(inspection_id) ON DELETE SET NULL
);

-- Create AI learning progress tracking table
CREATE TABLE IF NOT EXISTS ai_learning_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    learning_session_id VARCHAR(255) UNIQUE NOT NULL,
    feedback_count INT DEFAULT 0,
    accuracy_before DECIMAL(5,3),
    accuracy_after DECIMAL(5,3),
    improvement_percentage DECIMAL(5,2),
    learning_started_at TIMESTAMP NOT NULL,
    learning_completed_at TIMESTAMP,
    status ENUM('started', 'in_progress', 'completed', 'failed') DEFAULT 'started',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback aggregation view for analytics
CREATE OR REPLACE VIEW feedback_analytics AS
SELECT 
    DATE(hf.feedback_date) as feedback_date,
    hf.human_assessment,
    pi.risk_level as ai_assessment,
    COUNT(*) as count,
    CASE 
        WHEN hf.human_assessment = pi.risk_level THEN 'match' 
        ELSE 'mismatch' 
    END as assessment_match
FROM human_feedback hf
JOIN pipeline_inspections pi ON hf.inspection_id = pi.inspection_id
GROUP BY DATE(hf.feedback_date), hf.human_assessment, pi.risk_level;

-- Create maintenance analytics view
CREATE OR REPLACE VIEW maintenance_analytics AS
SELECT 
    DATE(maintenance_date) as scheduled_date,
    maintenance_type,
    priority,
    status,
    COUNT(*) as task_count,
    AVG(CASE WHEN status = 'completed' THEN 
        TIMESTAMPDIFF(DAY, created_at, updated_at) 
        ELSE NULL 
    END) as avg_completion_days
FROM maintenance_schedule
GROUP BY DATE(maintenance_date), maintenance_type, priority, status;

-- Insert default damage classes if not exists
INSERT IGNORE INTO yolo_damage_classes (id, class_name, description, severity_level) VALUES
(1, 'Crack', 'Retakan pada pipeline', 'medium'),
(2, 'Corrosion', 'Korosi pada permukaan pipeline', 'high'),
(3, 'Dent', 'Penyok atau deformasi fisik', 'medium'),
(4, 'Leak', 'Kebocoran pada pipeline', 'high'),
(5, 'Coating Damage', 'Kerusakan lapisan pelindung', 'low');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pipeline_inspections_composite ON pipeline_inspections(pipeline_id, inspection_date, risk_level);
CREATE INDEX IF NOT EXISTS idx_human_feedback_composite ON human_feedback(feedback_date, human_assessment, processed);
CREATE INDEX IF NOT EXISTS idx_maintenance_schedule_composite ON maintenance_schedule(maintenance_date, priority, status);

-- Create trigger to update processed flag when feedback is used for learning
DELIMITER //
CREATE TRIGGER IF NOT EXISTS update_feedback_processed
AFTER INSERT ON ai_learning_progress
FOR EACH ROW
BEGIN
    UPDATE human_feedback 
    SET processed = TRUE 
    WHERE created_at <= NEW.learning_started_at AND processed = FALSE;
END//
DELIMITER ;

-- Create procedure for automatic maintenance scheduling
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS AutoScheduleMaintenance(
    IN p_inspection_id VARCHAR(255),
    IN p_risk_level ENUM('LOW', 'MEDIUM', 'HIGH')
)
BEGIN
    DECLARE v_pipeline_id VARCHAR(255);
    DECLARE v_maintenance_date TIMESTAMP;
    DECLARE v_maintenance_type VARCHAR(50);
    DECLARE v_priority VARCHAR(50);
    
    -- Get pipeline info
    SELECT pipeline_id INTO v_pipeline_id 
    FROM pipeline_inspections 
    WHERE inspection_id = p_inspection_id;
    
    -- Calculate maintenance schedule based on risk level
    CASE p_risk_level
        WHEN 'HIGH' THEN
            SET v_maintenance_date = DATE_ADD(NOW(), INTERVAL 1 DAY);
            SET v_maintenance_type = 'emergency';
            SET v_priority = 'critical';
        WHEN 'MEDIUM' THEN
            SET v_maintenance_date = DATE_ADD(NOW(), INTERVAL 7 DAY);
            SET v_maintenance_type = 'repair';
            SET v_priority = 'urgent';
        WHEN 'LOW' THEN
            SET v_maintenance_date = DATE_ADD(NOW(), INTERVAL 30 DAY);
            SET v_maintenance_type = 'routine';
            SET v_priority = 'routine';
    END CASE;
    
    -- Insert maintenance task
    INSERT INTO maintenance_schedule 
    (pipeline_id, maintenance_date, maintenance_type, priority, auto_generated, inspection_id, description)
    VALUES 
    (v_pipeline_id, v_maintenance_date, v_maintenance_type, v_priority, TRUE, p_inspection_id, 
     CONCAT('Auto-scheduled based on ', p_risk_level, ' risk inspection'));
    
END//
DELIMITER ;
