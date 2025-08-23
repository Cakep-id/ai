-- Schema untuk sistem retraining AI Pipeline
-- Tambahan tabel untuk menyimpan dataset training

-- Tabel untuk menyimpan dataset training images
CREATE TABLE IF NOT EXISTS `yolo_training_datasets` (
  `dataset_id` int NOT NULL AUTO_INCREMENT,
  `dataset_name` varchar(100) NOT NULL,
  `description` text,
  `uploaded_by` varchar(100) NOT NULL,
  `upload_date` timestamp DEFAULT CURRENT_TIMESTAMP,
  `total_images` int DEFAULT 0,
  `status` enum('uploading','processing','ready','training','completed','failed') DEFAULT 'uploading',
  `dataset_version` varchar(20) DEFAULT '1.0',
  `is_active` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`dataset_id`),
  KEY `idx_status` (`status`),
  KEY `idx_upload_date` (`upload_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel untuk menyimpan individual training images dengan annotations
CREATE TABLE IF NOT EXISTS `yolo_training_images` (
  `image_id` int NOT NULL AUTO_INCREMENT,
  `dataset_id` int NOT NULL,
  `image_filename` varchar(255) NOT NULL,
  `image_path` varchar(500) NOT NULL,
  `image_width` int NOT NULL,
  `image_height` int NOT NULL,
  `damage_type` varchar(100) NOT NULL,
  `damage_severity` enum('LOW','MEDIUM','HIGH','CRITICAL') NOT NULL,
  `damage_description` text,
  `uploaded_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  `is_validated` tinyint(1) DEFAULT 0,
  `validation_notes` text,
  PRIMARY KEY (`image_id`),
  KEY `fk_dataset` (`dataset_id`),
  KEY `idx_damage_type` (`damage_type`),
  KEY `idx_severity` (`damage_severity`),
  CONSTRAINT `fk_dataset` FOREIGN KEY (`dataset_id`) REFERENCES `yolo_training_datasets` (`dataset_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel untuk menyimpan bounding box annotations (YOLO format)
CREATE TABLE IF NOT EXISTS `yolo_annotations` (
  `annotation_id` int NOT NULL AUTO_INCREMENT,
  `image_id` int NOT NULL,
  `class_id` int NOT NULL,
  `class_name` varchar(100) NOT NULL,
  `x_center` decimal(6,5) NOT NULL COMMENT 'Normalized x center (0-1)',
  `y_center` decimal(6,5) NOT NULL COMMENT 'Normalized y center (0-1)', 
  `width` decimal(6,5) NOT NULL COMMENT 'Normalized width (0-1)',
  `height` decimal(6,5) NOT NULL COMMENT 'Normalized height (0-1)',
  `confidence` decimal(4,3) DEFAULT 1.000,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`annotation_id`),
  KEY `fk_image` (`image_id`),
  KEY `idx_class` (`class_name`),
  CONSTRAINT `fk_image` FOREIGN KEY (`image_id`) REFERENCES `yolo_training_images` (`image_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel untuk menyimpan training sessions dan progress
CREATE TABLE IF NOT EXISTS `yolo_training_sessions` (
  `session_id` int NOT NULL AUTO_INCREMENT,
  `dataset_id` int NOT NULL,
  `session_name` varchar(100) NOT NULL,
  `started_by` varchar(100) NOT NULL,
  `start_time` timestamp DEFAULT CURRENT_TIMESTAMP,
  `end_time` timestamp NULL,
  `status` enum('preparing','training','completed','failed','cancelled') DEFAULT 'preparing',
  `epochs` int DEFAULT 100,
  `batch_size` int DEFAULT 16,
  `learning_rate` decimal(8,6) DEFAULT 0.001,
  `model_architecture` varchar(50) DEFAULT 'yolov8n',
  `training_accuracy` decimal(5,4) DEFAULT NULL,
  `validation_accuracy` decimal(5,4) DEFAULT NULL,
  `loss_value` decimal(10,6) DEFAULT NULL,
  `model_save_path` varchar(500) DEFAULT NULL,
  `training_logs` longtext,
  `error_message` text,
  PRIMARY KEY (`session_id`),
  KEY `fk_dataset_session` (`dataset_id`),
  KEY `idx_status_session` (`status`),
  CONSTRAINT `fk_dataset_session` FOREIGN KEY (`dataset_id`) REFERENCES `yolo_training_datasets` (`dataset_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel untuk menyimpan class mappings
CREATE TABLE IF NOT EXISTS `yolo_damage_classes` (
  `class_id` int NOT NULL AUTO_INCREMENT,
  `class_name` varchar(100) NOT NULL UNIQUE,
  `class_label_id` int NOT NULL UNIQUE COMMENT 'YOLO numeric class ID',
  `description` text,
  `risk_weight` decimal(3,2) DEFAULT 0.50 COMMENT 'Risk weight 0.0-1.0',
  `color_hex` varchar(7) DEFAULT '#FF0000' COMMENT 'Display color for annotations',
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`class_id`),
  KEY `idx_class_name` (`class_name`),
  KEY `idx_label_id` (`class_label_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default damage classes
INSERT INTO `yolo_damage_classes` (`class_name`, `class_label_id`, `description`, `risk_weight`, `color_hex`) VALUES
('korosi_ringan', 0, 'Korosi tingkat ringan pada permukaan', 0.30, '#FFA500'),
('korosi_sedang', 1, 'Korosi tingkat sedang', 0.60, '#FF8C00'),
('korosi_parah', 2, 'Korosi tingkat parah/kritis', 0.90, '#FF4500'),
('retak_permukaan', 3, 'Retakan pada permukaan', 0.50, '#8B0000'),
('retak_struktural', 4, 'Retakan struktural yang serius', 0.85, '#DC143C'),
('kebocoran', 5, 'Kebocoran fluid/gas', 0.80, '#0000FF'),
('keausan', 6, 'Keausan material', 0.40, '#800080'),
('deformasi', 7, 'Deformasi bentuk', 0.70, '#FF1493'),
('kontaminasi', 8, 'Kontaminasi permukaan', 0.25, '#32CD32'),
('patah', 9, 'Patahan material', 0.95, '#8B0000')
ON DUPLICATE KEY UPDATE description=VALUES(description);

-- Index untuk performa
CREATE INDEX idx_training_images_composite ON yolo_training_images(dataset_id, damage_type, damage_severity);
CREATE INDEX idx_annotations_composite ON yolo_annotations(image_id, class_name);
CREATE INDEX idx_training_sessions_composite ON yolo_training_sessions(dataset_id, status, start_time);
