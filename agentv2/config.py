# AgentV2 Configuration File
import os
from pathlib import Path
from env_loader import env

# Base Configuration
BASE_DIR = Path(__file__).parent
PROJECT_NAME = "AgentV2 - Advanced AI Asset Inspection System"
VERSION = "2.0.0"
DEBUG = env.get_bool("DEBUG", False)

# Database Configuration
DATABASE_CONFIG = {
    "host": env.get("DB_HOST", "localhost"),
    "port": env.get_int("DB_PORT", 3306),
    "user": env.get("DB_USER", "root"),
    "password": env.get("DB_PASSWORD", ""),
    "database": env.get("DB_NAME", "cakep_ews_v2"),
    "charset": env.get("DB_CHARSET", "utf8mb4"),
    "autocommit": True,
    "pool_size": env.get_int("DB_POOL_SIZE", 10),
    "max_overflow": env.get_int("DB_MAX_OVERFLOW", 20),
    "pool_timeout": env.get_int("DB_POOL_TIMEOUT", 30),
    "pool_recycle": env.get_int("DB_POOL_RECYCLE", 3600)
}

# API Configuration
API_CONFIG = {
    "host": env.get("API_HOST", "0.0.0.0"),
    "port": env.get_int("API_PORT", 8000),
    "reload": DEBUG,
    "workers": env.get_int("API_WORKERS", 4),
    "max_request_size": env.get_int("MAX_REQUEST_SIZE", 100 * 1024 * 1024),  # 100MB
    "cors_origins": env.get_list("CORS_ORIGINS", default=["*"]),
    "rate_limit": {
        "requests_per_minute": env.get_int("RATE_LIMIT_RPM", 60),
        "burst_size": env.get_int("RATE_LIMIT_BURST", 10)
    }
}

# File Storage Configuration
STORAGE_CONFIG = {
    "base_upload_dir": BASE_DIR / "uploads",
    "user_reports_dir": BASE_DIR / "uploads" / "user_reports",
    "training_images_dir": BASE_DIR / "uploads" / "training" / "images",
    "training_annotations_dir": BASE_DIR / "uploads" / "training" / "annotations",
    "models_dir": BASE_DIR / "models",
    "temp_dir": BASE_DIR / "temp",
    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", 50)),
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
    "image_quality": int(os.getenv("IMAGE_QUALITY", 95)),
    "thumbnail_size": (150, 150),
    "cleanup_temp_hours": int(os.getenv("CLEANUP_TEMP_HOURS", 24))
}

# YOLO Model Configuration
YOLO_CONFIG = {
    "model_path": BASE_DIR / "models" / "yolov8n.pt",
    "custom_model_path": BASE_DIR / "models" / "custom_yolo.pt",
    "confidence_threshold": float(os.getenv("YOLO_CONFIDENCE", 0.25)),
    "iou_threshold": float(os.getenv("YOLO_IOU", 0.45)),
    "max_detections": int(os.getenv("YOLO_MAX_DETECTIONS", 1000)),
    "device": os.getenv("YOLO_DEVICE", "cpu"),  # cpu, cuda, mps
    "half_precision": os.getenv("YOLO_HALF", "false").lower() == "true",
    "augment_inference": os.getenv("YOLO_AUGMENT", "false").lower() == "true",
    "classes": [
        "crack", "corrosion", "deformation", "hole", 
        "paint_loss", "rust", "scratch", "wear"
    ],
    "class_colors": {
        "crack": "#FF0000",      # Red
        "corrosion": "#FF8C00",  # Dark Orange
        "deformation": "#8B00FF", # Purple
        "hole": "#000000",        # Black
        "paint_loss": "#FFD700",  # Gold
        "rust": "#8B4513",        # Saddle Brown
        "scratch": "#00CED1",     # Dark Turquoise
        "wear": "#808080"         # Gray
    },
    "evaluation": {
        "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        "confidence_bins": 15,
        "monte_carlo_samples": 30,
        "temperature_scaling_lr": 0.01,
        "temperature_scaling_max_iter": 50
    }
}

# Risk Assessment Configuration
RISK_CONFIG = {
    "pixel_to_mm_ratio": float(os.getenv("PIXEL_MM_RATIO", 1.0)),
    "stress_concentration_factors": {
        "crack": 3.0,
        "hole": 2.5,
        "corrosion": 2.0,
        "deformation": 2.8,
        "paint_loss": 1.2,
        "rust": 1.8,
        "scratch": 1.5,
        "wear": 1.3
    },
    "severity_thresholds": {
        "area_mm2": {
            "low": 100,
            "medium": 500,
            "high": 1000,
            "critical": 2000
        },
        "aspect_ratio": {
            "low": 1.5,
            "medium": 3.0,
            "high": 5.0,
            "critical": 10.0
        },
        "proximity_mm": {
            "low": 50,
            "medium": 25,
            "high": 10,
            "critical": 5
        }
    },
    "risk_weights": {
        "damage_type": 0.3,
        "size": 0.25,
        "geometry": 0.2,
        "clustering": 0.15,
        "location": 0.1
    },
    "maintenance_priorities": {
        "critical": 1,  # Immediate (1-3 days)
        "high": 2,      # Urgent (1-2 weeks)
        "medium": 3,    # Moderate (1-3 months)
        "low": 4        # Routine (3-12 months)
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "default_epochs": int(os.getenv("TRAINING_EPOCHS", 100)),
    "default_batch_size": int(os.getenv("TRAINING_BATCH_SIZE", 16)),
    "default_image_size": int(os.getenv("TRAINING_IMAGE_SIZE", 640)),
    "default_learning_rate": float(os.getenv("TRAINING_LR", 0.01)),
    "patience": int(os.getenv("TRAINING_PATIENCE", 50)),
    "min_delta": float(os.getenv("TRAINING_MIN_DELTA", 0.001)),
    "save_period": int(os.getenv("TRAINING_SAVE_PERIOD", 10)),
    "val_split": float(os.getenv("TRAINING_VAL_SPLIT", 0.2)),
    "augmentation": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0
    },
    "optimizer": {
        "name": "SGD",
        "lr": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005
    },
    "scheduler": {
        "name": "linear",
        "lrf": 0.01
    },
    "monitoring": {
        "log_interval": 10,
        "eval_interval": 5,
        "save_best_only": True,
        "metric": "mAP50"
    }
}

# Background Tasks Configuration
TASK_CONFIG = {
    "max_workers": int(os.getenv("TASK_MAX_WORKERS", 4)),
    "task_timeout": int(os.getenv("TASK_TIMEOUT", 3600)),  # 1 hour
    "retry_attempts": int(os.getenv("TASK_RETRY_ATTEMPTS", 3)),
    "retry_delay": int(os.getenv("TASK_RETRY_DELAY", 60)),  # 1 minute
    "cleanup_interval": int(os.getenv("TASK_CLEANUP_INTERVAL", 300)),  # 5 minutes
    "max_queue_size": int(os.getenv("TASK_MAX_QUEUE_SIZE", 100))
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": BASE_DIR / "logs" / "agentv2.log",
    "max_file_size": int(os.getenv("LOG_MAX_FILE_SIZE", 10 * 1024 * 1024)),  # 10MB
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", 5)),
    "console_output": os.getenv("LOG_CONSOLE", "true").lower() == "true"
}

# Security Configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "agentv2-super-secret-key-change-in-production"),
    "algorithm": "HS256",
    "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE", 30)),
    "password_min_length": int(os.getenv("PASSWORD_MIN_LENGTH", 8)),
    "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", 5)),
    "lockout_duration": int(os.getenv("LOCKOUT_DURATION", 900)),  # 15 minutes
    "session_timeout": int(os.getenv("SESSION_TIMEOUT", 3600)),  # 1 hour
    "csrf_protection": os.getenv("CSRF_PROTECTION", "true").lower() == "true"
}

# Cache Configuration
CACHE_CONFIG = {
    "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
    "backend": os.getenv("CACHE_BACKEND", "memory"),  # memory, redis
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "default_timeout": int(os.getenv("CACHE_DEFAULT_TIMEOUT", 300)),  # 5 minutes
    "max_entries": int(os.getenv("CACHE_MAX_ENTRIES", 1000)),
    "key_prefix": "agentv2:"
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "enabled": os.getenv("MONITORING_ENABLED", "true").lower() == "true",
    "metrics_endpoint": "/metrics",
    "health_endpoint": "/health",
    "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true",
    "performance_tracking": os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true",
    "error_tracking": os.getenv("ERROR_TRACKING", "true").lower() == "true"
}

# Email Configuration (for notifications)
EMAIL_CONFIG = {
    "enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true",
    "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
    "smtp_port": int(os.getenv("SMTP_PORT", 587)),
    "smtp_username": os.getenv("SMTP_USERNAME", ""),
    "smtp_password": os.getenv("SMTP_PASSWORD", ""),
    "use_tls": os.getenv("SMTP_TLS", "true").lower() == "true",
    "from_email": os.getenv("FROM_EMAIL", "noreply@agentv2.local"),
    "admin_emails": os.getenv("ADMIN_EMAILS", "").split(",") if os.getenv("ADMIN_EMAILS") else []
}

# System Configuration
SYSTEM_CONFIG = {
    "maintenance_mode": os.getenv("MAINTENANCE_MODE", "false").lower() == "true",
    "maintenance_message": os.getenv("MAINTENANCE_MESSAGE", "System is under maintenance. Please try again later."),
    "timezone": os.getenv("TIMEZONE", "UTC"),
    "language": os.getenv("LANGUAGE", "en"),
    "backup_enabled": os.getenv("BACKUP_ENABLED", "true").lower() == "true",
    "backup_interval_hours": int(os.getenv("BACKUP_INTERVAL_HOURS", 24)),
    "backup_retention_days": int(os.getenv("BACKUP_RETENTION_DAYS", 30))
}

# Development Configuration
DEV_CONFIG = {
    "auto_reload": DEBUG,
    "profiling_enabled": os.getenv("PROFILING_ENABLED", "false").lower() == "true",
    "mock_data": os.getenv("MOCK_DATA", "false").lower() == "true",
    "debug_sql": os.getenv("DEBUG_SQL", "false").lower() == "true",
    "test_mode": os.getenv("TEST_MODE", "false").lower() == "true"
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        STORAGE_CONFIG["base_upload_dir"],
        STORAGE_CONFIG["user_reports_dir"],
        STORAGE_CONFIG["training_images_dir"],
        STORAGE_CONFIG["training_annotations_dir"],
        STORAGE_CONFIG["models_dir"],
        STORAGE_CONFIG["temp_dir"],
        LOGGING_CONFIG["file_path"].parent,
        BASE_DIR / "backups"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print(f"{PROJECT_NAME} v{VERSION} - Configuration loaded successfully")
    print(f"Debug mode: {DEBUG}")
    print(f"Database: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
    print(f"API: {API_CONFIG['host']}:{API_CONFIG['port']}")
