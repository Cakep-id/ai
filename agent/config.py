"""
Configuration management untuk CAKEP.id EWS AI Module
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings dari environment variables"""
    
    # Database Configuration
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=3306, env="DB_PORT")
    db_name: str = Field(default="cakep_ews", env="DB_NAME")
    db_user: str = Field(default="root", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")
    
    # Groq AI Configuration
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="mixtral-8x7b-32768", env="GROQ_MODEL")
    
    # YOLO Configuration
    yolo_model_path: str = Field(default="models/yolo_damage_detection.pt", env="YOLO_MODEL_PATH")
    yolo_conf_threshold: float = Field(default=0.25, env="YOLO_CONF_THRESHOLD")
    yolo_iou_threshold: float = Field(default=0.45, env="YOLO_IOU_THRESHOLD")
    yolo_device: str = Field(default="auto", env="YOLO_DEVICE")
    
    # Storage Configuration
    storage_dir: str = Field(default="./storage", env="STORAGE_DIR")
    max_file_size: str = Field(default="50MB", env="MAX_FILE_SIZE")
    allowed_image_types: str = Field(default="jpg,jpeg,png,bmp,tiff", env="ALLOWED_IMAGE_TYPES")
    
    # Risk Engine Configuration
    visual_weight: float = Field(default=0.6, env="VISUAL_WEIGHT")
    text_weight: float = Field(default=0.4, env="TEXT_WEIGHT")
    high_risk_threshold: float = Field(default=0.75, env="HIGH_RISK_THRESHOLD")
    medium_risk_threshold: float = Field(default=0.45, env="MEDIUM_RISK_THRESHOLD")
    
    # SLA Configuration
    high_risk_sla: int = Field(default=24, env="HIGH_RISK_SLA")
    medium_risk_sla: int = Field(default=72, env="MEDIUM_RISK_SLA")
    low_risk_sla: int = Field(default=168, env="LOW_RISK_SLA")
    
    # Scheduler Configuration
    scheduler_default_hours: int = Field(default=8, env="SCHEDULER_DEFAULT_HOURS")
    sla_critical_hours: int = Field(default=4, env="SLA_CRITICAL_HOURS")
    sla_high_hours: int = Field(default=24, env="SLA_HIGH_HOURS")
    sla_medium_hours: int = Field(default=72, env="SLA_MEDIUM_HOURS")
    sla_low_hours: int = Field(default=168, env="SLA_LOW_HOURS")
    
    # File Upload Configuration
    upload_max_size: str = Field(default="50MB", env="UPLOAD_MAX_SIZE")
    allowed_image_extensions: str = Field(default=".jpg,.jpeg,.png,.bmp,.tiff", env="ALLOWED_IMAGE_EXTENSIONS")
    allowed_annotation_extensions: str = Field(default=".txt,.xml,.json", env="ALLOWED_ANNOTATION_EXTENSIONS")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080", env="API_CORS_ORIGINS")
    
    # Security Configuration
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/ews_ai.log", env="LOG_FILE")
    
    # Development Configuration
    debug: bool = Field(default=True, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        protected_namespaces = ('settings_',)
    
    @property
    def database_url(self) -> str:
        """Get database connection URL"""
        return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as list"""
        return [origin.strip() for origin in self.api_cors_origins.split(",")]
    
    @property
    def allowed_image_types_list(self) -> List[str]:
        """Get allowed image types as list"""
        return [ext.strip() for ext in self.allowed_image_types.split(",")]
    
    @property
    def allowed_annotation_extensions_list(self) -> List[str]:
        """Get allowed annotation extensions as list"""
        return [ext.strip() for ext in self.allowed_annotation_extensions.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size to bytes"""
        size_str = self.max_file_size.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            # Assume bytes
            return int(size_str)
    
    @property
    def sla_hours_dict(self) -> dict:
        """Get SLA hours as dictionary"""
        return {
            'CRITICAL': 4,  # Fixed critical hours
            'HIGH': self.high_risk_sla,
            'MEDIUM': self.medium_risk_sla,
            'LOW': self.low_risk_sla
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"

# Global settings instance
settings = Settings()

# Validation functions
def validate_database_config() -> bool:
    """Validate database configuration"""
    required_fields = [
        settings.db_host,
        settings.db_name,
        settings.db_user
    ]
    return all(field for field in required_fields)

def validate_groq_config() -> bool:
    """Validate Groq API configuration"""
    return settings.groq_api_key is not None and len(settings.groq_api_key) > 0

def validate_yolo_config() -> bool:
    """Validate YOLO configuration"""
    return (
        settings.yolo_model_path and
        0.0 <= settings.yolo_conf_threshold <= 1.0
    )

def validate_risk_config() -> bool:
    """Validate risk engine configuration"""
    return (
        0.0 <= settings.visual_weight <= 1.0 and
        0.0 <= settings.text_weight <= 1.0 and
        abs(settings.visual_weight + settings.text_weight - 1.0) < 0.001
    )

def get_config_summary() -> dict:
    """Get configuration summary untuk debugging"""
    return {
        'database': {
            'host': settings.db_host,
            'port': settings.db_port,
            'name': settings.db_name,
            'user': settings.db_user,
            'password_set': bool(settings.db_password)
        },
        'groq': {
            'model': settings.groq_model,
            'api_key_set': bool(settings.groq_api_key)
        },
        'yolo': {
            'model_path': settings.yolo_model_path,
            'confidence_threshold': settings.yolo_confidence_threshold,
            'device': settings.yolo_device
        },
        'risk': {
            'visual_weight': settings.risk_visual_weight,
            'text_weight': settings.risk_text_weight
        },
        'api': {
            'host': settings.api_host,
            'port': settings.api_port,
            'debug': settings.debug,
            'environment': settings.environment
        },
        'validation': {
            'database_valid': validate_database_config(),
            'groq_valid': validate_groq_config(),
            'yolo_valid': validate_yolo_config(),
            'risk_valid': validate_risk_config()
        }
    }

# Setup logging directory
def setup_logging_directory():
    """Ensure logging directory exists"""
    log_dir = os.path.dirname(settings.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

# Initialize
setup_logging_directory()
