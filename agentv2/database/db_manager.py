"""
Database configuration and connection management for AgentV2
No static datasets - all data from human input and user reports
"""

import mysql.connector
from mysql.connector import pooling, Error
import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from datetime import datetime, date
import decimal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', 3306))
        self.database = os.getenv('DB_NAME', 'cakep_ews_v2')
        self.username = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.pool_size = int(os.getenv('DB_POOL_SIZE', 10))
        self.pool_name = 'agentv2_pool'
        
        # Connection pool configuration
        self.pool_config = {
            'pool_name': self.pool_name,
            'pool_size': self.pool_size,
            'pool_reset_session': True,
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': False,
            'time_zone': '+00:00'
        }

class DatabaseManager:
    """Database connection and query management"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(**self.config.pool_config)
            logger.info(f"Database pool initialized: {self.config.pool_name}")
        except Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    @contextmanager
    def get_cursor(self, dictionary=True, buffered=True):
        """Get database cursor with connection"""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=dictionary, buffered=buffered)
            try:
                yield cursor, connection
            except Error as e:
                connection.rollback()
                logger.error(f"Database cursor error: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute SELECT query"""
        try:
            with self.get_cursor() as (cursor, connection):
                cursor.execute(query, params or ())
                if fetch:
                    return cursor.fetchall()
                return None
        except Error as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute INSERT query and return last insert ID"""
        try:
            with self.get_cursor() as (cursor, connection):
                cursor.execute(query, params or ())
                connection.commit()
                return cursor.lastrowid
        except Error as e:
            logger.error(f"Insert execution error: {e}")
            raise
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute UPDATE/DELETE query and return affected rows"""
        try:
            with self.get_cursor() as (cursor, connection):
                cursor.execute(query, params or ())
                connection.commit()
                return cursor.rowcount
        except Error as e:
            logger.error(f"Update execution error: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries in batch"""
        try:
            with self.get_cursor() as (cursor, connection):
                cursor.executemany(query, params_list)
                connection.commit()
                return cursor.rowcount
        except Error as e:
            logger.error(f"Batch execution error: {e}")
            raise

class DataHandler:
    """Custom data type handling for database operations"""
    
    @staticmethod
    def serialize_json(data: Any) -> str:
        """Serialize data to JSON string for database storage"""
        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, decimal.Decimal):
                return float(obj)
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        return json.dumps(data, default=json_serializer, ensure_ascii=False)
    
    @staticmethod
    def deserialize_json(json_str: str) -> Any:
        """Deserialize JSON string from database"""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON deserialization error: {e}")
            return None
    
    @staticmethod
    def format_bbox(x1: float, y1: float, x2: float, y2: float) -> Dict[str, float]:
        """Format bounding box coordinates"""
        return {
            'x1': round(float(x1), 4),
            'y1': round(float(y1), 4),
            'x2': round(float(x2), 4),
            'y2': round(float(y2), 4),
            'width': round(float(x2 - x1), 4),
            'height': round(float(y2 - y1), 4)
        }
    
    @staticmethod
    def calculate_area(bbox: Dict[str, float]) -> int:
        """Calculate bounding box area in pixels"""
        return int(bbox['width'] * bbox['height'])
    
    @staticmethod
    def calculate_area_percentage(bbox_area: int, image_width: int, image_height: int) -> float:
        """Calculate damage area as percentage of total image"""
        total_area = image_width * image_height
        return round((bbox_area / total_area) * 100, 2) if total_area > 0 else 0.0

# Database Access Objects (DAOs)

class UserReportDAO:
    """Data access for user reports"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_report(self, report_data: Dict[str, Any]) -> str:
        """Create new user report"""
        query = """
        INSERT INTO user_reports 
        (report_id, user_identifier, asset_description, original_image_path)
        VALUES (%s, %s, %s, %s)
        """
        params = (
            report_data['report_id'],
            report_data.get('user_identifier'),
            report_data['asset_description'],
            report_data['original_image_path']
        )
        self.db.execute_insert(query, params)
        return report_data['report_id']
    
    def update_processing_status(self, report_id: str, status: str, 
                               yolo_path: str = None, final_path: str = None) -> bool:
        """Update report processing status and image paths"""
        query = """
        UPDATE user_reports 
        SET processing_status = %s, yolo_processed_image_path = %s, 
            final_analysis_image_path = %s
        WHERE report_id = %s
        """
        params = (status, yolo_path, final_path, report_id)
        return self.db.execute_update(query, params) > 0
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report by ID"""
        query = "SELECT * FROM user_reports WHERE report_id = %s"
        result = self.db.execute_query(query, (report_id,))
        return result[0] if result else None
    
    def get_reports_by_status(self, processing_status: str = None, 
                            validation_status: str = None) -> List[Dict[str, Any]]:
        """Get reports by status"""
        conditions = []
        params = []
        
        if processing_status:
            conditions.append("processing_status = %s")
            params.append(processing_status)
        
        if validation_status:
            conditions.append("validation_status = %s")
            params.append(validation_status)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM user_reports WHERE {where_clause} ORDER BY submission_date DESC"
        
        return self.db.execute_query(query, tuple(params))

class YOLODetectionDAO:
    """Data access for YOLO detections"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_detections(self, report_id: str, detections: List[Dict[str, Any]]) -> bool:
        """Save YOLO detection results"""
        if not detections:
            return True
        
        query = """
        INSERT INTO yolo_detections 
        (report_id, damage_class_id, confidence_score, bbox_x1, bbox_y1, 
         bbox_x2, bbox_y2, area_pixels, area_percentage, width_mm, height_mm, iou_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params_list = []
        for detection in detections:
            params_list.append((
                report_id,
                detection['damage_class_id'],
                detection['confidence_score'],
                detection['bbox_x1'],
                detection['bbox_y1'],
                detection['bbox_x2'],
                detection['bbox_y2'],
                detection['area_pixels'],
                detection['area_percentage'],
                detection.get('width_mm'),
                detection.get('height_mm'),
                detection.get('iou_score')
            ))
        
        return self.db.execute_many(query, params_list) > 0
    
    def get_detections(self, report_id: str) -> List[Dict[str, Any]]:
        """Get detections for a report"""
        query = """
        SELECT yd.*, dc.class_name, dc.description, dc.severity_weight
        FROM yolo_detections yd
        JOIN damage_classes dc ON yd.damage_class_id = dc.id
        WHERE yd.report_id = %s
        ORDER BY yd.confidence_score DESC
        """
        return self.db.execute_query(query, (report_id,))

class RiskAnalysisDAO:
    """Data access for risk analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_risk_analysis(self, risk_data: Dict[str, Any]) -> int:
        """Save risk analysis results"""
        query = """
        INSERT INTO risk_analysis 
        (report_id, overall_risk_score, risk_category, probability_score, 
         consequence_score, severity_calculation, geometry_based_severity, 
         calibrated_confidence, uncertainty_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            risk_data['report_id'],
            risk_data['overall_risk_score'],
            risk_data['risk_category'],
            risk_data['probability_score'],
            risk_data['consequence_score'],
            DataHandler.serialize_json(risk_data.get('severity_calculation')),
            risk_data.get('geometry_based_severity'),
            risk_data.get('calibrated_confidence'),
            risk_data.get('uncertainty_score')
        )
        return self.db.execute_insert(query, params)
    
    def get_risk_analysis(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get risk analysis for a report"""
        query = "SELECT * FROM risk_analysis WHERE report_id = %s"
        result = self.db.execute_query(query, (report_id,))
        if result:
            analysis = result[0]
            # Deserialize JSON fields
            if analysis.get('severity_calculation'):
                analysis['severity_calculation'] = DataHandler.deserialize_json(
                    analysis['severity_calculation']
                )
            return analysis
        return None

class TrainingSessionDAO:
    """Data access for training sessions"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create new training session"""
        query = """
        INSERT INTO training_sessions 
        (session_id, model_version, training_data_count, validation_data_count,
         epochs, learning_rate, batch_size, trainer_identifier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            session_data['session_id'],
            session_data['model_version'],
            session_data['training_data_count'],
            session_data['validation_data_count'],
            session_data['epochs'],
            session_data['learning_rate'],
            session_data['batch_size'],
            session_data.get('trainer_identifier')
        )
        self.db.execute_insert(query, params)
        return session_data['session_id']
    
    def update_session_completion(self, session_id: str, 
                                final_map_50: float, final_map_95: float,
                                model_path: str, training_logs: Dict) -> bool:
        """Update session with completion data"""
        query = """
        UPDATE training_sessions 
        SET status = 'completed', completed_at = NOW(), 
            final_map_50 = %s, final_map_95 = %s, 
            model_path = %s, training_logs = %s
        WHERE session_id = %s
        """
        params = (
            final_map_50, final_map_95, model_path,
            DataHandler.serialize_json(training_logs), session_id
        )
        return self.db.execute_update(query, params) > 0

class SystemConfigDAO:
    """Data access for system configuration"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_config(self, key: str) -> Any:
        """Get configuration value"""
        query = "SELECT config_value, data_type FROM system_config WHERE config_key = %s"
        result = self.db.execute_query(query, (key,))
        if result:
            value = result[0]['config_value']
            data_type = result[0]['data_type']
            
            if data_type == 'number':
                return float(value)
            elif data_type == 'boolean':
                return value.lower() in ('true', '1', 'yes')
            elif data_type == 'json':
                return DataHandler.deserialize_json(value)
            return value
        return None
    
    def set_config(self, key: str, value: Any, data_type: str, updated_by: str = None) -> bool:
        """Set configuration value"""
        query = """
        INSERT INTO system_config (config_key, config_value, data_type, updated_by)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        config_value = VALUES(config_value),
        data_type = VALUES(data_type),
        updated_by = VALUES(updated_by),
        updated_at = CURRENT_TIMESTAMP
        """
        
        if data_type == 'json':
            value = DataHandler.serialize_json(value)
        elif data_type == 'number':
            value = str(value)
        elif data_type == 'boolean':
            value = str(value).lower()
        
        params = (key, value, data_type, updated_by)
        return self.db.execute_update(query, params) > 0

# Database factory
def create_database_manager() -> DatabaseManager:
    """Create database manager instance"""
    config = DatabaseConfig()
    return DatabaseManager(config)

# DAOs factory
def create_daos(db_manager: DatabaseManager) -> Dict[str, Any]:
    """Create all DAO instances"""
    return {
        'user_reports': UserReportDAO(db_manager),
        'yolo_detections': YOLODetectionDAO(db_manager),
        'risk_analysis': RiskAnalysisDAO(db_manager),
        'training_sessions': TrainingSessionDAO(db_manager),
        'system_config': SystemConfigDAO(db_manager)
    }

# Test connection function
def test_connection():
    """Test database connection"""
    try:
        db_manager = create_database_manager()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the database connection
    if test_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
