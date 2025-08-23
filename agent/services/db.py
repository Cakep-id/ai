"""
Database service untuk CAKEP.id EWS
Mengelola koneksi MySQL dan operasi CRUD
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'cakep_ews'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# SQLAlchemy setup
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

class DatabaseService:
    """Service untuk mengelola koneksi dan operasi database"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.Base = declarative_base()
        self.metadata = MetaData()
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Inisialisasi koneksi database"""
        try:
            self.engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=os.getenv('DEBUG', 'False').lower() == 'true'
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager untuk database session"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test koneksi database"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Health check untuk database service"""
        try:
            start_time = datetime.now()
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "database": DB_CONFIG['database'],
                "host": DB_CONFIG['host']
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": DB_CONFIG['database'],
                "host": DB_CONFIG['host']
            }
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute SELECT query dan return hasil sebagai list of dict"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def fetch_all(self, query: str, params=None) -> List[tuple]:
        """Execute SELECT query dan return semua hasil sebagai list of tuples"""
        try:
            # Use direct connection for simple SELECT queries to avoid session issues
            with self.engine.connect() as connection:
                if params:
                    # Convert list/tuple params to positional parameters for raw SQL
                    if isinstance(params, (list, tuple)):
                        # For MySQL with direct connection, use tuple parameters
                        result = connection.execute(text(query), tuple(params))
                    else:
                        # Dict parameters
                        result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))
                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Fetch all failed: {e}")
            return []
    
    def fetch_one(self, query: str, params=None) -> Optional[tuple]:
        """Execute SELECT query dan return satu hasil sebagai tuple"""
        try:
            with self.engine.connect() as connection:
                if params:
                    # Convert list/tuple params to positional parameters for raw SQL
                    if isinstance(params, (list, tuple)):
                        # For MySQL with direct connection, use tuple parameters
                        result = connection.execute(text(query), tuple(params))
                    else:
                        # Dict parameters
                        result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))
                return result.fetchone()
        except SQLAlchemyError as e:
            logger.error(f"Fetch one failed: {e}")
            return None
    
    def execute_insert(self, query: str, params: Dict = None) -> Optional[int]:
        """Execute INSERT query dan return last insert ID"""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.lastrowid
        except SQLAlchemyError as e:
            logger.error(f"Insert execution failed: {e}")
            return None
    
    def execute_update(self, query: str, params: Dict = None) -> int:
        """Execute UPDATE/DELETE query dan return affected rows"""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"Update execution failed: {e}")
            return 0
    
    def create_tables(self):
        """Buat semua tabel yang diperlukan"""
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id INT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                location VARCHAR(120),
                criticality ENUM('LOW','MEDIUM','HIGH') DEFAULT 'MEDIUM',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id INT AUTO_INCREMENT PRIMARY KEY,
                asset_id INT NOT NULL,
                user_id INT NULL,
                description TEXT,
                image_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                status ENUM('NEW','ANALYZED','SCHEDULED','CLOSED') DEFAULT 'NEW',
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INT AUTO_INCREMENT PRIMARY KEY,
                report_id INT NOT NULL,
                label VARCHAR(100) NOT NULL,
                bbox JSON NULL,
                confidence FLOAT NOT NULL,
                model_ver VARCHAR(50),
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nlp_analyses (
                nlp_id INT AUTO_INCREMENT PRIMARY KEY,
                report_id INT NOT NULL,
                category VARCHAR(100),
                keyphrases JSON NULL,
                confidence FLOAT NOT NULL,
                model_ver VARCHAR(50),
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_scores (
                risk_id INT AUTO_INCREMENT PRIMARY KEY,
                report_id INT NOT NULL,
                visual_score FLOAT NOT NULL,
                text_score FLOAT NOT NULL,
                final_score FLOAT NOT NULL,
                risk_level ENUM('LOW','MEDIUM','HIGH') NOT NULL,
                rationale TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS repair_procedures (
                procedure_id INT AUTO_INCREMENT PRIMARY KEY,
                report_id INT NOT NULL,
                title VARCHAR(120) NOT NULL,
                steps JSON NOT NULL,
                priority ENUM('LOW','MEDIUM','HIGH') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS schedules (
                schedule_id INT AUTO_INCREMENT PRIMARY KEY,
                asset_id INT NOT NULL,
                report_id INT NOT NULL,
                type ENUM('INSPECTION','TEMP_FIX','PERMANENT_FIX','PREVENTIVE') DEFAULT 'INSPECTION',
                due_date DATETIME NOT NULL,
                frequency_days INT NULL,
                status ENUM('PLANNED','IN_PROGRESS','DONE','CANCELLED') DEFAULT 'PLANNED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE,
                FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_logs (
                training_id INT AUTO_INCREMENT PRIMARY KEY,
                model_type ENUM('YOLO','NLP') NOT NULL,
                dataset_info TEXT,
                accuracy FLOAT NULL,
                loss FLOAT NULL,
                model_ver VARCHAR(50) NOT NULL,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        try:
            with self.get_session() as session:
                for ddl in ddl_statements:
                    session.execute(text(ddl))
                logger.info("All tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    # Asset operations
    def create_asset(self, code: str, name: str, location: str = None, 
                    criticality: str = 'MEDIUM') -> Optional[int]:
        """Buat asset baru"""
        query = """
        INSERT INTO assets (code, name, location, criticality)
        VALUES (:code, :name, :location, :criticality)
        """
        return self.execute_insert(query, {
            'code': code,
            'name': name,
            'location': location,
            'criticality': criticality
        })
    
    def get_asset(self, asset_id: int) -> Optional[Dict]:
        """Ambil data asset berdasarkan ID"""
        query = "SELECT * FROM assets WHERE asset_id = :asset_id"
        result = self.execute_query(query, {'asset_id': asset_id})
        return result[0] if result else None
    
    def get_assets(self, limit: int = 100) -> List[Dict]:
        """Ambil semua assets"""
        query = "SELECT * FROM assets ORDER BY created_at DESC LIMIT :limit"
        return self.execute_query(query, {'limit': limit})
    
    # Report operations
    def create_report(self, asset_id: int, description: str = None, 
                     image_path: str = None, user_id: int = None) -> Optional[int]:
        """Buat report baru"""
        query = """
        INSERT INTO reports (asset_id, user_id, description, image_path)
        VALUES (:asset_id, :user_id, :description, :image_path)
        """
        return self.execute_insert(query, {
            'asset_id': asset_id,
            'user_id': user_id,
            'description': description,
            'image_path': image_path
        })
    
    def get_report(self, report_id: int) -> Optional[Dict]:
        """Ambil data report berdasarkan ID"""
        query = """
        SELECT r.*, a.code as asset_code, a.name as asset_name 
        FROM reports r 
        JOIN assets a ON r.asset_id = a.asset_id 
        WHERE r.report_id = :report_id
        """
        result = self.execute_query(query, {'report_id': report_id})
        return result[0] if result else None
    
    def update_report_status(self, report_id: int, status: str) -> bool:
        """Update status report"""
        query = "UPDATE reports SET status = :status WHERE report_id = :report_id"
        affected = self.execute_update(query, {
            'report_id': report_id,
            'status': status
        })
        return affected > 0
    
    # Detection operations
    def save_detection(self, report_id: int, label: str, confidence: float,
                      bbox: Dict = None, model_ver: str = None) -> Optional[int]:
        """Simpan hasil deteksi YOLO"""
        query = """
        INSERT INTO detections (report_id, label, bbox, confidence, model_ver)
        VALUES (:report_id, :label, :bbox, :confidence, :model_ver)
        """
        return self.execute_insert(query, {
            'report_id': report_id,
            'label': label,
            'bbox': json.dumps(bbox) if bbox else None,
            'confidence': confidence,
            'model_ver': model_ver
        })
    
    def get_detections(self, report_id: int) -> List[Dict]:
        """Ambil semua deteksi untuk report"""
        query = "SELECT * FROM detections WHERE report_id = :report_id ORDER BY confidence DESC"
        return self.execute_query(query, {'report_id': report_id})
    
    # NLP Analysis operations
    def save_nlp_analysis(self, report_id: int, category: str, confidence: float,
                         keyphrases: List = None, model_ver: str = None) -> Optional[int]:
        """Simpan hasil analisis NLP"""
        query = """
        INSERT INTO nlp_analyses (report_id, category, keyphrases, confidence, model_ver)
        VALUES (:report_id, :category, :keyphrases, :confidence, :model_ver)
        """
        return self.execute_insert(query, {
            'report_id': report_id,
            'category': category,
            'keyphrases': json.dumps(keyphrases) if keyphrases else None,
            'confidence': confidence,
            'model_ver': model_ver
        })
    
    def get_nlp_analysis(self, report_id: int) -> Optional[Dict]:
        """Ambil analisis NLP untuk report"""
        query = "SELECT * FROM nlp_analyses WHERE report_id = :report_id ORDER BY analyzed_at DESC LIMIT 1"
        result = self.execute_query(query, {'report_id': report_id})
        return result[0] if result else None
    
    # Risk Score operations
    def save_risk_score(self, report_id: int, visual_score: float, text_score: float,
                       final_score: float, risk_level: str, rationale: str = None) -> Optional[int]:
        """Simpan risk score"""
        query = """
        INSERT INTO risk_scores (report_id, visual_score, text_score, final_score, risk_level, rationale)
        VALUES (:report_id, :visual_score, :text_score, :final_score, :risk_level, :rationale)
        """
        return self.execute_insert(query, {
            'report_id': report_id,
            'visual_score': visual_score,
            'text_score': text_score,
            'final_score': final_score,
            'risk_level': risk_level,
            'rationale': rationale
        })
    
    def get_risk_score(self, report_id: int) -> Optional[Dict]:
        """Ambil risk score untuk report"""
        query = "SELECT * FROM risk_scores WHERE report_id = :report_id ORDER BY calculated_at DESC LIMIT 1"
        result = self.execute_query(query, {'report_id': report_id})
        return result[0] if result else None
    
    # Repair Procedures operations
    def save_repair_procedure(self, report_id: int, title: str, steps: List[str], 
                            priority: str) -> Optional[int]:
        """Simpan prosedur perbaikan"""
        query = """
        INSERT INTO repair_procedures (report_id, title, steps, priority)
        VALUES (:report_id, :title, :steps, :priority)
        """
        return self.execute_insert(query, {
            'report_id': report_id,
            'title': title,
            'steps': json.dumps(steps),
            'priority': priority
        })
    
    def get_repair_procedures(self, report_id: int) -> List[Dict]:
        """Ambil prosedur perbaikan untuk report"""
        query = "SELECT * FROM repair_procedures WHERE report_id = :report_id ORDER BY created_at DESC"
        return self.execute_query(query, {'report_id': report_id})
    
    # Schedule operations
    def save_schedule(self, asset_id: int, report_id: int, schedule_type: str,
                     due_date: datetime, frequency_days: int = None) -> Optional[int]:
        """Simpan jadwal pemeliharaan"""
        query = """
        INSERT INTO schedules (asset_id, report_id, type, due_date, frequency_days)
        VALUES (:asset_id, :report_id, :type, :due_date, :frequency_days)
        """
        return self.execute_insert(query, {
            'asset_id': asset_id,
            'report_id': report_id,
            'type': schedule_type,
            'due_date': due_date,
            'frequency_days': frequency_days
        })
    
    def get_schedules(self, asset_id: int = None, status: str = None) -> List[Dict]:
        """Ambil jadwal pemeliharaan"""
        query = "SELECT * FROM schedules WHERE 1=1"
        params = {}
        
        if asset_id:
            query += " AND asset_id = :asset_id"
            params['asset_id'] = asset_id
        
        if status:
            query += " AND status = :status"
            params['status'] = status
        
        query += " ORDER BY due_date ASC"
        return self.execute_query(query, params)
    
    def update_schedule_status(self, schedule_id: int, status: str) -> bool:
        """Update status jadwal"""
        query = "UPDATE schedules SET status = :status WHERE schedule_id = :schedule_id"
        affected = self.execute_update(query, {
            'schedule_id': schedule_id,
            'status': status
        })
        return affected > 0
    
    # Training logs operations
    def save_training_log(self, model_type: str, dataset_info: str, model_ver: str,
                         accuracy: float = None, loss: float = None) -> Optional[int]:
        """Simpan log training"""
        query = """
        INSERT INTO training_logs (model_type, dataset_info, accuracy, loss, model_ver)
        VALUES (:model_type, :dataset_info, :accuracy, :loss, :model_ver)
        """
        return self.execute_insert(query, {
            'model_type': model_type,
            'dataset_info': dataset_info,
            'accuracy': accuracy,
            'loss': loss,
            'model_ver': model_ver
        })
    
    def get_training_logs(self, model_type: str = None, limit: int = 50) -> List[Dict]:
        """Ambil log training"""
        query = "SELECT * FROM training_logs"
        params = {'limit': limit}
        
        if model_type:
            query += " WHERE model_type = :model_type"
            params['model_type'] = model_type
        
        query += " ORDER BY trained_at DESC LIMIT :limit"
        return self.execute_query(query, params)

# Singleton instance
db_service = DatabaseService()

# Helper functions untuk init
def init_database():
    """Inisialisasi database dan tabel"""
    try:
        db_service.create_tables()
        logger.info("Database initialized successfully")
        
        # Insert sample data jika belum ada
        assets = db_service.get_assets(limit=1)
        if not assets:
            # Sample assets
            sample_assets = [
                {
                    'code': 'PMP-001',
                    'name': 'Pompa Air Utama #1',
                    'location': 'Plant A - Ruang Pompa',
                    'criticality': 'HIGH'
                },
                {
                    'code': 'VLV-001',
                    'name': 'Valve Control #1',
                    'location': 'Plant A - Control Room',
                    'criticality': 'MEDIUM'
                },
                {
                    'code': 'MTR-001',
                    'name': 'Motor Listrik #1',
                    'location': 'Plant B - Workshop',
                    'criticality': 'MEDIUM'
                }
            ]
            
            for asset in sample_assets:
                db_service.create_asset(**asset)
            
            logger.info("Sample assets created")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    # Test database connection and initialization
    if db_service.test_connection():
        print("Database connection successful")
        init_database()
        print("Database initialization completed")
    else:
        print("Database connection failed")
