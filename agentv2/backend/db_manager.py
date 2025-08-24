"""
Database Manager for AgentV2 AI Asset Inspection System
Handles MySQL database operations with connection pooling and error handling
"""

import asyncio
import aiomysql
import json
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
import logging
from contextlib import asynccontextmanager

# Environment variable loading
try:
    from backend.env_loader import load_environment
    env_vars = load_environment()
    DB_CONFIG = env_vars.get('DATABASE', {})
except ImportError:
    print("Warning: Environment loader not available, using defaults")
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'agentv2_db'),
        'charset': 'utf8mb4'
    }

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Advanced database manager with connection pooling and error handling"""
    
    def __init__(self, pool_size: int = 10):
        """Initialize database manager"""
        self.pool = None
        self.pool_size = pool_size
        self.config = DB_CONFIG
        self.connected = False
        logger.info(f"Database Manager initialized for {self.config.get('host', 'localhost')}")
    
    async def connect(self) -> bool:
        """Create database connection pool"""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                db=self.config['database'],
                charset=self.config['charset'],
                autocommit=True,
                minsize=1,
                maxsize=self.pool_size
            )
            self.connected = True
            logger.info("Database connection pool created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.connected = False
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise Exception("Database not connected")
        
        async with self.pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_cursor(self):
        """Get database cursor with connection"""
        async with self.get_connection() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                yield cursor
    
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    result = await cursor.fetchall()
                    return result if result else []
                else:
                    # For INSERT, UPDATE, DELETE operations
                    await cursor.connection.commit()
                    return []
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    # Asset Management
    async def create_asset(self, asset_data: Dict[str, Any]) -> str:
        """Create new asset"""
        try:
            asset_id = asset_data.get('id') or f"AST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO assets (id, name, type, location, description, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    asset_id,
                    asset_data.get('name', ''),
                    asset_data.get('type', ''),
                    asset_data.get('location', ''),
                    asset_data.get('description', ''),
                    json.dumps(asset_data.get('metadata', {})),
                    datetime.now()
                ))
            
            logger.info(f"Asset created: {asset_id}")
            return asset_id
            
        except Exception as e:
            logger.error(f"Error creating asset: {e}")
            raise
    
    async def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset by ID"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM assets WHERE id = %s", (asset_id,))
                result = await cursor.fetchone()
                
                if result:
                    if result.get('metadata'):
                        result['metadata'] = json.loads(result['metadata'])
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting asset {asset_id}: {e}")
            return None
    
    async def list_assets(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List assets with pagination"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    SELECT * FROM assets 
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = await cursor.fetchall()
                
                for result in results:
                    if result.get('metadata'):
                        result['metadata'] = json.loads(result['metadata'])
                
                return results
                
        except Exception as e:
            logger.error(f"Error listing assets: {e}")
            return []
    
    async def update_asset(self, asset_id: str, update_data: Dict[str, Any]) -> bool:
        """Update asset"""
        try:
            fields = []
            values = []
            
            for key, value in update_data.items():
                if key not in ['id', 'created_at']:
                    if key == 'metadata':
                        value = json.dumps(value)
                    fields.append(f"{key} = %s")
                    values.append(value)
            
            if not fields:
                return False
            
            fields.append("updated_at = %s")
            values.append(datetime.now())
            values.append(asset_id)
            
            query = f"UPDATE assets SET {', '.join(fields)} WHERE id = %s"
            
            async with self.get_cursor() as cursor:
                await cursor.execute(query, values)
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating asset {asset_id}: {e}")
            return False
    
    # Inspection Management
    async def create_inspection(self, inspection_data: Dict[str, Any]) -> str:
        """Create new inspection"""
        try:
            inspection_id = inspection_data.get('id') or f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO inspections (
                        id, asset_id, inspector_id, inspection_date, 
                        type, status, description, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    inspection_id,
                    inspection_data.get('asset_id'),
                    inspection_data.get('inspector_id'),
                    inspection_data.get('inspection_date', datetime.now()),
                    inspection_data.get('type', 'AI_INSPECTION'),
                    inspection_data.get('status', 'CREATED'),
                    inspection_data.get('description', ''),
                    json.dumps(inspection_data.get('metadata', {})),
                    datetime.now()
                ))
            
            logger.info(f"Inspection created: {inspection_id}")
            return inspection_id
            
        except Exception as e:
            logger.error(f"Error creating inspection: {e}")
            raise
    
    async def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """Get inspection by ID"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM inspections WHERE id = %s", (inspection_id,))
                result = await cursor.fetchone()
                
                if result and result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting inspection {inspection_id}: {e}")
            return None
    
    async def list_inspections(self, asset_id: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List inspections with optional asset filter"""
        try:
            query = "SELECT * FROM inspections"
            params = []
            
            if asset_id:
                query += " WHERE asset_id = %s"
                params.append(asset_id)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            async with self.get_cursor() as cursor:
                await cursor.execute(query, params)
                results = await cursor.fetchall()
                
                for result in results:
                    if result.get('metadata'):
                        result['metadata'] = json.loads(result['metadata'])
                
                return results
                
        except Exception as e:
            logger.error(f"Error listing inspections: {e}")
            return []
    
    async def update_inspection(self, inspection_id: str, update_data: Dict[str, Any]) -> bool:
        """Update inspection"""
        try:
            fields = []
            values = []
            
            for key, value in update_data.items():
                if key not in ['id', 'created_at']:
                    if key == 'metadata':
                        value = json.dumps(value)
                    fields.append(f"{key} = %s")
                    values.append(value)
            
            if not fields:
                return False
            
            fields.append("updated_at = %s")
            values.append(datetime.now())
            values.append(inspection_id)
            
            query = f"UPDATE inspections SET {', '.join(fields)} WHERE id = %s"
            
            async with self.get_cursor() as cursor:
                await cursor.execute(query, values)
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating inspection {inspection_id}: {e}")
            return False
    
    # Detection Results Management
    async def save_detection_results(self, detection_data: Dict[str, Any]) -> str:
        """Save detection results"""
        try:
            detection_id = detection_data.get('id') or f"DET_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO detection_results (
                        id, inspection_id, model_version, detections,
                        confidence_threshold, processing_time, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    detection_id,
                    detection_data.get('inspection_id'),
                    detection_data.get('model_version', 'YOLOv8'),
                    json.dumps(detection_data.get('detections', [])),
                    detection_data.get('confidence_threshold', 0.5),
                    detection_data.get('processing_time', 0.0),
                    json.dumps(detection_data.get('metadata', {})),
                    datetime.now()
                ))
            
            logger.info(f"Detection results saved: {detection_id}")
            return detection_id
            
        except Exception as e:
            logger.error(f"Error saving detection results: {e}")
            raise
    
    async def get_detection_results(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get detection results by ID"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM detection_results WHERE id = %s", (detection_id,))
                result = await cursor.fetchone()
                
                if result:
                    if result.get('detections'):
                        result['detections'] = json.loads(result['detections'])
                    if result.get('metadata'):
                        result['metadata'] = json.loads(result['metadata'])
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting detection results {detection_id}: {e}")
            return None
    
    # Risk Analysis Management
    async def save_risk_analysis(self, risk_data: Dict[str, Any]) -> str:
        """Save risk analysis results"""
        try:
            risk_id = risk_data.get('id') or f"RSK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO risk_analysis (
                        id, inspection_id, risk_score, risk_category,
                        probability_score, consequence_score, analysis_data, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    risk_id,
                    risk_data.get('inspection_id'),
                    risk_data.get('risk_score', 0.0),
                    risk_data.get('risk_category', 'LOW'),
                    risk_data.get('probability_score', 0.0),
                    risk_data.get('consequence_score', 0.0),
                    json.dumps(risk_data.get('analysis_data', {})),
                    datetime.now()
                ))
            
            logger.info(f"Risk analysis saved: {risk_id}")
            return risk_id
            
        except Exception as e:
            logger.error(f"Error saving risk analysis: {e}")
            raise
    
    # Training Management
    async def save_training_session(self, training_data: Dict[str, Any]) -> str:
        """Save training session"""
        try:
            session_id = training_data.get('id') or f"TRN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO training_sessions (
                        id, trainer_id, model_version, training_data,
                        status, metrics, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    session_id,
                    training_data.get('trainer_id'),
                    training_data.get('model_version', 'YOLOv8'),
                    json.dumps(training_data.get('training_data', {})),
                    training_data.get('status', 'CREATED'),
                    json.dumps(training_data.get('metrics', {})),
                    datetime.now()
                ))
            
            logger.info(f"Training session saved: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error saving training session: {e}")
            raise
    
    async def get_training_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session by ID"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM training_sessions WHERE id = %s", (session_id,))
                result = await cursor.fetchone()
                
                if result:
                    if result.get('training_data'):
                        result['training_data'] = json.loads(result['training_data'])
                    if result.get('metrics'):
                        result['metrics'] = json.loads(result['metrics'])
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting training session {session_id}: {e}")
            return None
    
    async def list_training_sessions(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List training sessions"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    SELECT * FROM training_sessions 
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = await cursor.fetchall()
                
                for result in results:
                    if result.get('training_data'):
                        result['training_data'] = json.loads(result['training_data'])
                    if result.get('metrics'):
                        result['metrics'] = json.loads(result['metrics'])
                
                return results
                
        except Exception as e:
            logger.error(f"Error listing training sessions: {e}")
            return []
    
    # User Management
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user"""
        try:
            user_id = user_data.get('id') or f"USR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.get_cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO users (id, username, email, role, password_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    user_data.get('username'),
                    user_data.get('email'),
                    user_data.get('role', 'user'),
                    user_data.get('password_hash'),
                    datetime.now()
                ))
            
            logger.info(f"User created: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                return await cursor.fetchone()
                
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                return await cursor.fetchone()
                
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {e}")
            return None
    
    # Health Check and Statistics
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
                
                # Get table counts
                counts = {}
                tables = ['assets', 'inspections', 'detection_results', 'risk_analysis', 'training_sessions', 'users']
                
                for table in tables:
                    try:
                        await cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                        result = await cursor.fetchone()
                        counts[table] = result['count'] if result else 0
                    except:
                        counts[table] = 0
                
                return {
                    'status': 'healthy',
                    'connected': self.connected,
                    'timestamp': datetime.now().isoformat(),
                    'table_counts': counts
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'connected': self.connected,
            'pool_size': self.pool_size,
            'config': {k: v for k, v in self.config.items() if k != 'password'},
            'timestamp': datetime.now().isoformat()
        }

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_db():
        db = DatabaseManager()
        
        # Test connection (will likely fail without proper DB setup)
        connected = await db.connect()
        print(f"Database connected: {connected}")
        
        if connected:
            # Test health check
            health = await db.health_check()
            print(f"Health check: {health}")
            
            await db.disconnect()
        else:
            print("Database connection failed - this is expected in testing environment")
    
    asyncio.run(test_db())
