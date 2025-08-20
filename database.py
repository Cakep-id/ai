import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.database = os.getenv('DB_NAME', 'cakep_db')
        self.port = int(os.getenv('DB_PORT', 3306))
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                autocommit=True
            )
            
            if self.connection.is_connected():
                logger.info("✅ Database connection established")
                return True
                
        except Error as e:
            logger.error(f"❌ Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("📡 Database connection closed")
    
    def get_training_data(self, category=None):
        """Fetch training data from database"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if category:
                query = "SELECT * FROM training_data WHERE category = %s ORDER BY created_at DESC"
                cursor.execute(query, (category,))
            else:
                query = "SELECT * FROM training_data ORDER BY created_at DESC"
                cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            logger.info(f"📚 Fetched {len(results)} training data records")
            return results
            
        except Error as e:
            logger.error(f"❌ Error fetching training data: {e}")
            return []
    
    def test_connection(self):
        """Test database connectivity"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            return result is not None
            
        except Error as e:
            logger.error(f"❌ Database test failed: {e}")
            return False

# Global database instance
db = DatabaseConnection()
