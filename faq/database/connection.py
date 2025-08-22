"""
Database connection module untuk FAQ NLP System
Menggunakan SQLAlchemy dan mysql-connector-python
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Load environment variables
load_dotenv()

Base = declarative_base()

class FAQDataset(Base):
    __tablename__ = 'faq_dataset'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100), default='general')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationship
    variations = relationship("FAQVariations", back_populates="faq", cascade="all, delete-orphan")
    search_logs = relationship("SearchLogs", back_populates="faq")

class FAQVariations(Base):
    __tablename__ = 'faq_variations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    faq_id = Column(Integer, ForeignKey('faq_dataset.id', ondelete='CASCADE'), nullable=False)
    variation_question = Column(Text, nullable=False)
    similarity_score = Column(DECIMAL(5, 4), default=1.0000)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationship
    faq = relationship("FAQDataset", back_populates="variations")

class SearchLogs(Base):
    __tablename__ = 'search_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    search_query = Column(Text, nullable=False)
    result_faq_id = Column(Integer, ForeignKey('faq_dataset.id', ondelete='SET NULL'))
    similarity_score = Column(DECIMAL(5, 4))
    user_ip = Column(String(45))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationship
    faq = relationship("FAQDataset", back_populates="search_logs")

class DatabaseManager:
    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', 3306))
        self.db_user = os.getenv('DB_USER', 'root')
        self.db_password = os.getenv('DB_PASSWORD', '')
        self.db_name = os.getenv('DB_NAME', 'faq_nlp_system')
        
        # SQLAlchemy engine dan session
        self.engine = None
        self.Session = None
        self.init_database()
    
    def init_database(self):
        """Initialize database connection dan create tables"""
        try:
            # Create database URL
            database_url = f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            
            # Create engine
            self.engine = create_engine(database_url, echo=False)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if not exist
            Base.metadata.create_all(self.engine)
            
            print("Database connection initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.Session()
    
    def test_connection(self):
        """Test database connection"""
        try:
            connection = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                print(f"Connected to MySQL Server version: {version[0]}")
                return True
                
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
            
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def execute_raw_query(self, query, params=None):
        """Execute raw SQL query"""
        try:
            connection = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
            else:
                connection.commit()
                result = cursor.rowcount
                
            return result
            
        except Error as e:
            print(f"Error executing query: {e}")
            return None
            
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

# Global database manager instance
db_manager = DatabaseManager()
