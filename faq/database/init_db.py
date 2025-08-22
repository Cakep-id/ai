"""
Database initialization script
Run this after creating the database and updating .env file
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from database.connection import db_manager, Base
from services.faq_service import faq_service
import mysql.connector
from mysql.connector import Error

def create_database():
    """Create database if not exists"""
    try:
        # Connect without database name
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('DB_NAME', 'faq_nlp_system')}")
        print("Database created successfully!")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"Error creating database: {e}")
        return False
    
    return True

def init_tables():
    """Initialize database tables"""
    try:
        # Create tables
        Base.metadata.create_all(db_manager.engine)
        print("Tables created successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def insert_sample_data():
    """Insert sample FAQ data"""
    try:
        sample_faqs = [
            {
                'question': 'Bagaimana cara reset password?',
                'answer': 'Untuk reset password, klik "Lupa Password" di halaman login, masukkan email Anda, dan ikuti instruksi yang dikirim ke email.',
                'category': 'account',
                'variations': [
                    'Cara mengubah password',
                    'Lupa kata sandi',
                    'Reset kata sandi'
                ]
            },
            {
                'question': 'Apa itu machine learning?',
                'answer': 'Machine Learning adalah cabang dari artificial intelligence yang memungkinkan komputer belajar dan membuat keputusan tanpa diprogram secara eksplisit.',
                'category': 'technology',
                'variations': [
                    'Pengertian machine learning',
                    'Definisi ML',
                    'Apa yang dimaksud dengan machine learning'
                ]
            },
            {
                'question': 'Bagaimana cara menghubungi customer service?',
                'answer': 'Anda dapat menghubungi customer service melalui email: support@company.com atau telepon: 0800-1234-5678 (Senin-Jumat, 09:00-17:00).',
                'category': 'support',
                'variations': [
                    'Kontak customer service',
                    'Hubungi CS',
                    'Customer support'
                ]
            },
            {
                'question': 'Bagaimana cara membuat akun baru?',
                'answer': 'Untuk membuat akun baru, klik tombol "Daftar" di halaman utama, isi formulir pendaftaran dengan data yang valid, dan verifikasi email Anda.',
                'category': 'account',
                'variations': [
                    'Cara daftar akun',
                    'Registrasi akun baru',
                    'Buat akun'
                ]
            },
            {
                'question': 'Apa itu Natural Language Processing?',
                'answer': 'Natural Language Processing (NLP) adalah cabang AI yang fokus pada interaksi antara komputer dan bahasa manusia, memungkinkan mesin memahami, menginterpretasi, dan menghasilkan bahasa manusia.',
                'category': 'technology',
                'variations': [
                    'Pengertian NLP',
                    'Definisi Natural Language Processing',
                    'Apa yang dimaksud dengan NLP'
                ]
            }
        ]
        
        success_count = 0
        for faq_data in sample_faqs:
            result = faq_service.add_faq(
                question=faq_data['question'],
                answer=faq_data['answer'],
                category=faq_data['category'],
                variations=faq_data['variations']
            )
            
            if result['success']:
                success_count += 1
                print(f"✓ Added: {faq_data['question'][:50]}...")
            else:
                print(f"✗ Failed: {faq_data['question'][:50]}... - {result['message']}")
        
        print(f"\nSample data insertion completed! ({success_count}/{len(sample_faqs)} successful)")
        return success_count > 0
        
    except Exception as e:
        print(f"Error inserting sample data: {e}")
        return False

def main():
    print("========================================")
    print("FAQ NLP System Database Initialization")
    print("========================================\n")
    
    # Test environment variables
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease update your .env file with the required database credentials.")
        return False
    
    print("✓ Environment variables loaded")
    
    # Step 1: Create database
    print("\n[1/4] Creating database...")
    if not create_database():
        print("❌ Failed to create database")
        return False
    
    # Step 2: Test connection
    print("\n[2/4] Testing database connection...")
    if not db_manager.test_connection():
        print("❌ Failed to connect to database")
        print("Please check your database credentials in .env file")
        return False
    print("✓ Database connection successful")
    
    # Step 3: Initialize tables
    print("\n[3/4] Creating tables...")
    if not init_tables():
        print("❌ Failed to create tables")
        return False
    print("✓ Tables created successfully")
    
    # Step 4: Insert sample data
    print("\n[4/4] Inserting sample data...")
    if not insert_sample_data():
        print("❌ Failed to insert sample data")
        return False
    print("✓ Sample data inserted successfully")
    
    print("\n========================================")
    print("✅ Database initialization completed!")
    print("========================================")
    print("\nYou can now run the application with: python app.py")
    print("Or use the run.bat script")
    
    return True

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
