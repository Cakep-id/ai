#!/usr/bin/env python3
"""
Database Migration Script untuk YOLO Training Tables
Menambahkan tabel-tabel yang diperlukan untuk training system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.db import db_service
from loguru import logger

def run_migration():
    """
    Jalankan migrasi database untuk training system
    """
    try:
        logger.info("Starting database migration for YOLO training system...")
        
        # Read SQL file
        sql_file_path = "database/yolo_training_schema.sql"
        
        if not os.path.exists(sql_file_path):
            logger.error(f"SQL file not found: {sql_file_path}")
            return False
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split SQL content into individual statements
        sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        # Execute each statement
        for i, statement in enumerate(sql_statements):
            try:
                if statement.lower().startswith(('create table', 'create index', 'insert into')):
                    logger.info(f"Executing statement {i+1}/{len(sql_statements)}")
                    db_service.execute_query(statement)
                    logger.info(f"✅ Statement {i+1} executed successfully")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"⚠️ Statement {i+1} skipped (already exists)")
                else:
                    logger.error(f"❌ Error executing statement {i+1}: {e}")
                    raise
        
        logger.info("🎉 Database migration completed successfully!")
        
        # Verify tables were created
        verify_tables()
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

def verify_tables():
    """
    Verifikasi bahwa tabel-tabel telah dibuat dengan benar
    """
    try:
        tables_to_check = [
            'yolo_training_datasets',
            'yolo_training_images', 
            'yolo_annotations',
            'yolo_training_sessions',
            'yolo_damage_classes'
        ]
        
        logger.info("Verifying tables...")
        
        for table in tables_to_check:
            result = db_service.fetch_one(f"SHOW TABLES LIKE '{table}'")
            if result:
                logger.info(f"✅ Table '{table}' exists")
                
                # Check row count for damage classes
                if table == 'yolo_damage_classes':
                    count = db_service.fetch_one(f"SELECT COUNT(*) FROM {table}")
                    logger.info(f"   - {count[0]} damage classes inserted")
            else:
                logger.error(f"❌ Table '{table}' not found")
        
        logger.info("Table verification completed!")
        
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")

def show_damage_classes():
    """
    Tampilkan damage classes yang telah diinsert
    """
    try:
        logger.info("Damage classes in database:")
        
        classes = db_service.fetch_all(
            "SELECT class_name, class_label_id, description, risk_weight FROM yolo_damage_classes ORDER BY class_label_id"
        )
        
        for class_name, class_id, description, risk_weight in classes:
            logger.info(f"  [{class_id}] {class_name} - {description} (risk: {risk_weight})")
    
    except Exception as e:
        logger.error(f"Error showing damage classes: {e}")

if __name__ == "__main__":
    print("🚀 YOLO Training Database Migration")
    print("=" * 50)
    
    if run_migration():
        print("\n📊 Damage Classes:")
        show_damage_classes()
        print("\n✅ Migration completed successfully!")
        print("\nAnda sekarang dapat menggunakan:")
        print("- /training - Interface untuk dataset management")
        print("- /api/training/* - Training API endpoints")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)
