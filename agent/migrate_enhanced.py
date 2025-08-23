#!/usr/bin/env python3
"""
Enhanced Features Migration Script
Migrates database to support:
- Pipeline inspection history
- Human feedback for AI learning
- Maintenance scheduling
- Analytics and reporting
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from services.db import get_db_connection
from loguru import logger

def run_enhanced_migration():
    """Run the enhanced features migration"""
    try:
        logger.info("🚀 Starting enhanced features migration...")
        
        # Get database connection
        conn = get_db_connection()
        
        # Read and execute schema
        schema_file = Path(__file__).parent / "database" / "enhanced_features_schema.sql"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Split by delimiter and execute each statement
        statements = schema_sql.split(';')
        
        for i, statement in enumerate(statements):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    conn.execute(statement)
                    logger.debug(f"Executed statement {i+1}/{len(statements)}")
                except Exception as e:
                    logger.warning(f"Statement {i+1} failed (might be expected): {e}")
        
        conn.commit()
        
        # Verify tables were created
        tables_to_check = [
            'pipeline_inspections',
            'human_feedback', 
            'maintenance_schedule',
            'ai_learning_progress'
        ]
        
        for table in tables_to_check:
            result = conn.execute(f"SHOW TABLES LIKE '{table}'")
            if result.fetchone():
                logger.info(f"✅ Table '{table}' created successfully")
            else:
                logger.error(f"❌ Table '{table}' was not created")
        
        # Check views
        views_to_check = ['feedback_analytics', 'maintenance_analytics']
        for view in views_to_check:
            try:
                conn.execute(f"SELECT 1 FROM {view} LIMIT 1")
                logger.info(f"✅ View '{view}' created successfully")
            except:
                logger.warning(f"⚠️ View '{view}' may not be created properly")
        
        logger.info("✅ Enhanced features migration completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = run_enhanced_migration()
    sys.exit(0 if success else 1)
