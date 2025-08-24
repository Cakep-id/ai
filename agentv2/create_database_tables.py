#!/usr/bin/env python3
"""
Database schema creation script for AgentV2
"""
import asyncio
import aiomysql
from config import DATABASE_CONFIG

async def create_database_schema():
    """Create all database tables from schema.sql"""
    try:
        print("Reading schema file...")
        with open('database/schema.sql', 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        print("Connecting to database...")
        connection = await aiomysql.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'], 
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database=DATABASE_CONFIG['database'],
            charset=DATABASE_CONFIG['charset']
        )
        
        cursor = await connection.cursor()
        
        # Split schema into individual statements
        statements = []
        current_statement = ""
        delimiter = ";"
        
        for line in schema_sql.split('\n'):
            line = line.strip()
            
            # Handle DELIMITER changes
            if line.startswith('DELIMITER'):
                delimiter = line.split()[1]
                continue
            
            # Skip comments and empty lines
            if not line or line.startswith('--'):
                continue
                
            current_statement += line + " "
            
            # Check if statement is complete
            if line.endswith(delimiter):
                # Remove the delimiter
                current_statement = current_statement[:-len(delimiter)].strip()
                if current_statement:
                    statements.append(current_statement)
                current_statement = ""
                
                # Reset delimiter if it was changed
                if delimiter != ";":
                    delimiter = ";"
        
        print(f'Executing {len(statements)} SQL statements...')
        
        success_count = 0
        error_count = 0
        
        for i, statement in enumerate(statements):
            if statement:
                try:
                    await cursor.execute(statement)
                    print(f'✓ Statement {i+1}/{len(statements)}: Success')
                    success_count += 1
                except Exception as e:
                    print(f'✗ Statement {i+1}/{len(statements)}: Error - {str(e)}')
                    print(f'  Statement: {statement[:100]}...')
                    error_count += 1
        
        await connection.commit()
        await cursor.close()
        connection.close()
        
        print(f'\nDatabase schema creation completed!')
        print(f'Success: {success_count} statements')
        print(f'Errors: {error_count} statements')
        
        if error_count == 0:
            print('✓ All tables created successfully!')
        else:
            print(f'⚠ {error_count} errors encountered - some tables may already exist')
        
    except Exception as e:
        print(f'Database connection error: {str(e)}')
        return False
    
    return True

if __name__ == "__main__":
    print("=== AgentV2 Database Schema Creation ===")
    result = asyncio.run(create_database_schema())
    if result:
        print("Schema creation process completed.")
    else:
        print("Schema creation failed.")
