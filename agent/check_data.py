import sys
sys.path.append('.')

from services.db import db_service
import asyncio

async def check_data():
    try:
        # Check user reports
        reports = db_service.execute_query(
            "SELECT report_id, ai_risk_level, admin_status, description FROM user_reports LIMIT 5", 
            {}
        )
        print("User Reports:")
        for report in reports:
            print(f"ID: {report['report_id']}, Risk: {report['ai_risk_level']}, Status: {report['admin_status']}")
        print()
        
        # Check if we need to create test data
        if not reports:
            print("No reports found. Creating test data...")
            # Insert test report
            db_service.execute_query("""
                INSERT INTO user_reports 
                (report_id, reported_by_user_id, asset_id, description, ai_risk_level, admin_status, reported_at)
                VALUES 
                ('test-report-1', 1, 1, 'Test kerusakan untuk edit risk level', 'LOW', 'pending', NOW())
            """, {})
            print("Test data created!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_data())
