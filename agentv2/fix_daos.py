"""
Script to fix all daos references in main.py
"""

import re

def fix_main_py():
    with open('c:/programming/cakep/ai/agentv2/backend/main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace daos references with db_manager calls
    replacements = [
        # Process user report function
        (r"daos\['user_reports'\]\.update_processing_status\(report_id, \"processing\"\)", 
         "# Update processing status in db_manager if available\n        if db_manager:\n            await db_manager.update_inspection(report_id, {'status': 'processing'})"),
        
        (r"daos\['yolo_detections'\]\.save_detections\(report_id, detections\)", 
         "# Save detection results if db_manager available\n        if db_manager:\n            await db_manager.save_detection_results({'inspection_id': report_id, 'detections': detections})"),
        
        (r"daos\['risk_analysis'\]\.save_risk_analysis\(risk_results\)", 
         "# Save risk analysis if db_manager available\n        if db_manager:\n            await db_manager.save_risk_analysis(risk_results)"),
        
        (r"daos\['user_reports'\]\.update_processing_status\(\s*report_id, \"completed\",.*?\)", 
         "# Update completion status\n            if db_manager:\n                await db_manager.update_inspection(report_id, {'status': 'completed', 'yolo_results': json.dumps(detections), 'risk_analysis': json.dumps(risk_results)})"),
        
        (r"daos\['user_reports'\]\.update_processing_status\(report_id, \"failed\"\)", 
         "# Update failed status\n        if db_manager:\n            await db_manager.update_inspection(report_id, {'status': 'failed'})"),
        
        # Get functions
        (r"report = daos\['user_reports'\]\.get_report\(report_id\)", 
         "report = await db_manager.get_inspection(report_id) if db_manager else None"),
        
        (r"detections = daos\['yolo_detections'\]\.get_detections\(report_id\)", 
         "detections = [] # TODO: Implement get_detections from db_manager"),
        
        (r"risk_analysis = daos\['risk_analysis'\]\.get_risk_analysis\(report_id\)", 
         "risk_analysis = {} # TODO: Implement get_risk_analysis from db_manager"),
        
        (r"reports = daos\['user_reports'\]\.get_reports_by_status\(\)", 
         "reports = await db_manager.list_inspections() if db_manager else []"),
        
        (r"recent_reports = daos\['user_reports'\]\.get_reports_by_status\(\)\[:10\]", 
         "recent_reports = (await db_manager.list_inspections(limit=10)) if db_manager else []"),
        
        (r"reports = daos\['user_reports'\]\.get_reports_by_status\(\s*status.*?\)", 
         "reports = await db_manager.list_inspections(limit=50) if db_manager else []"),
        
        (r"report = daos\['user_reports'\]\.get_report\(report_id\)", 
         "report = await db_manager.get_inspection(report_id) if db_manager else None"),
        
        # Training
        (r"daos\['training_sessions'\]\.create_session\(session_data\)", 
         "# Save training session\n        if db_manager:\n            await db_manager.save_training_session(session_data)"),
        
        # Config
        (r"value = daos\['system_config'\]\.get_config\(key\)", 
         "value = None  # TODO: Implement system config in db_manager"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add training_service to globals
    content = content.replace(
        "global db_manager, yolo_service, risk_engine, report_service",
        "global db_manager, yolo_service, risk_engine, report_service, training_service, evaluation_service"
    )
    
    # Initialize training services
    content = content.replace(
        "        print(\"All services initialized successfully\")",
        """        # Initialize training services if available
        if TrainingService:
            training_service = TrainingService()
            print("Training Service initialized")
        
        if EvaluationService:
            evaluation_service = EvaluationService()
            print("Evaluation Service initialized")
        
        print("All services initialized successfully")"""
    )
    
    # Add training_service and evaluation_service to globals
    content = content.replace(
        "report_service = None",
        """report_service = None
training_service = None
evaluation_service = None"""
    )
    
    with open('c:/programming/cakep/ai/agentv2/backend/main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed all daos references in main.py")

if __name__ == "__main__":
    fix_main_py()
