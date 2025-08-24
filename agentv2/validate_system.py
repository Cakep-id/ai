"""
AgentV2 System Validation Script
Test all major components and functionality
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
import requests
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from database.db_manager import DatabaseManager
from ai_models.yolo_service import YOLOService
from ai_models.risk_engine import RiskEngine
from ai_models.training_service import TrainingService
from ai_models.evaluation_service import EvaluationService
from config import *

class SystemValidator:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def log_result(self, test_name, success, message="", error=None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        self.results[test_name] = {
            "success": success,
            "message": message,
            "error": str(error) if error else None
        }
        
        if not success and error:
            self.errors.append(f"{test_name}: {error}")
    
    def test_configuration(self):
        """Test configuration loading"""
        try:
            # Test all config sections
            configs = [
                DATABASE_CONFIG, API_CONFIG, STORAGE_CONFIG, 
                YOLO_CONFIG, RISK_CONFIG, TRAINING_CONFIG
            ]
            
            for config in configs:
                assert isinstance(config, dict), "Config must be dictionary"
            
            # Test directory creation
            create_directories()
            
            # Check critical directories exist
            for directory in [
                STORAGE_CONFIG["base_upload_dir"],
                STORAGE_CONFIG["models_dir"],
                LOGGING_CONFIG["file_path"].parent
            ]:
                assert directory.exists(), f"Directory {directory} not created"
            
            self.log_result("Configuration", True, "All configurations loaded successfully")
            
        except Exception as e:
            self.log_result("Configuration", False, "Configuration failed", e)
    
    def test_database_connection(self):
        """Test database connectivity"""
        try:
            db = DatabaseManager()
            db.test_connection()
            self.log_result("Database Connection", True, "Database connected successfully")
            
        except Exception as e:
            self.log_result("Database Connection", False, "Database connection failed", e)
    
    def test_database_schema(self):
        """Test database schema"""
        try:
            db = DatabaseManager()
            
            # Test critical tables exist
            critical_tables = [
                'user_reports', 'yolo_detections', 'risk_assessments',
                'training_sessions', 'model_metrics', 'model_calibration'
            ]
            
            connection = db.get_connection()
            cursor = connection.cursor()
            
            for table in critical_tables:
                cursor.execute(f"SHOW TABLES LIKE '{table}'")
                result = cursor.fetchone()
                assert result, f"Table {table} not found"
            
            cursor.close()
            connection.close()
            
            self.log_result("Database Schema", True, f"All {len(critical_tables)} critical tables found")
            
        except Exception as e:
            self.log_result("Database Schema", False, "Schema validation failed", e)
    
    def test_yolo_service(self):
        """Test YOLO service initialization"""
        try:
            yolo_service = YOLOService()
            
            # Test model loading
            assert yolo_service.model is not None, "YOLO model not loaded"
            
            # Test basic functionality (without actual image)
            device_info = yolo_service.get_device_info()
            assert 'device' in device_info, "Device info not available"
            
            self.log_result("YOLO Service", True, f"YOLO service initialized on {device_info['device']}")
            
        except Exception as e:
            self.log_result("YOLO Service", False, "YOLO service failed", e)
    
    def test_risk_engine(self):
        """Test risk assessment engine"""
        try:
            risk_engine = RiskEngine()
            
            # Test with dummy detection data
            dummy_detections = [
                {
                    'class_name': 'crack',
                    'confidence': 0.85,
                    'bbox': [100, 100, 200, 150],
                    'area_pixels': 5000
                }
            ]
            
            risk_assessment = risk_engine.assess_risk(
                detections=dummy_detections,
                image_dimensions=(640, 480),
                pixel_to_mm_ratio=1.0
            )
            
            assert 'overall_risk_score' in risk_assessment, "Risk score not calculated"
            assert 'risk_category' in risk_assessment, "Risk category not assigned"
            
            self.log_result("Risk Engine", True, f"Risk assessment: {risk_assessment['risk_category']}")
            
        except Exception as e:
            self.log_result("Risk Engine", False, "Risk engine failed", e)
    
    def test_training_service(self):
        """Test training service initialization"""
        try:
            training_service = TrainingService()
            
            # Test service initialization
            assert hasattr(training_service, 'db'), "Database connection not initialized"
            
            # Test basic functionality
            active_sessions = training_service.get_active_sessions()
            assert isinstance(active_sessions, list), "Active sessions not returned as list"
            
            self.log_result("Training Service", True, f"Training service initialized, {len(active_sessions)} active sessions")
            
        except Exception as e:
            self.log_result("Training Service", False, "Training service failed", e)
    
    def test_evaluation_service(self):
        """Test evaluation service"""
        try:
            evaluation_service = EvaluationService()
            
            # Test service initialization
            assert hasattr(evaluation_service, 'db'), "Database connection not initialized"
            
            self.log_result("Evaluation Service", True, "Evaluation service initialized")
            
        except Exception as e:
            self.log_result("Evaluation Service", False, "Evaluation service failed", e)
    
    def test_api_server(self):
        """Test API server startup (basic test)"""
        try:
            # Import main to test basic imports
            from backend.main import app
            
            assert app is not None, "FastAPI app not created"
            
            self.log_result("API Server", True, "FastAPI application created successfully")
            
        except Exception as e:
            self.log_result("API Server", False, "API server failed", e)
    
    def test_file_permissions(self):
        """Test file system permissions"""
        try:
            # Test write permissions in critical directories
            test_dirs = [
                STORAGE_CONFIG["base_upload_dir"],
                STORAGE_CONFIG["temp_dir"],
                LOGGING_CONFIG["file_path"].parent
            ]
            
            for directory in test_dirs:
                test_file = directory / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
            
            self.log_result("File Permissions", True, "Write permissions OK in all directories")
            
        except Exception as e:
            self.log_result("File Permissions", False, "File permission test failed", e)
    
    def test_model_files(self):
        """Test model file availability"""
        try:
            model_path = YOLO_CONFIG["model_path"]
            
            if not model_path.exists():
                # Try to download
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')  # This will download if not exists
                
            assert model_path.exists() or Path("yolov8n.pt").exists(), "YOLO model file not found"
            
            self.log_result("Model Files", True, "YOLO model available")
            
        except Exception as e:
            self.log_result("Model Files", False, "Model file test failed", e)
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 Starting AgentV2 System Validation")
        print("=" * 50)
        
        tests = [
            self.test_configuration,
            self.test_database_connection,
            self.test_database_schema,
            self.test_file_permissions,
            self.test_model_files,
            self.test_yolo_service,
            self.test_risk_engine,
            self.test_training_service,
            self.test_evaluation_service,
            self.test_api_server
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_result(test_name, False, "Unexpected error", e)
        
        print("\n" + "=" * 50)
        print("🏁 Validation Complete")
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.errors:
            print(f"\n❌ Errors found:")
            for error in self.errors:
                print(f"   - {error}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! System is ready to run.")
            return True
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please fix the issues above.")
            return False

def main():
    """Main validation function"""
    validator = SystemValidator()
    
    try:
        # Run validation
        success = asyncio.run(validator.run_all_tests())
        
        # Save results
        results_file = Path("validation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'success': success,
                'results': validator.results,
                'errors': validator.errors
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed results saved to {results_file}")
        
        # Exit code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error during validation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
