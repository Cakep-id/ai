"""
AI Models Package for AgentV2
Contains YOLO service, risk engine, training service, and evaluation service
"""

__version__ = "2.0.0"

# Import full services by default
try:
    from .yolo_service import YOLOService
    from .risk_engine import RiskEngine
    from .training_service import TrainingService
    from .evaluation_service import EvaluationService
    from .report_service import ReportService
    print("Full AI services loaded")
except ImportError as e:
    print(f"Warning: Could not load full AI services: {e}")
    print("Falling back to simple services...")
    try:
        from .yolo_service_simple import YOLOService
        from .risk_engine_simple import RiskEngine
        from .report_service_simple import ReportService
        TrainingService = None
        EvaluationService = None
        print("Simple AI services loaded as fallback")
    except ImportError as e2:
        print(f"Error: Could not load any AI services: {e2}")
        YOLOService = None
        RiskEngine = None
        TrainingService = None
        EvaluationService = None
        ReportService = None

__all__ = [
    "YOLOService",
    "RiskEngine", 
    "TrainingService",
    "EvaluationService",
    "ReportService"
]
