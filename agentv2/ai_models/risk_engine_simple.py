"""
Simple Risk Engine for testing
"""

import json
import math
from typing import Dict, List, Any, Optional
from datetime import datetime

class RiskEngine:
    """Simple risk assessment engine"""
    
    def __init__(self, daos=None):
        """Initialize risk engine"""
        self.daos = daos
        self.risk_weights = {
            "damage_type": 0.3,
            "size": 0.25,
            "geometry": 0.2,
            "clustering": 0.15,
            "location": 0.1
        }
        self.severity_factors = {
            "crack": 1.8,
            "corrosion": 1.5,
            "deformation": 1.2,
            "hole": 2.0,
            "paint_loss": 1.0,
            "rust": 1.3,
            "scratch": 0.8,
            "wear": 0.9
        }
    
    async def analyze_risk(self, report_id: str, yolo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk based on detection results"""
        try:
            detections = yolo_results.get('detections', [])
            
            if not detections:
                return self._create_low_risk_result(report_id)
            
            # Calculate individual detection risks
            detection_risks = []
            for detection in detections:
                risk = self._calculate_detection_risk(detection)
                detection_risks.append(risk)
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk(detection_risks)
            
            # Determine risk category
            risk_category = self._determine_risk_category(overall_risk)
            
            # Create risk analysis result
            risk_analysis = {
                'report_id': report_id,
                'overall_risk_score': round(overall_risk, 3),
                'risk_category': risk_category,
                'probability_score': round(overall_risk * 0.7, 4),
                'consequence_score': round(overall_risk * 0.8, 4),
                'severity_calculation': {
                    'detection_count': len(detections),
                    'max_individual_risk': max(detection_risks) if detection_risks else 0,
                    'avg_risk': sum(detection_risks) / len(detection_risks) if detection_risks else 0,
                    'risk_factors': self._get_risk_factors(detections)
                },
                'geometry_based_severity': round(overall_risk, 3),
                'calibrated_confidence': round(overall_risk * 0.9, 4),
                'uncertainty_score': round(0.1, 4),  # Mock uncertainty
                'recommendations': self._generate_recommendations(risk_category, detections)
            }
            
            return risk_analysis
            
        except Exception as e:
            print(f"Error in risk analysis: {e}")
            return self._create_error_result(report_id, str(e))
    
    def _calculate_detection_risk(self, detection: Dict[str, Any]) -> float:
        """Calculate risk for individual detection"""
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0.5)
        area_percentage = detection.get('area_percentage', 1.0)
        
        # Base severity from damage type
        base_severity = self.severity_factors.get(class_name, 1.0)
        
        # Size factor (larger damage = higher risk)
        size_factor = min(area_percentage / 5.0, 2.0)  # Cap at 2x
        
        # Confidence factor
        confidence_factor = confidence
        
        # Combine factors
        risk_score = base_severity * (1 + size_factor) * confidence_factor
        
        return min(risk_score, 5.0)  # Cap at 5.0
    
    def _calculate_overall_risk(self, detection_risks: List[float]) -> float:
        """Calculate overall risk from individual detection risks"""
        if not detection_risks:
            return 0.1
        
        # Use weighted combination
        max_risk = max(detection_risks)
        avg_risk = sum(detection_risks) / len(detection_risks)
        count_factor = min(len(detection_risks) / 5.0, 1.5)  # More detections = higher risk
        
        overall_risk = (max_risk * 0.6 + avg_risk * 0.4) * count_factor
        
        return min(overall_risk, 5.0)
    
    def _determine_risk_category(self, risk_score: float) -> str:
        """Determine risk category based on score"""
        if risk_score >= 4.0:
            return "CRITICAL"
        elif risk_score >= 3.0:
            return "HIGH"
        elif risk_score >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_risk_factors(self, detections: List[Dict]) -> Dict[str, Any]:
        """Get risk factors from detections"""
        damage_types = [d.get('class_name') for d in detections]
        unique_types = list(set(damage_types))
        
        return {
            'damage_types': unique_types,
            'damage_count': len(detections),
            'unique_damage_types': len(unique_types),
            'max_confidence': max([d.get('confidence', 0) for d in detections]) if detections else 0,
            'total_affected_area': sum([d.get('area_percentage', 0) for d in detections])
        }
    
    def _generate_recommendations(self, risk_category: str, detections: List[Dict]) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        
        if risk_category == "CRITICAL":
            recommendations.extend([
                "Immediate inspection and repair required",
                "Stop operations until repairs are completed",
                "Engage emergency maintenance team"
            ])
        elif risk_category == "HIGH":
            recommendations.extend([
                "Schedule urgent repair within 1-2 weeks",
                "Monitor closely for deterioration",
                "Consider temporary protective measures"
            ])
        elif risk_category == "MEDIUM":
            recommendations.extend([
                "Schedule maintenance within 1-3 months",
                "Regular monitoring recommended",
                "Plan for repair during next maintenance window"
            ])
        else:
            recommendations.extend([
                "Include in routine maintenance schedule",
                "Monitor during regular inspections"
            ])
        
        # Add specific recommendations based on damage types
        damage_types = [d.get('class_name') for d in detections]
        if 'crack' in damage_types:
            recommendations.append("Monitor crack propagation")
        if 'corrosion' in damage_types:
            recommendations.append("Apply anti-corrosion treatment")
        if 'leak' in damage_types:
            recommendations.append("Check system pressure and seals")
        
        return recommendations
    
    def _create_low_risk_result(self, report_id: str) -> Dict[str, Any]:
        """Create result for no detections (low risk)"""
        return {
            'report_id': report_id,
            'overall_risk_score': 0.1,
            'risk_category': 'LOW',
            'probability_score': 0.05,
            'consequence_score': 0.1,
            'severity_calculation': {
                'detection_count': 0,
                'message': 'No significant damage detected'
            },
            'geometry_based_severity': 0.1,
            'calibrated_confidence': 0.95,
            'uncertainty_score': 0.05,
            'recommendations': ['Asset appears to be in good condition', 'Continue regular monitoring']
        }
    
    def _create_error_result(self, report_id: str, error_msg: str) -> Dict[str, Any]:
        """Create result for error cases"""
        return {
            'report_id': report_id,
            'overall_risk_score': 2.5,
            'risk_category': 'MEDIUM',
            'probability_score': 0.5,
            'consequence_score': 0.5,
            'severity_calculation': {
                'error': error_msg,
                'message': 'Risk analysis failed, using conservative estimate'
            },
            'geometry_based_severity': 2.5,
            'calibrated_confidence': 0.3,
            'uncertainty_score': 0.7,
            'recommendations': ['Manual inspection recommended due to analysis error']
        }

# For testing
if __name__ == "__main__":
    engine = RiskEngine()
    print("Risk Engine initialized")
    
    # Test with mock detection
    mock_yolo_results = {
        'detections': [
            {
                'class_name': 'crack',
                'confidence': 0.85,
                'area_percentage': 3.2
            }
        ]
    }
    
    import asyncio
    result = asyncio.run(engine.analyze_risk("TEST_001", mock_yolo_results))
    print(json.dumps(result, indent=2))
