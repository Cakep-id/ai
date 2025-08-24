"""
Report Service for AgentV2 AI Asset Inspection System
Generates comprehensive inspection reports with AI analysis
"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ReportService:
    """Advanced report generation service"""
    
    def __init__(self, daos=None):
        """Initialize report service"""
        self.daos = daos
        self.report_templates = self._load_report_templates()
        logger.info("Report Service initialized")
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load report templates"""
        return {
            "inspection": {
                "sections": [
                    "executive_summary",
                    "detection_details", 
                    "risk_assessment",
                    "recommendations",
                    "technical_details",
                    "compliance_info"
                ],
                "format": "comprehensive"
            },
            "summary": {
                "sections": [
                    "key_findings",
                    "risk_level",
                    "immediate_actions"
                ],
                "format": "condensed"
            }
        }
    
    async def generate_inspection_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive inspection report"""
        try:
            # Extract input data
            report_id = report_data.get('report_id', str(uuid.uuid4()))
            asset_id = report_data.get('asset_id', 'UNKNOWN')
            inspection_date = report_data.get('inspection_date', datetime.now().isoformat())
            yolo_results = report_data.get('yolo_results', {})
            risk_analysis = report_data.get('risk_analysis', {})
            
            # Generate report sections
            executive_summary = self._generate_executive_summary(yolo_results, risk_analysis)
            detection_details = self._format_detection_details(yolo_results)
            risk_assessment = self._format_risk_assessment(risk_analysis)
            recommendations = self._generate_recommendations(risk_analysis, yolo_results)
            technical_details = self._generate_technical_details(yolo_results, risk_analysis)
            compliance_info = self._generate_compliance_info(risk_analysis)
            
            # Compile full report
            report = {
                'report_id': report_id,
                'asset_id': asset_id,
                'inspection_date': inspection_date,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'COMPREHENSIVE_INSPECTION',
                'status': 'COMPLETED',
                'version': '2.0',
                
                # Main sections
                'executive_summary': executive_summary,
                'detection_details': detection_details,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'technical_details': technical_details,
                'compliance_info': compliance_info,
                
                # Metadata
                'metadata': {
                    'ai_confidence': self._calculate_overall_confidence(yolo_results),
                    'analysis_quality': self._assess_analysis_quality(yolo_results, risk_analysis),
                    'data_completeness': self._assess_data_completeness(yolo_results, risk_analysis),
                    'processing_notes': report_data.get('processing_notes', [])
                }
            }
            
            # Save report if DAO available
            if self.daos and 'reports' in self.daos:
                await self.daos['reports'].save_report(report)
            
            return {
                'success': True,
                'report': report,
                'report_id': report_id
            }
            
        except Exception as e:
            logger.error(f"Error generating inspection report: {e}")
            return {
                'success': False,
                'error': str(e),
                'report_id': report_data.get('report_id', 'ERROR')
            }
    
    def _generate_executive_summary(self, yolo_results: Dict, risk_analysis: Dict) -> Dict[str, Any]:
        """Generate executive summary"""
        detections = yolo_results.get('detections', [])
        risk_category = risk_analysis.get('risk_category', 'UNKNOWN')
        risk_score = risk_analysis.get('overall_risk_score', 0)
        
        # Key findings
        key_findings = []
        if detections:
            damage_types = list(set([d.get('class_name') for d in detections]))
            key_findings.append(f"Detected {len(detections)} instances of damage across {len(damage_types)} categories")
            
            if 'crack' in damage_types:
                key_findings.append("Structural cracking identified - requires attention")
            if 'corrosion' in damage_types:
                key_findings.append("Corrosion damage present - monitor progression")
            if 'leak' in damage_types:
                key_findings.append("Potential leak points detected - immediate inspection recommended")
        else:
            key_findings.append("No significant damage detected during AI inspection")
        
        # Overall condition assessment
        condition_map = {
            'CRITICAL': {'condition': 'POOR', 'urgency': 'IMMEDIATE'},
            'HIGH': {'condition': 'FAIR', 'urgency': 'URGENT'},
            'MEDIUM': {'condition': 'GOOD', 'urgency': 'SCHEDULED'},
            'LOW': {'condition': 'EXCELLENT', 'urgency': 'ROUTINE'}
        }
        
        assessment = condition_map.get(risk_category, {'condition': 'UNKNOWN', 'urgency': 'REVIEW'})
        
        return {
            'overall_condition': assessment['condition'],
            'risk_level': risk_category,
            'risk_score': round(risk_score, 2),
            'action_urgency': assessment['urgency'],
            'key_findings': key_findings,
            'detection_count': len(detections),
            'inspection_quality': 'HIGH' if len(detections) > 0 else 'STANDARD'
        }
    
    def _format_detection_details(self, yolo_results: Dict) -> List[Dict[str, Any]]:
        """Format detailed detection information"""
        detections = yolo_results.get('detections', [])
        formatted_detections = []
        
        for i, detection in enumerate(detections):
            formatted_detections.append({
                'detection_id': i + 1,
                'damage_type': detection.get('class_name', 'unknown'),
                'confidence': round(detection.get('confidence', 0), 3),
                'severity': self._determine_detection_severity(detection),
                'location': {
                    'coordinates': {
                        'x': detection.get('x', 0),
                        'y': detection.get('y', 0),
                        'width': detection.get('width', 0),
                        'height': detection.get('height', 0)
                    },
                    'area_percentage': round(detection.get('area_percentage', 0), 2),
                    'description': self._generate_location_description(detection)
                },
                'characteristics': self._analyze_damage_characteristics(detection),
                'priority': self._determine_repair_priority(detection)
            })
        
        return formatted_detections
    
    def _format_risk_assessment(self, risk_analysis: Dict) -> Dict[str, Any]:
        """Format comprehensive risk assessment"""
        return {
            'overall_risk': {
                'score': risk_analysis.get('overall_risk_score', 0),
                'category': risk_analysis.get('risk_category', 'UNKNOWN'),
                'confidence': risk_analysis.get('calibrated_confidence', 0),
                'uncertainty': risk_analysis.get('uncertainty_score', 0)
            },
            'component_risks': {
                'probability': risk_analysis.get('probability_score', 0),
                'consequence': risk_analysis.get('consequence_score', 0),
                'detectability': risk_analysis.get('detectability_score', 0.8)  # AI has high detectability
            },
            'risk_factors': risk_analysis.get('severity_calculation', {}),
            'risk_drivers': self._identify_risk_drivers(risk_analysis),
            'mitigation_effectiveness': self._assess_mitigation_potential(risk_analysis)
        }
    
    def _generate_recommendations(self, risk_analysis: Dict, yolo_results: Dict) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        detections = yolo_results.get('detections', [])
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        # Immediate actions
        immediate_actions = []
        if risk_category == 'CRITICAL':
            immediate_actions.extend([
                "Stop operations immediately",
                "Implement emergency safety measures",
                "Notify emergency response team",
                "Isolate affected area"
            ])
        elif risk_category == 'HIGH':
            immediate_actions.extend([
                "Schedule urgent inspection within 24-48 hours",
                "Implement temporary safety measures",
                "Monitor conditions closely",
                "Prepare for emergency response if needed"
            ])
        
        # Short-term actions (1-4 weeks)
        short_term_actions = []
        if risk_category in ['CRITICAL', 'HIGH']:
            short_term_actions.extend([
                "Conduct detailed engineering assessment",
                "Develop repair/replacement plan",
                "Procure necessary materials and resources"
            ])
        elif risk_category == 'MEDIUM':
            short_term_actions.extend([
                "Schedule maintenance within 1-3 months",
                "Increase inspection frequency",
                "Plan repair during next maintenance window"
            ])
        
        # Long-term actions (1-12 months)
        long_term_actions = []
        damage_types = [d.get('class_name') for d in detections]
        
        if 'corrosion' in damage_types:
            long_term_actions.extend([
                "Implement corrosion prevention program",
                "Review and improve coating systems",
                "Monitor environmental factors"
            ])
        
        if 'crack' in damage_types:
            long_term_actions.extend([
                "Implement structural monitoring program",
                "Review loading conditions",
                "Consider design modifications"
            ])
        
        # Maintenance optimization
        maintenance_recommendations = self._generate_maintenance_recommendations(detections, risk_analysis)
        
        return {
            'immediate_actions': immediate_actions,
            'short_term_actions': short_term_actions,
            'long_term_actions': long_term_actions,
            'maintenance_optimization': maintenance_recommendations,
            'monitoring_requirements': self._generate_monitoring_requirements(risk_analysis),
            'resource_requirements': self._estimate_resource_requirements(risk_analysis, detections)
        }
    
    def _generate_technical_details(self, yolo_results: Dict, risk_analysis: Dict) -> Dict[str, Any]:
        """Generate technical analysis details"""
        return {
            'ai_analysis': {
                'model_version': yolo_results.get('model_info', {}).get('version', 'YOLOv8'),
                'detection_threshold': yolo_results.get('model_info', {}).get('threshold', 0.5),
                'processing_time': yolo_results.get('processing_time', 0),
                'image_quality': yolo_results.get('image_quality', 'GOOD'),
                'detection_confidence_distribution': self._analyze_confidence_distribution(yolo_results)
            },
            'risk_calculation': {
                'methodology': 'AI-Enhanced Risk Assessment v2.0',
                'factors_considered': list(risk_analysis.get('severity_calculation', {}).keys()),
                'calibration_status': 'CALIBRATED',
                'uncertainty_quantification': 'ENABLED'
            },
            'quality_metrics': {
                'detection_quality': self._assess_detection_quality(yolo_results),
                'risk_analysis_quality': self._assess_risk_quality(risk_analysis),
                'overall_confidence': self._calculate_overall_confidence(yolo_results)
            }
        }
    
    def _generate_compliance_info(self, risk_analysis: Dict) -> Dict[str, Any]:
        """Generate compliance and regulatory information"""
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        # Determine applicable standards
        applicable_standards = []
        if risk_category in ['HIGH', 'CRITICAL']:
            applicable_standards.extend(['ISO 55000', 'API 570', 'ASME B31.3'])
        
        return {
            'applicable_standards': applicable_standards,
            'compliance_status': 'REQUIRES_REVIEW' if risk_category in ['HIGH', 'CRITICAL'] else 'COMPLIANT',
            'regulatory_requirements': self._determine_regulatory_requirements(risk_analysis),
            'documentation_requirements': self._determine_documentation_needs(risk_analysis),
            'inspection_frequency': self._recommend_inspection_frequency(risk_analysis)
        }
    
    # Helper methods
    def _determine_detection_severity(self, detection: Dict) -> str:
        """Determine severity of individual detection"""
        confidence = detection.get('confidence', 0)
        area = detection.get('area_percentage', 0)
        damage_type = detection.get('class_name', '')
        
        # Base severity mapping
        severity_map = {
            'crack': 'HIGH',
            'leak': 'CRITICAL',
            'hole': 'HIGH', 
            'corrosion': 'MEDIUM',
            'deformation': 'MEDIUM',
            'wear': 'LOW',
            'scratch': 'LOW'
        }
        
        base_severity = severity_map.get(damage_type, 'MEDIUM')
        
        # Adjust based on confidence and area
        if confidence > 0.8 and area > 5.0:
            if base_severity == 'LOW':
                return 'MEDIUM'
            elif base_severity == 'MEDIUM':
                return 'HIGH'
            elif base_severity == 'HIGH':
                return 'CRITICAL'
        
        return base_severity
    
    def _generate_location_description(self, detection: Dict) -> str:
        """Generate human-readable location description"""
        x, y = detection.get('x', 0), detection.get('y', 0)
        
        # Simple quadrant-based description
        if x < 320 and y < 320:
            return "Upper left quadrant"
        elif x >= 320 and y < 320:
            return "Upper right quadrant"
        elif x < 320 and y >= 320:
            return "Lower left quadrant"
        else:
            return "Lower right quadrant"
    
    def _analyze_damage_characteristics(self, detection: Dict) -> Dict[str, Any]:
        """Analyze characteristics of detected damage"""
        return {
            'size_category': 'LARGE' if detection.get('area_percentage', 0) > 5 else 'SMALL',
            'shape_aspect_ratio': detection.get('width', 1) / detection.get('height', 1),
            'position_risk': 'HIGH' if detection.get('y', 0) < 200 else 'MEDIUM',  # Top areas more critical
            'isolation_level': 'ISOLATED'  # Would need clustering analysis for this
        }
    
    def _determine_repair_priority(self, detection: Dict) -> str:
        """Determine repair priority for detection"""
        severity = self._determine_detection_severity(detection)
        confidence = detection.get('confidence', 0)
        
        if severity == 'CRITICAL' and confidence > 0.8:
            return 'P1_IMMEDIATE'
        elif severity == 'HIGH' and confidence > 0.7:
            return 'P2_URGENT'
        elif severity == 'MEDIUM':
            return 'P3_SCHEDULED'
        else:
            return 'P4_ROUTINE'
    
    def _identify_risk_drivers(self, risk_analysis: Dict) -> List[str]:
        """Identify primary risk drivers"""
        drivers = []
        severity_calc = risk_analysis.get('severity_calculation', {})
        
        if severity_calc.get('detection_count', 0) > 5:
            drivers.append("Multiple damage instances")
        if severity_calc.get('max_individual_risk', 0) > 3.0:
            drivers.append("High-severity individual damage")
        if risk_analysis.get('uncertainty_score', 0) > 0.3:
            drivers.append("Analysis uncertainty")
        
        return drivers
    
    def _assess_mitigation_potential(self, risk_analysis: Dict) -> str:
        """Assess potential for risk mitigation"""
        risk_score = risk_analysis.get('overall_risk_score', 0)
        
        if risk_score < 2.0:
            return 'EXCELLENT'
        elif risk_score < 3.0:
            return 'GOOD'
        elif risk_score < 4.0:
            return 'MODERATE'
        else:
            return 'LIMITED'
    
    def _generate_maintenance_recommendations(self, detections: List, risk_analysis: Dict) -> Dict[str, Any]:
        """Generate maintenance optimization recommendations"""
        damage_types = [d.get('class_name') for d in detections]
        
        recommendations = {
            'frequency_adjustment': 'INCREASE' if len(detections) > 3 else 'MAINTAIN',
            'method_improvements': [],
            'technology_upgrades': [],
            'cost_optimization': []
        }
        
        if 'corrosion' in damage_types:
            recommendations['method_improvements'].append("Implement cathodic protection")
            recommendations['technology_upgrades'].append("Deploy corrosion sensors")
        
        return recommendations
    
    def _generate_monitoring_requirements(self, risk_analysis: Dict) -> Dict[str, Any]:
        """Generate monitoring requirements"""
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        frequency_map = {
            'CRITICAL': 'DAILY',
            'HIGH': 'WEEKLY', 
            'MEDIUM': 'MONTHLY',
            'LOW': 'QUARTERLY'
        }
        
        return {
            'inspection_frequency': frequency_map.get(risk_category, 'QUARTERLY'),
            'monitoring_methods': ['Visual inspection', 'AI-powered analysis'],
            'key_parameters': ['Damage progression', 'New damage emergence'],
            'alert_thresholds': self._define_alert_thresholds(risk_analysis)
        }
    
    def _estimate_resource_requirements(self, risk_analysis: Dict, detections: List) -> Dict[str, Any]:
        """Estimate resource requirements"""
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        base_costs = {
            'CRITICAL': {'inspection': 10000, 'repair': 50000},
            'HIGH': {'inspection': 5000, 'repair': 25000},
            'MEDIUM': {'inspection': 2000, 'repair': 10000},
            'LOW': {'inspection': 500, 'repair': 2000}
        }
        
        costs = base_costs.get(risk_category, base_costs['LOW'])
        
        return {
            'estimated_inspection_cost': costs['inspection'],
            'estimated_repair_cost': costs['repair'] * len(detections),
            'timeline_estimate': self._estimate_timeline(risk_category),
            'skill_requirements': self._determine_skill_requirements(detections),
            'equipment_needs': self._determine_equipment_needs(detections)
        }
    
    def _determine_regulatory_requirements(self, risk_analysis: Dict) -> List[str]:
        """Determine applicable regulatory requirements"""
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        if risk_category in ['HIGH', 'CRITICAL']:
            return [
                "Notify regulatory authority within 24 hours",
                "Submit incident report within 30 days",
                "Implement corrective actions",
                "Schedule follow-up inspection"
            ]
        elif risk_category == 'MEDIUM':
            return [
                "Document findings in maintenance records",
                "Schedule corrective actions within compliance window"
            ]
        else:
            return ["Standard record keeping requirements"]
    
    def _determine_documentation_needs(self, risk_analysis: Dict) -> List[str]:
        """Determine documentation requirements"""
        return [
            "Detailed inspection report",
            "Photographic evidence",
            "AI analysis results", 
            "Risk assessment documentation",
            "Maintenance recommendations"
        ]
    
    def _recommend_inspection_frequency(self, risk_analysis: Dict) -> str:
        """Recommend inspection frequency"""
        risk_category = risk_analysis.get('risk_category', 'LOW')
        
        frequency_map = {
            'CRITICAL': 'Monthly',
            'HIGH': 'Quarterly',
            'MEDIUM': 'Semi-annually', 
            'LOW': 'Annually'
        }
        
        return frequency_map.get(risk_category, 'Annually')
    
    def _analyze_confidence_distribution(self, yolo_results: Dict) -> Dict[str, Any]:
        """Analyze confidence score distribution"""
        detections = yolo_results.get('detections', [])
        if not detections:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        confidences = [d.get('confidence', 0) for d in detections]
        
        return {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'count': len(confidences)
        }
    
    def _assess_detection_quality(self, yolo_results: Dict) -> str:
        """Assess quality of detection results"""
        detections = yolo_results.get('detections', [])
        
        if not detections:
            return 'NO_DETECTIONS'
        
        avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
        
        if avg_confidence > 0.8:
            return 'HIGH'
        elif avg_confidence > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_risk_quality(self, risk_analysis: Dict) -> str:
        """Assess quality of risk analysis"""
        uncertainty = risk_analysis.get('uncertainty_score', 1.0)
        
        if uncertainty < 0.2:
            return 'HIGH'
        elif uncertainty < 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_overall_confidence(self, yolo_results: Dict) -> float:
        """Calculate overall confidence in analysis"""
        detections = yolo_results.get('detections', [])
        
        if not detections:
            return 0.95  # High confidence in "no damage"
        
        avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
        return round(avg_confidence, 3)
    
    def _assess_analysis_quality(self, yolo_results: Dict, risk_analysis: Dict) -> str:
        """Assess overall analysis quality"""
        detection_quality = self._assess_detection_quality(yolo_results)
        risk_quality = self._assess_risk_quality(risk_analysis)
        
        if detection_quality == 'HIGH' and risk_quality == 'HIGH':
            return 'EXCELLENT'
        elif detection_quality in ['HIGH', 'MEDIUM'] and risk_quality in ['HIGH', 'MEDIUM']:
            return 'GOOD'
        else:
            return 'ACCEPTABLE'
    
    def _assess_data_completeness(self, yolo_results: Dict, risk_analysis: Dict) -> str:
        """Assess completeness of analysis data"""
        has_detections = len(yolo_results.get('detections', [])) > 0
        has_risk_score = 'overall_risk_score' in risk_analysis
        has_recommendations = 'recommendations' in risk_analysis
        
        completeness_score = sum([has_detections, has_risk_score, has_recommendations])
        
        if completeness_score >= 3:
            return 'COMPLETE'
        elif completeness_score >= 2:
            return 'MOSTLY_COMPLETE'
        else:
            return 'PARTIAL'
    
    def _define_alert_thresholds(self, risk_analysis: Dict) -> Dict[str, float]:
        """Define alert thresholds for monitoring"""
        current_risk = risk_analysis.get('overall_risk_score', 0)
        
        return {
            'risk_increase_threshold': current_risk * 1.2,
            'new_damage_confidence_threshold': 0.7,
            'damage_size_increase_threshold': 1.5
        }
    
    def _estimate_timeline(self, risk_category: str) -> str:
        """Estimate timeline for repairs"""
        timeline_map = {
            'CRITICAL': '1-3 days',
            'HIGH': '1-2 weeks',
            'MEDIUM': '1-3 months',
            'LOW': '3-12 months'
        }
        
        return timeline_map.get(risk_category, '3-12 months')
    
    def _determine_skill_requirements(self, detections: List) -> List[str]:
        """Determine required skills for repairs"""
        damage_types = [d.get('class_name') for d in detections]
        skills = set()
        
        if 'crack' in damage_types:
            skills.update(['Welding', 'Structural repair'])
        if 'corrosion' in damage_types:
            skills.update(['Surface preparation', 'Coating application'])
        if 'leak' in damage_types:
            skills.update(['Pipe fitting', 'Pressure testing'])
        
        return list(skills) if skills else ['General maintenance']
    
    def _determine_equipment_needs(self, detections: List) -> List[str]:
        """Determine required equipment for repairs"""
        damage_types = [d.get('class_name') for d in detections]
        equipment = set()
        
        if 'crack' in damage_types:
            equipment.update(['Welding equipment', 'Grinding tools'])
        if 'corrosion' in damage_types:
            equipment.update(['Sandblasting equipment', 'Spray guns'])
        if 'leak' in damage_types:
            equipment.update(['Pipe cutters', 'Pressure testing equipment'])
        
        return list(equipment) if equipment else ['Basic hand tools']
    
    # Public report access methods
    async def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report by ID"""
        if self.daos and 'reports' in self.daos:
            return await self.daos['reports'].get_report(report_id)
        return None
    
    async def list_reports(self, asset_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List reports with optional asset filter"""
        if self.daos and 'reports' in self.daos:
            return await self.daos['reports'].list_reports(asset_id=asset_id, limit=limit)
        return []
    
    async def generate_summary_report(self, report_ids: List[str]) -> Dict[str, Any]:
        """Generate summary report from multiple inspection reports"""
        try:
            reports = []
            for report_id in report_ids:
                report = await self.get_report_by_id(report_id)
                if report:
                    reports.append(report)
            
            if not reports:
                return {'success': False, 'error': 'No reports found'}
            
            # Aggregate data
            summary = {
                'summary_id': str(uuid.uuid4()),
                'generated_at': datetime.now().isoformat(),
                'report_count': len(reports),
                'date_range': {
                    'start': min(r.get('inspection_date', '') for r in reports),
                    'end': max(r.get('inspection_date', '') for r in reports)
                },
                'aggregated_findings': self._aggregate_findings(reports),
                'trend_analysis': self._analyze_trends(reports),
                'fleet_risk_assessment': self._assess_fleet_risk(reports)
            }
            
            return {'success': True, 'summary': summary}
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {'success': False, 'error': str(e)}
    
    def _aggregate_findings(self, reports: List[Dict]) -> Dict[str, Any]:
        """Aggregate findings across multiple reports"""
        total_detections = 0
        damage_types = {}
        risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for report in reports:
            detections = report.get('detection_details', [])
            total_detections += len(detections)
            
            # Count damage types
            for detection in detections:
                damage_type = detection.get('damage_type', 'unknown')
                damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
            
            # Count risk categories
            risk_cat = report.get('risk_assessment', {}).get('overall_risk', {}).get('category', 'LOW')
            risk_distribution[risk_cat] = risk_distribution.get(risk_cat, 0) + 1
        
        return {
            'total_detections': total_detections,
            'damage_type_distribution': damage_types,
            'risk_category_distribution': risk_distribution,
            'most_common_damage': max(damage_types.items(), key=lambda x: x[1])[0] if damage_types else None
        }
    
    def _analyze_trends(self, reports: List[Dict]) -> Dict[str, Any]:
        """Analyze trends across reports"""
        # Sort reports by date
        sorted_reports = sorted(reports, key=lambda x: x.get('inspection_date', ''))
        
        if len(sorted_reports) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        # Compare first and last reports
        first_risk = sorted_reports[0].get('risk_assessment', {}).get('overall_risk', {}).get('score', 0)
        last_risk = sorted_reports[-1].get('risk_assessment', {}).get('overall_risk', {}).get('score', 0)
        
        trend = 'STABLE'
        if last_risk > first_risk * 1.2:
            trend = 'DETERIORATING'
        elif last_risk < first_risk * 0.8:
            trend = 'IMPROVING'
        
        return {
            'trend': trend,
            'risk_change': last_risk - first_risk,
            'change_percentage': ((last_risk - first_risk) / first_risk * 100) if first_risk > 0 else 0
        }
    
    def _assess_fleet_risk(self, reports: List[Dict]) -> Dict[str, Any]:
        """Assess overall fleet/system risk"""
        if not reports:
            return {'overall_risk': 'UNKNOWN'}
        
        # Calculate weighted average risk
        total_risk = sum(r.get('risk_assessment', {}).get('overall_risk', {}).get('score', 0) for r in reports)
        avg_risk = total_risk / len(reports)
        
        # Determine fleet risk category
        if avg_risk >= 4.0:
            fleet_risk = 'CRITICAL'
        elif avg_risk >= 3.0:
            fleet_risk = 'HIGH'
        elif avg_risk >= 2.0:
            fleet_risk = 'MEDIUM'
        else:
            fleet_risk = 'LOW'
        
        return {
            'overall_risk': fleet_risk,
            'average_risk_score': round(avg_risk, 2),
            'asset_count': len(reports),
            'high_risk_assets': len([r for r in reports if r.get('risk_assessment', {}).get('overall_risk', {}).get('category') in ['HIGH', 'CRITICAL']])
        }

# For testing
if __name__ == "__main__":
    service = ReportService()
    print("Report Service initialized")
    
    # Test report generation
    import asyncio
    
    mock_data = {
        'report_id': 'TEST_REPORT_001',
        'asset_id': 'ASSET_001',
        'yolo_results': {
            'detections': [
                {
                    'class_name': 'crack',
                    'confidence': 0.85,
                    'x': 100, 'y': 100, 'width': 50, 'height': 20,
                    'area_percentage': 2.5
                }
            ],
            'processing_time': 2.3
        },
        'risk_analysis': {
            'overall_risk_score': 2.8,
            'risk_category': 'MEDIUM',
            'probability_score': 0.7,
            'consequence_score': 0.8,
            'recommendations': ['Schedule maintenance', 'Monitor closely']
        }
    }
    
    async def test():
        result = await service.generate_inspection_report(mock_data)
        print(json.dumps(result, indent=2)[:500] + "...")
    
    asyncio.run(test())
