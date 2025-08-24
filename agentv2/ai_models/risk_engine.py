"""
Advanced Risk Engine for AgentV2
Implements geometry-based severity calculations with:
- Physics-based damage assessment
- Multi-dimensional risk analysis
- Consequence modeling
- Asset criticality evaluation
- Maintenance priority optimization
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import math
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DamageType(Enum):
    CORROSION = "corrosion"
    DENT = "dent"
    CRACK = "crack"
    COATING_LOSS = "coating_loss"
    LEAK = "leak"
    WEAR = "wear"

@dataclass
class GeometryAnalysis:
    """Geometry-based damage analysis results"""
    area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    circularity: float
    solidity: float
    extent: float
    major_axis_mm: float
    minor_axis_mm: float
    orientation_deg: float
    centroid: Tuple[float, float]
    bounding_rect_area_mm2: float
    convex_hull_area_mm2: float
    
@dataclass
class SeverityFactors:
    """Severity calculation factors"""
    size_factor: float  # Based on damage size
    shape_factor: float  # Based on damage geometry
    location_factor: float  # Based on damage location
    propagation_factor: float  # Based on damage type progression risk
    stress_concentration_factor: float  # Based on geometry and loading
    material_factor: float  # Based on material properties
    environmental_factor: float  # Based on operating conditions

@dataclass
class ConsequenceAnalysis:
    """Consequence analysis results"""
    safety_consequence: float  # Risk to personnel safety
    environmental_consequence: float  # Environmental impact risk
    production_consequence: float  # Production loss risk
    equipment_consequence: float  # Equipment damage risk
    cost_consequence: float  # Financial impact
    reputation_consequence: float  # Reputation impact

@dataclass
class RiskAssessment:
    """Complete risk assessment"""
    overall_risk_score: float
    risk_category: RiskLevel
    probability_score: float
    consequence_score: float
    geometry_based_severity: float
    severity_factors: SeverityFactors
    consequence_analysis: ConsequenceAnalysis
    confidence_interval: Tuple[float, float]
    uncertainty_score: float
    recommended_actions: List[str]
    maintenance_priority: int
    estimated_remaining_life: Optional[float]

class GeometryCalculator:
    """Advanced geometry calculations for damage assessment"""
    
    def __init__(self, pixel_to_mm_ratio: float = 1.0):
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
    
    def analyze_damage_geometry(self, contour: np.ndarray, 
                              image_shape: Tuple[int, int]) -> GeometryAnalysis:
        """
        Comprehensive geometric analysis of damage region
        """
        # Basic measurements
        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels * (self.pixel_to_mm_ratio ** 2)
        
        perimeter_pixels = cv2.arcLength(contour, True)
        perimeter_mm = perimeter_pixels * self.pixel_to_mm_ratio
        
        # Bounding rectangle
        rect = cv2.boundingRect(contour)
        bounding_area_pixels = rect[2] * rect[3]
        bounding_area_mm2 = bounding_area_pixels * (self.pixel_to_mm_ratio ** 2)
        
        # Fitted ellipse (if contour has enough points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, (major_axis, minor_axis), angle = ellipse
            major_axis_mm = major_axis * self.pixel_to_mm_ratio
            minor_axis_mm = minor_axis * self.pixel_to_mm_ratio
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            orientation_deg = angle
            centroid = center
        else:
            # Fallback for small contours
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                centroid = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
            else:
                centroid = (rect[0] + rect[2]/2, rect[1] + rect[3]/2)
            
            major_axis_mm = max(rect[2], rect[3]) * self.pixel_to_mm_ratio
            minor_axis_mm = min(rect[2], rect[3]) * self.pixel_to_mm_ratio
            aspect_ratio = major_axis_mm / minor_axis_mm if minor_axis_mm > 0 else 1.0
            orientation_deg = 0.0
        
        # Shape descriptors
        circularity = 4 * math.pi * area_pixels / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area_pixels = cv2.contourArea(hull)
        hull_area_mm2 = hull_area_pixels * (self.pixel_to_mm_ratio ** 2)
        solidity = area_pixels / hull_area_pixels if hull_area_pixels > 0 else 0
        
        # Extent (ratio of contour area to bounding rectangle area)
        extent = area_pixels / bounding_area_pixels if bounding_area_pixels > 0 else 0
        
        return GeometryAnalysis(
            area_mm2=area_mm2,
            perimeter_mm=perimeter_mm,
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            solidity=solidity,
            extent=extent,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            orientation_deg=orientation_deg,
            centroid=centroid,
            bounding_rect_area_mm2=bounding_area_mm2,
            convex_hull_area_mm2=hull_area_mm2
        )
    
    def calculate_stress_concentration_factor(self, geometry: GeometryAnalysis, 
                                            damage_type: DamageType) -> float:
        """
        Calculate stress concentration factor based on damage geometry
        """
        if damage_type == DamageType.CRACK:
            # For cracks, stress concentration depends on crack tip radius and length
            # Simplified model: Kt = 1 + 2*sqrt(a/r) where a is crack length, r is tip radius
            crack_length = geometry.major_axis_mm
            tip_radius = 0.1  # Assume sharp crack tip
            kt = 1 + 2 * math.sqrt(crack_length / tip_radius) if tip_radius > 0 else 3.0
            return min(kt, 10.0)  # Cap at reasonable value
            
        elif damage_type == DamageType.DENT:
            # For dents, stress concentration depends on depth-to-diameter ratio
            # Assume depth is related to aspect ratio
            depth_ratio = 1.0 / geometry.aspect_ratio if geometry.aspect_ratio > 1 else geometry.aspect_ratio
            kt = 1 + 2 * depth_ratio
            return min(kt, 4.0)
            
        elif damage_type == DamageType.CORROSION:
            # For corrosion, stress concentration depends on pit shape
            # Circular pits have lower stress concentration than irregular shapes
            kt = 1 + (1 / geometry.circularity - 1) * 0.5 if geometry.circularity > 0 else 2.0
            return min(kt, 3.0)
            
        else:
            # Default stress concentration for other damage types
            return 1.5
    
    def calculate_location_factor(self, centroid: Tuple[float, float], 
                                image_shape: Tuple[int, int],
                                critical_zones: List[Dict[str, Any]] = None) -> float:
        """
        Calculate location-based risk factor
        """
        image_height, image_width = image_shape
        x, y = centroid
        
        # Base location factor based on position (edges are more critical)
        edge_distance_x = min(x, image_width - x) / (image_width / 2)
        edge_distance_y = min(y, image_height - y) / (image_height / 2)
        edge_factor = 1.0 + (1.0 - min(edge_distance_x, edge_distance_y)) * 0.5
        
        # Critical zones factor
        zone_factor = 1.0
        if critical_zones:
            for zone in critical_zones:
                zone_x, zone_y = zone.get('center', (image_width/2, image_height/2))
                zone_radius = zone.get('radius', image_width * 0.2)
                distance = math.sqrt((x - zone_x)**2 + (y - zone_y)**2)
                
                if distance < zone_radius:
                    zone_severity = zone.get('severity_multiplier', 1.5)
                    zone_factor = max(zone_factor, zone_severity)
        
        return edge_factor * zone_factor

class SeverityCalculator:
    """Calculate damage severity using physics-based models"""
    
    def __init__(self):
        self.geometry_calculator = GeometryCalculator()
        
        # Damage type specific parameters
        self.damage_parameters = {
            DamageType.CORROSION: {
                'growth_rate': 0.1,  # mm/year
                'stress_multiplier': 1.2,
                'fatigue_factor': 1.5
            },
            DamageType.CRACK: {
                'growth_rate': 0.5,  # mm/year
                'stress_multiplier': 3.0,
                'fatigue_factor': 2.5
            },
            DamageType.DENT: {
                'growth_rate': 0.01,  # mm/year
                'stress_multiplier': 1.8,
                'fatigue_factor': 1.3
            },
            DamageType.COATING_LOSS: {
                'growth_rate': 0.05,  # mm/year
                'stress_multiplier': 1.1,
                'fatigue_factor': 1.2
            },
            DamageType.LEAK: {
                'growth_rate': 1.0,  # mm/year
                'stress_multiplier': 2.0,
                'fatigue_factor': 2.0
            },
            DamageType.WEAR: {
                'growth_rate': 0.02,  # mm/year
                'stress_multiplier': 1.3,
                'fatigue_factor': 1.4
            }
        }
    
    def calculate_severity_factors(self, geometry: GeometryAnalysis,
                                 damage_type: DamageType,
                                 image_shape: Tuple[int, int],
                                 material_properties: Dict[str, float] = None,
                                 operating_conditions: Dict[str, float] = None) -> SeverityFactors:
        """
        Calculate comprehensive severity factors
        """
        # Size factor (normalized by typical damage sizes)
        typical_sizes = {
            DamageType.CORROSION: 100.0,  # mm²
            DamageType.CRACK: 50.0,
            DamageType.DENT: 200.0,
            DamageType.COATING_LOSS: 500.0,
            DamageType.LEAK: 25.0,
            DamageType.WEAR: 150.0
        }
        
        typical_size = typical_sizes.get(damage_type, 100.0)
        size_factor = min(geometry.area_mm2 / typical_size, 5.0)  # Cap at 5x typical
        
        # Shape factor based on geometry
        if damage_type == DamageType.CRACK:
            # Long, thin cracks are more severe
            shape_factor = geometry.aspect_ratio * (1.0 - geometry.circularity)
        elif damage_type == DamageType.CORROSION:
            # Irregular corrosion is more severe
            shape_factor = (1.0 - geometry.circularity) * (1.0 - geometry.solidity)
        else:
            # General shape irregularity
            shape_factor = 1.0 - geometry.circularity * geometry.solidity
        
        shape_factor = max(0.1, min(shape_factor, 2.0))  # Bound between 0.1 and 2.0
        
        # Location factor
        location_factor = self.geometry_calculator.calculate_location_factor(
            geometry.centroid, image_shape
        )
        
        # Propagation factor based on damage type
        params = self.damage_parameters.get(damage_type, self.damage_parameters[DamageType.CORROSION])
        propagation_factor = 1.0 + params['growth_rate'] * size_factor
        
        # Stress concentration factor
        stress_concentration_factor = self.geometry_calculator.calculate_stress_concentration_factor(
            geometry, damage_type
        )
        
        # Material factor
        material_factor = 1.0
        if material_properties:
            yield_strength = material_properties.get('yield_strength_mpa', 250.0)
            fatigue_strength = material_properties.get('fatigue_strength_mpa', 150.0)
            # Lower strength materials are more susceptible
            material_factor = 300.0 / yield_strength  # Normalized to typical steel
        
        # Environmental factor
        environmental_factor = 1.0
        if operating_conditions:
            temperature = operating_conditions.get('temperature_c', 20.0)
            humidity = operating_conditions.get('humidity_percent', 50.0)
            pressure = operating_conditions.get('pressure_bar', 1.0)
            
            # High temperature increases risk
            temp_factor = 1.0 + max(0, (temperature - 20) / 100)
            # High humidity increases corrosion risk
            humidity_factor = 1.0 + max(0, (humidity - 50) / 100)
            # High pressure increases stress
            pressure_factor = 1.0 + max(0, (pressure - 1) / 10)
            
            environmental_factor = temp_factor * humidity_factor * pressure_factor
        
        return SeverityFactors(
            size_factor=size_factor,
            shape_factor=shape_factor,
            location_factor=location_factor,
            propagation_factor=propagation_factor,
            stress_concentration_factor=stress_concentration_factor,
            material_factor=material_factor,
            environmental_factor=environmental_factor
        )
    
    def calculate_geometry_based_severity(self, severity_factors: SeverityFactors) -> float:
        """
        Calculate overall geometry-based severity score
        """
        # Weighted combination of severity factors
        weights = {
            'size': 0.25,
            'shape': 0.15,
            'location': 0.15,
            'propagation': 0.20,
            'stress_concentration': 0.15,
            'material': 0.05,
            'environmental': 0.05
        }
        
        severity = (
            weights['size'] * severity_factors.size_factor +
            weights['shape'] * severity_factors.shape_factor +
            weights['location'] * severity_factors.location_factor +
            weights['propagation'] * severity_factors.propagation_factor +
            weights['stress_concentration'] * severity_factors.stress_concentration_factor +
            weights['material'] * severity_factors.material_factor +
            weights['environmental'] * severity_factors.environmental_factor
        )
        
        return min(severity, 10.0)  # Cap at maximum severity of 10

class ConsequenceCalculator:
    """Calculate consequences of damage"""
    
    def __init__(self):
        self.consequence_weights = {
            'safety': 0.30,
            'environmental': 0.20,
            'production': 0.20,
            'equipment': 0.15,
            'cost': 0.10,
            'reputation': 0.05
        }
    
    def calculate_consequences(self, geometry_severity: float,
                            damage_type: DamageType,
                            asset_properties: Dict[str, Any] = None) -> ConsequenceAnalysis:
        """
        Calculate multi-dimensional consequences
        """
        # Default asset properties
        if asset_properties is None:
            asset_properties = {
                'asset_type': 'pressure_vessel',
                'criticality': 'medium',
                'safety_class': 'normal',
                'environmental_sensitivity': 'medium'
            }
        
        base_consequence = geometry_severity / 10.0  # Normalize to 0-1
        
        # Safety consequence
        safety_multipliers = {
            DamageType.CRACK: 3.0,
            DamageType.LEAK: 3.0,
            DamageType.CORROSION: 2.0,
            DamageType.DENT: 1.5,
            DamageType.WEAR: 1.2,
            DamageType.COATING_LOSS: 1.0
        }
        
        safety_consequence = base_consequence * safety_multipliers.get(damage_type, 1.5)
        
        # Adjust for asset safety class
        safety_class = asset_properties.get('safety_class', 'normal')
        if safety_class == 'critical':
            safety_consequence *= 2.0
        elif safety_class == 'high':
            safety_consequence *= 1.5
        
        # Environmental consequence
        environmental_multipliers = {
            DamageType.LEAK: 3.0,
            DamageType.CORROSION: 2.0,
            DamageType.CRACK: 1.5,
            DamageType.COATING_LOSS: 1.2,
            DamageType.DENT: 1.0,
            DamageType.WEAR: 1.0
        }
        
        environmental_consequence = base_consequence * environmental_multipliers.get(damage_type, 1.0)
        
        # Adjust for environmental sensitivity
        env_sensitivity = asset_properties.get('environmental_sensitivity', 'medium')
        if env_sensitivity == 'high':
            environmental_consequence *= 2.0
        elif env_sensitivity == 'critical':
            environmental_consequence *= 3.0
        
        # Production consequence
        production_multipliers = {
            DamageType.LEAK: 2.5,
            DamageType.CRACK: 2.0,
            DamageType.CORROSION: 1.8,
            DamageType.DENT: 1.5,
            DamageType.WEAR: 1.3,
            DamageType.COATING_LOSS: 1.0
        }
        
        production_consequence = base_consequence * production_multipliers.get(damage_type, 1.5)
        
        # Equipment consequence
        equipment_multipliers = {
            DamageType.CRACK: 2.5,
            DamageType.CORROSION: 2.0,
            DamageType.WEAR: 1.8,
            DamageType.DENT: 1.5,
            DamageType.LEAK: 1.3,
            DamageType.COATING_LOSS: 1.0
        }
        
        equipment_consequence = base_consequence * equipment_multipliers.get(damage_type, 1.5)
        
        # Cost consequence (related to repair costs)
        cost_multipliers = {
            DamageType.CRACK: 2.0,
            DamageType.CORROSION: 1.8,
            DamageType.DENT: 1.5,
            DamageType.WEAR: 1.3,
            DamageType.LEAK: 1.2,
            DamageType.COATING_LOSS: 1.0
        }
        
        cost_consequence = base_consequence * cost_multipliers.get(damage_type, 1.5)
        
        # Reputation consequence
        reputation_consequence = base_consequence * 1.2 if damage_type in [DamageType.LEAK, DamageType.CRACK] else base_consequence
        
        return ConsequenceAnalysis(
            safety_consequence=min(safety_consequence, 1.0),
            environmental_consequence=min(environmental_consequence, 1.0),
            production_consequence=min(production_consequence, 1.0),
            equipment_consequence=min(equipment_consequence, 1.0),
            cost_consequence=min(cost_consequence, 1.0),
            reputation_consequence=min(reputation_consequence, 1.0)
        )
    
    def calculate_overall_consequence(self, consequence_analysis: ConsequenceAnalysis) -> float:
        """Calculate weighted overall consequence score"""
        overall = (
            self.consequence_weights['safety'] * consequence_analysis.safety_consequence +
            self.consequence_weights['environmental'] * consequence_analysis.environmental_consequence +
            self.consequence_weights['production'] * consequence_analysis.production_consequence +
            self.consequence_weights['equipment'] * consequence_analysis.equipment_consequence +
            self.consequence_weights['cost'] * consequence_analysis.cost_consequence +
            self.consequence_weights['reputation'] * consequence_analysis.reputation_consequence
        )
        
        return overall

class RiskEngine:
    """Main risk analysis engine"""
    
    def __init__(self, daos: Dict[str, Any]):
        self.daos = daos
        self.severity_calculator = SeverityCalculator()
        self.consequence_calculator = ConsequenceCalculator()
        self.geometry_calculator = GeometryCalculator()
        
        # Load system configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load system configuration"""
        try:
            # Get pixel to mm ratio
            self.pixel_to_mm_ratio = self.daos['system_config'].get_config('pixel_to_mm_ratio') or 1.0
            self.geometry_calculator.pixel_to_mm_ratio = self.pixel_to_mm_ratio
            
            # Get risk calculation weights
            self.risk_weights = self.daos['system_config'].get_config('risk_calculation_weights') or {
                'geometry': 0.4,
                'confidence': 0.3,
                'consequence': 0.3
            }
            
            logger.info("Risk engine configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load configuration, using defaults: {e}")
            self.pixel_to_mm_ratio = 1.0
            self.risk_weights = {'geometry': 0.4, 'confidence': 0.3, 'consequence': 0.3}
    
    async def analyze_risk(self, report_id: str, yolo_results: Dict[str, Any],
                         material_properties: Dict[str, float] = None,
                         operating_conditions: Dict[str, float] = None,
                         asset_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive risk analysis for a damage report
        """
        try:
            logger.info(f"Starting risk analysis for report {report_id}")
            
            detections = yolo_results.get('detections', [])
            image_path = yolo_results.get('image_path')
            
            if not detections:
                return self._create_no_damage_result(report_id)
            
            # Load image for geometry analysis
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_shape = image.shape[:2]
            
            # Analyze each detection
            detection_analyses = []
            total_risk_score = 0.0
            max_severity = 0.0
            confidence_scores = []
            
            for detection in detections:
                analysis = await self._analyze_single_detection(
                    detection, image, image_shape,
                    material_properties, operating_conditions, asset_properties
                )
                detection_analyses.append(analysis)
                total_risk_score += analysis['risk_score']
                max_severity = max(max_severity, analysis['geometry_based_severity'])
                confidence_scores.append(detection['confidence'])
            
            # Calculate overall risk metrics
            avg_confidence = np.mean(confidence_scores)
            overall_geometry_severity = max_severity  # Use maximum severity
            
            # Calculate overall consequence (use worst case)
            worst_consequence = max([a['consequence_score'] for a in detection_analyses])
            
            # Calculate probability score (based on confidence and detection count)
            probability_score = avg_confidence * min(1.0, len(detections) / 3.0)  # Normalize for multiple detections
            
            # Calculate overall risk score using weighted combination
            overall_risk_score = (
                self.risk_weights['geometry'] * overall_geometry_severity +
                self.risk_weights['confidence'] * probability_score +
                self.risk_weights['consequence'] * worst_consequence
            )
            
            # Determine risk category
            risk_category = self._determine_risk_category(overall_risk_score)
            
            # Calculate uncertainty (standard deviation of detection confidences)
            uncertainty_score = np.std(confidence_scores) if len(confidence_scores) > 1 else 0.0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                detection_analyses, risk_category, overall_risk_score
            )
            
            # Calculate maintenance priority
            maintenance_priority = self._calculate_maintenance_priority(
                risk_category, overall_risk_score, len(detections)
            )
            
            # Estimate remaining useful life
            remaining_life = self._estimate_remaining_life(
                detection_analyses, overall_geometry_severity
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                overall_risk_score, uncertainty_score
            )
            
            result = {
                'report_id': report_id,
                'overall_risk_score': round(overall_risk_score, 4),
                'risk_category': risk_category.value,
                'probability_score': round(probability_score, 4),
                'consequence_score': round(worst_consequence, 4),
                'geometry_based_severity': round(overall_geometry_severity, 4),
                'calibrated_confidence': round(avg_confidence, 4),
                'uncertainty_score': round(uncertainty_score, 4),
                'severity_calculation': {
                    'method': 'geometry_based_physics_model',
                    'weights': self.risk_weights,
                    'detection_count': len(detections),
                    'max_geometry_severity': round(max_severity, 4),
                    'avg_confidence': round(avg_confidence, 4),
                    'worst_consequence': round(worst_consequence, 4),
                    'detections': detection_analyses
                },
                'recommended_actions': recommendations,
                'maintenance_priority': maintenance_priority,
                'estimated_remaining_life_days': remaining_life,
                'confidence_interval': confidence_interval
            }
            
            logger.info(f"Risk analysis completed for {report_id}: {risk_category.value} risk")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk analysis for {report_id}: {e}")
            raise
    
    async def _analyze_single_detection(self, detection: Dict[str, Any],
                                      image: np.ndarray, image_shape: Tuple[int, int],
                                      material_properties: Dict[str, float],
                                      operating_conditions: Dict[str, float],
                                      asset_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single detection for risk"""
        
        # Extract bounding box and create contour approximation
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Create rectangular contour for geometry analysis
        contour = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.int32)
        
        # Get damage type
        class_id = detection['class_id']
        damage_types = {
            0: DamageType.CORROSION,
            1: DamageType.DENT,
            2: DamageType.CRACK,
            3: DamageType.COATING_LOSS,
            4: DamageType.LEAK,
            5: DamageType.WEAR
        }
        damage_type = damage_types.get(class_id, DamageType.CORROSION)
        
        # Geometry analysis
        geometry = self.geometry_calculator.analyze_damage_geometry(contour, image_shape)
        
        # Severity calculation
        severity_factors = self.severity_calculator.calculate_severity_factors(
            geometry, damage_type, image_shape, material_properties, operating_conditions
        )
        
        geometry_severity = self.severity_calculator.calculate_geometry_based_severity(severity_factors)
        
        # Consequence analysis
        consequence_analysis = self.consequence_calculator.calculate_consequences(
            geometry_severity, damage_type, asset_properties
        )
        
        overall_consequence = self.consequence_calculator.calculate_overall_consequence(consequence_analysis)
        
        # Risk score for this detection
        confidence = detection['confidence']
        risk_score = (
            0.4 * geometry_severity +
            0.3 * confidence +
            0.3 * overall_consequence
        )
        
        return {
            'class_id': class_id,
            'damage_type': damage_type.value,
            'confidence': confidence,
            'geometry_analysis': {
                'area_mm2': geometry.area_mm2,
                'perimeter_mm': geometry.perimeter_mm,
                'aspect_ratio': geometry.aspect_ratio,
                'circularity': geometry.circularity,
                'major_axis_mm': geometry.major_axis_mm,
                'minor_axis_mm': geometry.minor_axis_mm
            },
            'severity_factors': {
                'size_factor': severity_factors.size_factor,
                'shape_factor': severity_factors.shape_factor,
                'location_factor': severity_factors.location_factor,
                'stress_concentration_factor': severity_factors.stress_concentration_factor
            },
            'geometry_based_severity': geometry_severity,
            'consequence_analysis': {
                'safety': consequence_analysis.safety_consequence,
                'environmental': consequence_analysis.environmental_consequence,
                'production': consequence_analysis.production_consequence,
                'equipment': consequence_analysis.equipment_consequence
            },
            'consequence_score': overall_consequence,
            'risk_score': risk_score
        }
    
    def _create_no_damage_result(self, report_id: str) -> Dict[str, Any]:
        """Create result for case with no damage detected"""
        return {
            'report_id': report_id,
            'overall_risk_score': 0.0,
            'risk_category': RiskLevel.LOW.value,
            'probability_score': 0.0,
            'consequence_score': 0.0,
            'geometry_based_severity': 0.0,
            'calibrated_confidence': 0.0,
            'uncertainty_score': 0.0,
            'severity_calculation': {
                'method': 'no_damage_detected',
                'detection_count': 0
            }
        }
    
    def _determine_risk_category(self, risk_score: float) -> RiskLevel:
        """Determine risk category based on score"""
        if risk_score >= 7.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 5.0:
            return RiskLevel.HIGH
        elif risk_score >= 2.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(self, detection_analyses: List[Dict[str, Any]],
                                risk_category: RiskLevel, 
                                overall_risk_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_category == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Stop operations and perform emergency inspection",
                "Contact qualified inspection personnel immediately",
                "Implement emergency safety protocols",
                "Consider equipment isolation or shutdown"
            ])
        elif risk_category == RiskLevel.HIGH:
            recommendations.extend([
                "Schedule detailed inspection within 24-48 hours",
                "Increase monitoring frequency",
                "Prepare for potential repair activities",
                "Review operating parameters and reduce stress if possible"
            ])
        elif risk_category == RiskLevel.MEDIUM:
            recommendations.extend([
                "Schedule inspection within 1-2 weeks",
                "Monitor damage progression",
                "Plan maintenance during next scheduled shutdown",
                "Consider preventive measures"
            ])
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Include in next scheduled inspection",
                "Document for trend analysis"
            ])
        
        # Add specific recommendations based on damage types
        damage_types_found = set([analysis['damage_type'] for analysis in detection_analyses])
        
        if 'crack' in damage_types_found:
            recommendations.append("Perform crack length measurement and growth monitoring")
        if 'corrosion' in damage_types_found:
            recommendations.append("Consider corrosion protection measures")
        if 'leak' in damage_types_found:
            recommendations.append("Check for source of leakage and implement containment")
        
        return recommendations
    
    def _calculate_maintenance_priority(self, risk_category: RiskLevel,
                                      risk_score: float, detection_count: int) -> int:
        """Calculate maintenance priority (1=highest, 5=lowest)"""
        if risk_category == RiskLevel.CRITICAL:
            return 1
        elif risk_category == RiskLevel.HIGH:
            return 2
        elif risk_category == RiskLevel.MEDIUM:
            return 3 if detection_count > 2 else 4
        else:
            return 5
    
    def _estimate_remaining_life(self, detection_analyses: List[Dict[str, Any]],
                               geometry_severity: float) -> Optional[float]:
        """Estimate remaining useful life in days"""
        if not detection_analyses:
            return None
        
        # Simple model based on damage severity and type
        worst_analysis = max(detection_analyses, key=lambda x: x['geometry_based_severity'])
        damage_type = worst_analysis['damage_type']
        
        # Base remaining life estimates (days)
        base_life = {
            'crack': 30,
            'corrosion': 180,
            'dent': 365,
            'leak': 7,
            'wear': 90,
            'coating_loss': 270
        }
        
        base = base_life.get(damage_type, 180)
        severity_factor = max(0.1, 1.0 - geometry_severity / 10.0)
        
        remaining = base * severity_factor
        return max(1, int(remaining))  # At least 1 day
    
    def _calculate_confidence_interval(self, risk_score: float, 
                                     uncertainty: float) -> Tuple[float, float]:
        """Calculate 95% confidence interval for risk score"""
        margin = 1.96 * uncertainty  # 95% confidence
        lower = max(0.0, risk_score - margin)
        upper = min(10.0, risk_score + margin)
        return (round(lower, 4), round(upper, 4))

# Factory function
def create_risk_engine(daos: Dict[str, Any]) -> RiskEngine:
    """Create risk engine instance"""
    return RiskEngine(daos)

if __name__ == "__main__":
    # Test the risk engine
    logger.info("Risk engine module loaded successfully")
