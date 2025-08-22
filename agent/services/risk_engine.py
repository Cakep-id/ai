"""
Risk Engine untuk CAKEP.id EWS
Menggabungkan hasil CV dan NLP untuk menghasilkan risk assessment
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RiskEngine:
    """Engine untuk menghitung risk assessment dari hasil CV dan NLP"""
    
    def __init__(self):
        # Konfigurasi weights
        self.visual_weight = float(os.getenv('VISUAL_WEIGHT', 0.6))
        self.text_weight = float(os.getenv('TEXT_WEIGHT', 0.4))
        
        # Threshold untuk risk levels
        self.high_threshold = float(os.getenv('HIGH_RISK_THRESHOLD', 0.75))
        self.medium_threshold = float(os.getenv('MEDIUM_RISK_THRESHOLD', 0.45))
        
        # Repair procedure templates
        self.procedure_templates = {
            'structural_damage': {
                'title': 'Prosedur Perbaikan Kerusakan Struktural',
                'base_steps': [
                    'Isolasi area kerja dan pasang safety barrier',
                    'Lakukan assessment detail tingkat kerusakan',
                    'Konsultasi dengan structural engineer',
                    'Siapkan material perbaikan yang sesuai',
                    'Lakukan perbaikan sesuai standar teknis',
                    'Uji integritas struktur setelah perbaikan',
                    'Dokumentasi hasil perbaikan'
                ]
            },
            'fluid_leak': {
                'title': 'Prosedur Perbaikan Kebocoran',
                'base_steps': [
                    'Isolasi sistem dan turunkan tekanan',
                    'Verifikasi tekanan sistem sudah aman',
                    'Identifikasi sumber kebocoran',
                    'Pasang clamp sementara jika diperlukan',
                    'Ganti seal, gasket, atau komponen yang rusak',
                    'Uji tekanan sistem setelah perbaikan',
                    'Monitor kebocoran selama 24 jam'
                ]
            },
            'corrosion': {
                'title': 'Prosedur Penanganan Korosi',
                'base_steps': [
                    'Bersihkan area korosi dari kotoran',
                    'Ukur ketebalan material dengan ultrasonic',
                    'Evaluasi tingkat korosi dan sisa lifetime',
                    'Aplikasikan treatment anti-korosi',
                    'Ganti komponen jika korosi parah',
                    'Dokumentasi kondisi dan treatment'
                ]
            },
            'wear_tear': {
                'title': 'Prosedur Penanganan Keausan',
                'base_steps': [
                    'Inspeksi detail komponen yang aus',
                    'Periksa alignment dan balancing sistem',
                    'Evaluasi penyebab keausan berlebih',
                    'Ganti komponen yang habis pakai',
                    'Adjust parameter operasi',
                    'Lakukan preventive maintenance'
                ]
            },
            'operational': {
                'title': 'Prosedur Troubleshooting Operasional',
                'base_steps': [
                    'Periksa parameter operasi normal',
                    'Lakukan diagnostic sistematis',
                    'Identifikasi akar masalah',
                    'Adjust setting operasi',
                    'Monitor performance berkelanjutan',
                    'Update maintenance schedule'
                ]
            },
            'contamination': {
                'title': 'Prosedur Pembersihan Kontaminasi',
                'base_steps': [
                    'Identifikasi jenis kontaminan',
                    'Siapkan cleaning agent yang sesuai',
                    'Bersihkan kontaminan dengan aman',
                    'Periksa sistem filtrasi',
                    'Lakukan flushing sistem',
                    'Verifikasi kebersihan sistem'
                ]
            }
        }
        
        # Asset criticality multipliers
        self.criticality_multipliers = {
            'HIGH': 1.3,
            'MEDIUM': 1.0,
            'LOW': 0.8
        }
        
        logger.info(f"Risk Engine initialized - Visual weight: {self.visual_weight}, Text weight: {self.text_weight}")
    
    def aggregate_risk(self, report_id: int, cv_results: Dict, nlp_results: Dict, 
                      asset_criticality: str = 'MEDIUM') -> Dict[str, Any]:
        """
        Agregasi risk assessment dari hasil CV dan NLP
        
        Args:
            report_id: ID report
            cv_results: Hasil deteksi YOLO
            nlp_results: Hasil analisis NLP
            asset_criticality: Tingkat kritis aset (HIGH/MEDIUM/LOW)
        
        Returns:
            Dict dengan risk assessment lengkap
        """
        try:
            logger.info(f"Starting risk aggregation for report {report_id}")
            
            # Extract scores
            visual_score = self._extract_visual_score(cv_results)
            text_score = self._extract_text_score(nlp_results)
            
            # Calculate final risk score
            final_score = self._calculate_final_score(
                visual_score, 
                text_score, 
                asset_criticality
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_score)
            
            # Generate rationale
            rationale = self._generate_rationale(cv_results, nlp_results, final_score)
            
            # Generate repair procedures
            procedures = self._generate_procedures(cv_results, nlp_results, risk_level)
            
            # Calculate impact metrics
            impact_metrics = self._calculate_impact_metrics(
                risk_level, 
                asset_criticality,
                cv_results,
                nlp_results
            )
            
            result = {
                'report_id': report_id,
                'risk_score': round(final_score, 3),
                'risk_level': risk_level,
                'visual_score': round(visual_score, 3),
                'text_score': round(text_score, 3),
                'rationale': rationale,
                'procedures': procedures,
                'impact_metrics': impact_metrics,
                'asset_criticality': asset_criticality,
                'calculated_at': datetime.now().isoformat(),
                'engine_version': self._get_engine_version()
            }
            
            logger.info(f"Risk aggregation completed: {risk_level} risk (score: {final_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Risk aggregation failed: {e}")
            return {
                'report_id': report_id,
                'risk_score': 0.5,
                'risk_level': 'MEDIUM',
                'visual_score': 0.0,
                'text_score': 0.0,
                'rationale': f"Error in risk calculation: {str(e)}",
                'procedures': [],
                'error': str(e)
            }
    
    def _extract_visual_score(self, cv_results: Dict) -> float:
        """Extract visual score dari hasil YOLO"""
        if not cv_results or not cv_results.get('success'):
            return 0.0
        
        detections = cv_results.get('detections', [])
        if not detections:
            return 0.0
        
        # Ambil detection dengan confidence tertinggi
        max_confidence = max(d.get('confidence', 0) for d in detections)
        
        # Cari detection dengan risk score tertinggi
        risk_scores = [d.get('risk_score', d.get('confidence', 0) * 0.5) for d in detections]
        max_risk_score = max(risk_scores) if risk_scores else 0.0
        
        # Gabungkan confidence dan risk score
        visual_score = (max_confidence * 0.4) + (max_risk_score * 0.6)
        
        return min(visual_score, 1.0)
    
    def _extract_text_score(self, nlp_results: Dict) -> float:
        """Extract text score dari hasil NLP"""
        if not nlp_results or not nlp_results.get('success'):
            return 0.0
        
        base_confidence = nlp_results.get('confidence', 0.0)
        severity = nlp_results.get('severity', 'medium')
        category = nlp_results.get('category', 'unknown')
        
        # Severity multiplier
        severity_multiplier = {
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }.get(severity, 1.0)
        
        # Category risk mapping
        category_risk_map = {
            'structural': 0.9,
            'fluid_leak': 0.85,
            'corrosion': 0.7,
            'wear_tear': 0.6,
            'operational': 0.5,
            'contamination': 0.3
        }
        
        category_risk = category_risk_map.get(category, 0.5)
        
        # Calculate text score
        text_score = base_confidence * severity_multiplier * category_risk
        
        return min(text_score, 1.0)
    
    def _calculate_final_score(self, visual_score: float, text_score: float, 
                              asset_criticality: str) -> float:
        """Hitung final risk score"""
        
        # Weighted average
        base_score = (visual_score * self.visual_weight) + (text_score * self.text_weight)
        
        # Apply asset criticality multiplier
        criticality_multiplier = self.criticality_multipliers.get(asset_criticality, 1.0)
        final_score = base_score * criticality_multiplier
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, final_score))
    
    def _determine_risk_level(self, final_score: float) -> str:
        """Tentukan risk level berdasarkan score"""
        if final_score >= self.high_threshold:
            return 'HIGH'
        elif final_score >= self.medium_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_rationale(self, cv_results: Dict, nlp_results: Dict, 
                           final_score: float) -> str:
        """Generate rationale untuk risk assessment"""
        
        rationale_parts = []
        
        # Visual analysis part
        if cv_results and cv_results.get('success'):
            detections = cv_results.get('detections', [])
            if detections:
                top_detection = max(detections, key=lambda x: x.get('confidence', 0))
                rationale_parts.append(
                    f"YOLO: {top_detection.get('label', 'unknown')} "
                    f"conf={top_detection.get('confidence', 0):.2f}"
                )
        
        # Text analysis part
        if nlp_results and nlp_results.get('success'):
            category = nlp_results.get('category', 'unknown')
            confidence = nlp_results.get('confidence', 0)
            keyphrases = nlp_results.get('keyphrases', [])
            
            rationale_parts.append(f"NLP: kategori '{category}' conf={confidence:.2f}")
            
            if keyphrases:
                key_words = ', '.join(keyphrases[:3])  # Top 3 keyphrases
                rationale_parts.append(f"kata kunci: {key_words}")
        
        # Final assessment
        risk_desc = {
            'HIGH': 'risiko tinggi memerlukan tindakan segera',
            'MEDIUM': 'risiko sedang perlu monitoring',
            'LOW': 'risiko rendah untuk maintenance rutin'
        }
        
        final_desc = risk_desc.get(self._determine_risk_level(final_score), 'perlu evaluasi')
        rationale_parts.append(f"Assessment: {final_desc}")
        
        return '; '.join(rationale_parts)
    
    def _generate_procedures(self, cv_results: Dict, nlp_results: Dict, 
                           risk_level: str) -> List[Dict[str, Any]]:
        """Generate repair procedures berdasarkan analisis"""
        
        procedures = []
        
        # Tentukan template berdasarkan hasil analisis
        primary_template = self._select_primary_template(cv_results, nlp_results)
        
        if primary_template:
            template_info = self.procedure_templates[primary_template]
            
            # Customize steps berdasarkan risk level
            customized_steps = self._customize_procedure_steps(
                template_info['base_steps'].copy(),
                risk_level,
                cv_results,
                nlp_results
            )
            
            procedure = {
                'title': template_info['title'],
                'steps': customized_steps,
                'priority': risk_level,
                'estimated_duration': self._estimate_duration(len(customized_steps), risk_level),
                'required_skills': self._get_required_skills(primary_template),
                'safety_notes': self._get_safety_notes(primary_template, risk_level)
            }
            
            procedures.append(procedure)
        
        # Tambahkan procedure umum jika diperlukan
        if risk_level == 'HIGH':
            procedures.append(self._get_emergency_procedure())
        
        return procedures
    
    def _select_primary_template(self, cv_results: Dict, nlp_results: Dict) -> Optional[str]:
        """Pilih template procedure utama"""
        
        # Priority berdasarkan NLP category
        if nlp_results and nlp_results.get('success'):
            nlp_category = nlp_results.get('category', '')
            
            category_template_map = {
                'structural': 'structural_damage',
                'fluid_leak': 'fluid_leak',
                'corrosion': 'corrosion',
                'wear_tear': 'wear_tear',
                'operational': 'operational',
                'contamination': 'contamination'
            }
            
            template = category_template_map.get(nlp_category)
            if template:
                return template
        
        # Fallback berdasarkan CV results
        if cv_results and cv_results.get('success'):
            detections = cv_results.get('detections', [])
            if detections:
                top_label = detections[0].get('original_label', '').lower()
                
                if any(word in top_label for word in ['crack', 'fracture']):
                    return 'structural_damage'
                elif any(word in top_label for word in ['leak', 'fluid']):
                    return 'fluid_leak'
                elif any(word in top_label for word in ['corrosion', 'rust']):
                    return 'corrosion'
                elif any(word in top_label for word in ['wear', 'erosion']):
                    return 'wear_tear'
        
        return 'operational'  # Default fallback
    
    def _customize_procedure_steps(self, base_steps: List[str], risk_level: str,
                                  cv_results: Dict, nlp_results: Dict) -> List[str]:
        """Customize procedure steps berdasarkan kondisi spesifik"""
        
        customized_steps = base_steps.copy()
        
        # Tambahkan safety steps untuk HIGH risk
        if risk_level == 'HIGH':
            safety_step = 'CRITICAL: Pastikan semua safety protocol diikuti dan dapatkan approval supervisor'
            customized_steps.insert(0, safety_step)
            
            # Tambahkan response time requirement
            time_step = 'Target penyelesaian: maksimal 24 jam dari deteksi'
            customized_steps.append(time_step)
        
        # Tambahkan spesific steps berdasarkan deteksi
        if cv_results and cv_results.get('detections'):
            detection_count = len(cv_results['detections'])
            if detection_count > 1:
                multi_step = f'Perhatian: terdeteksi {detection_count} area masalah, prioritaskan yang paling kritis'
                customized_steps.insert(1, multi_step)
        
        # Tambahkan monitoring steps berdasarkan severity
        if nlp_results and nlp_results.get('severity') == 'high':
            monitor_step = 'Lakukan monitoring kontinyu setiap 2 jam selama 24 jam pertama'
            customized_steps.append(monitor_step)
        
        return customized_steps
    
    def _estimate_duration(self, step_count: int, risk_level: str) -> str:
        """Estimasi durasi perbaikan"""
        base_hours = step_count * 0.5  # 30 menit per step average
        
        risk_multiplier = {
            'HIGH': 1.5,    # Lebih hati-hati
            'MEDIUM': 1.0,
            'LOW': 0.8      # Lebih cepat
        }.get(risk_level, 1.0)
        
        total_hours = base_hours * risk_multiplier
        
        if total_hours < 1:
            return "< 1 jam"
        elif total_hours < 8:
            return f"{total_hours:.1f} jam"
        elif total_hours < 24:
            return f"{total_hours:.0f} jam"
        else:
            days = total_hours / 8  # 8 jam kerja per hari
            return f"{days:.1f} hari kerja"
    
    def _get_required_skills(self, template: str) -> List[str]:
        """Dapatkan skill yang diperlukan"""
        skill_map = {
            'structural_damage': ['structural engineer', 'welder', 'safety inspector'],
            'fluid_leak': ['mechanical technician', 'pipe fitter', 'pressure tester'],
            'corrosion': ['corrosion specialist', 'surface treatment', 'coating applicator'],
            'wear_tear': ['mechanical engineer', 'alignment specialist', 'vibration analyst'],
            'operational': ['operations engineer', 'control system technician'],
            'contamination': ['chemical specialist', 'cleaning technician']
        }
        
        return skill_map.get(template, ['maintenance technician'])
    
    def _get_safety_notes(self, template: str, risk_level: str) -> List[str]:
        """Dapatkan catatan safety"""
        base_safety = ['Gunakan APD lengkap', 'Ikuti LOTO procedure']
        
        template_safety = {
            'structural_damage': ['Waspadai kemungkinan collapse', 'Evakuasi area sekitar'],
            'fluid_leak': ['Isolasi sistem bertekanan', 'Waspadai zat berbahaya'],
            'corrosion': ['Ventilasi yang memadai', 'Hindari kontak langsung'],
            'operational': ['Monitor parameter safety', 'Siaga emergency shutdown']
        }
        
        safety_notes = base_safety + template_safety.get(template, [])
        
        if risk_level == 'HIGH':
            safety_notes.append('WAJIB: Koordinasi dengan safety supervisor')
            safety_notes.append('Siapkan emergency response team')
        
        return safety_notes
    
    def _get_emergency_procedure(self) -> Dict[str, Any]:
        """Dapatkan emergency procedure untuk HIGH risk"""
        return {
            'title': 'Prosedur Emergency Response',
            'steps': [
                'Isolasi area dan evakuasi personel non-essential',
                'Aktivasi emergency response team',
                'Lakukan emergency shutdown jika diperlukan',
                'Hubungi management dan safety department',
                'Dokumentasi kondisi emergency',
                'Lakukan temporary fix untuk stabilisasi',
                'Koordinasi dengan vendor expert jika diperlukan'
            ],
            'priority': 'CRITICAL',
            'estimated_duration': '2-4 jam',
            'required_skills': ['emergency coordinator', 'safety officer', 'senior engineer'],
            'safety_notes': [
                'PRIORITAS UTAMA: Keselamatan personel',
                'Jangan bekerja sendirian',
                'Komunikasi real-time dengan control room'
            ]
        }
    
    def _calculate_impact_metrics(self, risk_level: str, asset_criticality: str,
                                 cv_results: Dict, nlp_results: Dict) -> Dict[str, Any]:
        """Hitung impact metrics untuk decision making"""
        
        # Base impact berdasarkan risk level
        impact_base = {
            'HIGH': {'safety': 0.9, 'operational': 0.8, 'financial': 0.7},
            'MEDIUM': {'safety': 0.6, 'operational': 0.5, 'financial': 0.4},
            'LOW': {'safety': 0.3, 'operational': 0.2, 'financial': 0.2}
        }.get(risk_level, {'safety': 0.5, 'operational': 0.3, 'financial': 0.3})
        
        # Asset criticality multiplier
        crit_multiplier = self.criticality_multipliers.get(asset_criticality, 1.0)
        
        # Calculate impacts
        safety_impact = min(impact_base['safety'] * crit_multiplier, 1.0)
        operational_impact = min(impact_base['operational'] * crit_multiplier, 1.0)
        financial_impact = min(impact_base['financial'] * crit_multiplier, 1.0)
        
        # Estimate downtime (hours)
        downtime_map = {
            'HIGH': 24,
            'MEDIUM': 8,
            'LOW': 2
        }
        estimated_downtime = downtime_map.get(risk_level, 8) * crit_multiplier
        
        return {
            'safety_impact': round(safety_impact, 2),
            'operational_impact': round(operational_impact, 2),
            'financial_impact': round(financial_impact, 2),
            'estimated_downtime_hours': round(estimated_downtime, 1),
            'business_continuity_risk': self._assess_business_continuity(
                operational_impact, asset_criticality
            )
        }
    
    def _assess_business_continuity(self, operational_impact: float, 
                                   asset_criticality: str) -> str:
        """Assess business continuity risk"""
        if asset_criticality == 'HIGH' and operational_impact > 0.7:
            return 'CRITICAL - May halt operations'
        elif operational_impact > 0.5:
            return 'SIGNIFICANT - Reduced capacity'
        elif operational_impact > 0.3:
            return 'MODERATE - Minor disruption'
        else:
            return 'MINIMAL - Normal operations'
    
    def _get_engine_version(self) -> str:
        """Get risk engine version"""
        return f"RiskEngine_v1.0_{datetime.now().strftime('%Y%m%d')}"
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current risk engine configuration"""
        return {
            'visual_weight': self.visual_weight,
            'text_weight': self.text_weight,
            'high_threshold': self.high_threshold,
            'medium_threshold': self.medium_threshold,
            'criticality_multipliers': self.criticality_multipliers,
            'available_templates': list(self.procedure_templates.keys()),
            'engine_version': self._get_engine_version()
        }

# Singleton instance
risk_engine = RiskEngine()

if __name__ == "__main__":
    # Test risk engine
    print("Risk Engine Test")
    print("Configuration:", json.dumps(risk_engine.get_configuration(), indent=2))
    
    # Sample test data
    sample_cv = {
        'success': True,
        'detections': [
            {'label': 'crack', 'confidence': 0.85, 'risk_score': 0.75}
        ]
    }
    
    sample_nlp = {
        'success': True,
        'category': 'structural',
        'confidence': 0.8,
        'severity': 'high',
        'keyphrases': ['retak', 'parah']
    }
    
    result = risk_engine.aggregate_risk(123, sample_cv, sample_nlp, 'HIGH')
    print("Sample Risk Assessment:", json.dumps(result, indent=2, ensure_ascii=False))
