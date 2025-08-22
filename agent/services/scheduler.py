"""
Scheduler Service untuk CAKEP.id EWS
Auto generate jadwal pemeliharaan berdasarkan risk assessment
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SchedulerService:
    """Service untuk generate dan manage jadwal pemeliharaan"""
    
    def __init__(self):
        # SLA configuration (dalam jam)
        self.sla_hours = {
            'HIGH': int(os.getenv('HIGH_RISK_SLA', 24)),      # 24 jam
            'MEDIUM': int(os.getenv('MEDIUM_RISK_SLA', 72)),  # 72 jam  
            'LOW': int(os.getenv('LOW_RISK_SLA', 168))        # 168 jam (7 hari)
        }
        
        # Follow-up frequencies (dalam hari)
        self.followup_frequencies = {
            'HIGH': 7,      # Setiap 7 hari
            'MEDIUM': 14,   # Setiap 14 hari
            'LOW': 30       # Setiap 30 hari
        }
        
        # Work schedule configuration
        self.work_hours = {
            'start': 7,     # 07:00
            'end': 17       # 17:00
        }
        
        # Schedule type mapping berdasarkan risk dan kategori
        self.schedule_type_mapping = {
            'HIGH': {
                'structural': 'PERMANENT_FIX',
                'fluid_leak': 'TEMP_FIX',
                'corrosion': 'PERMANENT_FIX',
                'operational': 'INSPECTION',
                'default': 'TEMP_FIX'
            },
            'MEDIUM': {
                'structural': 'PERMANENT_FIX',
                'fluid_leak': 'PERMANENT_FIX',
                'corrosion': 'PERMANENT_FIX',
                'wear_tear': 'PREVENTIVE',
                'operational': 'INSPECTION',
                'default': 'INSPECTION'
            },
            'LOW': {
                'wear_tear': 'PREVENTIVE',
                'contamination': 'PREVENTIVE',
                'operational': 'INSPECTION',
                'default': 'PREVENTIVE'
            }
        }
        
        # Priority weights untuk scheduling conflicts
        self.priority_weights = {
            'HIGH': 100,
            'MEDIUM': 50,
            'LOW': 20
        }
        
        logger.info("Scheduler Service initialized")
    
    def generate_schedule(self, report_id: int, asset_id: int, risk_assessment: Dict,
                         asset_criticality: str = 'MEDIUM') -> Dict[str, Any]:
        """
        Generate jadwal pemeliharaan berdasarkan risk assessment
        
        Args:
            report_id: ID report
            asset_id: ID asset
            risk_assessment: Hasil risk assessment
            asset_criticality: Tingkat kritikalitas asset
        
        Returns:
            Dict dengan jadwal yang digenerate
        """
        try:
            logger.info(f"Generating schedule for report {report_id}, asset {asset_id}")
            
            risk_level = risk_assessment.get('risk_level', 'MEDIUM')
            risk_score = risk_assessment.get('risk_score', 0.5)
            
            # Determine schedule type
            schedule_type = self._determine_schedule_type(risk_assessment)
            
            # Calculate due date
            due_date = self._calculate_due_date(risk_level, asset_criticality)
            
            # Determine frequency for follow-up
            frequency_days = self._determine_frequency(risk_level, schedule_type)
            
            # Generate work order details
            work_order = self._generate_work_order(
                asset_id,
                risk_assessment,
                schedule_type,
                due_date
            )
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(
                risk_level,
                risk_score,
                asset_criticality,
                due_date
            )
            
            # Estimate resources needed
            resources = self._estimate_resources(risk_assessment, schedule_type)
            
            schedule_data = {
                'asset_id': asset_id,
                'report_id': report_id,
                'type': schedule_type,
                'due_date': due_date.isoformat(),
                'frequency_days': frequency_days,
                'status': 'PLANNED',
                'priority_score': priority_score,
                'work_order': work_order,
                'resources': resources,
                'sla_deadline': due_date.isoformat(),
                'risk_context': {
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'asset_criticality': asset_criticality
                },
                'created_at': datetime.now().isoformat(),
                'scheduler_version': self._get_scheduler_version()
            }
            
            # Generate follow-up schedules jika diperlukan
            if frequency_days and frequency_days > 0:
                followup_schedules = self._generate_followup_schedules(
                    asset_id,
                    report_id,
                    due_date,
                    frequency_days,
                    risk_level
                )
                schedule_data['followup_schedules'] = followup_schedules
            
            logger.info(f"Schedule generated: {schedule_type} due {due_date.strftime('%Y-%m-%d %H:%M')}")
            return schedule_data
            
        except Exception as e:
            logger.error(f"Failed to generate schedule: {e}")
            return {
                'error': str(e),
                'asset_id': asset_id,
                'report_id': report_id,
                'status': 'ERROR'
            }
    
    def _determine_schedule_type(self, risk_assessment: Dict) -> str:
        """Tentukan tipe schedule berdasarkan risk assessment"""
        
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        
        # Cari kategori dari NLP results atau procedures
        category = 'default'
        procedures = risk_assessment.get('procedures', [])
        
        if procedures:
            # Extract category dari procedure title
            title = procedures[0].get('title', '').lower()
            if 'struktural' in title or 'structural' in title:
                category = 'structural'
            elif 'kebocoran' in title or 'leak' in title:
                category = 'fluid_leak'
            elif 'korosi' in title or 'corrosion' in title:
                category = 'corrosion'
            elif 'keausan' in title or 'wear' in title:
                category = 'wear_tear'
            elif 'operasional' in title or 'operational' in title:
                category = 'operational'
            elif 'kontaminasi' in title or 'contamination' in title:
                category = 'contamination'
        
        # Get schedule type dari mapping
        risk_mapping = self.schedule_type_mapping.get(risk_level, {})
        schedule_type = risk_mapping.get(category, risk_mapping.get('default', 'INSPECTION'))
        
        return schedule_type
    
    def _calculate_due_date(self, risk_level: str, asset_criticality: str) -> datetime:
        """Hitung due date berdasarkan SLA"""
        
        base_hours = self.sla_hours.get(risk_level, 72)
        
        # Adjust berdasarkan asset criticality
        criticality_multiplier = {
            'HIGH': 0.8,    # Lebih cepat
            'MEDIUM': 1.0,
            'LOW': 1.2      # Lebih lambat
        }.get(asset_criticality, 1.0)
        
        adjusted_hours = base_hours * criticality_multiplier
        
        # Calculate due date dari sekarang
        now = datetime.now()
        due_date = now + timedelta(hours=adjusted_hours)
        
        # Adjust ke jam kerja jika bukan HIGH risk
        if risk_level != 'HIGH':
            due_date = self._adjust_to_work_hours(due_date)
        
        return due_date
    
    def _adjust_to_work_hours(self, target_date: datetime) -> datetime:
        """Adjust tanggal ke jam kerja normal"""
        
        # Jika weekend, pindah ke Senin
        if target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            days_to_monday = 7 - target_date.weekday()
            target_date = target_date + timedelta(days=days_to_monday)
        
        # Adjust jam ke jam kerja
        if target_date.hour < self.work_hours['start']:
            target_date = target_date.replace(hour=self.work_hours['start'], minute=0)
        elif target_date.hour >= self.work_hours['end']:
            # Pindah ke hari kerja berikutnya
            target_date = target_date + timedelta(days=1)
            target_date = target_date.replace(hour=self.work_hours['start'], minute=0)
            
            # Check lagi kalau jadi weekend
            if target_date.weekday() >= 5:
                days_to_monday = 7 - target_date.weekday()
                target_date = target_date + timedelta(days=days_to_monday)
        
        return target_date
    
    def _determine_frequency(self, risk_level: str, schedule_type: str) -> Optional[int]:
        """Tentukan frequency untuk follow-up inspections"""
        
        # Tidak perlu follow-up untuk TEMP_FIX
        if schedule_type == 'TEMP_FIX':
            return None
        
        base_frequency = self.followup_frequencies.get(risk_level, 14)
        
        # Adjust berdasarkan schedule type
        type_multiplier = {
            'INSPECTION': 1.0,
            'PREVENTIVE': 2.0,      # Lebih jarang
            'PERMANENT_FIX': 0.5    # Lebih sering monitoring
        }.get(schedule_type, 1.0)
        
        frequency = int(base_frequency * type_multiplier)
        
        return max(frequency, 1)  # Minimal 1 hari
    
    def _generate_work_order(self, asset_id: int, risk_assessment: Dict,
                           schedule_type: str, due_date: datetime) -> Dict[str, Any]:
        """Generate work order details"""
        
        procedures = risk_assessment.get('procedures', [])
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        
        # Basic work order info
        work_order = {
            'title': self._generate_work_order_title(schedule_type, procedures),
            'description': self._generate_work_order_description(risk_assessment),
            'type': schedule_type,
            'priority': risk_level,
            'estimated_duration': self._estimate_work_duration(procedures, schedule_type),
            'due_date': due_date.isoformat(),
            'safety_requirements': self._get_safety_requirements(risk_assessment),
            'tools_required': self._get_required_tools(procedures),
            'parts_required': self._get_required_parts(risk_assessment),
            'skills_required': self._get_required_skills(procedures)
        }
        
        return work_order
    
    def _generate_work_order_title(self, schedule_type: str, procedures: List[Dict]) -> str:
        """Generate work order title"""
        
        type_titles = {
            'INSPECTION': 'Inspeksi',
            'TEMP_FIX': 'Perbaikan Sementara',
            'PERMANENT_FIX': 'Perbaikan Permanen',
            'PREVENTIVE': 'Maintenance Preventif'
        }
        
        base_title = type_titles.get(schedule_type, 'Maintenance')
        
        if procedures and len(procedures) > 0:
            procedure_type = procedures[0].get('title', '')
            if procedure_type:
                # Extract jenis perbaikan dari title
                if 'struktural' in procedure_type.lower():
                    base_title += ' - Struktural'
                elif 'kebocoran' in procedure_type.lower():
                    base_title += ' - Sistem Fluida'
                elif 'korosi' in procedure_type.lower():
                    base_title += ' - Korosi'
                elif 'keausan' in procedure_type.lower():
                    base_title += ' - Keausan'
                elif 'operasional' in procedure_type.lower():
                    base_title += ' - Operasional'
        
        return base_title
    
    def _generate_work_order_description(self, risk_assessment: Dict) -> str:
        """Generate work order description"""
        
        rationale = risk_assessment.get('rationale', '')
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        risk_score = risk_assessment.get('risk_score', 0.5)
        
        description = f"Risk Level: {risk_level} (Score: {risk_score:.2f})\n"
        description += f"Assessment: {rationale}\n\n"
        
        # Tambahkan procedure steps jika ada
        procedures = risk_assessment.get('procedures', [])
        if procedures:
            main_procedure = procedures[0]
            steps = main_procedure.get('steps', [])
            if steps:
                description += "Langkah Perbaikan:\n"
                for i, step in enumerate(steps[:5], 1):  # Max 5 steps
                    description += f"{i}. {step}\n"
                
                if len(steps) > 5:
                    description += f"... dan {len(steps) - 5} langkah lainnya\n"
        
        return description
    
    def _estimate_work_duration(self, procedures: List[Dict], schedule_type: str) -> str:
        """Estimasi durasi pekerjaan"""
        
        if procedures and len(procedures) > 0:
            # Ambil estimasi dari procedure
            estimated = procedures[0].get('estimated_duration', '')
            if estimated:
                return estimated
        
        # Default estimasi berdasarkan type
        duration_map = {
            'INSPECTION': '2-4 jam',
            'TEMP_FIX': '4-8 jam',
            'PERMANENT_FIX': '1-3 hari',
            'PREVENTIVE': '4-6 jam'
        }
        
        return duration_map.get(schedule_type, '4 jam')
    
    def _get_safety_requirements(self, risk_assessment: Dict) -> List[str]:
        """Dapatkan safety requirements"""
        
        basic_safety = ['APD lengkap', 'LOTO procedure']
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        
        if risk_level == 'HIGH':
            basic_safety.extend([
                'Safety supervisor approval',
                'Emergency response team standby',
                'Area isolation'
            ])
        
        # Tambahkan dari procedures jika ada
        procedures = risk_assessment.get('procedures', [])
        if procedures:
            for procedure in procedures:
                safety_notes = procedure.get('safety_notes', [])
                basic_safety.extend(safety_notes)
        
        return list(set(basic_safety))  # Remove duplicates
    
    def _get_required_tools(self, procedures: List[Dict]) -> List[str]:
        """Dapatkan tools yang diperlukan"""
        
        basic_tools = ['Basic hand tools', 'Measuring instruments']
        
        # Tools berdasarkan procedure type
        for procedure in procedures:
            title = procedure.get('title', '').lower()
            
            if 'struktural' in title:
                basic_tools.extend(['Welding equipment', 'Structural measurement tools'])
            elif 'kebocoran' in title:
                basic_tools.extend(['Pressure testing kit', 'Pipe repair tools'])
            elif 'korosi' in title:
                basic_tools.extend(['Surface preparation tools', 'Coating equipment'])
            elif 'keausan' in title:
                basic_tools.extend(['Alignment tools', 'Vibration meter'])
        
        return list(set(basic_tools))
    
    def _get_required_parts(self, risk_assessment: Dict) -> List[str]:
        """Estimasi parts yang mungkin diperlukan"""
        
        parts = []
        procedures = risk_assessment.get('procedures', [])
        
        for procedure in procedures:
            title = procedure.get('title', '').lower()
            
            if 'kebocoran' in title:
                parts.extend(['Gasket', 'Seal', 'O-ring'])
            elif 'korosi' in title:
                parts.extend(['Anti-corrosion coating', 'Metal patch'])
            elif 'keausan' in title:
                parts.extend(['Bearing', 'Coupling', 'Filter'])
        
        return list(set(parts)) if parts else ['TBD - akan ditentukan saat inspeksi']
    
    def _get_required_skills(self, procedures: List[Dict]) -> List[str]:
        """Dapatkan skill requirements"""
        
        skills = set(['Maintenance technician'])
        
        for procedure in procedures:
            required_skills = procedure.get('required_skills', [])
            skills.update(required_skills)
        
        return list(skills)
    
    def _calculate_priority_score(self, risk_level: str, risk_score: float,
                                asset_criticality: str, due_date: datetime) -> int:
        """Hitung priority score untuk scheduling"""
        
        # Base priority dari risk level
        base_priority = self.priority_weights.get(risk_level, 50)
        
        # Risk score multiplier
        risk_multiplier = risk_score  # 0.0 - 1.0
        
        # Asset criticality multiplier
        asset_multiplier = {
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.7
        }.get(asset_criticality, 1.0)
        
        # Time urgency (semakin dekat due date, semakin tinggi priority)
        now = datetime.now()
        hours_until_due = (due_date - now).total_seconds() / 3600
        
        if hours_until_due <= 0:
            time_multiplier = 2.0  # Overdue
        elif hours_until_due <= 24:
            time_multiplier = 1.5  # Very urgent
        elif hours_until_due <= 72:
            time_multiplier = 1.2  # Urgent
        else:
            time_multiplier = 1.0   # Normal
        
        # Calculate final priority score
        priority_score = int(base_priority * risk_multiplier * asset_multiplier * time_multiplier)
        
        return max(1, min(1000, priority_score))  # Clamp between 1-1000
    
    def _estimate_resources(self, risk_assessment: Dict, schedule_type: str) -> Dict[str, Any]:
        """Estimasi resources yang diperlukan"""
        
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        procedures = risk_assessment.get('procedures', [])
        
        # Base resource requirements
        resources = {
            'technicians_required': 1,
            'supervisor_required': False,
            'specialist_required': False,
            'equipment_required': ['basic_tools'],
            'estimated_cost': 'TBD'
        }
        
        # Adjust berdasarkan risk level
        if risk_level == 'HIGH':
            resources['technicians_required'] = 2
            resources['supervisor_required'] = True
            resources['specialist_required'] = True
        elif risk_level == 'MEDIUM':
            resources['technicians_required'] = 1
            resources['supervisor_required'] = False
        
        # Adjust berdasarkan schedule type
        if schedule_type == 'PERMANENT_FIX':
            resources['technicians_required'] += 1
            resources['specialist_required'] = True
        
        # Add equipment dari procedures
        for procedure in procedures:
            title = procedure.get('title', '').lower()
            if 'struktural' in title:
                resources['equipment_required'].append('welding_equipment')
            elif 'kebocoran' in title:
                resources['equipment_required'].append('pressure_test_kit')
        
        return resources
    
    def _generate_followup_schedules(self, asset_id: int, report_id: int,
                                   initial_due_date: datetime, frequency_days: int,
                                   risk_level: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate jadwal follow-up inspections"""
        
        followup_schedules = []
        
        for i in range(1, count + 1):
            followup_date = initial_due_date + timedelta(days=frequency_days * i)
            followup_date = self._adjust_to_work_hours(followup_date)
            
            followup = {
                'asset_id': asset_id,
                'report_id': report_id,
                'type': 'INSPECTION',
                'due_date': followup_date.isoformat(),
                'frequency_days': frequency_days,
                'status': 'PLANNED',
                'sequence': i,
                'title': f'Follow-up Inspection #{i}',
                'description': f'Scheduled follow-up inspection berdasarkan {risk_level} risk assessment',
                'estimated_duration': '1-2 jam'
            }
            
            followup_schedules.append(followup)
        
        return followup_schedules
    
    def update_schedule_status(self, schedule_id: int, new_status: str,
                             completion_notes: str = None) -> Dict[str, Any]:
        """Update status schedule"""
        
        valid_statuses = ['PLANNED', 'IN_PROGRESS', 'DONE', 'CANCELLED']
        
        if new_status not in valid_statuses:
            return {
                'success': False,
                'error': f'Invalid status. Must be one of: {valid_statuses}'
            }
        
        update_data = {
            'schedule_id': schedule_id,
            'status': new_status,
            'updated_at': datetime.now().isoformat()
        }
        
        if completion_notes:
            update_data['completion_notes'] = completion_notes
        
        if new_status == 'DONE':
            update_data['completed_at'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'update_data': update_data
        }
    
    def get_overdue_schedules(self, hours_overdue: int = 0) -> Dict[str, Any]:
        """Dapatkan jadwal yang overdue"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_overdue)
        
        # Ini akan digunakan dengan database query
        query_info = {
            'cutoff_time': cutoff_time.isoformat(),
            'statuses': ['PLANNED', 'IN_PROGRESS'],
            'order_by': 'priority_score DESC, due_date ASC'
        }
        
        return {
            'query_info': query_info,
            'cutoff_time': cutoff_time.isoformat()
        }
    
    def _get_scheduler_version(self) -> str:
        """Get scheduler version"""
        return f"Scheduler_v1.0_{datetime.now().strftime('%Y%m%d')}"
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current scheduler configuration"""
        return {
            'sla_hours': self.sla_hours,
            'followup_frequencies': self.followup_frequencies,
            'work_hours': self.work_hours,
            'schedule_type_mapping': self.schedule_type_mapping,
            'priority_weights': self.priority_weights,
            'scheduler_version': self._get_scheduler_version()
        }

# Singleton instance
scheduler_service = SchedulerService()

if __name__ == "__main__":
    # Test scheduler service
    print("Scheduler Service Test")
    print("Configuration:", scheduler_service.get_configuration())
    
    # Sample risk assessment
    sample_risk = {
        'risk_level': 'HIGH',
        'risk_score': 0.85,
        'procedures': [
            {
                'title': 'Prosedur Perbaikan Kebocoran',
                'estimated_duration': '4-6 jam',
                'required_skills': ['pipe fitter', 'pressure tester']
            }
        ],
        'rationale': 'YOLO: leak conf=0.90; NLP: kebocoran parah'
    }
    
    schedule = scheduler_service.generate_schedule(
        report_id=123,
        asset_id=456,
        risk_assessment=sample_risk,
        asset_criticality='HIGH'
    )
    
    print("Sample Schedule:", json.dumps(schedule, indent=2, ensure_ascii=False))
