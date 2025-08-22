"""
Groq Service untuk analisis NLP
Menggunakan Groq AI API untuk analisis teks deskripsi kerusakan
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from groq import Groq
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqService:
    """Service untuk analisis NLP menggunakan Groq AI"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.client = None
        
        # Category mapping untuk kerusakan
        self.damage_categories = {
            'structural': {
                'keywords': ['retak', 'crack', 'patah', 'fracture', 'bengkok', 'deformasi'],
                'confidence_boost': 0.2,
                'risk_level': 0.8
            },
            'corrosion': {
                'keywords': ['karat', 'korosi', 'rust', 'corrosion', 'oksidasi'],
                'confidence_boost': 0.15,
                'risk_level': 0.7
            },
            'fluid_leak': {
                'keywords': ['bocor', 'leak', 'rembes', 'tetes', 'mengalir', 'keluar'],
                'confidence_boost': 0.25,
                'risk_level': 0.85
            },
            'wear_tear': {
                'keywords': ['aus', 'wear', 'terkikis', 'habis', 'tipis', 'erosi'],
                'confidence_boost': 0.1,
                'risk_level': 0.6
            },
            'contamination': {
                'keywords': ['kotor', 'kontaminasi', 'minyak', 'debu', 'oli', 'cemaran'],
                'confidence_boost': 0.05,
                'risk_level': 0.3
            },
            'operational': {
                'keywords': ['panas', 'berisik', 'getaran', 'tidak normal', 'berhenti'],
                'confidence_boost': 0.15,
                'risk_level': 0.5
            }
        }
        
        # Severity indicators
        self.severity_indicators = {
            'high': ['parah', 'besar', 'serius', 'urgent', 'emergency', 'critical', 'bahaya'],
            'medium': ['sedang', 'cukup', 'moderate', 'normal', 'standar'],
            'low': ['kecil', 'ringan', 'minor', 'sedikit', 'slight']
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inisialisasi Groq client"""
        try:
            if not self.api_key:
                logger.warning("Groq API key not found in environment variables")
                return
            
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def analyze(self, text: str, asset_context: str = None) -> Dict[str, Any]:
        """
        Analisis teks deskripsi untuk kategori dan indikasi kerusakan
        
        Args:
            text: Teks deskripsi kerusakan
            asset_context: Konteks aset (opsional) seperti "pompa air", "valve"
        
        Returns:
            {
                'success': bool,
                'category': str,
                'confidence': float,
                'keyphrases': List[str],
                'severity': str,
                'risk_indicators': List[str],
                'recommendations': List[str],
                'model_version': str
            }
        """
        try:
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'Empty text input',
                    'category': 'unknown',
                    'confidence': 0.0
                }
            
            # Pre-processing analysis
            preprocessed = self._preprocess_text(text)
            keyword_analysis = self._analyze_keywords(preprocessed['text'])
            
            # Groq AI analysis jika client tersedia
            ai_analysis = {}
            if self.client:
                ai_analysis = self._groq_analysis(preprocessed['text'], asset_context)
            
            # Combine hasil analisis
            final_result = self._combine_analysis(
                text=preprocessed['text'],
                keyword_analysis=keyword_analysis,
                ai_analysis=ai_analysis,
                asset_context=asset_context
            )
            
            logger.info(f"NLP analysis completed: category={final_result['category']}, confidence={final_result['confidence']:.2f}")
            return final_result
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'category': 'unknown',
                'confidence': 0.0,
                'keyphrases': [],
                'severity': 'unknown',
                'risk_indicators': [],
                'recommendations': []
            }
    
    async def analyze_damage_description(self, description: str, asset_context: str = None) -> Dict[str, Any]:
        """
        Analisis deskripsi kerusakan untuk menentukan kategori dan tingkat kerusakan
        Method ini digunakan oleh user_endpoints.py
        
        Args:
            description: Deskripsi kerusakan dari user
            asset_context: Konteks aset (opsional)
        
        Returns:
            Dict dengan category, confidence, severity, dll.
        """
        try:
            # Gunakan method analyze yang sudah ada
            result = self.analyze(description, asset_context)
            
            # Format hasil untuk compatibility dengan endpoint
            return {
                'success': result.get('success', True),
                'category': result.get('category', 'unknown'),
                'detected_damage': result.get('category', 'Unknown Damage'),
                'confidence': result.get('confidence', 0.5),
                'severity': result.get('severity', 'medium'),
                'risk_indicators': result.get('risk_indicators', []),
                'recommendations': result.get('recommendations', []),
                'keyphrases': result.get('keyphrases', [])
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_damage_description: {e}")
            return {
                'success': False,
                'category': 'unknown',
                'detected_damage': 'Unknown Damage', 
                'confidence': 0.0,
                'severity': 'unknown',
                'risk_indicators': [],
                'recommendations': [],
                'keyphrases': []
            }
    
    async def generate_repair_procedures(self, damage_description: str, cv_results: Dict = None, risk_level: str = 'MEDIUM') -> List[Dict[str, Any]]:
        """
        Generate repair procedures berdasarkan damage description dan CV results
        Method ini dipanggil oleh user_endpoints.py
        
        Args:
            damage_description: Deskripsi kerusakan
            cv_results: Hasil computer vision (opsional)
            risk_level: Level risiko (LOW/MEDIUM/HIGH/CRITICAL)
        
        Returns:
            List prosedur perbaikan dengan format standar
        """
        try:
            if not self.client:
                logger.warning("Groq client not available, using fallback procedures")
                return self._get_fallback_procedures(damage_description, risk_level)
            
            # Buat context untuk generate procedures
            context = {
                'damage_description': damage_description,
                'risk_level': risk_level,
                'cv_results': cv_results or {},
                'asset_type': 'equipment'  # default
            }
            
            # Extract asset type dari deskripsi jika memungkinkan
            description_lower = damage_description.lower()
            if 'pompa' in description_lower:
                context['asset_type'] = 'pompa'
            elif 'pipa' in description_lower:
                context['asset_type'] = 'pipa'
            elif 'motor' in description_lower:
                context['asset_type'] = 'motor'
            elif 'tangki' in description_lower:
                context['asset_type'] = 'tangki'
            
            # Generate procedures menggunakan AI
            prompt = self._create_repair_procedure_prompt(context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert maintenance engineer. Generate detailed repair procedures in Indonesian."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            if response.choices and response.choices[0].message:
                procedures_text = response.choices[0].message.content
                return self._parse_procedures_text(procedures_text, risk_level)
            else:
                return self._get_fallback_procedures(damage_description, risk_level)
                
        except Exception as e:
            logger.error(f"Error generating repair procedures: {e}")
            return self._get_fallback_procedures(damage_description, risk_level)
    
    def _create_repair_procedure_prompt(self, context: Dict) -> str:
        """Create prompt untuk generate repair procedures"""
        damage_desc = context.get('damage_description', '')
        risk_level = context.get('risk_level', 'MEDIUM')
        asset_type = context.get('asset_type', 'equipment')
        
        prompt = f"""Buatkan prosedur perbaikan detail untuk kerusakan berikut:

INFORMASI:
- Asset Type: {asset_type}
- Risk Level: {risk_level}
- Deskripsi Kerusakan: {damage_desc}

REQUIREMENTS:
1. Buat prosedur step-by-step yang mudah diikuti teknisi
2. Sesuaikan detail dengan risk level {risk_level}
3. Sertakan safety precautions
4. Estimasi waktu dan material yang diperlukan
5. Gunakan Bahasa Indonesia yang jelas

Format output sebagai list prosedur dengan:
- Langkah-langkah detail
- Estimasi waktu
- Material/tools needed
- Safety notes

Contoh format:
1. Matikan sistem dan pastikan keamanan area kerja (15 menit)
2. Siapkan tools: kunci pas, seal baru, pembersih (5 menit)
3. Lepas komponen yang rusak dengan hati-hati (30 menit)
dst."""
        
        return prompt
    
    def _parse_procedures_text(self, text: str, risk_level: str) -> List[Dict[str, Any]]:
        """Parse AI response menjadi structured procedures"""
        procedures = []
        
        # Split berdasarkan numbering
        lines = text.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Cek apakah line ini adalah langkah baru (dimulai dengan angka)
            if re.match(r'^\d+\.', line):
                if current_step:
                    # Proses langkah sebelumnya
                    procedures.append(self._create_procedure_item(current_step, len(procedures) + 1, risk_level))
                current_step = line
            else:
                # Tambahkan ke langkah saat ini
                if current_step:
                    current_step += f" {line}"
        
        # Proses langkah terakhir
        if current_step:
            procedures.append(self._create_procedure_item(current_step, len(procedures) + 1, risk_level))
        
        return procedures if procedures else self._get_fallback_procedures("", risk_level)
    
    def _create_procedure_item(self, step_text: str, step_number: int, risk_level: str) -> Dict[str, Any]:
        """Create structured procedure item"""
        # Extract time estimate jika ada
        time_match = re.search(r'\((\d+)\s*(?:menit|min|jam|hour)', step_text.lower())
        estimated_minutes = int(time_match.group(1)) if time_match else (30 if risk_level in ['HIGH', 'CRITICAL'] else 15)
        
        return {
            'step': step_number,
            'description': step_text,
            'estimated_time_minutes': estimated_minutes,
            'safety_level': risk_level,
            'required_tools': [],
            'materials': [],
            'safety_notes': "Ikuti prosedur keselamatan standar" if risk_level in ['HIGH', 'CRITICAL'] else ""
        }
    
    def _get_fallback_procedures(self, damage_description: str, risk_level: str) -> List[Dict[str, Any]]:
        """Fallback procedures jika AI tidak tersedia"""
        base_procedures = [
            {
                'step': 1,
                'description': '1. Matikan sistem dan pastikan keamanan area kerja',
                'estimated_time_minutes': 15,
                'safety_level': risk_level,
                'required_tools': ['Kunci pembuka', 'APD'],
                'materials': [],
                'safety_notes': 'Pastikan sistem benar-benar mati sebelum memulai'
            },
            {
                'step': 2,
                'description': '2. Inspeksi visual dan identifikasi masalah utama',
                'estimated_time_minutes': 20,
                'safety_level': risk_level,
                'required_tools': ['Senter', 'Kamera'],
                'materials': [],
                'safety_notes': 'Gunakan APD lengkap'
            },
            {
                'step': 3,
                'description': '3. Siapkan spare parts dan tools yang diperlukan',
                'estimated_time_minutes': 10,
                'safety_level': risk_level,
                'required_tools': ['Tools sesuai kebutuhan'],
                'materials': ['Spare parts', 'Consumables'],
                'safety_notes': 'Pastikan spare parts sesuai spesifikasi'
            },
            {
                'step': 4,
                'description': '4. Lakukan perbaikan sesuai prosedur standar',
                'estimated_time_minutes': 60 if risk_level in ['HIGH', 'CRITICAL'] else 30,
                'safety_level': risk_level,
                'required_tools': [],
                'materials': [],
                'safety_notes': 'Ikuti manual maintenance dengan teliti'
            },
            {
                'step': 5,
                'description': '5. Test dan verifikasi hasil perbaikan',
                'estimated_time_minutes': 15,
                'safety_level': risk_level,
                'required_tools': ['Testing equipment'],
                'materials': [],
                'safety_notes': 'Pastikan semua parameter normal sebelum operasi'
            }
        ]
        
        return base_procedures
    
    def _preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocessing teks"""
        # Bersihkan teks
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Extract nomor dan ukuran
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        units = re.findall(r'\d+\s*(mm|cm|m|inch|bar|psi|degree|°c)', text.lower())
        
        return {
            'text': cleaned,
            'original': text,
            'numbers': numbers,
            'units': units,
            'word_count': len(cleaned.split())
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analisis berdasarkan keyword matching"""
        category_scores = {}
        found_keywords = []
        
        # Score setiap kategori berdasarkan keyword
        for category, info in self.damage_categories.items():
            score = 0
            category_keywords = []
            
            for keyword in info['keywords']:
                if keyword in text:
                    score += 1
                    category_keywords.append(keyword)
                    found_keywords.append(keyword)
            
            if score > 0:
                # Normalize score dengan boost
                normalized_score = min(score / len(info['keywords']) + info['confidence_boost'], 1.0)
                category_scores[category] = {
                    'score': normalized_score,
                    'keywords': category_keywords,
                    'risk_level': info['risk_level']
                }
        
        # Tentukan kategori terbaik
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k]['score'])
            best_score = category_scores[best_category]['score']
        else:
            best_category = 'unknown'
            best_score = 0.0
        
        # Analisis severity
        severity = self._analyze_severity(text)
        
        return {
            'category': best_category,
            'confidence': best_score,
            'all_scores': category_scores,
            'keywords': found_keywords,
            'severity': severity
        }
    
    def _analyze_severity(self, text: str) -> str:
        """Analisis tingkat keparahan dari teks"""
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    return severity
        
        return 'medium'  # Default
    
    def _groq_analysis(self, text: str, asset_context: str = None) -> Dict[str, Any]:
        """Analisis menggunakan Groq AI"""
        try:
            # Buat prompt yang terstruktur
            context_info = f" pada {asset_context}" if asset_context else ""
            
            prompt = f"""
Analisis deskripsi kerusakan berikut{context_info}:

"{text}"

Berikan analisis dalam format JSON dengan field berikut:
1. category: kategori kerusakan (structural/corrosion/fluid_leak/wear_tear/contamination/operational)
2. severity: tingkat keparahan (high/medium/low)
3. risk_indicators: daftar indikator risiko yang ditemukan
4. keyphrases: kata kunci penting terkait kerusakan
5. recommendations: rekomendasi tindakan singkat
6. confidence: tingkat kepercayaan analisis (0.0-1.0)

Fokus pada aspek teknis dan keamanan aset industri.
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Anda adalah expert dalam analisis kerusakan aset industri. Berikan analisis yang akurat dan praktis."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Extract JSON dari response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                
                # Validasi dan normalisasi
                ai_result['confidence'] = max(0.0, min(1.0, float(ai_result.get('confidence', 0.5))))
                ai_result['category'] = ai_result.get('category', 'unknown').lower()
                ai_result['severity'] = ai_result.get('severity', 'medium').lower()
                
                return ai_result
            else:
                # Fallback parsing
                return self._parse_text_response(response_text)
                
        except Exception as e:
            logger.error(f"Groq analysis failed: {e}")
            return {}
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response text jika JSON parsing gagal"""
        try:
            # Simple keyword extraction
            categories = ['structural', 'corrosion', 'fluid_leak', 'wear_tear', 'contamination', 'operational']
            severities = ['high', 'medium', 'low']
            
            found_category = 'unknown'
            found_severity = 'medium'
            
            response_lower = response_text.lower()
            
            for category in categories:
                if category in response_lower:
                    found_category = category
                    break
            
            for severity in severities:
                if severity in response_lower:
                    found_severity = severity
                    break
            
            # Extract recommendations (sentences with "rekomendasi" atau "saran")
            recommendations = []
            sentences = response_text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['rekomendasi', 'saran', 'recommend', 'should']):
                    recommendations.append(sentence.strip())
            
            return {
                'category': found_category,
                'severity': found_severity,
                'confidence': 0.6,  # Medium confidence untuk fallback
                'recommendations': recommendations[:3],  # Max 3 recommendations
                'keyphrases': [],
                'risk_indicators': []
            }
            
        except Exception as e:
            logger.error(f"Text response parsing failed: {e}")
            return {}
    
    def _combine_analysis(self, text: str, keyword_analysis: Dict, ai_analysis: Dict, 
                         asset_context: str = None) -> Dict[str, Any]:
        """Gabungkan hasil analisis keyword dan AI"""
        
        # Pilih kategori terbaik
        keyword_conf = keyword_analysis.get('confidence', 0.0)
        ai_conf = ai_analysis.get('confidence', 0.0)
        
        if ai_conf > keyword_conf and ai_analysis:
            # Gunakan hasil AI
            final_category = ai_analysis.get('category', keyword_analysis['category'])
            base_confidence = ai_conf
            keyphrases = ai_analysis.get('keyphrases', [])
            severity = ai_analysis.get('severity', keyword_analysis['severity'])
            recommendations = ai_analysis.get('recommendations', [])
            risk_indicators = ai_analysis.get('risk_indicators', [])
        else:
            # Gunakan hasil keyword
            final_category = keyword_analysis['category']
            base_confidence = keyword_conf
            keyphrases = keyword_analysis['keywords']
            severity = keyword_analysis['severity']
            recommendations = self._generate_default_recommendations(final_category)
            risk_indicators = self._extract_risk_indicators(text)
        
        # Adjust confidence berdasarkan berbagai faktor
        final_confidence = self._calculate_final_confidence(
            base_confidence, 
            text, 
            final_category,
            asset_context
        )
        
        return {
            'success': True,
            'category': final_category,
            'confidence': final_confidence,
            'keyphrases': keyphrases[:10],  # Max 10 keyphrases
            'severity': severity,
            'risk_indicators': risk_indicators[:5],  # Max 5 indicators
            'recommendations': recommendations[:3],  # Max 3 recommendations
            'model_version': self._get_model_version(),
            'analysis_method': 'combined' if ai_analysis else 'keyword_only',
            'processing_time': datetime.now().isoformat()
        }
    
    def _calculate_final_confidence(self, base_confidence: float, text: str, 
                                  category: str, asset_context: str = None) -> float:
        """Hitung confidence score final dengan berbagai faktor"""
        confidence = base_confidence
        
        # Boost jika ada konteks aset yang sesuai
        if asset_context and category in ['fluid_leak', 'operational']:
            if any(word in asset_context.lower() for word in ['pompa', 'valve', 'pipa']):
                confidence += 0.1
        
        # Boost jika ada angka/measurement (lebih spesifik)
        if re.search(r'\d+', text):
            confidence += 0.05
        
        # Penalty jika teks terlalu pendek
        if len(text.split()) < 3:
            confidence -= 0.2
        
        # Boost jika teks cukup detail
        if len(text.split()) > 10:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_risk_indicators(self, text: str) -> List[str]:
        """Extract indikator risiko dari teks"""
        risk_patterns = [
            r'tidak (berfungsi|normal|bekerja)',
            r'(parah|besar|serius)',
            r'(bocor|rembes|keluar)',
            r'(retak|patah|rusak)',
            r'(panas|overheat|berlebih)',
            r'(berhenti|mati|tidak jalan)'
        ]
        
        indicators = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, text.lower())
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    def _generate_default_recommendations(self, category: str) -> List[str]:
        """Generate rekomendasi default berdasarkan kategori"""
        recommendations_map = {
            'structural': [
                'Lakukan inspeksi detail pada area yang retak',
                'Hentikan operasi jika kerusakan struktural parah',
                'Konsultasi dengan engineer struktural'
            ],
            'corrosion': [
                'Bersihkan area korosi dan aplikasikan coating protection',
                'Periksa ketebalan material dengan ultrasonic testing',
                'Ganti komponen jika korosi sudah parah'
            ],
            'fluid_leak': [
                'Isolasi sistem dan turunkan tekanan',
                'Pasang clamp sementara jika memungkinkan',
                'Ganti seal atau gasket yang rusak'
            ],
            'wear_tear': [
                'Lakukan preventive maintenance lebih sering',
                'Periksa alignment dan balancing',
                'Ganti komponen yang sudah habis'
            ],
            'contamination': [
                'Bersihkan kontaminan dengan solvent yang sesuai',
                'Periksa sistem filtrasi',
                'Lakukan flushing sistem jika perlu'
            ],
            'operational': [
                'Periksa parameter operasi normal',
                'Lakukan troubleshooting sistematis',
                'Monitor kondisi secara berkelanjutan'
            ]
        }
        
        return recommendations_map.get(category, ['Lakukan inspeksi lebih detail'])
    
    def get_text_score(self, analysis_result: Dict) -> float:
        """
        Hitung text score untuk risk engine
        Berdasarkan confidence dan severity
        """
        if not analysis_result.get('success'):
            return 0.0
        
        base_confidence = analysis_result.get('confidence', 0.0)
        severity = analysis_result.get('severity', 'medium')
        category = analysis_result.get('category', 'unknown')
        
        # Severity multiplier
        severity_multiplier = {
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }.get(severity, 1.0)
        
        # Category risk factor
        category_risk = self.damage_categories.get(category, {}).get('risk_level', 0.5)
        
        # Combine factors
        text_score = base_confidence * severity_multiplier * category_risk
        
        return max(0.0, min(1.0, text_score))
    
    def _get_model_version(self) -> str:
        """Get model version info"""
        return f"Groq_{self.model}_{datetime.now().strftime('%Y%m%d')}"
    
    def test_connection(self) -> Dict[str, Any]:
        """Test koneksi ke Groq API"""
        try:
            if not self.client:
                return {
                    'success': False,
                    'error': 'Groq client not initialized'
                }
            
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Test connection"}
                ],
                max_tokens=10
            )
            
            return {
                'success': True,
                'model': self.model,
                'response_length': len(response.choices[0].message.content)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_repair_procedure(self, context: Dict) -> Dict:
        """
        Generate repair procedure menggunakan Groq AI
        
        Args:
            context: Dictionary berisi informasi kerusakan dan asset
            
        Returns:
            Dictionary dengan procedure dan estimasi
        """
        try:
            if not self._initialize_client():
                return {'success': False, 'error': 'Groq client not initialized'}
            
            # Buat prompt untuk generate procedure
            prompt = self._create_procedure_prompt(context)
            
            logger.info("Generating repair procedure with Groq AI")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Anda adalah expert maintenance engineer yang akan memberikan prosedur perbaikan detail untuk equipment industri. Berikan response dalam format JSON dengan field: procedure (text detail), estimated_hours (number), estimated_cost (number), required_skills (array), materials (array), safety_notes (text)."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response dari Groq
            try:
                parsed_response = json.loads(response_text)
                
                return {
                    'success': True,
                    'procedure': parsed_response.get('procedure', 'Prosedur tidak dapat di-generate'),
                    'estimated_hours': parsed_response.get('estimated_hours', 4),
                    'estimated_cost': parsed_response.get('estimated_cost', 0),
                    'required_skills': parsed_response.get('required_skills', []),
                    'materials': parsed_response.get('materials', []),
                    'safety_notes': parsed_response.get('safety_notes', ''),
                    'model_version': self._get_model_version(),
                    'generated_at': datetime.now().isoformat()
                }
                
            except json.JSONDecodeError:
                # Jika response bukan JSON, ambil sebagai text biasa
                return {
                    'success': True,
                    'procedure': response_text,
                    'estimated_hours': self._estimate_hours_from_text(response_text),
                    'estimated_cost': 0,
                    'required_skills': [],
                    'materials': [],
                    'safety_notes': '',
                    'model_version': self._get_model_version(),
                    'generated_at': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Groq procedure generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_procedure_prompt(self, context: Dict) -> str:
        """Create prompt untuk generate repair procedure"""
        
        risk_level = context.get('risk_level', 'MEDIUM')
        asset_type = context.get('asset_type', 'equipment')
        damage_category = context.get('damage_category', 'general')
        
        prompt = f"""Buatkan prosedur perbaikan detail untuk:

INFORMASI ASSET:
- Tipe Asset: {asset_type}
- Risk Level: {risk_level}
- Kategori Kerusakan: {damage_category}
"""
        
        # Tambahkan info visual damage jika ada
        if 'visual_damages' in context:
            prompt += "\nDAMAGE DETECTION (Computer Vision):\n"
            for damage in context['visual_damages']:
                prompt += f"- {damage['type']} (confidence: {damage['confidence']:.2f}, severity: {damage['severity']})\n"
        
        # Tambahkan info text analysis jika ada
        if 'text_analysis' in context:
            text_info = context['text_analysis']
            prompt += f"\nANALISIS TEKS:\n"
            prompt += f"- Kategori: {text_info.get('category', 'N/A')}\n"
            prompt += f"- Confidence: {text_info.get('confidence', 0):.2f}\n"
            if text_info.get('keyphrases'):
                prompt += f"- Keywords: {', '.join(text_info['keyphrases'])}\n"
        
        prompt += f"""
REQUIREMENTS:
1. Buat prosedur perbaikan step-by-step yang detail dan praktis
2. Sesuaikan dengan risk level {risk_level} - semakin tinggi risk semakin detail dan hati-hati
3. Sertakan estimasi waktu, biaya, skill required, dan material needed
4. Berikan safety notes yang komprehensif
5. Format output dalam JSON dengan structure yang diminta

Berikan response dalam Bahasa Indonesia yang professional dan mudah dipahami teknisi lapangan."""
        
        return prompt
    
    def _estimate_hours_from_text(self, text: str) -> float:
        """Estimate hours dari text jika tidak ada dalam JSON"""
        text_lower = text.lower()
        
        # Simple heuristic berdasarkan keywords
        if any(word in text_lower for word in ['emergency', 'critical', 'segera', 'urgent']):
            return 2.0
        elif any(word in text_lower for word in ['kompleks', 'complex', 'detail', 'thorough']):
            return 8.0
        elif any(word in text_lower for word in ['simple', 'sederhana', 'quick', 'cepat']):
            return 1.0
        else:
            return 4.0  # default

# Singleton instance
groq_service = GroqService()

if __name__ == "__main__":
    # Test Groq service
    print("Groq Service Test")
    
    # Test connection
    connection_test = groq_service.test_connection()
    print("Connection Test:", connection_test)
    
    # Test analysis
    test_text = "Pompa air mengalami kebocoran pada seal dan terdengar suara berisik tidak normal"
    result = groq_service.analyze(test_text, "pompa air")
    print("Analysis Result:", json.dumps(result, indent=2, ensure_ascii=False))
