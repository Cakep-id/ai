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
