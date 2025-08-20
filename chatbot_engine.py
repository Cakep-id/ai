from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

load_dotenv()

logger = logging.getLogger(__name__)

class ChatbotEngine:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
        self.model = None
        self.knowledge_base = {
            'faq': [],
            'assistant': []
        }
        self.embeddings = {
            'faq': [],
            'assistant': []
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"🔄 Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise e
    
    def update_knowledge_base(self, training_data: List[Dict]):
        """Update knowledge base with new training data"""
        try:
            # Reset knowledge base
            self.knowledge_base = {
                'faq': [],
                'assistant': []
            }
            
            # Organize data by category
            for item in training_data:
                category = item.get('category', 'faq')
                self.knowledge_base[category].append({
                    'id': item.get('id'),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'created_at': item.get('created_at')
                })
            
            # Generate embeddings for questions
            self._generate_embeddings()
            
            faq_count = len(self.knowledge_base['faq'])
            assistant_count = len(self.knowledge_base['assistant'])
            logger.info(f"📚 Knowledge base updated: {faq_count} FAQ, {assistant_count} Assistant items")
            
        except Exception as e:
            logger.error(f"❌ Error updating knowledge base: {e}")
            raise e
    
    def _generate_embeddings(self):
        """Generate embeddings for all questions in knowledge base"""
        try:
            self.embeddings = {
                'faq': [],
                'assistant': []
            }
            
            for category in ['faq', 'assistant']:
                if self.knowledge_base[category]:
                    questions = [item['question'] for item in self.knowledge_base[category]]
                    embeddings = self.model.encode(questions)
                    self.embeddings[category] = embeddings
                    logger.info(f"🧠 Generated {len(embeddings)} embeddings for {category}")
                else:
                    self.embeddings[category] = np.array([])
                    
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise e
    
    def find_best_answer(self, question: str, category: str = 'faq') -> Dict:
        """Find the best answer for a given question"""
        try:
            # Validate category
            if category not in ['faq', 'assistant']:
                category = 'faq'
            
            # Check if we have data for this category
            if not self.knowledge_base[category] or len(self.embeddings[category]) == 0:
                return {
                    'answer': self._get_fallback_answer(question, category),
                    'confidence': 0.0,
                    'source': 'fallback',
                    'matched_question': None
                }
            
            # Generate embedding for input question
            question_embedding = self.model.encode([question])
            
            # Calculate similarity with all questions in the category
            similarities = cosine_similarity(question_embedding, self.embeddings[category])[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # Check if similarity is above threshold
            if best_similarity >= self.similarity_threshold:
                best_match = self.knowledge_base[category][best_idx]
                return {
                    'answer': best_match['answer'],
                    'confidence': float(best_similarity),
                    'source': 'knowledge_base',
                    'matched_question': best_match['question'],
                    'data_id': best_match['id']
                }
            else:
                return {
                    'answer': self._get_fallback_answer(question, category),
                    'confidence': float(best_similarity),
                    'source': 'fallback',
                    'matched_question': None
                }
                
        except Exception as e:
            logger.error(f"❌ Error finding best answer: {e}")
            return {
                'answer': self._get_error_answer(),
                'confidence': 0.0,
                'source': 'error',
                'matched_question': None
            }
    
    def _get_fallback_answer(self, question: str, category: str) -> str:
        """Generate fallback answer when no good match is found"""
        question_lower = question.lower()
        
        if category == 'faq':
            if any(word in question_lower for word in ['cakep', 'platform', 'aplikasi']):
                return "Cakep.id adalah platform manajemen aset migas berbasis AI. Untuk informasi lebih detail, silakan hubungi tim support kami."
            
            if any(word in question_lower for word in ['laporan', 'report', 'melaporkan']):
                return "Untuk melaporkan kerusakan atau masalah, silakan gunakan fitur 'Buat Laporan' di dashboard Anda."
            
            if any(word in question_lower for word in ['cara', 'bagaimana', 'how']):
                return "Silakan rujuk ke dokumentasi pengguna atau hubungi tim support untuk panduan lebih detail."
        
        elif category == 'assistant':
            if any(word in question_lower for word in ['analisis', 'analyze', 'foto', 'gambar']):
                return "Saya siap membantu menganalisis data atau foto yang Anda berikan. Silakan upload file yang ingin dianalisis."
            
            if any(word in question_lower for word in ['jadwal', 'schedule', 'pemeliharaan']):
                return "Saya dapat membantu membuat jadwal pemeliharaan. Silakan berikan detail aset yang ingin dijadwalkan."
            
            if any(word in question_lower for word in ['bantuan', 'help', 'tolong']):
                return "Saya di sini untuk membantu Anda. Silakan jelaskan apa yang Anda butuhkan dan saya akan berusaha membantu."
        
        return f"Maaf, saya belum memiliki informasi yang tepat untuk pertanyaan Anda. Silakan hubungi tim support untuk bantuan lebih lanjut."
    
    def _get_error_answer(self) -> str:
        """Return error message when system fails"""
        return "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi atau hubungi tim support."
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            'total_items': len(self.knowledge_base['faq']) + len(self.knowledge_base['assistant']),
            'faq_items': len(self.knowledge_base['faq']),
            'assistant_items': len(self.knowledge_base['assistant']),
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'embeddings_generated': {
                'faq': len(self.embeddings['faq']) > 0,
                'assistant': len(self.embeddings['assistant']) > 0
            }
        }

# Global chatbot instance
chatbot = ChatbotEngine()
