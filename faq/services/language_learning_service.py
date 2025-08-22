"""
Language Learning Service untuk AI Chatbot
Mempelajari gaya bahasa user dan menyesuaikan response
"""

import json
import re
from collections import Counter, defaultdict
from database.connection import db_manager
from services.auth_service import UserInteraction
from sqlalchemy import desc

class LanguageLearningService:
    def __init__(self):
        self.db_manager = db_manager
        
        # Pattern bahasa Indonesia
        self.formal_patterns = [
            r'\b(anda|bapak|ibu|saudara)\b',
            r'\b(mohon|silakan|terima kasih)\b',
            r'\b(selamat pagi|selamat siang|selamat sore|selamat malam)\b'
        ]
        
        self.informal_patterns = [
            r'\b(kamu|lu|gue|gw|lo)\b',
            r'\b(gimana|gmn|bgmn|kayak|kyk)\b',
            r'\b(oke|ok|ya|yah|deh|dong|sih)\b'
        ]
        
        self.question_patterns = [
            r'\b(bagaimana|gimana|gmn|bgmn)\b',
            r'\b(apa|apakah)\b',
            r'\b(dimana|dmn|mana)\b',
            r'\b(kapan|kpn)\b',
            r'\b(kenapa|knp|mengapa)\b'
        ]
        
    def analyze_user_language(self, message):
        """Analyze gaya bahasa user dari pesan"""
        message_lower = message.lower()
        
        analysis = {
            'formality': self._detect_formality(message_lower),
            'question_style': self._detect_question_style(message_lower),
            'common_words': self._extract_common_words(message_lower),
            'punctuation_style': self._analyze_punctuation(message),
            'abbreviation_usage': self._detect_abbreviations(message_lower),
            'emotion_level': self._detect_emotion_level(message)
        }
        
        return analysis
    
    def _detect_formality(self, message):
        """Detect tingkat formalitas"""
        formal_score = 0
        informal_score = 0
        
        for pattern in self.formal_patterns:
            formal_score += len(re.findall(pattern, message))
        
        for pattern in self.informal_patterns:
            informal_score += len(re.findall(pattern, message))
        
        if formal_score > informal_score:
            return 'formal'
        elif informal_score > formal_score:
            return 'informal'
        else:
            return 'neutral'
    
    def _detect_question_style(self, message):
        """Detect style pertanyaan"""
        for pattern in self.question_patterns:
            if re.search(pattern, message):
                return 'direct'
        
        if '?' in message:
            return 'interrogative'
        
        return 'statement'
    
    def _extract_common_words(self, message):
        """Extract kata-kata umum yang sering digunakan"""
        words = re.findall(r'\b\w+\b', message)
        return [word for word in words if len(word) > 2]
    
    def _analyze_punctuation(self, message):
        """Analyze penggunaan tanda baca"""
        return {
            'exclamation': message.count('!'),
            'question': message.count('?'),
            'ellipsis': message.count('...'),
            'comma': message.count(',')
        }
    
    def _detect_abbreviations(self, message):
        """Detect penggunaan singkatan"""
        abbreviations = ['gmn', 'bgmn', 'gw', 'lu', 'kyk', 'dmn', 'kpn', 'knp']
        found_abbrev = []
        
        for abbrev in abbreviations:
            if abbrev in message:
                found_abbrev.append(abbrev)
        
        return found_abbrev
    
    def _detect_emotion_level(self, message):
        """Detect tingkat emosi dari pesan"""
        excited_words = ['wow', 'wah', 'keren', 'hebat', 'amazing', 'bagus']
        frustrated_words = ['aduh', 'duh', 'capek', 'susah', 'ribet', 'error']
        
        excited_count = sum(1 for word in excited_words if word in message.lower())
        frustrated_count = sum(1 for word in frustrated_words if word in message.lower())
        
        if excited_count > frustrated_count:
            return 'excited'
        elif frustrated_count > excited_count:
            return 'frustrated'
        else:
            return 'neutral'
    
    def learn_from_interaction(self, session_id, user_message, bot_response, user_feedback=None):
        """Simpan interaksi untuk pembelajaran"""
        session = self.db_manager.get_session()
        
        try:
            # Analyze language style
            language_style = self.analyze_user_language(user_message)
            
            # Save interaction
            interaction = UserInteraction(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                user_feedback=user_feedback,
                language_style=json.dumps(language_style)
            )
            
            session.add(interaction)
            session.commit()
            
            return {
                'success': True,
                'language_style': language_style
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error learning from interaction: {str(e)}'
            }
        finally:
            session.close()
    
    def get_user_language_profile(self, session_id, limit=10):
        """Get profil bahasa user berdasarkan interaksi sebelumnya"""
        session = self.db_manager.get_session()
        
        try:
            interactions = session.query(UserInteraction).filter(
                UserInteraction.session_id == session_id
            ).order_by(desc(UserInteraction.created_at)).limit(limit).all()
            
            if not interactions:
                return self._get_default_profile()
            
            # Aggregate language patterns
            formality_scores = []
            question_styles = []
            common_words = []
            abbreviation_usage = []
            emotion_levels = []
            
            for interaction in interactions:
                if interaction.language_style:
                    try:
                        style = json.loads(interaction.language_style)
                        formality_scores.append(style.get('formality', 'neutral'))
                        question_styles.append(style.get('question_style', 'statement'))
                        common_words.extend(style.get('common_words', []))
                        abbreviation_usage.extend(style.get('abbreviation_usage', []))
                        emotion_levels.append(style.get('emotion_level', 'neutral'))
                    except:
                        continue
            
            # Create profile
            profile = {
                'preferred_formality': Counter(formality_scores).most_common(1)[0][0] if formality_scores else 'neutral',
                'common_question_style': Counter(question_styles).most_common(1)[0][0] if question_styles else 'statement',
                'frequently_used_words': [word for word, count in Counter(common_words).most_common(10)],
                'uses_abbreviations': len(abbreviation_usage) > 0,
                'typical_emotion': Counter(emotion_levels).most_common(1)[0][0] if emotion_levels else 'neutral',
                'interaction_count': len(interactions)
            }
            
            return profile
            
        except Exception as e:
            return self._get_default_profile()
        finally:
            session.close()
    
    def _get_default_profile(self):
        """Default language profile"""
        return {
            'preferred_formality': 'neutral',
            'common_question_style': 'direct',
            'frequently_used_words': [],
            'uses_abbreviations': False,
            'typical_emotion': 'neutral',
            'interaction_count': 0
        }
    
    def adapt_response(self, base_response, language_profile):
        """Adapt response berdasarkan profil bahasa user"""
        adapted_response = base_response
        
        # Adapt formality
        if language_profile['preferred_formality'] == 'formal':
            adapted_response = self._formalize_response(adapted_response)
        elif language_profile['preferred_formality'] == 'informal':
            adapted_response = self._informalize_response(adapted_response)
        
        # Adapt emotion
        if language_profile['typical_emotion'] == 'excited':
            adapted_response = self._add_enthusiasm(adapted_response)
        elif language_profile['typical_emotion'] == 'frustrated':
            adapted_response = self._add_empathy(adapted_response)
        
        # Add personal touch
        if language_profile['interaction_count'] > 3:
            adapted_response = self._add_familiarity(adapted_response)
        
        return adapted_response
    
    def _formalize_response(self, response):
        """Make response more formal"""
        replacements = {
            'kamu': 'Anda',
            'gimana': 'bagaimana',
            'oke': 'baik',
            'ya': 'silakan'
        }
        
        for informal, formal in replacements.items():
            response = re.sub(r'\b' + informal + r'\b', formal, response, flags=re.IGNORECASE)
        
        return response
    
    def _informalize_response(self, response):
        """Make response more informal"""
        replacements = {
            'Anda': 'kamu',
            'bagaimana': 'gimana',
            'silakan': 'ayo'
        }
        
        for formal, informal in replacements.items():
            response = re.sub(r'\b' + formal + r'\b', informal, response, flags=re.IGNORECASE)
        
        return response
    
    def _add_enthusiasm(self, response):
        """Add enthusiasm to response"""
        enthusiasm_words = ['!', 'hebat', 'keren', 'bagus sekali']
        if not any(word in response for word in enthusiasm_words):
            response += ' 😊'
        
        return response
    
    def _add_empathy(self, response):
        """Add empathy to response"""
        if 'maaf' not in response.lower():
            response = 'Saya memahami, ' + response.lower()
        
        return response
    
    def _add_familiarity(self, response):
        """Add familiarity for returning users"""
        familiarity_prefixes = [
            'Seperti yang sudah kita bahas sebelumnya, ',
            'Kamu sudah sering bertanya, jadi ',
            'Berdasarkan percakapan kita sebelumnya, '
        ]
        
        import random
        if random.random() < 0.3:  # 30% chance
            prefix = random.choice(familiarity_prefixes)
            response = prefix + response.lower()
        
        return response

# Global language learning service instance
language_learning_service = LanguageLearningService()
