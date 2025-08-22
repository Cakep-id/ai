"""
Language Learning Service untuk AI Chatbot
Mempelajari gaya bahasa dan preferensi user
"""

import re
import json
from collections import Counter, defaultdict
from database.connection import db_manager, UserSessions, UserLanguagePatterns, ChatFeedback
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import uuid

class LanguageLearningService:
    def __init__(self):
        self.db_manager = db_manager
        
        # Common Indonesian patterns
        self.greeting_patterns = [
            r'\b(hai|halo|selamat|permisi|hei)\b',
            r'\b(pagi|siang|sore|malam)\b',
            r'\b(terima kasih|makasih|thanks)\b'
        ]
        
        self.question_patterns = [
            r'\b(bagaimana|gimana|gmn|bgmn)\b',
            r'\b(apakah|apa|kenapa|mengapa)\b',
            r'\b(dimana|dmn|kapan|siapa)\b',
            r'\b(bisakah|bisa|tolong|mohon)\b'
        ]
        
        self.formality_indicators = {
            'formal': [r'\bAnda\b', r'\bsaudara\b', r'\bmohon\b', r'\bterima kasih\b'],
            'informal': [r'\bkamu\b', r'\blu\b', r'\bmakasih\b', r'\bgimana\b', r'\bgmn\b']
        }
    
    def create_or_get_session(self, session_id=None, user_role='user'):
        """Create atau get user session"""
        session = self.db_manager.get_session()
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Check if session exists
            user_session = session.query(UserSessions).filter(
                UserSessions.session_id == session_id
            ).first()
            
            if not user_session:
                user_session = UserSessions(
                    session_id=session_id,
                    user_role=user_role,
                    is_active=True
                )
                session.add(user_session)
                session.commit()
            else:
                # Update last activity
                user_session.last_activity = datetime.now()
                session.commit()
            
            return {
                'success': True,
                'session_id': session_id,
                'user_role': user_session.user_role
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error creating session: {str(e)}'
            }
        finally:
            session.close()
    
    def analyze_user_message(self, message, session_id):
        """Analyze user message untuk pattern learning"""
        patterns = {}
        
        # Analyze greeting patterns
        greeting_found = []
        for pattern in self.greeting_patterns:
            matches = re.findall(pattern, message.lower())
            if matches:
                greeting_found.extend(matches)
        
        if greeting_found:
            patterns['greeting'] = greeting_found
        
        # Analyze question patterns
        question_style = []
        for pattern in self.question_patterns:
            matches = re.findall(pattern, message.lower())
            if matches:
                question_style.extend(matches)
        
        if question_style:
            patterns['question_style'] = question_style
        
        # Analyze formality level
        formal_score = 0
        informal_score = 0
        
        for pattern in self.formality_indicators['formal']:
            if re.search(pattern, message):
                formal_score += 1
        
        for pattern in self.formality_indicators['informal']:
            if re.search(pattern, message):
                informal_score += 1
        
        if formal_score > informal_score:
            patterns['formality'] = 'formal'
        elif informal_score > formal_score:
            patterns['formality'] = 'informal'
        else:
            patterns['formality'] = 'neutral'
        
        # Analyze vocabulary preferences
        words = message.lower().split()
        common_words = [word for word in words if len(word) > 3]
        if common_words:
            patterns['vocabulary'] = common_words[:5]  # Top 5 words
        
        # Save patterns to database
        self.save_language_patterns(session_id, patterns)
        
        return patterns
    
    def save_language_patterns(self, session_id, patterns):
        """Save learned patterns to database"""
        session = self.db_manager.get_session()
        
        try:
            for pattern_type, pattern_value in patterns.items():
                # Convert to string if list
                if isinstance(pattern_value, list):
                    pattern_value = json.dumps(pattern_value)
                
                # Check if pattern exists
                existing_pattern = session.query(UserLanguagePatterns).filter(
                    UserLanguagePatterns.session_id == session_id,
                    UserLanguagePatterns.pattern_type == pattern_type,
                    UserLanguagePatterns.pattern_value == pattern_value
                ).first()
                
                if existing_pattern:
                    # Increment frequency
                    existing_pattern.frequency += 1
                    existing_pattern.updated_at = datetime.now()
                else:
                    # Create new pattern
                    new_pattern = UserLanguagePatterns(
                        session_id=session_id,
                        pattern_type=pattern_type,
                        pattern_value=pattern_value,
                        frequency=1,
                        confidence_score=0.5
                    )
                    session.add(new_pattern)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Error saving language patterns: {e}")
        finally:
            session.close()
    
    def get_user_language_profile(self, session_id):
        """Get user's language profile"""
        session = self.db_manager.get_session()
        
        try:
            patterns = session.query(UserLanguagePatterns).filter(
                UserLanguagePatterns.session_id == session_id
            ).all()
            
            profile = defaultdict(list)
            
            for pattern in patterns:
                try:
                    value = json.loads(pattern.pattern_value)
                except:
                    value = pattern.pattern_value
                
                profile[pattern.pattern_type].append({
                    'value': value,
                    'frequency': pattern.frequency,
                    'confidence': float(pattern.confidence_score)
                })
            
            return dict(profile)
            
        except Exception as e:
            print(f"Error getting language profile: {e}")
            return {}
        finally:
            session.close()
    
    def adapt_response_style(self, response, session_id):
        """Adapt response style based on user's language patterns"""
        profile = self.get_user_language_profile(session_id)
        
        if not profile:
            return response
        
        # Adapt formality
        formality_patterns = profile.get('formality', [])
        if formality_patterns:
            most_common_formality = max(formality_patterns, key=lambda x: x['frequency'])['value']
            
            if most_common_formality == 'formal':
                # Make response more formal
                response = response.replace('kamu', 'Anda')
                response = response.replace('gimana', 'bagaimana')
                response = response.replace('makasih', 'terima kasih')
            elif most_common_formality == 'informal':
                # Make response more informal
                response = response.replace('Anda', 'kamu')
                response = response.replace('bagaimana', 'gimana')
                response = response.replace('terima kasih', 'makasih')
        
        # Add personalized greeting if user often uses greetings
        greeting_patterns = profile.get('greeting', [])
        if greeting_patterns and not any(greet in response.lower() for greet in ['hai', 'halo', 'terima kasih']):
            most_common_greeting = max(greeting_patterns, key=lambda x: x['frequency'])['value']
            if isinstance(most_common_greeting, list) and most_common_greeting:
                greeting = most_common_greeting[0]
                if greeting in ['makasih', 'terima kasih', 'thanks']:
                    response = f"Sama-sama! {response}"
        
        return response
    
    def save_chat_feedback(self, session_id, user_message, bot_response, faq_id=None, 
                          feedback_type='neutral', feedback_text=None, similarity_score=None):
        """Save chat feedback for learning"""
        session = self.db_manager.get_session()
        
        try:
            feedback = ChatFeedback(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                faq_id=faq_id,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                similarity_score=similarity_score
            )
            
            session.add(feedback)
            session.commit()
            
            return {'success': True}
            
        except Exception as e:
            session.rollback()
            return {'success': False, 'message': str(e)}
        finally:
            session.close()
    
    def get_learning_insights(self, session_id=None, days=7):
        """Get learning insights for admin"""
        session = self.db_manager.get_session()
        
        try:
            # Date filter
            date_filter = datetime.now() - timedelta(days=days)
            
            # Query patterns
            query = session.query(UserLanguagePatterns)
            if session_id:
                query = query.filter(UserLanguagePatterns.session_id == session_id)
            
            patterns = query.filter(
                UserLanguagePatterns.created_at >= date_filter
            ).all()
            
            # Analyze insights
            insights = {
                'total_sessions': len(set([p.session_id for p in patterns])),
                'pattern_summary': defaultdict(int),
                'formality_distribution': defaultdict(int),
                'common_greetings': defaultdict(int),
                'question_styles': defaultdict(int)
            }
            
            for pattern in patterns:
                insights['pattern_summary'][pattern.pattern_type] += pattern.frequency
                
                if pattern.pattern_type == 'formality':
                    insights['formality_distribution'][pattern.pattern_value] += pattern.frequency
                elif pattern.pattern_type == 'greeting':
                    try:
                        greetings = json.loads(pattern.pattern_value)
                        for greeting in greetings:
                            insights['common_greetings'][greeting] += pattern.frequency
                    except:
                        insights['common_greetings'][pattern.pattern_value] += pattern.frequency
                elif pattern.pattern_type == 'question_style':
                    try:
                        styles = json.loads(pattern.pattern_value)
                        for style in styles:
                            insights['question_styles'][style] += pattern.frequency
                    except:
                        insights['question_styles'][pattern.pattern_value] += pattern.frequency
            
            # Convert defaultdict to regular dict
            for key in insights:
                if isinstance(insights[key], defaultdict):
                    insights[key] = dict(insights[key])
            
            return {'success': True, 'data': insights}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
        finally:
            session.close()

# Global language learning service instance
language_service = LanguageLearningService()
