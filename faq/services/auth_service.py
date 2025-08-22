"""
Authentication and Role Management Service
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from database.connection import db_manager, Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='user')  # 'admin' or 'user'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

class UserSession(Base):
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.current_timestamp())

class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    user_feedback = Column(String(20), nullable=True)  # 'helpful', 'not_helpful'
    language_style = Column(Text, nullable=True)  # JSON untuk menyimpan pattern bahasa
    created_at = Column(DateTime, default=func.current_timestamp())

class AuthService:
    def __init__(self):
        self.db_manager = db_manager
    
    def hash_password(self, password):
        """Hash password dengan salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password, password_hash):
        """Verify password"""
        try:
            salt, hash_value = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_value == password_hash_check.hex()
        except:
            return False
    
    def create_user(self, username, email, password, role='user'):
        """Create new user"""
        session = self.db_manager.get_session()
        
        try:
            # Check if user exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return {
                    'success': False,
                    'message': 'Username atau email sudah digunakan'
                }
            
            # Create user
            password_hash = self.hash_password(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role
            )
            
            session.add(new_user)
            session.commit()
            
            return {
                'success': True,
                'message': 'User berhasil dibuat',
                'user_id': new_user.id
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error creating user: {str(e)}'
            }
        finally:
            session.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user dan create session"""
        session = self.db_manager.get_session()
        
        try:
            user = session.query(User).filter(
                (User.username == username) | (User.email == username)
            ).filter(User.is_active == True).first()
            
            if not user or not self.verify_password(password, user.password_hash):
                return {
                    'success': False,
                    'message': 'Username atau password salah'
                }
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            user_session = UserSession(
                user_id=user.id,
                session_token=session_token,
                expires_at=expires_at
            )
            
            session.add(user_session)
            session.commit()
            
            return {
                'success': True,
                'message': 'Login berhasil',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role
                },
                'session_token': session_token
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error during authentication: {str(e)}'
            }
        finally:
            session.close()
    
    def verify_session(self, session_token):
        """Verify session token"""
        session = self.db_manager.get_session()
        
        try:
            user_session = session.query(UserSession).filter(
                UserSession.session_token == session_token,
                UserSession.expires_at > datetime.now()
            ).first()
            
            if not user_session:
                return None
            
            user = session.query(User).filter(User.id == user_session.user_id).first()
            
            if not user or not user.is_active:
                return None
            
            return {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
            
        except Exception as e:
            return None
        finally:
            session.close()
    
    def logout(self, session_token):
        """Logout user"""
        session = self.db_manager.get_session()
        
        try:
            session.query(UserSession).filter(
                UserSession.session_token == session_token
            ).delete()
            session.commit()
            
            return {
                'success': True,
                'message': 'Logout berhasil'
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error during logout: {str(e)}'
            }
        finally:
            session.close()

# Global auth service instance
auth_service = AuthService()
