"""
Flask Backend untuk FAQ NLP System
Main application file
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from functools import wraps

# Import modules
from database.connection import db_manager
from services.faq_service import faq_service
from services.language_service import language_service
from nlp.processor import nlp_processor

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='frontend', static_folder='frontend/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key_here')

# Enable CORS
CORS(app)

def get_client_ip():
    """Get client IP address"""
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR']

def get_or_create_session():
    """Get or create user session"""
    if 'user_session_id' not in session:
        # Create new session
        result = language_service.create_or_get_session()
        if result['success']:
            session['user_session_id'] = result['session_id']
            session['user_role'] = result['user_role']
        else:
            session['user_session_id'] = None
            session['user_role'] = 'user'
    
    return session.get('user_session_id'), session.get('user_role', 'user')

def require_admin(f):
    """Decorator untuk require admin access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id, user_role = get_or_create_session()
        if user_role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Admin access required'
            }), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Serve main page - redirect to chat for users"""
    session_id, user_role = get_or_create_session()
    
    if user_role == 'admin':
        return render_template('admin.html')
    else:
        return render_template('chat.html')

@app.route('/admin')
def admin():
    """Admin interface"""
    session_id, user_role = get_or_create_session()
    
    # For demo purposes, set role to admin
    session['user_role'] = 'admin'
    
    return render_template('admin.html')

@app.route('/chat')
def chat():
    """User chat interface"""
    session_id, user_role = get_or_create_session()
    session['user_role'] = 'user'
    
    return render_template('chat.html')

@app.route('/api/session', methods=['GET', 'POST'])
def manage_session():
    """Manage user session"""
    if request.method == 'POST':
        data = request.get_json() or {}
        user_role = data.get('role', 'user')
        
        result = language_service.create_or_get_session(user_role=user_role)
        if result['success']:
            session['user_session_id'] = result['session_id']
            session['user_role'] = result['user_role']
        
        return jsonify(result)
    
    else:
        session_id, user_role = get_or_create_session()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'user_role': user_role
        })
                'success': False,
                'message': 'Akses admin diperlukan'
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Serve admin interface"""
    return render_template('admin.html')

@app.route('/user')
def user():
    """Serve user interface"""
    return render_template('user.html')

# Authentication endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong'
            }), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'user').strip()
        
        if not username or not email or not password:
            return jsonify({
                'success': False,
                'message': 'Username, email, dan password harus diisi'
            }), 400
        
        result = auth_service.create_user(username, email, password, role)
        
        return jsonify(result), 201 if result['success'] else 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong'
            }), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({
                'success': False,
                'message': 'Username dan password harus diisi'
            }), 400
        
        result = auth_service.authenticate_user(username, password)
        
        return jsonify(result), 200 if result['success'] else 401
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """Logout user"""
    try:
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        
        result = auth_service.logout(token)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user info"""
    return jsonify({
        'success': True,
        'user': request.current_user
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = db_manager.test_connection()
        
        return jsonify({
            'status': 'healthy' if db_status else 'unhealthy',
            'database': 'connected' if db_status else 'disconnected',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }), 200 if db_status else 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/faq', methods=['POST'])
@require_auth
@require_admin
def add_faq():
    """Add new FAQ - Admin only"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong'
            }), 400
        
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        category = data.get('category', 'general').strip()
        variations = data.get('variations', [])
        
        # Validasi input
        if not question:
            return jsonify({
                'success': False,
                'message': 'Pertanyaan tidak boleh kosong'
            }), 400
        
        if not answer:
            return jsonify({
                'success': False,
                'message': 'Jawaban tidak boleh kosong'
            }), 400
        
        # Process variations
        processed_variations = []
        if variations:
            if isinstance(variations, str):
                # Split string variations by newline atau comma
                variations = [v.strip() for v in variations.replace(',', '\n').split('\n') if v.strip()]
            
            for var in variations:
                if isinstance(var, str) and var.strip():
                    processed_variations.append(var.strip())
        
        # Add FAQ
        result = faq_service.add_faq(
            question=question,
            answer=answer,
            category=category,
            variations=processed_variations
        )
        
        return jsonify(result), 201 if result['success'] else 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/faq', methods=['GET'])
def get_faqs():
    """Get all FAQs"""
    try:
        include_variations = request.args.get('include_variations', 'true').lower() == 'true'
        category = request.args.get('category')
        is_active = request.args.get('is_active')
        
        # Convert is_active parameter
        if is_active is not None:
            is_active = is_active.lower() == 'true'
        
        result = faq_service.get_all_faqs(
            include_variations=include_variations,
            category=category,
            is_active=is_active
        )
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}',
            'data': []
        }), 500

@app.route('/api/faq/<int:faq_id>', methods=['GET'])
def get_faq(faq_id):
    """Get FAQ by ID"""
    try:
        include_variations = request.args.get('include_variations', 'true').lower() == 'true'
        
        result = faq_service.get_faq_by_id(faq_id, include_variations)
        
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/faq/<int:faq_id>', methods=['PUT'])
@require_auth
@require_admin
def update_faq(faq_id):
    """Update FAQ by ID - Admin only"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong'
            }), 400
        
        question = data.get('question')
        answer = data.get('answer')
        category = data.get('category')
        is_active = data.get('is_active')
        variations = data.get('variations')
        
        # Process variations
        processed_variations = None
        if variations is not None:
            processed_variations = []
            if isinstance(variations, str):
                variations = [v.strip() for v in variations.replace(',', '\n').split('\n') if v.strip()]
            
            for var in variations:
                if isinstance(var, str) and var.strip():
                    processed_variations.append(var.strip())
        
        result = faq_service.update_faq(
            faq_id=faq_id,
            question=question,
            answer=answer,
            category=category,
            is_active=is_active,
            variations=processed_variations
        )
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/faq/<int:faq_id>', methods=['DELETE'])
@require_auth
@require_admin
def delete_faq(faq_id):
    """Delete FAQ by ID - Admin only"""
    try:
        result = faq_service.delete_faq(faq_id)
        
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/search', methods=['POST'])
def search_faq():
    """Search FAQ dengan language learning"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong',
                'results': []
            }), 400
        
        query = data.get('query', '').strip()
        threshold = float(data.get('threshold', 0.2))
        max_results = int(data.get('max_results', 10))
        session_id = data.get('session_id', 'anonymous')
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Query pencarian tidak boleh kosong',
                'results': []
            }), 400
        
        # Get client IP
        user_ip = get_client_ip()
        
        # Search FAQ
        result = faq_service.search_faq(
            query=query,
            threshold=threshold,
            max_results=max_results,
            user_ip=user_ip
        )
        
        # If result found, adapt response based on user language profile
        if result['results']:
            # Get user language profile
            language_profile = language_learning_service.get_user_language_profile(session_id)
            
            # Adapt response
            for faq_result in result['results']:
                original_answer = faq_result['answer']
                adapted_answer = language_learning_service.adapt_response(
                    original_answer, 
                    language_profile
                )
                faq_result['answer'] = adapted_answer
                faq_result['language_adapted'] = True
                faq_result['user_profile'] = language_profile
            
            # Learn from interaction
            if result['results']:
                best_result = result['results'][0]
                language_learning_service.learn_from_interaction(
                    session_id=session_id,
                    user_message=query,
                    bot_response=best_result['answer']
                )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}',
            'results': []
        }), 500

@app.route('/api/search', methods=['GET'])
def search_faq_get():
    """Search FAQ via GET parameter"""
    try:
        query = request.args.get('q', '').strip()
        threshold = float(request.args.get('threshold', 0.3))
        max_results = int(request.args.get('max_results', 10))
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Query pencarian tidak boleh kosong',
                'results': []
            }), 400
        
        # Get client IP
        user_ip = get_client_ip()
        
        result = faq_service.search_faq(
            query=query,
            threshold=threshold,
            max_results=max_results,
            user_ip=user_ip
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}',
            'results': []
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories"""
    try:
        # Query distinct categories dari database
        result = db_manager.execute_raw_query(
            "SELECT DISTINCT category FROM faq_dataset WHERE is_active = TRUE ORDER BY category"
        )
        
        categories = [row['category'] for row in result] if result else []
        
        return jsonify({
            'success': True,
            'data': categories
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}',
            'data': []
        }), 500

@app.route('/api/statistics', methods=['GET'])
@require_auth
@require_admin
def get_statistics():
    """Get search statistics - Admin only"""
    try:
        limit = int(request.args.get('limit', 100))
        
        result = faq_service.get_search_statistics(limit)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
@require_auth
@require_admin
def train_model():
    """Endpoint untuk training model (placeholder) - Admin only"""
    try:
        # Untuk sekarang, ini hanya placeholder
        # Bisa diimplementasikan untuk retrain TF-IDF vectorizer
        # atau model NLP lainnya
        
        return jsonify({
            'success': True,
            'message': 'Model training completed (placeholder)',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        }), 500

# Language Learning Endpoints
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback untuk response"""
    try:
        data = request.get_json()
        
        session_id = data.get('session_id', 'anonymous')
        user_message = data.get('user_message', '')
        bot_response = data.get('bot_response', '')
        feedback = data.get('feedback', '')  # 'helpful' or 'not_helpful'
        
        result = language_learning_service.learn_from_interaction(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            user_feedback=feedback
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error submitting feedback: {str(e)}'
        }), 500

@app.route('/api/language-profile/<session_id>', methods=['GET'])
def get_language_profile(session_id):
    """Get user language profile"""
    try:
        profile = language_learning_service.get_user_language_profile(session_id)
        
        return jsonify({
            'success': True,
            'profile': profile
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting language profile: {str(e)}'
        }), 500

@app.route('/api/admin/interactions', methods=['GET'])
@require_auth
@require_admin
def get_user_interactions():
    """Get user interactions untuk admin analysis"""
    try:
        from services.language_learning_service import UserInteraction
        from sqlalchemy import desc
        
        session = db_manager.get_session()
        
        limit = int(request.args.get('limit', 50))
        
        interactions = session.query(UserInteraction).order_by(
            desc(UserInteraction.created_at)
        ).limit(limit).all()
        
        result = []
        for interaction in interactions:
            result.append({
                'id': interaction.id,
                'session_id': interaction.session_id,
                'user_message': interaction.user_message,
                'bot_response': interaction.bot_response,
                'user_feedback': interaction.user_feedback,
                'language_style': json.loads(interaction.language_style) if interaction.language_style else None,
                'created_at': interaction.created_at.isoformat() if interaction.created_at else None
            })
        
        session.close()
        
        return jsonify({
            'success': True,
            'data': result,
            'total': len(result)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting interactions: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({
        'success': False,
        'message': 'Endpoint tidak ditemukan'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """405 error handler"""
    return jsonify({
        'success': False,
        'message': 'Method tidak diizinkan'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Test database connection on startup
    print("Testing database connection...")
    if db_manager.test_connection():
        print("Database connected successfully!")
    else:
        print("Warning: Database connection failed!")
    
    # Get configuration
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting FAQ NLP System on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
