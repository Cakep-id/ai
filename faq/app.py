"""
Flask Backend untuk FAQ NLP System dengan AI Language Learning
Main application file
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from datetime import datetime

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

# Routes
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
    session['user_role'] = 'admin'
    return render_template('admin.html')

@app.route('/chat')
def chat():
    """User chat interface"""
    session['user_role'] = 'user'
    return render_template('chat.html')

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
            'version': '2.0.0'
        }), 200 if db_status else 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """AI Chat endpoint dengan language learning"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Data tidak boleh kosong'
            }), 400
        
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Pesan tidak boleh kosong'
            }), 400
        
        # Get session
        session_id, user_role = get_or_create_session()
        user_ip = get_client_ip()
        
        # Analyze user message untuk language learning
        language_service.analyze_user_message(query, session_id)
        
        # Search for FAQ dengan threshold rendah untuk chat
        search_result = faq_service.search_faq(
            query=query,
            threshold=0.15,  # Threshold rendah untuk chat yang lebih fleksibel
            max_results=1,
            user_ip=user_ip
        )
        
        if search_result['success'] and search_result['results']:
            result = search_result['results'][0]
            
            # Adapt response style based on user's language patterns
            adapted_answer = language_service.adapt_response_style(
                result['answer'], 
                session_id
            )
            
            # Save chat feedback
            language_service.save_chat_feedback(
                session_id=session_id,
                user_message=query,
                bot_response=adapted_answer,
                faq_id=result['faq_id'],
                feedback_type='helpful',
                similarity_score=result['similarity_score']
            )
            
            return jsonify({
                'success': True,
                'response': adapted_answer,
                'metadata': {
                    'matched_question': result['question'],
                    'similarity_score': result['similarity_score'],
                    'match_type': result['match_type'],
                    'faq_id': result['faq_id']
                }
            })
        
        else:
            # No FAQ found, return helpful response
            response = ("Maaf, saya belum memiliki informasi tentang pertanyaan tersebut. "
                       "Apakah Anda bisa menggunakan kata kunci yang berbeda atau "
                       "menghubungi customer service untuk bantuan lebih lanjut?")
            
            # Adapt response style
            adapted_response = language_service.adapt_response_style(response, session_id)
            
            # Save chat feedback
            language_service.save_chat_feedback(
                session_id=session_id,
                user_message=query,
                bot_response=adapted_response,
                feedback_type='not_helpful'
            )
            
            return jsonify({
                'success': True,
                'response': adapted_response,
                'metadata': {
                    'matched_question': None,
                    'similarity_score': 0,
                    'match_type': 'no_match'
                }
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
        }), 500

@app.route('/api/chat/feedback', methods=['POST'])
def chat_feedback():
    """Endpoint untuk user feedback pada chat"""
    try:
        data = request.get_json()
        session_id, user_role = get_or_create_session()
        
        feedback_type = data.get('feedback_type', 'neutral')  # helpful, not_helpful, neutral
        feedback_text = data.get('feedback_text', '')
        user_message = data.get('user_message', '')
        bot_response = data.get('bot_response', '')
        faq_id = data.get('faq_id')
        
        result = language_service.save_chat_feedback(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            faq_id=faq_id,
            feedback_type=feedback_type,
            feedback_text=feedback_text
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error saving feedback: {str(e)}'
        }), 500

# FAQ Management Endpoints (Admin only for some)
@app.route('/api/faq', methods=['POST'])
def add_faq():
    """Add new FAQ (Admin only)"""
    session_id, user_role = get_or_create_session()
    
    if user_role != 'admin':
        return jsonify({
            'success': False,
            'message': 'Admin access required'
        }), 403
    
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

@app.route('/api/search', methods=['POST'])
def search_faq():
    """Search FAQ"""
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

@app.route('/api/language/insights', methods=['GET'])
def get_language_insights():
    """Get language learning insights (Admin only)"""
    session_id, user_role = get_or_create_session()
    
    if user_role != 'admin':
        return jsonify({
            'success': False,
            'message': 'Admin access required'
        }), 403
    
    try:
        days = int(request.args.get('days', 7))
        target_session = request.args.get('session_id')
        
        result = language_service.get_learning_insights(
            session_id=target_session,
            days=days
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting insights: {str(e)}'
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get search statistics (Admin only)"""
    session_id, user_role = get_or_create_session()
    
    if user_role != 'admin':
        return jsonify({
            'success': False,
            'message': 'Admin access required'
        }), 403
    
    try:
        limit = int(request.args.get('limit', 100))
        
        result = faq_service.get_search_statistics(limit)
        
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error server: {str(e)}'
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
    
    print(f"Starting FAQ NLP System v2.0 on http://{host}:{port}")
    print("Features: AI Language Learning, Admin/User Modes")
    app.run(host=host, port=port, debug=debug)
