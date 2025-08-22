"""
Flask Backend untuk FAQ NLP System
Main application file
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Import modules
from database.connection import db_manager
from services.faq_service import faq_service
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

@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template('index.html')

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
def add_faq():
    """Add new FAQ"""
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
def update_faq(faq_id):
    """Update FAQ by ID"""
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
def delete_faq(faq_id):
    """Delete FAQ by ID"""
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
        threshold = float(data.get('threshold', 0.3))
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
def get_statistics():
    """Get search statistics"""
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
def train_model():
    """Endpoint untuk training model (placeholder)"""
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
