"""
NLP API Endpoints
Endpoints untuk analisis teks menggunakan Groq AI
"""

from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from services import groq_service, db_service

router = APIRouter()

# Pydantic models
class AnalysisRequest(BaseModel):
    text: str = Field(..., description="Teks deskripsi kerusakan", min_length=1, max_length=2000)
    asset_context: Optional[str] = Field(None, description="Konteks aset (misal: pompa air, valve)")
    report_id: Optional[int] = Field(None, description="Report ID untuk logging")
    save_result: bool = Field(True, description="Simpan hasil ke database")

class AnalysisResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    keyphrases: List[str]
    severity: str
    risk_indicators: List[str]
    recommendations: List[str]
    text_score: float
    model_version: str
    analysis_method: str
    error: Optional[str] = None

class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List teks untuk analisis batch")
    asset_context: Optional[str] = Field(None, description="Konteks aset")

class CategoryStatsResponse(BaseModel):
    category: str
    count: int
    avg_confidence: float
    avg_text_score: float

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """
    Analisis teks deskripsi kerusakan menggunakan Groq AI
    
    - **text**: Teks deskripsi kerusakan (1-2000 karakter)
    - **asset_context**: Konteks aset untuk analisis yang lebih akurat
    - **report_id**: ID report untuk logging ke database
    - **save_result**: Simpan hasil ke database (default: True)
    """
    try:
        # Validasi input text
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        logger.info(f"Analyzing text: {request.text[:100]}...")
        
        # Run NLP analysis
        analysis_result = groq_service.analyze(
            text=request.text,
            asset_context=request.asset_context
        )
        
        if not analysis_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        # Calculate text score untuk risk engine
        text_score = groq_service.get_text_score(analysis_result)
        analysis_result['text_score'] = text_score
        
        # Simpan hasil ke database jika diminta dan ada report_id
        if request.save_result and request.report_id:
            try:
                db_service.save_nlp_analysis(
                    report_id=request.report_id,
                    category=analysis_result['category'],
                    confidence=analysis_result['confidence'],
                    keyphrases=analysis_result['keyphrases'],
                    model_ver=analysis_result['model_version']
                )
                
                logger.info(f"Saved NLP analysis to database for report {request.report_id}")
                
            except Exception as e:
                logger.error(f"Failed to save NLP analysis: {e}")
                # Don't fail the request, just log the error
        
        return AnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analisis batch untuk multiple teks
    
    - **texts**: List teks untuk dianalisis
    - **asset_context**: Konteks aset (opsional)
    """
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts list cannot be empty"
            )
        
        if len(request.texts) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 50 texts per request"
            )
        
        logger.info(f"Analyzing batch of {len(request.texts)} texts")
        
        results = []
        
        for i, text in enumerate(request.texts):
            try:
                if not text.strip():
                    results.append({
                        'index': i,
                        'success': False,
                        'error': 'Empty text'
                    })
                    continue
                
                analysis_result = groq_service.analyze(
                    text=text,
                    asset_context=request.asset_context
                )
                
                analysis_result['index'] = i
                analysis_result['text_score'] = groq_service.get_text_score(analysis_result)
                
                results.append(analysis_result)
                
            except Exception as e:
                logger.error(f"Failed to analyze text {i}: {e}")
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary statistics
        successful_results = [r for r in results if r.get('success', False)]
        
        summary = {
            'total_texts': len(request.texts),
            'successful': len(successful_results),
            'failed': len(request.texts) - len(successful_results),
            'avg_confidence': sum(r.get('confidence', 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            'categories': {}
        }
        
        # Category distribution
        for result in successful_results:
            category = result.get('category', 'unknown')
            if category in summary['categories']:
                summary['categories'][category] += 1
            else:
                summary['categories'][category] = 1
        
        return {
            'success': True,
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/categories")
async def get_damage_categories():
    """Get daftar kategori kerusakan yang didukung"""
    try:
        # Get categories dari groq service
        categories = groq_service.damage_categories
        
        category_info = []
        for category, info in categories.items():
            category_info.append({
                'category': category,
                'keywords': info['keywords'],
                'risk_level': info['risk_level'],
                'confidence_boost': info['confidence_boost']
            })
        
        # Get statistics dari database
        try:
            stats_query = """
            SELECT 
                category,
                COUNT(*) as analysis_count,
                AVG(confidence) as avg_confidence
            FROM nlp_analyses 
            WHERE analyzed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY category
            ORDER BY analysis_count DESC
            """
            
            db_stats = db_service.execute_query(stats_query)
            
            # Merge dengan category info
            stats_map = {stat['category']: stat for stat in db_stats}
            for cat_info in category_info:
                category = cat_info['category']
                if category in stats_map:
                    cat_info['usage_stats'] = stats_map[category]
                else:
                    cat_info['usage_stats'] = {
                        'analysis_count': 0,
                        'avg_confidence': 0.0
                    }
        
        except Exception as e:
            logger.warning(f"Failed to get category stats: {e}")
            for cat_info in category_info:
                cat_info['usage_stats'] = {
                    'analysis_count': 0,
                    'avg_confidence': 0.0
                }
        
        return {
            'success': True,
            'categories': category_info,
            'total_categories': len(category_info),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get categories failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.get("/stats")
async def get_analysis_stats():
    """Get statistik analisis NLP dari database"""
    try:
        # Category distribution
        category_query = """
        SELECT 
            category,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            MAX(confidence) as max_confidence,
            MIN(confidence) as min_confidence
        FROM nlp_analyses 
        WHERE analyzed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY category
        ORDER BY count DESC
        """
        
        category_stats = db_service.execute_query(category_query)
        
        # Severity distribution (if available)
        recent_query = """
        SELECT 
            n.*,
            r.asset_id,
            r.description as report_description
        FROM nlp_analyses n
        JOIN reports r ON n.report_id = r.report_id
        ORDER BY n.analyzed_at DESC
        LIMIT 20
        """
        
        recent_analyses = db_service.execute_query(recent_query)
        
        # Time series data (last 7 days)
        time_series_query = """
        SELECT 
            DATE(analyzed_at) as analysis_date,
            COUNT(*) as daily_count,
            AVG(confidence) as daily_avg_confidence
        FROM nlp_analyses 
        WHERE analyzed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY DATE(analyzed_at)
        ORDER BY analysis_date
        """
        
        time_series = db_service.execute_query(time_series_query)
        
        return {
            'success': True,
            'stats': {
                'category_distribution': category_stats,
                'recent_analyses': recent_analyses,
                'time_series': time_series,
                'total_analyses': sum(stat['count'] for stat in category_stats),
                'unique_categories': len(category_stats)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get analysis stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis stats: {str(e)}")

@router.get("/test-connection")
async def test_groq_connection():
    """Test koneksi ke Groq AI API"""
    try:
        connection_test = groq_service.test_connection()
        
        return {
            'success': True,
            'connection_test': connection_test,
            'service_info': {
                'model': groq_service.model,
                'api_configured': groq_service.api_key is not None
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

@router.post("/retrain")
async def retrain_nlp_model():
    """
    Placeholder untuk NLP model retraining
    Saat ini Groq menggunakan model cloud, jadi tidak ada retraining lokal
    """
    try:
        # Untuk implementasi future: custom NLP model training
        return {
            'success': True,
            'message': 'NLP model menggunakan Groq AI cloud service. Retraining tidak diperlukan.',
            'model_info': {
                'type': 'cloud_service',
                'provider': 'Groq',
                'model': groq_service.model,
                'version': groq_service._get_model_version()
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"NLP retrain endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain endpoint error: {str(e)}")

@router.get("/keyphrases/common")
async def get_common_keyphrases(limit: int = 20):
    """Get keyphrases yang paling sering muncul"""
    try:
        # Query untuk common keyphrases
        query = """
        SELECT 
            JSON_UNQUOTE(JSON_EXTRACT(keyphrases, CONCAT('$[', idx, ']'))) as keyphrase,
            COUNT(*) as frequency
        FROM (
            SELECT 
                keyphrases,
                0 as idx UNION ALL SELECT keyphrases, 1 UNION ALL SELECT keyphrases, 2 
                UNION ALL SELECT keyphrases, 3 UNION ALL SELECT keyphrases, 4
        ) t
        JOIN nlp_analyses n ON t.keyphrases = n.keyphrases
        WHERE JSON_UNQUOTE(JSON_EXTRACT(keyphrases, CONCAT('$[', idx, ']'))) IS NOT NULL
        AND analyzed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY keyphrase
        HAVING keyphrase != 'null'
        ORDER BY frequency DESC
        LIMIT :limit
        """
        
        common_keyphrases = db_service.execute_query(query, {'limit': limit})
        
        return {
            'success': True,
            'common_keyphrases': common_keyphrases,
            'total_found': len(common_keyphrases),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get common keyphrases failed: {e}")
        # Return empty result instead of error untuk optional feature
        return {
            'success': True,
            'common_keyphrases': [],
            'total_found': 0,
            'note': 'Keyphrases extraction temporarily unavailable',
            'timestamp': datetime.now().isoformat()
        }
