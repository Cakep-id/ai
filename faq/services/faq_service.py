"""
FAQ Service module untuk business logic
"""

from database.connection import db_manager, FAQDataset, FAQVariations, SearchLogs
from nlp.processor import nlp_processor
from sqlalchemy import or_, desc
from sqlalchemy.orm import joinedload
import json
from datetime import datetime

class FAQService:
    def __init__(self):
        self.db_manager = db_manager
    
    def add_faq(self, question, answer, category='general', variations=None):
        """Add new FAQ dengan variations opsional"""
        session = self.db_manager.get_session()
        
        try:
            # Create main FAQ
            new_faq = FAQDataset(
                question=question.strip(),
                answer=answer.strip(),
                category=category.strip() if category else 'general'
            )
            
            session.add(new_faq)
            session.flush()  # Get ID
            
            # Add variations jika ada
            if variations and isinstance(variations, list):
                for variation in variations:
                    if variation.strip():
                        # Calculate similarity dengan main question
                        similarity = nlp_processor.calculate_similarity(
                            variation.strip(), 
                            [question.strip()]
                        )[0] if question.strip() else 1.0
                        
                        new_variation = FAQVariations(
                            faq_id=new_faq.id,
                            variation_question=variation.strip(),
                            similarity_score=round(similarity, 4)
                        )
                        session.add(new_variation)
            
            session.commit()
            
            return {
                'success': True,
                'message': 'FAQ berhasil ditambahkan',
                'faq_id': new_faq.id
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error menambahkan FAQ: {str(e)}'
            }
        finally:
            session.close()
    
    def get_all_faqs(self, include_variations=True, category=None, is_active=True):
        """Get semua FAQ dengan variations"""
        session = self.db_manager.get_session()
        
        try:
            query = session.query(FAQDataset)
            
            if include_variations:
                query = query.options(joinedload(FAQDataset.variations))
            
            if category:
                query = query.filter(FAQDataset.category == category)
            
            if is_active is not None:
                query = query.filter(FAQDataset.is_active == is_active)
            
            query = query.order_by(desc(FAQDataset.created_at))
            faqs = query.all()
            
            result = []
            for faq in faqs:
                faq_data = {
                    'id': faq.id,
                    'question': faq.question,
                    'answer': faq.answer,
                    'category': faq.category,
                    'is_active': faq.is_active,
                    'created_at': faq.created_at.isoformat() if faq.created_at else None,
                    'updated_at': faq.updated_at.isoformat() if faq.updated_at else None
                }
                
                if include_variations:
                    faq_data['variations'] = []
                    for var in faq.variations:
                        faq_data['variations'].append({
                            'id': var.id,
                            'variation_question': var.variation_question,
                            'similarity_score': float(var.similarity_score),
                            'created_at': var.created_at.isoformat() if var.created_at else None
                        })
                
                result.append(faq_data)
            
            return {
                'success': True,
                'data': result,
                'total': len(result)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error mengambil FAQ: {str(e)}',
                'data': []
            }
        finally:
            session.close()
    
    def search_faq(self, query, threshold=0.2, max_results=10, user_ip=None):
        """Search FAQ berdasarkan query dengan NLP - improved version"""
        if not query or not query.strip():
            return {
                'success': False,
                'message': 'Query tidak boleh kosong',
                'results': []
            }
        
        session = self.db_manager.get_session()
        
        try:
            # Get semua FAQ dengan variations
            faqs_data = self.get_all_faqs(include_variations=True)
            
            if not faqs_data['success'] or not faqs_data['data']:
                return {
                    'success': True,
                    'message': 'Tidak ada FAQ ditemukan',
                    'results': []
                }
            
            # Debug: Print available FAQs
            print(f"Available FAQs: {len(faqs_data['data'])}")
            for faq in faqs_data['data'][:3]:  # Print first 3 for debugging
                print(f"- {faq['question'][:50]}...")
            
            # Find best matches menggunakan improved NLP
            best_match = nlp_processor.find_best_match(
                query.strip(), 
                faqs_data['data'], 
                threshold
            )
            
            results = []
            
            if best_match:
                faq_data = best_match['faq']
                
                result = {
                    'faq_id': faq_data['id'],
                    'question': faq_data['question'],
                    'answer': faq_data['answer'],
                    'category': faq_data['category'],
                    'similarity_score': round(best_match['similarity_score'], 4),
                    'matched_question': best_match['matched_question'],
                    'match_type': best_match['match_type']
                }
                
                results.append(result)
                
                # Log search
                self._log_search(session, query.strip(), faq_data['id'], 
                               best_match['similarity_score'], user_ip)
                
                print(f"Match found: {best_match['match_type']} - Score: {best_match['similarity_score']}")
            else:
                # Log search tanpa hasil
                self._log_search(session, query.strip(), None, 0.0, user_ip)
                print(f"No match found for: {query}")
            
            return {
                'success': True,
                'message': f'Ditemukan {len(results)} hasil pencarian',
                'query': query.strip(),
                'results': results
            }
            
        except Exception as e:
            print(f"Search error: {e}")
            return {
                'success': False,
                'message': f'Error searching FAQ: {str(e)}',
                'results': []
            }
        finally:
            session.close()
    
    def _log_search(self, session, query, faq_id, similarity_score, user_ip):
        """Log pencarian ke database"""
        try:
            search_log = SearchLogs(
                search_query=query,
                result_faq_id=faq_id,
                similarity_score=round(similarity_score, 4) if similarity_score else None,
                user_ip=user_ip
            )
            session.add(search_log)
            session.commit()
        except Exception as e:
            print(f"Error logging search: {e}")
            session.rollback()
    
    def get_faq_by_id(self, faq_id, include_variations=True):
        """Get FAQ berdasarkan ID"""
        session = self.db_manager.get_session()
        
        try:
            query = session.query(FAQDataset).filter(FAQDataset.id == faq_id)
            
            if include_variations:
                query = query.options(joinedload(FAQDataset.variations))
            
            faq = query.first()
            
            if not faq:
                return {
                    'success': False,
                    'message': 'FAQ tidak ditemukan'
                }
            
            faq_data = {
                'id': faq.id,
                'question': faq.question,
                'answer': faq.answer,
                'category': faq.category,
                'is_active': faq.is_active,
                'created_at': faq.created_at.isoformat() if faq.created_at else None,
                'updated_at': faq.updated_at.isoformat() if faq.updated_at else None
            }
            
            if include_variations:
                faq_data['variations'] = []
                for var in faq.variations:
                    faq_data['variations'].append({
                        'id': var.id,
                        'variation_question': var.variation_question,
                        'similarity_score': float(var.similarity_score),
                        'created_at': var.created_at.isoformat() if var.created_at else None
                    })
            
            return {
                'success': True,
                'data': faq_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error mengambil FAQ: {str(e)}'
            }
        finally:
            session.close()
    
    def update_faq(self, faq_id, question=None, answer=None, category=None, 
                   is_active=None, variations=None):
        """Update FAQ berdasarkan ID"""
        session = self.db_manager.get_session()
        
        try:
            faq = session.query(FAQDataset).filter(FAQDataset.id == faq_id).first()
            
            if not faq:
                return {
                    'success': False,
                    'message': 'FAQ tidak ditemukan'
                }
            
            # Update fields
            if question is not None:
                faq.question = question.strip()
            if answer is not None:
                faq.answer = answer.strip()
            if category is not None:
                faq.category = category.strip()
            if is_active is not None:
                faq.is_active = is_active
            
            # Update variations jika ada
            if variations is not None:
                # Delete existing variations
                session.query(FAQVariations).filter(FAQVariations.faq_id == faq_id).delete()
                
                # Add new variations
                if isinstance(variations, list):
                    for variation in variations:
                        if variation.strip():
                            similarity = nlp_processor.calculate_similarity(
                                variation.strip(), 
                                [faq.question]
                            )[0] if faq.question else 1.0
                            
                            new_variation = FAQVariations(
                                faq_id=faq_id,
                                variation_question=variation.strip(),
                                similarity_score=round(similarity, 4)
                            )
                            session.add(new_variation)
            
            session.commit()
            
            return {
                'success': True,
                'message': 'FAQ berhasil diupdate',
                'faq_id': faq_id
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error updating FAQ: {str(e)}'
            }
        finally:
            session.close()
    
    def delete_faq(self, faq_id):
        """Delete FAQ berdasarkan ID"""
        session = self.db_manager.get_session()
        
        try:
            faq = session.query(FAQDataset).filter(FAQDataset.id == faq_id).first()
            
            if not faq:
                return {
                    'success': False,
                    'message': 'FAQ tidak ditemukan'
                }
            
            session.delete(faq)
            session.commit()
            
            return {
                'success': True,
                'message': 'FAQ berhasil dihapus'
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error deleting FAQ: {str(e)}'
            }
        finally:
            session.close()
    
    def get_search_statistics(self, limit=100):
        """Get statistik pencarian"""
        session = self.db_manager.get_session()
        
        try:
            # Total searches
            total_searches = session.query(SearchLogs).count()
            
            # Successful searches (yang menemukan hasil)
            successful_searches = session.query(SearchLogs).filter(
                SearchLogs.result_faq_id.isnot(None)
            ).count()
            
            # Recent searches
            recent_searches = session.query(SearchLogs).order_by(
                desc(SearchLogs.created_at)
            ).limit(limit).all()
            
            # Popular FAQs
            popular_faqs = session.query(
                SearchLogs.result_faq_id,
                session.query(SearchLogs).filter(
                    SearchLogs.result_faq_id == SearchLogs.result_faq_id
                ).count().label('search_count')
            ).filter(
                SearchLogs.result_faq_id.isnot(None)
            ).group_by(SearchLogs.result_faq_id).order_by(
                desc('search_count')
            ).limit(10).all()
            
            return {
                'success': True,
                'data': {
                    'total_searches': total_searches,
                    'successful_searches': successful_searches,
                    'success_rate': round(successful_searches / total_searches * 100, 2) if total_searches > 0 else 0,
                    'recent_searches': [
                        {
                            'query': log.search_query,
                            'found_result': log.result_faq_id is not None,
                            'similarity_score': float(log.similarity_score) if log.similarity_score else 0,
                            'created_at': log.created_at.isoformat() if log.created_at else None
                        }
                        for log in recent_searches
                    ],
                    'popular_faq_ids': [faq_id for faq_id, count in popular_faqs]
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error getting statistics: {str(e)}'
            }
        finally:
            session.close()

# Global FAQ service instance
faq_service = FAQService()
