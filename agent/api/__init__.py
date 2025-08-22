"""
API Package
FastAPI endpoints untuk CAKEP.id EWS AI Module
"""

from . import cv_endpoints
from . import nlp_endpoints  
from . import risk_endpoints
from . import schedule_endpoints
from . import admin_endpoints

__all__ = [
    'cv_endpoints',
    'nlp_endpoints',
    'risk_endpoints', 
    'schedule_endpoints',
    'admin_endpoints'
]
