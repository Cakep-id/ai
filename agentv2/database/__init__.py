"""
Database Package for AgentV2
Contains database manager and data access objects
"""

__version__ = "2.0.0"

from .db_manager import DatabaseManager, create_database_manager, create_daos

__all__ = [
    "DatabaseManager",
    "create_database_manager", 
    "create_daos"
]
