"""
Simple Database Manager for testing
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class DatabaseManager:
    """Simple database manager with mock data for testing"""
    
    def __init__(self):
        """Initialize database manager"""
        self.connected = False
        self.mock_data = {
            'assets': {},
            'inspections': {},
            'reports': {},
            'users': {
                'admin': {
                    'id': 'admin',
                    'username': 'admin',
                    'role': 'admin',
                    'password_hash': 'mock_hash',
                    'created_at': datetime.now().isoformat()
                }
            }
        }
        print("Database Manager initialized (mock mode)")
    
    async def connect(self):
        """Connect to database (mock)"""
        try:
            self.connected = True
            print("Database connected (mock)")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database (mock)"""
        self.connected = False
        print("Database disconnected (mock)")
    
    # Asset methods
    async def create_asset(self, asset_data: Dict[str, Any]) -> str:
        """Create new asset"""
        asset_id = asset_data.get('id', f"ASSET_{len(self.mock_data['assets']) + 1:03d}")
        self.mock_data['assets'][asset_id] = {
            **asset_data,
            'id': asset_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"Asset created: {asset_id}")
        return asset_id
    
    async def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset by ID"""
        return self.mock_data['assets'].get(asset_id)
    
    async def list_assets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List assets"""
        assets = list(self.mock_data['assets'].values())
        return assets[:limit]
    
    async def update_asset(self, asset_id: str, update_data: Dict[str, Any]) -> bool:
        """Update asset"""
        if asset_id in self.mock_data['assets']:
            self.mock_data['assets'][asset_id].update(update_data)
            self.mock_data['assets'][asset_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False
    
    # Inspection methods
    async def create_inspection(self, inspection_data: Dict[str, Any]) -> str:
        """Create new inspection"""
        inspection_id = inspection_data.get('id', f"INSP_{len(self.mock_data['inspections']) + 1:03d}")
        self.mock_data['inspections'][inspection_id] = {
            **inspection_data,
            'id': inspection_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"Inspection created: {inspection_id}")
        return inspection_id
    
    async def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """Get inspection by ID"""
        return self.mock_data['inspections'].get(inspection_id)
    
    async def list_inspections(self, asset_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List inspections"""
        inspections = list(self.mock_data['inspections'].values())
        if asset_id:
            inspections = [i for i in inspections if i.get('asset_id') == asset_id]
        return inspections[:limit]
    
    async def update_inspection(self, inspection_id: str, update_data: Dict[str, Any]) -> bool:
        """Update inspection"""
        if inspection_id in self.mock_data['inspections']:
            self.mock_data['inspections'][inspection_id].update(update_data)
            self.mock_data['inspections'][inspection_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False
    
    # Report methods
    async def create_report(self, report_data: Dict[str, Any]) -> str:
        """Create new report"""
        report_id = report_data.get('id', f"RPT_{len(self.mock_data['reports']) + 1:03d}")
        self.mock_data['reports'][report_id] = {
            **report_data,
            'id': report_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"Report created: {report_id}")
        return report_id
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report by ID"""
        return self.mock_data['reports'].get(report_id)
    
    async def list_reports(self, asset_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List reports"""
        reports = list(self.mock_data['reports'].values())
        if asset_id:
            reports = [r for r in reports if r.get('asset_id') == asset_id]
        return reports[:limit]
    
    # User methods
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user"""
        user_id = user_data.get('id', f"USER_{len(self.mock_data['users']) + 1:03d}")
        self.mock_data['users'][user_id] = {
            **user_data,
            'id': user_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"User created: {user_id}")
        return user_id
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        return self.mock_data['users'].get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        for user in self.mock_data['users'].values():
            if user.get('username') == username:
                return user
        return None
    
    async def list_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List users"""
        users = list(self.mock_data['users'].values())
        return users[:limit]
    
    # Detection methods
    async def save_detection_results(self, detection_data: Dict[str, Any]) -> str:
        """Save detection results"""
        detection_id = f"DET_{len(self.mock_data.get('detections', {})) + 1:03d}"
        if 'detections' not in self.mock_data:
            self.mock_data['detections'] = {}
        
        self.mock_data['detections'][detection_id] = {
            **detection_data,
            'id': detection_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"Detection results saved: {detection_id}")
        return detection_id
    
    async def get_detection_results(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get detection results by ID"""
        return self.mock_data.get('detections', {}).get(detection_id)
    
    # Training methods
    async def save_training_session(self, training_data: Dict[str, Any]) -> str:
        """Save training session"""
        session_id = f"TRN_{len(self.mock_data.get('training_sessions', {})) + 1:03d}"
        if 'training_sessions' not in self.mock_data:
            self.mock_data['training_sessions'] = {}
        
        self.mock_data['training_sessions'][session_id] = {
            **training_data,
            'id': session_id,
            'created_at': datetime.now().isoformat()
        }
        print(f"Training session saved: {session_id}")
        return session_id
    
    async def get_training_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session by ID"""
        return self.mock_data.get('training_sessions', {}).get(session_id)
    
    async def list_training_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List training sessions"""
        sessions = list(self.mock_data.get('training_sessions', {}).values())
        return sessions[:limit]
    
    # Utility methods
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        return {
            'status': 'healthy' if self.connected else 'disconnected',
            'mode': 'mock',
            'data_counts': {
                'assets': len(self.mock_data['assets']),
                'inspections': len(self.mock_data['inspections']),
                'reports': len(self.mock_data['reports']),
                'users': len(self.mock_data['users'])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_assets': len(self.mock_data['assets']),
            'total_inspections': len(self.mock_data['inspections']),
            'total_reports': len(self.mock_data['reports']),
            'total_users': len(self.mock_data['users']),
            'mode': 'mock'
        }

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_db():
        db = DatabaseManager()
        await db.connect()
        
        # Test asset creation
        asset_id = await db.create_asset({
            'name': 'Test Asset',
            'type': 'pipeline',
            'location': 'Test Location'
        })
        
        asset = await db.get_asset(asset_id)
        print(f"Created asset: {asset}")
        
        # Test health check
        health = await db.health_check()
        print(f"Health check: {health}")
        
        await db.disconnect()
    
    asyncio.run(test_db())
