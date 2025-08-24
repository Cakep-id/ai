"""
Environment loader for AgentV2
Load environment variables from .env file
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class EnvLoader:
    """Environment variables loader and manager"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.env_vars = {}
        self.load_env()
    
    def load_env(self) -> None:
        """Load environment variables from file"""
        if not self.env_file.exists():
            print(f"Warning: {self.env_file} not found. Using system environment variables only.")
            return
        
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Set environment variable
                        os.environ[key] = value
                        self.env_vars[key] = value
                    else:
                        print(f"Warning: Invalid line {line_num} in {self.env_file}: {line}")
        
        except Exception as e:
            print(f"Error loading {self.env_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> str:
        """Get environment variable with fallback"""
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get environment variable as integer"""
        try:
            return int(self.get(key, str(default)))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get environment variable as float"""
        try:
            return float(self.get(key, str(default)))
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean"""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_list(self, key: str, separator: str = ',', default: Optional[list] = None) -> list:
        """Get environment variable as list"""
        if default is None:
            default = []
        
        value = self.get(key, '')
        if not value:
            return default
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def require(self, key: str) -> str:
        """Get required environment variable, raise error if not found"""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' not found")
        return value
    
    def validate_database_config(self) -> Dict[str, Any]:
        """Validate database configuration"""
        try:
            config = {
                'host': self.require('DB_HOST'),
                'port': self.get_int('DB_PORT', 3306),
                'user': self.require('DB_USER'),
                'password': self.get('DB_PASSWORD', ''),
                'database': self.require('DB_NAME'),
                'charset': self.get('DB_CHARSET', 'utf8mb4')
            }
            
            # Validate port range
            if not (1 <= config['port'] <= 65535):
                raise ValueError("DB_PORT must be between 1 and 65535")
            
            return config
        
        except ValueError as e:
            raise ValueError(f"Database configuration error: {e}")
    
    def validate_api_config(self) -> Dict[str, Any]:
        """Validate API configuration"""
        try:
            config = {
                'host': self.get('API_HOST', '0.0.0.0'),
                'port': self.get_int('API_PORT', 8000),
                'debug': self.get_bool('DEBUG', False),
                'workers': self.get_int('API_WORKERS', 4)
            }
            
            # Validate port range
            if not (1 <= config['port'] <= 65535):
                raise ValueError("API_PORT must be between 1 and 65535")
            
            return config
        
        except ValueError as e:
            raise ValueError(f"API configuration error: {e}")
    
    def print_config_summary(self) -> None:
        """Print configuration summary"""
        print("=== AgentV2 Configuration Summary ===")
        
        try:
            db_config = self.validate_database_config()
            print(f"Database: {db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        except ValueError as e:
            print(f"Database: ERROR - {e}")
        
        try:
            api_config = self.validate_api_config()
            print(f"API Server: {api_config['host']}:{api_config['port']} (debug: {api_config['debug']})")
        except ValueError as e:
            print(f"API Server: ERROR - {e}")
        
        print(f"YOLO Device: {self.get('YOLO_DEVICE', 'cpu')}")
        print(f"Max File Size: {self.get('MAX_FILE_SIZE_MB', '50')} MB")
        print(f"Log Level: {self.get('LOG_LEVEL', 'INFO')}")
        print("=====================================")

# Global environment loader instance
env = EnvLoader()

# Convenience functions
def get_env(key: str, default: Any = None) -> str:
    """Get environment variable"""
    return env.get(key, default)

def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer"""
    return env.get_int(key, default)

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float"""
    return env.get_float(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean"""
    return env.get_bool(key, default)

def get_env_list(key: str, separator: str = ',', default: Optional[list] = None) -> list:
    """Get environment variable as list"""
    return env.get_list(key, separator, default)

if __name__ == "__main__":
    # Test the environment loader
    env.print_config_summary()
