#!/usr/bin/env python3
"""
Environment Setup Script for AgentV2
Helps configure environment variables and validate settings
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

def load_env_file(filepath: str) -> Dict[str, str]:
    """Load environment variables from file"""
    env_vars = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def validate_env_vars(env_vars: Dict[str, str]) -> Dict[str, Any]:
    """Validate environment variables"""
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Required variables
    required_vars = [
        'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_NAME',
        'API_HOST', 'API_PORT', 'SECRET_KEY'
    ]
    
    for var in required_vars:
        if not env_vars.get(var):
            validation_results['errors'].append(f"Missing required variable: {var}")
            validation_results['valid'] = False
    
    # Validate database port
    try:
        db_port = int(env_vars.get('DB_PORT', '3306'))
        if not (1 <= db_port <= 65535):
            validation_results['errors'].append("DB_PORT must be between 1 and 65535")
            validation_results['valid'] = False
    except ValueError:
        validation_results['errors'].append("DB_PORT must be a valid integer")
        validation_results['valid'] = False
    
    # Validate API port
    try:
        api_port = int(env_vars.get('API_PORT', '8000'))
        if not (1 <= api_port <= 65535):
            validation_results['errors'].append("API_PORT must be between 1 and 65535")
            validation_results['valid'] = False
    except ValueError:
        validation_results['errors'].append("API_PORT must be a valid integer")
        validation_results['valid'] = False
    
    # Check for default/weak secret key
    secret_key = env_vars.get('SECRET_KEY', '')
    if 'change-in-production' in secret_key.lower() or len(secret_key) < 32:
        validation_results['warnings'].append("SECRET_KEY should be changed for production use")
    
    # Check YOLO device setting
    yolo_device = env_vars.get('YOLO_DEVICE', 'cpu').lower()
    if yolo_device not in ['cpu', 'cuda', 'mps']:
        validation_results['warnings'].append("YOLO_DEVICE should be 'cpu', 'cuda', or 'mps'")
    
    # Check file size limit
    try:
        max_file_size = int(env_vars.get('MAX_FILE_SIZE_MB', '50'))
        if max_file_size > 100:
            validation_results['warnings'].append("MAX_FILE_SIZE_MB is quite large, may cause memory issues")
    except ValueError:
        validation_results['errors'].append("MAX_FILE_SIZE_MB must be a valid integer")
        validation_results['valid'] = False
    
    return validation_results

def create_env_file():
    """Create .env file from template"""
    base_dir = Path(__file__).parent
    env_example_path = base_dir / '.env.example'
    env_path = base_dir / '.env'
    
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Keeping existing .env file.")
            return
    
    if env_example_path.exists():
        # Copy from example
        with open(env_example_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Prompt for key values
        print("\nConfiguring key environment variables...")
        
        db_password = input("Enter MySQL root password (leave empty if none): ")
        secret_key = input("Enter secret key (leave empty for default): ")
        
        if db_password:
            content = content.replace('DB_PASSWORD=', f'DB_PASSWORD={db_password}')
        
        if secret_key:
            content = content.replace('SECRET_KEY=agentv2-super-secret-key-change-in-production-2024', 
                                    f'SECRET_KEY={secret_key}')
        
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created .env file at {env_path}")
    else:
        print("❌ .env.example file not found!")
        return False
    
    return True

def main():
    """Main setup function"""
    print("=== AgentV2 Environment Setup ===\n")
    
    base_dir = Path(__file__).parent
    env_path = base_dir / '.env'
    
    # Check if .env exists
    if not env_path.exists():
        print("No .env file found.")
        create_new = input("Create new .env file? (Y/n): ")
        if create_new.lower() != 'n':
            if not create_env_file():
                sys.exit(1)
        else:
            print("Please create .env file manually using .env.example as template.")
            sys.exit(1)
    
    # Load and validate environment variables
    print("Loading environment variables...")
    env_vars = load_env_file(str(env_path))
    
    if not env_vars:
        print("❌ No environment variables found in .env file!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(env_vars)} environment variables")
    
    # Validate environment
    print("\nValidating environment configuration...")
    validation = validate_env_vars(env_vars)
    
    if validation['errors']:
        print("❌ Validation errors found:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['valid']:
        print("✅ Environment validation passed!")
        
        # Show key configuration
        print("\n=== Current Configuration ===")
        print(f"Database: {env_vars.get('DB_HOST')}:{env_vars.get('DB_PORT')}/{env_vars.get('DB_NAME')}")
        print(f"API Server: {env_vars.get('API_HOST')}:{env_vars.get('API_PORT')}")
        print(f"Debug Mode: {env_vars.get('DEBUG', 'false')}")
        print(f"YOLO Device: {env_vars.get('YOLO_DEVICE', 'cpu')}")
        print(f"Max File Size: {env_vars.get('MAX_FILE_SIZE_MB', '50')} MB")
        
        print("\n✅ Environment setup completed successfully!")
        print("Next steps:")
        print("1. Ensure MySQL is running")
        print("2. Run setup_database.bat to create database")
        print("3. Run setup.bat to install dependencies")
        print("4. Run run.bat to start the application")
    else:
        print("❌ Environment validation failed!")
        print("Please fix the errors above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
