@echo off
echo ===============================================
echo AgentV2 - Starting Application
echo ===============================================
echo.

echo Activating virtual environment...
if exist "agentv2_env\Scripts\activate.bat" (
    call agentv2_env\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Loading environment variables...
if exist ".env" (
    echo Environment file found.
) else (
    echo WARNING: .env file not found. Using default configuration.
)

echo.
echo Creating necessary directories...
python config.py

echo.
echo Checking database connection...
python -c "
try:
    from backend.db_manager import DatabaseManager
    db = DatabaseManager()
    db.test_connection()
    print('Database connection: OK')
except Exception as e:
    print(f'Database connection: FAILED - {e}')
    print('Please check your database configuration in .env file')
"

echo.
echo Starting AgentV2 backend server...
echo.
echo Available interfaces:
echo - User Interface: http://localhost:8000/user.html
echo - Admin Dashboard: http://localhost:8000/admin.html  
echo - Trainer Platform: http://localhost:8000/trainer.html
echo - API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python backend/main.py

pause
